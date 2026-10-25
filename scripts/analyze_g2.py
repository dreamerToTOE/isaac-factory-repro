"""CPU-only analysis of G2 evidence, with explicit missing-data denominators.

This script never imports Isaac Sim or loads checkpoints. TensorBoard extraction
can run in the existing learning Python while Matplotlib runs in system Python.
No package installation or modification of the training environment is required.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

try:
    from evaluation_reporting import wilson_interval
except ImportError:
    from scripts.evaluation_reporting import wilson_interval


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def descriptive(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None,
                "q25": None, "q75": None}
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "q25": quartiles[0], "q75": quartiles[2]}


def extract_scalars(directory: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    result = {}
    for tag in sorted(events.Tags()["scalars"]):
        result[tag] = [{"step": event.step, "wall_time": event.wall_time,
                        "value": event.value if math.isfinite(event.value) else None,
                        "finite": math.isfinite(event.value)} for event in events.Scalars(tag)]
    return result


def read_scalars(train_log: Path, tensorboard_python: Path | None, issues: dict) -> dict:
    directory = train_log / "summaries" if (train_log / "summaries").is_dir() else train_log
    event_files = sorted(directory.glob("events.out.tfevents.*")) if directory.is_dir() else []
    if not event_files:
        issues["warnings"].append("No TensorBoard event files found; scalar evidence is missing.")
        return {"directory": str(directory.resolve()), "files": [], "tags": {},
                "all_logged_scalars_finite": None}
    if tensorboard_python:
        command = [str(tensorboard_python), str(Path(__file__).resolve()),
                   "--export-scalars", str(directory.resolve())]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("TensorBoard extraction failed: " + result.stderr[-4000:])
        tags = json.loads(result.stdout)
    else:
        tags = extract_scalars(directory)
    bad = {tag: [v for v in values if not v["finite"]] for tag, values in tags.items()
           if any(not v["finite"] for v in values)}
    if bad:
        issues["errors"].append("TensorBoard contains nonfinite scalar values: " + ", ".join(bad))
    if not tags:
        issues["warnings"].append("TensorBoard event files contain no scalar tags.")
    return {"directory": str(directory.resolve()), "files": [identity(p) for p in event_files],
            "tags": tags, "all_logged_scalars_finite": not bad if tags else None,
            "nonfinite": bad}


def read_audit(path: Path | None, issues: dict) -> tuple[dict, list[dict]]:
    if path is None:
        issues["warnings"].append("No epoch audit provided; efficiency uses official TensorBoard timing if available.")
        return {}, []
    if not path.is_file():
        issues["warnings"].append("Epoch audit does not exist: " + str(path))
        return {}, []
    text = path.read_text()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            data = {"epochs": data}
    except json.JSONDecodeError:
        # Only complete JSONL lines are usable. A truncated final line is noted.
        records = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                issues["warnings"].append(f"Ignored incomplete audit JSONL line {number}: {path}")
        data = {"epochs": [x for x in records if "epoch" in x and
                           ("environment_transitions" in x or "official_total_seconds" in x)],
                "other_records": [x for x in records if "epoch" not in x]}
    if not isinstance(data, dict):
        raise ValueError("audit must be a JSON object, epoch list, or JSONL records")
    epochs = data.get("epochs", [])
    if not isinstance(epochs, list):
        raise ValueError("audit epochs must be a list")
    epochs_by_id = {}
    for record in epochs:
        epoch = int(record["epoch"])
        if epoch in epochs_by_id:
            issues["errors"].append(f"Duplicate epoch {epoch} in audit; an uninterrupted run cannot contain duplicate epochs.")
            continue
        epochs_by_id[epoch] = record
    return dict(data, input_identity=identity(path)), [epochs_by_id[k] for k in sorted(epochs_by_id)]


def read_monitor(directory: Path, issues: dict) -> tuple[dict, list[dict]]:
    path = directory / "gpu.csv"
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if not path.is_file():
        issues["warnings"].append("NVML CSV is missing: " + str(path))
        return {"sample_count": 0, "input_result": result}, []
    samples = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for number, row in enumerate(reader, 2):
            try:
                sample = {name: (float(value) if value and value.strip() else None)
                          for name, value in row.items() if name is not None}
            except ValueError:
                issues["warnings"].append(f"Ignored malformed/incomplete NVML CSV line {number}.")
                continue
            if not finite_number(sample.get("elapsed_seconds")) or not finite_number(sample.get("total_used_MiB")):
                issues["warnings"].append(f"Ignored NVML CSV line {number} without finite elapsed/used memory.")
                continue
            if any(value is not None and not finite_number(value) for value in sample.values()):
                issues["errors"].append(f"Nonfinite NVML reading at line {number}.")
                continue
            samples.append(sample)
    if any(b["elapsed_seconds"] <= a["elapsed_seconds"] for a, b in zip(samples, samples[1:])):
        issues["errors"].append("NVML sample time is not strictly increasing.")
    intervals = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])]
    summary = {"input_csv": identity(path), "input_result": result, "sample_count": len(samples),
               "median_actual_interval_seconds": statistics.median(intervals) if intervals else None,
               "max_actual_interval_seconds": max(intervals, default=None),
               "elapsed_seconds_first": samples[0]["elapsed_seconds"] if samples else None,
               "elapsed_seconds_last": samples[-1]["elapsed_seconds"] if samples else None,
               "peak_total_gpu_used_MiB": max((x["total_used_MiB"] for x in samples), default=None),
               "min_free_MiB": min((x["free_MiB"] for x in samples if finite_number(x.get("free_MiB"))), default=None),
               "peak_test_compute_used_MiB": max((x["test_compute_used_MiB"] for x in samples
                                                  if finite_number(x.get("test_compute_used_MiB"))), default=None),
               "sampled_maximum_note": "10 Hz samples can miss shorter peaks; total used includes desktop; driver reserved memory is separate."}
    if result.get("stop_reason"):
        issues["errors"].append("Monitor stopped workload: " + result["stop_reason"])
    if result.get("monitor_errors"):
        issues["errors"].append("NVML monitor reported errors.")
    if result.get("returncode") not in (None, 0):
        issues["errors"].append("Monitored process returned a nonzero status.")
    return summary, samples


def utc_timestamp(value: str) -> float:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("monitor timestamps must include a UTC offset")
    return parsed.timestamp()


def read_monitor_segments(path: Path, issues: dict) -> tuple[dict, list[dict]]:
    """Combine observed samples without treating an observer gap as coverage.

    Each CSV retains its own elapsed-time origin. Wall evidence is supplied
    independently; a successful replacement guard exit is not a child exit code.
    Coverage is the union of sampled spans, excluding gaps longer than three
    median intervals (with a 0.5 s floor), not proof of continuous observation.
    """
    manifest = json.loads(path.read_text())
    wall = manifest.get("wall_evidence", {})
    start = utc_timestamp(wall["start_utc"])
    end = utc_timestamp(wall["end_utc"]) if wall.get("end_utc") else None
    if end is not None and end < start:
        raise ValueError("monitor wall end precedes wall start")
    for field in ("start_uncertainty_seconds", "end_uncertainty_seconds"):
        value = wall.get(field, 0)
        if not finite_number(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    returncode = wall.get("returncode")
    if returncode is not None and (not isinstance(returncode, int) or isinstance(returncode, bool)):
        raise ValueError("wall returncode must be an integer or null")
    if returncode is None:
        issues["warnings"].append("Training child exit code is unavailable; observer exit status is not substituted.")
    elif returncode:
        issues["errors"].append("Training child has a reported nonzero exit status.")
    if end is None:
        issues["warnings"].append("Process disappearance time is unavailable; end-to-end wall duration is unknown.")
    segments, samples, spans, names = [], [], [], set()
    for index, segment in enumerate(manifest.get("segments", [])):
        name = segment.get("name", f"segment_{index + 1}")
        if name in names:
            raise ValueError("monitor segment names must be unique")
        names.add(name)
        origin = utc_timestamp(segment["elapsed_origin_utc"])
        uncertainty = segment.get("elapsed_origin_uncertainty_seconds", 0)
        if not finite_number(uncertainty) or uncertainty < 0:
            raise ValueError("elapsed origin uncertainty must be finite and nonnegative")
        directory = Path(segment["directory"])
        if not directory.is_absolute():
            directory = path.parent / directory
        summary, rows = read_monitor(directory, issues)
        summary.update({"name": name, "elapsed_origin_utc": segment["elapsed_origin_utc"],
                        "elapsed_origin_source": segment.get("elapsed_origin_source"),
                        "elapsed_origin_uncertainty_seconds": uncertainty})
        intervals = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(rows, rows[1:])]
        gap_threshold = max(.5, 3 * statistics.median(intervals)) if intervals else .5
        summary["coverage_gap_threshold_seconds"] = gap_threshold
        piece, previous, piece_start = 0, None, None
        for row in rows:
            absolute = origin + row["elapsed_seconds"]
            if previous is not None and absolute - previous > gap_threshold:
                spans.append((piece_start, previous, uncertainty))
                piece += 1
                piece_start = None
            if piece_start is None:
                piece_start = absolute
            samples.append(dict(row, segment_elapsed_seconds=row["elapsed_seconds"],
                                elapsed_seconds=absolute - start, unix_seconds=absolute,
                                monitor_segment=name, coverage_piece=f"{name}:{piece}"))
            previous = absolute
        if previous is not None:
            spans.append((piece_start, previous, uncertainty))
            summary["first_sample_utc"] = datetime.datetime.fromtimestamp(origin + rows[0]["elapsed_seconds"], datetime.timezone.utc).isoformat()
            summary["last_sample_utc"] = datetime.datetime.fromtimestamp(previous, datetime.timezone.utc).isoformat()
        segments.append(summary)
    if not segments:
        issues["warnings"].append("Monitor manifest has no segments.")
    samples.sort(key=lambda row: row["elapsed_seconds"])
    # Union only the spans containing observations. Do not interpolate across
    # segment boundaries or long gaps within one CSV.
    merged = []
    for lo, hi, uncertainty in sorted(spans):
        if end is not None:
            hi = min(hi, end)
        lo = max(lo, start)
        if hi < lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]), max(uncertainty, merged[-1][2]))
        else:
            merged.append((lo, hi, uncertainty))
    gaps, cursor = [], start
    previous_uncertainty = wall.get("start_uncertainty_seconds", 0)
    for lo, hi, uncertainty in merged:
        if lo > cursor:
            gaps.append({"start_elapsed_seconds": cursor - start, "end_elapsed_seconds": lo - start,
                         "duration_seconds": lo - cursor,
                         "duration_uncertainty_seconds": previous_uncertainty + uncertainty})
        cursor, previous_uncertainty = hi, uncertainty
    if end is not None and end > cursor:
        gaps.append({"start_elapsed_seconds": cursor - start, "end_elapsed_seconds": end - start,
                     "duration_seconds": end - cursor,
                     "duration_uncertainty_seconds": previous_uncertainty + wall.get("end_uncertainty_seconds", 0)})
    if gaps:
        issues["warnings"].append("GPU observation has uncovered intervals; reported extrema apply only to observed samples.")
    duration = end - start if end is not None else None
    coverage = sum(hi - lo for lo, hi, _ in merged)
    result = {"input_manifest": identity(path), "segments": segments,
              "wall_evidence": wall, "wall_duration_seconds": duration,
              "wall_duration_uncertainty_seconds": wall.get("start_uncertainty_seconds", 0) + wall.get("end_uncertainty_seconds", 0) if duration is not None else None,
              "training_child_returncode": returncode, "training_child_exit_code_available": returncode is not None,
              "sample_count": len(samples), "sampled_span_coverage_seconds": coverage,
              "sampled_span_coverage_fraction": coverage / duration if duration else None,
              "coverage_gaps": gaps,
              "elapsed_seconds_first": samples[0]["elapsed_seconds"] if samples else None,
              "elapsed_seconds_last": samples[-1]["elapsed_seconds"] if samples else None,
              "peak_total_gpu_used_MiB": max((r["total_used_MiB"] for r in samples), default=None),
              "min_free_MiB": min((r["free_MiB"] for r in samples if finite_number(r.get("free_MiB"))), default=None),
              "peak_test_compute_used_MiB": max((r["test_compute_used_MiB"] for r in samples if finite_number(r.get("test_compute_used_MiB"))), default=None),
              "extrema_scope": "observed_samples_only; missing intervals may contain larger used-memory or smaller free-memory values",
              "coverage_note": "Union of sampled spans, excluding segment gaps and within-segment intervals above the recorded threshold. Ordinary sampling intervals are included; continuous coverage is not asserted.",
              "sampled_maximum_note": "10 Hz samples can miss shorter peaks; total used includes desktop; driver reserved memory is separate.",
              "notes": manifest.get("notes", [])}
    return result, samples


def checkpoint_epoch(payload: dict, path: Path) -> int | None:
    identities = []
    for key in ("checkpoint_epoch", "epoch"):
        if payload.get(key) is not None:
            identities.append(int(payload[key]))
    checkpoint = str(payload.get("checkpoint", ""))
    for text in (checkpoint, str(path)):
        matches = re.findall(r"(?:epoch|ep)[_-]?(\d+)(?=[^0-9]|$)", text, re.I)
        if matches:
            identities.append(int(matches[-1]))
    if len(set(identities)) > 1:
        raise ValueError("checkpoint epoch disagrees between metadata, checkpoint basename, or evaluation path")
    return identities[0] if identities else None


def summarize_episodes(episodes: list[dict]) -> dict:
    n = len(episodes)
    ever = sum(e["ever_success"] for e in episodes)
    terminal = sum(e["terminal_success"] for e in episodes)
    first_steps = [e["first_success_step"] for e in episodes if e["ever_success"]]
    first_seconds = [e["first_success_time_seconds"] for e in episodes
                     if e["ever_success"] and e.get("first_success_time_seconds") is not None]
    return {"episodes_completed": n, "ever_success_count": ever,
            "ever_success_rate": ever / n if n else None, "ever_success_wilson95": wilson_interval(ever, n),
            "terminal_success_count": terminal, "terminal_success_rate": terminal / n if n else None,
            "terminal_success_wilson95": wilson_interval(terminal, n),
            "transient_only_count": sum(e["ever_success"] and not e["terminal_success"] for e in episodes),
            "first_success_step_conditional": descriptive(first_steps),
            "first_success_time_seconds_conditional": descriptive(first_seconds)}


def validate_episodes(payload: dict) -> list[dict]:
    seed = int(payload["seed"])
    episodes, identifiers = [], set()
    for item in payload["episodes"]:
        record = dict(item)
        for key in ("ever_success", "terminal_success"):
            if not isinstance(record.get(key), bool):
                raise ValueError(f"episode {key} must be a bool")
        if record["terminal_success"] and not record["ever_success"]:
            raise ValueError("terminal success without ever success")
        if int(record.get("seed", seed)) != seed:
            raise ValueError("episode seed differs from evaluation seed")
        key = (int(record["env_id"]), int(record["episode_id"]))
        if key in identifiers:
            raise ValueError("duplicate env_id/episode_id within evaluation")
        identifiers.add(key)
        first = record.get("first_success_step")
        if record["ever_success"]:
            if not finite_number(first) or first < 1 or first != int(first):
                raise ValueError("successful episode must have a positive integer first_success_step")
            if first > record["steps"]:
                raise ValueError("first success occurs after episode end")
        elif first is not None:
            raise ValueError("unsuccessful episode has first_success_step")
        seconds = record.get("first_success_time_seconds")
        if seconds is None and first is not None and finite_number(payload.get("step_dt_seconds")):
            seconds = first * payload["step_dt_seconds"]
        if seconds is not None and (not finite_number(seconds) or seconds < 0):
            raise ValueError("first success seconds must be finite/nonnegative or null")
        record["first_success_time_seconds"] = seconds
        record["seed"] = seed
        episodes.append(record)
    declared = payload.get("episodes_completed", (payload.get("aggregate") or {}).get("episodes_completed"))
    if declared is not None and int(declared) != len(episodes):
        raise ValueError("declared denominator differs from actual episode records")
    aggregate = payload.get("aggregate") or {}
    actual = summarize_episodes(episodes)
    for field in ("ever_success_count", "terminal_success_count"):
        if field in aggregate and aggregate[field] != actual[field]:
            raise ValueError(f"declared {field} differs from actual episode records")
    return episodes


def read_evaluations(directory: Path, expected_epochs: list[int], seeds: list[int], eval_envs: int,
                     final_episodes_per_env: int, issues: dict) -> dict:
    candidates = {}
    if not directory.is_dir():
        issues["warnings"].append("Evaluation directory missing: " + str(directory))
    for path in sorted(directory.rglob("*.json")) if directory.is_dir() else []:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            issues["warnings"].append("Ignored unreadable evaluation JSON: " + str(path))
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list) or "seed" not in payload:
            continue
        try:
            epoch = checkpoint_epoch(payload, path)
        except ValueError as error:
            issues["errors"].append(f"Invalid checkpoint identity {path}: {error}")
            continue
        if epoch is None:
            issues["warnings"].append("Evaluation has no identifiable checkpoint epoch: " + str(path))
            continue
        try:
            episodes = validate_episodes(payload)
        except (ValueError, KeyError, TypeError) as error:
            issues["errors"].append(f"Invalid evaluation {path}: {error}")
            continue
        if episodes and payload.get("deterministic") is not True:
            issues["errors"].append(f"Evaluation with completed episodes is not confirmed deterministic: {path}")
            continue
        if episodes and payload.get("official_success_capture") != "pre_reset_factory_get_curr_successes":
            issues["errors"].append(f"Evaluation does not confirm the independent pre-reset capture: {path}")
            continue
        key = (epoch, int(payload["seed"]))
        candidate = {"path": path, "payload": payload, "episodes": episodes}
        candidates.setdefault(key, []).append(candidate)

    selected = {}
    for key, options in candidates.items():
        # Complete output and its atomic progress snapshot contain the same
        # episodes. They must never be added as two independent evaluations.
        hashes = {x["payload"].get("checkpoint_sha256") for x in options if x["payload"].get("checkpoint_sha256")}
        if len(hashes) > 1:
            issues["errors"].append(f"Multiple checkpoint hashes for epoch/seed {key}; cannot pool them.")
            continue
        options.sort(key=lambda x: (x["payload"].get("status") == "complete", len(x["episodes"]),
                                    ".partial." not in x["path"].name), reverse=True)
        chosen = options[0]
        chosen_set = {(e["env_id"], e["episode_id"]): e for e in chosen["episodes"]}
        for other in options[1:]:
            for record in other["episodes"]:
                identity_key = (record["env_id"], record["episode_id"])
                if identity_key in chosen_set and record != chosen_set[identity_key]:
                    issues["errors"].append(f"Conflicting duplicate episode {identity_key} for epoch/seed {key}.")
        selected[key] = chosen

    unexpected_seeds = sorted({seed for _, seed in selected} - set(seeds))
    if unexpected_seeds:
        issues["warnings"].append("Unregistered evaluation seeds appear in per-seed rows but are excluded from the primary aggregate: " + str(unexpected_seeds))

    rows = []
    final_epoch = max(expected_epochs)
    for epoch in sorted(set(expected_epochs) | {k[0] for k in selected}):
        pooled, per_seed = [], []
        epoch_hashes = {value["payload"].get("checkpoint_sha256") for (ep, seed), value in selected.items()
                        if ep == epoch and seed in seeds and value["payload"].get("checkpoint_sha256")}
        if len(epoch_hashes) > 1:
            issues["errors"].append(f"Multiple checkpoint hashes across preregistered seeds at epoch {epoch}; aggregate cannot establish one checkpoint policy.")
        requested_per_seed = eval_envs * (final_episodes_per_env if epoch == final_epoch else 1)
        for seed in sorted(set(seeds) | {key[1] for key in selected if key[0] == epoch}):
            chosen = selected.get((epoch, seed))
            if chosen is None:
                per_seed.append({"seed": seed, "status": "missing", "episodes_requested": requested_per_seed,
                                 **summarize_episodes([])})
                continue
            payload = chosen["payload"]
            episodes = chosen["episodes"]
            if seed in seeds and len(epoch_hashes) <= 1:
                pooled.extend(episodes)
            per_seed.append({"seed": seed, "status": payload.get("status", "unknown"),
                             "episodes_requested": requested_per_seed,
                             "meets_preregistered_target": len(episodes) >= requested_per_seed,
                             "input_identity": identity(chosen["path"]),
                             "checkpoint": payload.get("checkpoint"),
                             "checkpoint_sha256": payload.get("checkpoint_sha256"),
                             "deterministic": payload.get("deterministic"),
                             "official_success_capture": payload.get("official_success_capture"),
                             "auto_reset_order_verified": payload.get("auto_reset_order_verified"),
                             **summarize_episodes(episodes)})
        rows.append({"checkpoint_epoch": epoch, "episodes_requested": requested_per_seed * len(seeds),
                     "complete_preregistered_seeds": all((epoch, s) in selected and
                                                        selected[(epoch, s)]["payload"].get("status") == "complete" and
                                                        len(selected[(epoch, s)]["episodes"]) >= requested_per_seed for s in seeds),
                     "per_seed": per_seed, "aggregate": summarize_episodes(pooled)})
    return {"expected_checkpoint_epochs": expected_epochs, "preregistered_seeds": seeds,
            "num_envs_per_seed": eval_envs,
            "final_episodes_per_env": final_episodes_per_env, "checkpoints": rows,
            "denominator_rule": "Only actual completed episode records from preregistered seeds in each selected evaluator output; unfinished episodes excluded; complete and partial duplicate outputs are not pooled. Additional seeds remain separate per-seed rows.",
            "first_success_rule": "Conditional on ever-successful completed episodes; failed episodes are excluded, never encoded as zero seconds or timeout times.",
            "interval_rule": "Two-sided 95% Wilson score intervals over episode counts. Descriptive only: vector environments and episodes share policy/seeds; intervals do not quantify training-seed variability."}


def scalar_values(tags: dict, name: str) -> list[float]:
    return [x["value"] for x in tags.get(name, []) if x["finite"]]


def training_summary(scalars: dict, audit: dict, epochs: list[dict], monitor: dict,
                     num_envs: int | None, horizon: int | None, issues: dict) -> dict:
    tags = scalars["tags"]
    scalar_epochs = scalar_values(tags, "info/epochs")
    last_epoch = max([int(x["epoch"]) for x in epochs] + [int(x) for x in scalar_epochs], default=None)
    runtime = audit.get("runtime", {})
    n = num_envs if num_envs is not None else runtime.get("num_actors")
    h = horizon if horizon is not None else runtime.get("horizon_length")
    record_ids = [int(x["epoch"]) for x in epochs]
    complete_epoch_sequence = bool(last_epoch is not None and record_ids == list(range(1, last_epoch + 1)))
    if epochs and not complete_epoch_sequence:
        issues["warnings"].append("Epoch audit is not a complete contiguous sequence from epoch 1.")
    official_seconds = [e["official_total_seconds"] for e in epochs if finite_number(e.get("official_total_seconds"))]
    env_seconds = [e["env_step_seconds"] for e in epochs if finite_number(e.get("env_step_seconds"))]
    transitions = [e["environment_transitions"] for e in epochs if finite_number(e.get("environment_transitions"))]
    source = "audit_sum" if complete_epoch_sequence and len(transitions) == len(epochs) else None
    total_transitions = sum(transitions) if source else None
    if total_transitions is None and n and h and last_epoch:
        total_transitions = n * h * last_epoch
        source = "derived_from_epoch_count_x_explicit_num_envs_x_horizon; verify uninterrupted completion separately"
    total_time = sum(official_seconds) if complete_epoch_sequence and len(official_seconds) == len(epochs) else None
    timing_source = "audit_official_total_seconds_sum" if total_time else None
    # Official PPO logs rollout (inference included) and optimization durations
    # once per epoch; their sum is the official PPO total-time denominator.
    rollout = scalar_values(tags, "performance/step_inference_time")
    optimization = scalar_values(tags, "performance/rl_update_time")
    if total_time is None and len(rollout) == len(optimization) == last_epoch:
        total_time = sum(rollout) + sum(optimization)
        timing_source = "sum_official_tensorboard_rollout_and_rl_update_times"
    env_total = sum(env_seconds) if complete_epoch_sequence and len(env_seconds) == len(epochs) else None
    if env_total is None:
        env_scalar_times = scalar_values(tags, "performance/step_time")
        if len(env_scalar_times) == last_epoch:
            env_total = sum(env_scalar_times)
    duration = monitor.get("wall_duration_seconds", monitor.get("input_result", {}).get("duration_seconds"))
    official_stable_times = [e["official_total_seconds"] for e in epochs if int(e["epoch"]) >= 11
                             and finite_number(e.get("official_total_seconds"))]
    stable_transitions = [e["environment_transitions"] for e in epochs if int(e["epoch"]) >= 11
                          and finite_number(e.get("official_total_seconds"))]
    stable_source = "epochs_11_onward_audit"
    if not official_stable_times and len(rollout) == len(optimization) and len(rollout) > 10 and n and h:
        official_stable_times = [r + o for r, o in zip(rollout[10:], optimization[10:])]
        stable_transitions = [n * h] * len(official_stable_times)
        stable_source = "official_tensorboard_epochs_11_onward"
    return {"last_observed_epoch": last_epoch, "audit_epoch_count": len(epochs),
            "audit_contiguous_from_epoch_1": complete_epoch_sequence,
            "num_envs": n, "horizon_length": h, "environment_transitions": total_transitions,
            "transition_count_source": source, "official_ppo_total_seconds": total_time,
            "official_ppo_timing_source": timing_source,
            "official_ppo_aggregate_fps": (total_transitions / total_time if total_transitions and total_time else None),
            "environment_step_aggregate_fps": (total_transitions / env_total if total_transitions and env_total else None),
            "launched_process_wall_seconds": duration,
            "launched_process_wall_uncertainty_seconds": monitor.get("wall_duration_uncertainty_seconds"),
            "wall_measurement_evidence": monitor.get("wall_evidence", monitor.get("input_result", {})),
            "end_to_end_transitions_per_wall_second": (total_transitions / duration if total_transitions and duration else None),
            "stable_epoch_range": ([11, last_epoch] if official_stable_times else None),
            "stable_timing_source": stable_source if official_stable_times else None,
            "stable_aggregate_total_fps": (sum(stable_transitions) / sum(official_stable_times) if official_stable_times else None),
            "stable_epoch_seconds": descriptive(official_stable_times),
            "all_logged_scalars_finite": scalars["all_logged_scalars_finite"],
            "scalar_tag_count": len(tags),
            "reward_summary": {name: descriptive(scalar_values(tags, name))
                               for name in ("rewards/iter", "shaped_rewards/iter")},
            "timing_note": "Total PPO time = rollout/inference + optimization. Process wall includes startup, initialization, checkpoint I/O, shutdown and monitoring overhead. Neither is the policy's simulated elapsed time.",
            "gradient_note": "Finite logged losses do not prove finite applied gradients. Report gradient/AMP observations separately from the observer evidence."}


def plots(output: Path, scalars: dict, epochs: list[dict], memory: list[dict], evaluations: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25,
                         "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})
    artifacts = []

    def save(fig, name):
        for suffix in ("png", "pdf"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=180)
            artifacts.append(path.name)
        plt.close(fig)

    def curve(ax, tag, label=None, xkey="step"):
        points = [x for x in scalars["tags"].get(tag, []) if x["finite"]]
        if points:
            ax.plot([x[xkey] for x in points], [x["value"] for x in points], label=label or tag)
        return bool(points)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
    for tag, label in (("rewards/iter", "Raw episodic reward"),
                       ("shaped_rewards/iter", "Shaped return (includes timeout value bootstrap)")):
        curve(axes[0], tag, label)
    curve(axes[1], "successes/iter", "Latest logged episode-end success (training)")
    if axes[0].lines:
        axes[0].legend(loc="best", fontsize=8)
    else:
        axes[0].text(.5, .5, "No logged episodic reward", transform=axes[0].transAxes, ha="center")
    axes[0].set(ylabel="Logged episodic sum", title="Official training scalars; reward is not insertion success")
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    axes[1].set(xlabel="PPO epoch (official /iter step)", ylabel="Success fraction", ylim=(-.02, 1.02))
    save(fig, "training_reward")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    groups = [(axes[0, 0], ("losses/a_loss",), "Actor PPO loss"),
              (axes[0, 1], ("losses/cval_loss", "losses/c_loss"), "Value losses"),
              (axes[0, 2], ("losses/entropy",), "Policy entropy"),
              (axes[1, 0], ("info/kl",), "Policy KL divergence"),
              (axes[1, 1], ("info/last_lr", "info/cval_lr"), "Adaptive learning rates"),
              (axes[1, 2], ("losses/bounds_loss",), "Action bounds loss")]
    for ax, names, title in groups:
        for tag in names:
            curve(ax, tag)
        ax.set(title=title, xlabel="Official TensorBoard step")
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "Missing", transform=ax.transAxes, ha="center")
    fig.suptitle("Raw logged scalars; central-value and actor step offsets retained", fontsize=11)
    fig.tight_layout()
    save(fig, "training_losses")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5))
    if epochs:
        for key, label in (("total_fps", "PPO total"), ("env_step_fps", "Environment stepping"),
                           ("rollout_fps", "Rollout + inference")):
            rows = [r for r in epochs if finite_number(r.get(key))]
            if rows:
                axes[0].plot([r["epoch"] for r in rows], [r[key] for r in rows], label=label)
        axes[0].set_xlabel("PPO epoch")
    else:
        for tag in ("performance/step_inference_rl_update_fps", "performance/step_fps"):
            curve(axes[0], tag)
        axes[0].set_xlabel("Official TensorBoard step")
    axes[0].set(ylabel="Environment transitions / s", title="Throughput (different timing denominators)")
    if axes[0].lines:
        axes[0].legend(fontsize=8)
    if memory:
        # One envelope per time bin retains memory extrema while avoiding huge
        # vector PDFs. Exact maxima/minima remain in summary.json from all rows.
        pieces = {}
        span = max(1.0, memory[-1]["elapsed_seconds"] / 1500)
        for row in memory:
            bins = pieces.setdefault(row.get("coverage_piece", "original"), {})
            bins.setdefault(int(row["elapsed_seconds"] / span), []).append(row)
        for index, bins in enumerate(pieces.values()):
            mid, low, high, free_low = [], [], [], []
            for rows in bins.values():
                mid.append(statistics.mean(x["elapsed_seconds"] for x in rows) / 3600)
                low.append(min(x["total_used_MiB"] for x in rows))
                high.append(max(x["total_used_MiB"] for x in rows))
                free = [x["free_MiB"] for x in rows if finite_number(x.get("free_MiB"))]
                free_low.append(min(free) if free else math.nan)
            axes[1].fill_between(mid, low, high, color="tab:blue", alpha=.5,
                                 label="Whole-GPU observed used min/max envelope" if index == 0 else None)
            axes[1].plot(mid, free_low, color="tab:green", linewidth=.8,
                         label="Observed free-memory minimum per bin" if index == 0 else None)
        axes[1].axhline(1024, color="tab:red", linestyle="--", linewidth=.8, label="Free-memory guard = 1024 MiB")
        axes[1].legend(fontsize=7)
    axes[1].set(xlabel="Launched-process elapsed time (hours)", ylabel="MiB", title="NVML sampled memory; desktop included; gaps are unobserved")
    fig.tight_layout()
    save(fig, "throughput_memory")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = [r for r in evaluations["checkpoints"] if r["aggregate"]["episodes_completed"]]
    for key, offset, label, color in (("ever_success", -.8, "Ever-success", "tab:blue"),
                                      ("terminal_success", .8, "Terminal-success", "tab:orange")):
        if rows:
            x = [r["checkpoint_epoch"] + offset for r in rows]
            y = [r["aggregate"][key + "_rate"] for r in rows]
            lo = [a - r["aggregate"][key + "_wilson95"][0] for a, r in zip(y, rows)]
            hi = [r["aggregate"][key + "_wilson95"][1] - a for a, r in zip(y, rows)]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=3, label=label, color=color)
    for row in evaluations["checkpoints"]:
        count = row["aggregate"]["episodes_completed"]
        ax.text(row["checkpoint_epoch"], 1.035, f"n={count}" if count else "missing", ha="center", fontsize=8)
    ax.set(xlabel="Checkpoint epoch", ylabel="Completed-episode success fraction", ylim=(-.04, 1.12),
           title="Deterministic evaluation; episode-binomial Wilson 95% intervals")
    ax.set_xticks(evaluations["expected_checkpoint_epochs"])
    if rows:
        ax.legend(loc="lower right", fontsize=8)
    save(fig, "checkpoint_success")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, field, title in ((axes[0], "first_success_step_conditional", "First-success policy step"),
                             (axes[1], "first_success_time_seconds_conditional", "First-success simulated seconds")):
        rows = [r for r in evaluations["checkpoints"] if r["aggregate"][field]["n"]]
        if rows:
            ax.errorbar([r["checkpoint_epoch"] for r in rows], [r["aggregate"][field]["median"] for r in rows],
                        yerr=[[r["aggregate"][field]["median"] - r["aggregate"][field]["q25"] for r in rows],
                              [r["aggregate"][field]["q75"] - r["aggregate"][field]["median"] for r in rows]],
                        marker="o", capsize=4, label="Aggregate median and IQR")
            for seed in evaluations["preregistered_seeds"]:
                selected = [(r, next((p for p in r["per_seed"] if p["seed"] == seed), None)) for r in rows]
                selected = [(r, p) for r, p in selected if p and p[field]["n"]]
                if selected:
                    ax.plot([r["checkpoint_epoch"] for r, p in selected], [p[field]["median"] for r, p in selected],
                            "--", linewidth=.8, marker=".", label=f"Seed {seed} median")
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "No successful completed episodes\nConditional time is undefined", transform=ax.transAxes, ha="center")
        ax.set(title=title, xlabel="Checkpoint epoch")
        ax.set_xticks(evaluations["expected_checkpoint_epochs"])
    fig.suptitle("Conditional on ever-success; failed episodes are excluded", fontsize=11)
    fig.tight_layout()
    save(fig, "first_success_conditional")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path, required=True)
    monitor_input = parser.add_mutually_exclusive_group(required=True)
    monitor_input.add_argument("--monitor", type=Path)
    monitor_input.add_argument("--monitor-manifest", type=Path,
                               help="Optional explicit segment UTC origins and independent wall/exit evidence.")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--tensorboard-python", type=Path,
                        help="Existing Python with TensorBoard; current Python needs Matplotlib only.")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--horizon-length", type=int)
    parser.add_argument("--expected-epochs", default="50,100,150,200")
    parser.add_argument("--eval-seeds", default="1001,1002,1003")
    parser.add_argument("--eval-num-envs", type=int, default=32)
    parser.add_argument("--final-episodes-per-env", type=int, default=3)
    args = parser.parse_args()
    expected_epochs = sorted(set(int(x) for x in args.expected_epochs.split(",")))
    seeds = sorted(set(int(x) for x in args.eval_seeds.split(",")))
    if not expected_epochs or not seeds or min(expected_epochs) < 1 or args.eval_num_envs < 1 or args.final_episodes_per_env < 1:
        parser.error("positive checkpoint epochs, evaluation environments and episode counts required")
    if args.output.exists():
        parser.error("--output exists; use a new directory to preserve previous analysis")
    args.output.mkdir(parents=True)
    issues = {"errors": [], "warnings": []}
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    scalars = read_scalars(args.train_log, args.tensorboard_python, issues)
    audit_path = args.audit
    if audit_path is None:
        audit_candidates = [args.train_log / "training_audit.json"]
        if args.monitor is not None:
            audit_candidates.insert(0, args.monitor / "training_audit.json")
        for path in audit_candidates:
            if path.is_file():
                audit_path = path
                break
    audit, epoch_rows = read_audit(audit_path, issues)
    monitor, memory_rows = (read_monitor_segments(args.monitor_manifest, issues) if args.monitor_manifest
                            else read_monitor(args.monitor, issues))
    evaluations = read_evaluations(args.eval_root, expected_epochs, seeds, args.eval_num_envs,
                                   args.final_episodes_per_env, issues)
    train = training_summary(scalars, audit, epoch_rows, monitor, args.num_envs, args.horizon_length, issues)
    write_json(args.output / "tensorboard_scalars.json", scalars)
    write_json(args.output / "evaluation_summary.json", evaluations)
    summary = {"schema_version": 1, "training": train, "monitor": monitor,
               "evaluation": evaluations, "issues": issues,
               "manifest_identity": identity(args.manifest) if args.manifest else None,
               "manifest_recorded_without_assuming_schema": manifest,
               "analysis_script_identity": identity(Path(__file__)),
               "audit_identity": audit.get("input_identity"),
               "audit_runtime": audit.get("runtime"),
               "audit_optimizer_steps": audit.get("optimizer_steps"),
               "audit_instrumentation_errors": audit.get("instrumentation_errors"),
               "audit_exit_error": audit.get("exit_error"),
               "scientific_limits": ["One training seed; evaluation seeds do not replicate training.",
                                     "No claim of insertion quality from reward or finite losses.",
                                     "No zeros are inserted for missing evaluations.",
                                     "Checkpoint evaluation is post hoc; no G2 tuning is performed."]}
    artifacts = plots(args.output, scalars, epoch_rows, memory_rows, evaluations)
    summary["plots"] = artifacts
    write_json(args.output / "summary.json", summary)
    with (args.output / "evaluation_per_seed.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint_epoch", "seed", "status", "episodes_completed", "episodes_requested",
                         "ever_success_count", "ever_success_rate", "terminal_success_count", "terminal_success_rate",
                         "conditional_first_success_n", "conditional_first_success_step_mean", "conditional_first_success_seconds_mean"])
        for row in evaluations["checkpoints"]:
            for seed in row["per_seed"]:
                writer.writerow([row["checkpoint_epoch"], seed["seed"], seed["status"], seed["episodes_completed"],
                                 seed["episodes_requested"], seed["ever_success_count"], seed["ever_success_rate"],
                                 seed["terminal_success_count"], seed["terminal_success_rate"],
                                 seed["first_success_step_conditional"]["n"], seed["first_success_step_conditional"]["mean"],
                                 seed["first_success_time_seconds_conditional"]["mean"]])
    print(json.dumps({"output": str(args.output.resolve()), "errors": issues["errors"],
                      "warnings": issues["warnings"], "training_epoch": train["last_observed_epoch"],
                      "evaluation_denominators": {r["checkpoint_epoch"]: r["aggregate"]["episodes_completed"]
                                                  for r in evaluations["checkpoints"]}}, sort_keys=True))
    if issues["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    # Kept private so the extraction subprocess does not import Matplotlib or
    # require all analysis CLI arguments in the learning environment.
    if len(sys.argv) == 3 and sys.argv[1] == "--export-scalars":
        print(json.dumps(extract_scalars(Path(sys.argv[2])), allow_nan=False))
    else:
        main()
