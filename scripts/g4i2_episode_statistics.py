"""Streaming CPU-only per-episode statistics from actual G4-I2 raw traces.

A raw file is scanned once per seed. Minimal single-environment action/controller
rows are spooled to temporary disk, while batch timing rows are retained once per
policy step. Only one episode's spool is loaded for each summary call. The raw
observations, controller inputs and official success masks remain in the external
trace; failure references contain byte/line bounds, SHA256, env and step filters.
No inference, GPU operation or simulator import is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from scripts.g4i2_analysis import (classify_episode, statistics, summarize_actions,
                                 summarize_controller, summarize_episodes, summarize_timings)

_ACTION_FIELDS = dict(raw_mu=6, clipped=6, ema=6, physical_position_increment_m=3,
                      physical_rotation_increment_rad=3)
_CONTROLLER_FIELDS = dict(target_position=3, target_quaternion=4, requested_torque=7,
                          software_torque=7, effort_limits=7)
_OPTIONAL_FIELDS = dict(task_space_pd_output=6, position_constraint_active=3,
                        orientation_constraint_active=3, applied_sim_effort=7)


def _read_episodes(path):
    value = json.loads(Path(path).read_text())
    value = value if isinstance(value, list) else value["episodes"]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("episodes input must be a list or {episodes: list}")
    return value


def _select_matrix(value, env_id, width, label):
    if not isinstance(value, list) or env_id >= len(value) or not isinstance(value[env_id], list) or len(value[env_id]) != width:
        raise ValueError(f"{label}: missing env_id {env_id} or width != {width}")
    if any(not isinstance(item,(int,float,bool)) or not math.isfinite(item) for item in value[env_id]):
        raise ValueError(f"{label}: selected env row must contain finite numeric values")
    return [value[env_id]]


def _single_env(row, env_id):
    required = _ACTION_FIELDS if row["kind"] == "action" else _CONTROLLER_FIELDS
    result = {"kind": row["kind"], "policy_step": row["policy_step"]}
    if "substep" in row:
        result["substep"] = row["substep"]
    for key, width in required.items():
        result[key] = _select_matrix(row[key], env_id, width, key)
    if row["kind"] == "controller":
        for key, width in _OPTIONAL_FIELDS.items():
            if key in row:
                result[key] = (_select_matrix(row[key], env_id, width, key)
                               if isinstance(row[key], list) else row[key])
    return result


def _positive_int(value, label, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{label} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _specifications(episodes):
    identities = set()
    specs = []
    for ordinal, episode in enumerate(episodes):
        classify_episode(episode)
        env_id = _positive_int(episode["env_id"], "env_id", zero=True)
        episode_id = _positive_int(episode["episode_id"], "episode_id", zero=True)
        seed = episode.get("evaluation_seed", episode.get("seed", "UNKNOWN"))
        identity = (seed, env_id, episode_id)
        if identity in identities:
            raise ValueError("duplicate episode identity")
        identities.add(identity)
        first = _positive_int(episode["trace"]["first_policy_step"], "first_policy_step")
        last = _positive_int(episode["trace"]["last_policy_step"], "last_policy_step")
        if last < first or episode.get("steps", last-first+1) != last-first+1:
            raise ValueError("episode steps disagree with its inclusive trace range")
        specs.append(dict(ordinal=ordinal, episode=episode, env_id=env_id, first=first, last=last,
                          counts=defaultdict(Counter), controller_substeps=defaultdict(set), success={},
                          start_byte=None, end_byte=None, start_line=None, end_line=None,
                          observation_steps=set(), normalized_steps=set(), errors=[]))
    for left in specs:
        for right in specs[left["ordinal"]+1:]:
            if left["env_id"] == right["env_id"] and max(left["first"],right["first"]) <= min(left["last"],right["last"]):
                raise ValueError("episode ranges overlap for the same environment")
    return specs


def _device_statistics(rows):
    values = defaultdict(list)
    for row in rows:
        for section in row.get("event_sections", []):
            if section.get("succeeded", True) and section.get("device_ms") is not None:
                values[section["stage"]].append(section["device_ms"])
    return {key: statistics(value) for key,value in sorted(values.items())}


def _coverage(spec, timings, expected_controller_substeps):
    expected = list(range(spec["first"], spec["last"]+1))
    errors = list(spec["errors"])
    missing = {}
    for kind in ("action", "controller", "success_state"):
        missing[kind] = [step for step in expected if spec["counts"][kind][step] == 0]
        if missing[kind]: errors.append(f"MISSING_{kind.upper()}_STEPS:{missing[kind]}")
    missing["timing"] = [step for step in expected if step not in timings]
    if missing["timing"]: errors.append(f"MISSING_TIMING_STEPS:{missing['timing']}")
    for kind in ("action", "success_state"):
        duplicated = [step for step in expected if spec["counts"][kind][step] > 1]
        if duplicated: errors.append(f"DUPLICATE_{kind.upper()}_STEPS:{duplicated}")
    if expected_controller_substeps is not None:
        bad = {step:spec["counts"]["controller"][step] for step in expected
               if spec["counts"]["controller"][step] != expected_controller_substeps}
        if bad: errors.append(f"CONTROLLER_SUBSTEP_COUNTS:{bad}")
    if any(len(spec["controller_substeps"][step]) != spec["counts"]["controller"][step] for step in expected):
        errors.append("DUPLICATE_OR_MISSING_CONTROLLER_SUBSTEP_IDENTITIES")
    for key in ("observation", "normalized"):
        omitted = [step for step in expected if step not in spec[key+"_steps"]]
        if omitted: errors.append(f"MISSING_FULL_{key.upper()}_STEPS:{omitted}")
    success_rows = [spec["success"][step] for step in expected if step in spec["success"]]
    if len(success_rows) == len(expected):
        flags = [row["success"] for row in success_rows]
        original = spec["episode"]
        if any(flags) != original["ever_success"] or flags[-1] != original["terminal_success"]:
            errors.append("RAW_SUCCESS_FLAGS_DISAGREE_WITH_EPISODE")
        first_success = next((index+1 for index,flag in enumerate(flags) if flag), None)
        if "first_success_step" in original and original["first_success_step"] != first_success:
            errors.append("RAW_FIRST_SUCCESS_STEP_DISAGREES_WITH_EPISODE")
        if [row["step_number"] for row in success_rows] != list(range(1,len(expected)+1)):
            errors.append("RAW_EPISODE_STEP_COUNTERS_DISAGREE")
    return dict(status="COMPLETE" if not errors else "INCOMPLETE", errors=errors,
                expected_policy_steps=len(expected), first_policy_step=spec["first"],last_policy_step=spec["last"],
                action_records=sum(spec["counts"]["action"].values()),
                controller_records=sum(spec["counts"]["controller"].values()),
                success_state_records=sum(spec["counts"]["success_state"].values()),
                timing_records=sum(step in timings for step in expected),
                missing_policy_steps=missing, expected_controller_substeps=expected_controller_substeps,
                full_observation_and_normalized_width=19)


def _write_json_exclusive(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv(episodes, path):
    columns = ("evaluation_seed", "env_id", "episode_id", "steps", "ever_success", "terminal_success",
               "first_success_time_seconds", "termination_reason", "failure_category", "statistics_coverage",
               "action_statistics", "controller_statistics", "timing_statistics", "gpu_event_statistics",
               "initial_randomized_configuration", "pre_reset_terminal_state", "raw_trace_reference")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for episode in episodes:
            stats = episode["statistics"]
            row = {key:episode.get(key, "UNKNOWN") for key in columns[:8]}
            row.update(failure_category=classify_episode(episode)["category"],
                       statistics_coverage=stats["coverage"]["status"])
            for output_key, key in (("action_statistics","actions"),("controller_statistics","controller"),
                                    ("timing_statistics","timing"),("gpu_event_statistics","device_sections_ms")):
                row[output_key] = json.dumps(stats[key],allow_nan=False)
            for key in ("initial_randomized_configuration","pre_reset_terminal_state","raw_trace_reference"):
                row[key] = json.dumps(episode.get(key,"UNKNOWN"),allow_nan=False)
            writer.writerow(row)


def process_seed(trace_path, episodes_path, output_dir, *, expected_controller_substeps=8,
                 scratch_dir=None, strict=True):
    """Scan one seed and exclusively create enriched episode JSON/CSV + coverage.

    Warmup flags are retained in raw evidence but explicitly disabled when calling
    summarize_timings for scientific episode statistics. Every timing row belongs
    to its batch decision once per episode, never once per controller substep.
    Failure evidence remains an external immutable trace reference, not an
    invented reproduction or a cropped simulation trajectory.
    """
    trace_path, episodes_path, output_dir = Path(trace_path), Path(episodes_path), Path(output_dir)
    episodes = _read_episodes(episodes_path)
    if not episodes: raise ValueError("at least one completed episode is required")
    specs = _specifications(episodes)
    if expected_controller_substeps is not None:
        _positive_int(expected_controller_substeps,"expected_controller_substeps")
    if output_dir.exists(): raise FileExistsError(output_dir)
    by_env = defaultdict(list)
    for spec in specs: by_env[spec["env_id"]].append(spec)
    timings = {}
    digest = hashlib.sha256()
    total_lines = 0
    enriched = []
    with tempfile.TemporaryDirectory(prefix="g4i2_episode_statistics_",dir=scratch_dir) as temporary:
        handles = {}
        try:
            with trace_path.open("rb") as stream:
                while True:
                    start_byte = stream.tell()
                    line = stream.readline()
                    if not line: break
                    digest.update(line)
                    total_lines += 1
                    if not line.strip(): continue
                    try: row = json.loads(line)
                    except (ValueError,UnicodeDecodeError) as exc:
                        raise ValueError(f"invalid JSON at {trace_path}:{total_lines}") from exc
                    kind = row.get("kind")
                    if kind not in ("action","controller","timing","success_state"): continue
                    step = _positive_int(row["policy_step"], "policy_step")
                    if kind == "timing":
                        if step in timings: raise ValueError(f"duplicate batch timing for policy step {step}")
                        timings[step] = row
                    for env_id, candidates in by_env.items():
                        for spec in candidates:
                            if not spec["first"] <= step <= spec["last"]: continue
                            if spec["start_byte"] is None:
                                spec["start_byte"],spec["start_line"] = start_byte,total_lines
                            spec["end_byte"],spec["end_line"] = stream.tell(),total_lines
                            if kind == "timing": continue
                            spec["counts"][kind][step] += 1
                            if kind == "success_state":
                                capture = row["capture"]
                                success = capture["success"][env_id]
                                if type(success) is not bool: raise ValueError("official success mask must be boolean")
                                spec["success"][step] = dict(success=success,step_number=capture["step_numbers"][env_id])
                                continue
                            selected = _single_env(row,env_id)
                            if kind == "action":
                                for key in ("observation","normalized"):
                                    if key in row:
                                        _select_matrix(row[key],env_id,19,key)
                                        spec[key+"_steps"].add(step)
                            else:
                                spec["controller_substeps"][step].add(row.get("substep"))
                            ordinal = spec["ordinal"]
                            if ordinal not in handles:
                                handles[ordinal] = (Path(temporary)/f"{ordinal}.jsonl").open("w")
                            handles[ordinal].write(json.dumps(selected,allow_nan=False)+"\n")
        finally:
            for handle in handles.values(): handle.close()
        trace_hash = digest.hexdigest()
        for spec in specs:
            spool_path = Path(temporary)/f"{spec['ordinal']}.jsonl"
            records = ([json.loads(line) for line in spool_path.read_text().splitlines()] if spool_path.exists() else [])
            episode_timings = [{**timings[step],"warmup":False} for step in range(spec["first"],spec["last"]+1) if step in timings]
            coverage = _coverage(spec,timings,expected_controller_substeps)
            if strict and coverage["errors"]:
                raise ValueError(f"incomplete episode env={spec['env_id']} episode={spec['episode']['episode_id']}: " + "; ".join(coverage["errors"]))
            statistic = dict(actions=summarize_actions([row for row in records if row["kind"] == "action"]),
                             controller=summarize_controller([row for row in records if row["kind"] == "controller"]),
                             timing=summarize_timings(episode_timings), device_sections_ms=_device_statistics(episode_timings),
                             coverage=coverage,
                             warmup_policy="included in scientific episode statistics; raw flag remains unchanged",
                             original_warmup_steps=sum(bool(timings[step].get("warmup",False)) for step in range(spec["first"],spec["last"]+1) if step in timings),
                             timing_scope="one measured batch timing per policy step; no division by env count and no per-controller duplication; command intervals restricted to this episode")
            reference = dict(path=str(trace_path.resolve()),sha256=trace_hash,
                             byte_range_start_inclusive=spec["start_byte"],byte_range_end_exclusive=spec["end_byte"],
                             line_range_start_inclusive=spec["start_line"],line_range_end_inclusive=spec["end_line"],
                             env_id=spec["env_id"],first_policy_step=spec["first"],last_policy_step=spec["last"],
                             filter="select the indicated env row in each action/controller/success_state; retain timing once per batch step",
                             retained_evidence="complete original observation, normalized observation, action chain, controller diagnostics and official pre-reset success state",
                             coverage_status=coverage["status"],failure_episode=not spec["episode"]["terminal_success"])
            enriched.append(dict(spec["episode"],statistics=statistic,raw_trace_reference=reference))
    result = dict(schema="g4i2_episode_statistics_v1",source_trace=str(trace_path.resolve()),source_trace_sha256=digest.hexdigest(),
                  source_trace_lines=total_lines,source_episodes=str(episodes_path.resolve()),
                  source_episodes_sha256=hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
                  episodes=enriched,episode_count=len(enriched),
                  failures=sum(not episode["terminal_success"] for episode in enriched),
                  coverage_status="COMPLETE" if all(episode["statistics"]["coverage"]["status"] == "COMPLETE" for episode in enriched) else "INCOMPLETE",
                  processing="one raw seed file scanned at a time; reduced per-env disk spools; one episode loaded for summary",
                  failure_trace_policy="original external trace retained and hashed; no synthetic or re-executed failures")
    output_dir.mkdir(parents=True,exist_ok=False)
    _write_json_exclusive(output_dir/"episodes_with_statistics.json",result)
    _write_csv(enriched,output_dir/"episodes_with_statistics.csv")
    _write_json_exclusive(output_dir/"coverage.json",dict(status=result["coverage_status"],episodes=[dict(env_id=episode["env_id"],episode_id=episode["episode_id"],**episode["statistics"]["coverage"]) for episode in enriched]))
    return result


def process_runs(run_dirs, output_dir, **kwargs):
    """Process seed directories sequentially and combine enriched episode rows."""
    output_dir = Path(output_dir)
    run_dirs = list(run_dirs)
    if not run_dirs: raise ValueError("at least one run directory is required")
    output_dir.mkdir(parents=True,exist_ok=False)
    episodes, sources = [], []
    identities = set()
    for index,run_dir in enumerate(run_dirs):
        run_dir = Path(run_dir)
        result = process_seed(run_dir/"i2_trace.jsonl",run_dir/"episodes.json",output_dir/f"seed_{index:02d}",**kwargs)
        for episode in result["episodes"]:
            identity = (episode.get("evaluation_seed",episode.get("seed","UNKNOWN")),episode["env_id"],episode["episode_id"])
            if identity in identities: raise ValueError("duplicate episode identity across input runs")
            identities.add(identity)
        episodes.extend(result["episodes"])
        sources.append({key:result[key] for key in ("source_trace","source_trace_sha256","source_episodes_sha256","coverage_status","episode_count")})
    aggregate = summarize_episodes(episodes)
    aggregate.pop("episodes")
    combined = dict(schema="g4i2_episode_statistics_multiseed_v1",sources=sources,episodes=episodes,
                    aggregate=aggregate,processing="sequential per-seed streaming; raw traces never jointly loaded")
    _write_json_exclusive(output_dir/"episodes_with_statistics.json",combined)
    _write_csv(episodes,output_dir/"episodes_with_statistics.csv")
    return combined


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,action="append",required=True,help="actual runtime directory containing i2_trace.jsonl and episodes.json; repeat for seeds")
    parser.add_argument("--output",type=Path,required=True,help="new exclusive output directory")
    parser.add_argument("--scratch-dir",type=Path)
    parser.add_argument("--expected-controller-substeps",type=int,default=8)
    parser.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args()
    result=process_runs(args.run_dir,args.output,expected_controller_substeps=args.expected_controller_substeps,
                        scratch_dir=args.scratch_dir,strict=not args.allow_incomplete)
    print(json.dumps({"episodes":len(result["episodes"]),"output":str(args.output),"aggregate":{key:value for key,value in result["aggregate"].items() if key != "episodes"}},allow_nan=False))


if __name__ == "__main__": main()
