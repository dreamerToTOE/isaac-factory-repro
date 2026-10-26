"""CPU-only analysis of G4-I2 records; never imports Isaac, CUDA or a policy.

All numerical inputs must be finite. Warmup is excluded from performance
summaries but retained in the source JSONL. Percentiles use NumPy's linear
interpolation and standard deviation is the population statistic (ddof=0).
Unknown diagnostics remain UNKNOWN; requested effort is never renamed applied.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.evaluation_reporting import atomic_write_json, wilson_interval

POLICY_INTERVAL_MS = 1000.0 / 15.0
ACTION_DIMENSIONS = ("translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z")


def statistics(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("statistics require one-dimensional finite values")
    if not len(array):
        return dict(n=0, mean=None, p50=None, p95=None, p99=None, min=None, max=None, std=None)
    return dict(n=len(array), mean=float(np.mean(array)), p50=float(np.percentile(array, 50)),
                p95=float(np.percentile(array, 95)), p99=float(np.percentile(array, 99)),
                min=float(np.min(array)), max=float(np.max(array)), std=float(np.std(array, ddof=0)))


def _number(value, label, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric, not boolean")
    value = float(value)
    if not np.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _matrix(value, width, label):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must have finite shape [N,{width}], N>=1")
    return array


def _stack(records, key, width):
    return np.concatenate([_matrix(record[key], width, key) for record in records], axis=0)


def _dimension_statistics(array):
    return [statistics(array[:, index]) for index in range(array.shape[1])]


def summarize_timings(records: list[dict], deadline_ms=POLICY_INTERVAL_MS) -> dict:
    """Summarize measured rows, preserving timer names and actual denominators.

    ``wall_command_monotonic_ns`` is a timestamp at the actual software command
    boundary, not a neural-network completion timestamp. ``control_loop_ms`` is
    the measured whole-loop wall time and ``sim_dt_s``/``simdt`` its simulated
    advancement. Neither frequency nor real-time factor is derived from forward
    latency. A missing clock remains unknown. Record order must be chronological.
    """
    deadline_ms = _number(deadline_ms, "deadline_ms", nonnegative=True)
    if deadline_ms == 0:
        raise ValueError("deadline_ms must be positive")
    selected = []
    for record in records:
        warmup = record.get("warmup", False)
        if type(warmup) is not bool:
            raise ValueError("warmup must be boolean")
        times = record.get("timings_ms", {})
        if not isinstance(times, dict):
            raise ValueError("timings_ms must be a mapping")
        for name, value in times.items():
            _number(value, f"timings_ms.{name}", nonnegative=True)
        if not warmup:
            selected.append(record)
    names = sorted({name for record in selected for name in record.get("timings_ms", {})})
    stages = {name: statistics(record["timings_ms"][name] for record in selected if name in record.get("timings_ms", {}))
              for name in names}
    commands = [record.get("wall_command_monotonic_ns") for record in selected]
    available_commands = [value for value in commands if value is not None]
    for value in available_commands:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("command timestamps must be nonnegative integer monotonic nanoseconds")
    if any(right <= left for left, right in zip(available_commands, available_commands[1:])):
        raise ValueError("command timestamps must be strictly increasing")
    # Do not bridge missing timestamp records: that would merge two intervals.
    intervals = [(right-left)/1e6 for left, right in zip(commands, commands[1:]) if left is not None and right is not None]
    loops = [_number(record["control_loop_ms"], "control_loop_ms", nonnegative=True)
             for record in selected if record.get("control_loop_ms") is not None]
    simulated = []
    aligned_loops = []
    for record in selected:
        dt = record.get("sim_dt_s", record.get("simdt"))
        if dt is not None and record.get("control_loop_ms") is not None:
            simulated.append(_number(dt, "sim_dt_s", nonnegative=True))
            aligned_loops.append(_number(record["control_loop_ms"], "control_loop_ms", nonnegative=True))
    interval_stats = statistics(intervals)
    def misses(values):
        count = sum(value > deadline_ms + 1e-9 for value in values)
        return dict(count=count, n=len(values), rate=count / len(values) if values else None)
    latency_key = next((key for key in ("observation_to_command", "total_observation_to_command", "observation_to_command_ms")
                        if key in stages), None)
    latencies = [record["timings_ms"][latency_key] for record in selected
                 if latency_key in record.get("timings_ms", {})] if latency_key else []
    return dict(records_total=len(records), warmup_records=sum(record.get("warmup", False) for record in records),
                measured_records=len(selected), statistics_convention="linear percentiles; population std ddof=0",
                deadline_ms=deadline_ms, stages_ms=stages, command_intervals_ms=interval_stats,
                command_deadline_misses=misses(intervals), observation_to_command_deadline_misses=misses(latencies),
                control_loop_ms=statistics(loops),
                wall_clock_command_frequency_hz=(1000/interval_stats["mean"] if intervals and interval_stats["mean"] > 0 else None),
                real_time_factor=(sum(simulated)/(sum(aligned_loops)/1000) if sum(aligned_loops) > 0 else None),
                real_time_factor_aligned_samples=len(simulated), simulated_policy_frequency_hz=15,
                simulated_physics_frequency_hz=120,
                deadline_behavior="diagnostic only; no holding, rejecting, stopping, or hardware safety inferred",
                timing_scope="timer names and synchronization semantics are defined by the runtime manifest")


def summarize_actions(records: list[dict]) -> dict:
    if not records:
        return dict(records=0, samples=0, dimensions=[], status="UNKNOWN_NO_ACTION_RECORDS")
    for record in records:
        lengths = [len(_matrix(record[key], width, key)) for key, width in
                   (("raw_mu", 6), ("clipped", 6), ("ema", 6),
                    ("physical_position_increment_m", 3), ("physical_rotation_increment_rad", 3))]
        if len(set(lengths)) != 1:
            raise ValueError("each action record must retain its complete aligned processing chain")
    raw, clipped, ema = [_stack(records, key, 6) for key in ("raw_mu", "clipped", "ema")]
    position = _stack(records, "physical_position_increment_m", 3)
    rotation = _stack(records, "physical_rotation_increment_rad", 3)
    if not (raw.shape == clipped.shape == ema.shape and len(position) == len(raw) == len(rotation)):
        raise ValueError("action chain sample counts must agree")
    if np.any(np.abs(clipped) > 1 + 1e-7):
        raise ValueError("clipped actions exceed the original [-1,1] bound")
    expected = np.clip(raw, -1, 1)
    if not np.allclose(expected, clipped, atol=1e-7, rtol=1e-7):
        raise ValueError("clipped action is inconsistent with the recorded raw mu")
    physical = np.concatenate((position, rotation), axis=1)
    dimensions = []
    for index, name in enumerate(ACTION_DIMENSIONS):
        clipped_count = int(np.count_nonzero(np.abs(raw[:, index]) > 1))
        saturated_count = int(np.count_nonzero(np.abs(clipped[:, index]) >= 1))
        dimensions.append(dict(index=index, name=name, n=len(raw), raw_mu=statistics(raw[:, index]),
                               clipped=statistics(clipped[:, index]), ema=statistics(ema[:, index]),
                               physical_increment=statistics(physical[:, index]),
                               physical_unit="m" if index < 3 else "rad",
                               raw_outside_unit_count=clipped_count, raw_outside_unit_rate=clipped_count/len(raw),
                               clipped_at_unit_count=saturated_count, clipped_at_unit_rate=saturated_count/len(raw)))
    return dict(records=len(records), samples=len(raw), dimensions=dimensions,
                ema_policy="recorded original EMA; no additional clipping applied by analysis",
                clipping_definition="abs(raw_mu) > 1", saturation_definition="abs(clipped) >= 1")


def summarize_controller(records: list[dict]) -> dict:
    if not records:
        return dict(records=0, samples=0, status="UNKNOWN_NO_CONTROLLER_RECORDS")
    for record in records:
        lengths = [len(_matrix(record[key], width, key)) for key, width in
                   (("target_position", 3), ("target_quaternion", 4), ("requested_torque", 7),
                    ("software_torque", 7), ("effort_limits", 7))]
        if len(set(lengths)) != 1:
            raise ValueError("each controller record must contain aligned samples")
    position = _stack(records, "target_position", 3)
    quaternion = _stack(records, "target_quaternion", 4)
    requested = _stack(records, "requested_torque", 7)
    software = _stack(records, "software_torque", 7)
    limits = _stack(records, "effort_limits", 7)
    counts = {len(value) for value in (position, quaternion, requested, software, limits)}
    if len(counts) != 1 or np.any(limits <= 0):
        raise ValueError("controller sample counts must agree; effort limits must be positive")
    if np.any(np.abs(np.linalg.norm(quaternion, axis=1)-1) > 1e-4):
        raise ValueError("target quaternion is not unit length")
    # A configured actuator limit is not proof that the solver applied a torque.
    joint_stats = []
    changed = np.abs(requested-software) > 1e-7
    for index in range(7):
        joint_stats.append(dict(index=index, n=len(requested), requested_torque=statistics(requested[:, index]),
                               software_torque=statistics(software[:, index]), effort_limits=statistics(limits[:, index]),
                               software_changed_count=int(changed[:, index].sum()),
                               software_changed_rate=float(changed[:, index].mean()),
                               requested_above_configured_limit_count=int((np.abs(requested[:, index]) > limits[:, index]).sum()),
                               requested_above_configured_limit_rate=float((np.abs(requested[:, index]) > limits[:, index]).mean()),
                               software_at_or_above_configured_limit_count=int((np.abs(software[:, index]) >= limits[:, index]).sum()),
                               software_at_or_above_configured_limit_rate=float((np.abs(software[:, index]) >= limits[:, index]).mean())))
    constraints = {}
    for key, width in (("position_constraint_active", 3), ("orientation_constraint_active", 3)):
        rows = [record for record in records if key in record and record[key] is not None]
        if rows:
            values = _stack(rows, key, width)
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f"{key} must contain only booleans or 0/1")
            constraints[key] = [dict(index=i, n=len(values), count=int(values[:, i].sum()), rate=float(values[:, i].mean()))
                                for i in range(width)]
        else:
            constraints[key] = "UNKNOWN"
    result = dict(records=len(records), samples=len(requested), target_position_m=_dimension_statistics(position),
                  target_quaternion_wxyz=_dimension_statistics(quaternion), joints=joint_stats, constraints=constraints,
                  requested_vs_applied="requested and software-clipped efforts do not establish actual simulated execution")
    for key, width in (("task_space_pd_output", 6), ("applied_sim_effort", 7)):
        rows = [record for record in records if key in record and record[key] is not None and not isinstance(record[key], str)]
        result[key] = (_dimension_statistics(_stack(rows, key, width)) if rows else "UNKNOWN")
        result[key+"_records"] = len(rows)
    return result


def classify_episode(episode: dict) -> dict:
    """Describe predicate history; geometric residuals are evidence, not causes."""
    ever, terminal = episode["ever_success"], episode["terminal_success"]
    if type(ever) is not bool or type(terminal) is not bool or (terminal and not ever):
        raise ValueError("episode success flags must be consistent booleans")
    first = episode.get("first_success_time_seconds", episode.get("first_success_time_s"))
    if first is not None:
        first = _number(first, "first success time", nonnegative=True)
        if not ever:
            raise ValueError("never-success episode cannot have first-success time")
    return dict(category=("terminal_success" if terminal else "lost_success_before_terminal" if ever else "never_success"),
                first_success_time_seconds=first,
                terminal_kind=episode.get("terminal_kind", episode.get("termination_reason", "UNKNOWN")),
                pre_reset_terminal_state=episode.get("pre_reset_terminal_state", episode.get("terminal_state", "UNKNOWN")),
                recorded_geometric_errors=episode.get("geometric_errors", "UNKNOWN"),
                interpretation="descriptive predicate classification; recorded errors do not establish a causal failure mechanism")


def summarize_episodes(episodes: list[dict], *, pairing_verified=False, archived_successes=280, archived_n=288) -> dict:
    if pairing_verified:
        raise ValueError("paired comparison requires a separate verified per-episode matching analysis")
    classified = []
    identities = set()
    for episode in episodes:
        if "evaluation_seed" in episode and "episode_id" in episode:
            identity = (episode["evaluation_seed"], episode.get("env_id", 0), episode["episode_id"])
            if identity in identities:
                raise ValueError("duplicate completed episode identity")
            identities.add(identity)
        classified.append(dict(episode, failure_analysis=classify_episode(episode)))
    n = len(episodes)
    ever = sum(row["ever_success"] for row in episodes)
    terminal = sum(row["terminal_success"] for row in episodes)
    archived_interval = wilson_interval(archived_successes, archived_n)
    archived_rate = archived_successes / archived_n if archived_n else None
    first = [row["failure_analysis"]["first_success_time_seconds"] for row in classified
             if row["failure_analysis"]["first_success_time_seconds"] is not None]
    return dict(episodes_completed=n, ever_success_count=ever, ever_success_rate=ever/n if n else None,
                terminal_success_count=terminal, terminal_success_rate=terminal/n if n else None,
                ever_success_wilson95=wilson_interval(ever, n), terminal_success_wilson95=wilson_interval(terminal, n),
                first_success_time_seconds=statistics(first),
                failure_categories=dict(Counter(row["failure_analysis"]["category"] for row in classified)), episodes=classified,
                archived_comparison=dict(kind="descriptive_unpaired", archived_successes=archived_successes,
                                         archived_n=archived_n, archived_rate=archived_rate,
                                         archived_wilson95=archived_interval,
                                         standalone_minus_archived_terminal_rate=(terminal/n-archived_rate if n and archived_rate is not None else None),
                                         limitation="initial conditions and RNG streams are not verified paired; no paired significance or per-episode agreement claim"),
                uncertainty="episode-binomial intervals are descriptive, not training-seed uncertainty")


def write_episode_csv(episodes: list[dict], output_path: Path) -> None:
    """Export completed episodes with original denominators and source evidence."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("evaluation_seed", "env_id", "episode_id", "steps", "ever_success", "terminal_success",
               "first_success_time_seconds", "terminal_kind", "failure_category", "initial_randomized_configuration",
               "pre_reset_terminal_state", "recorded_geometric_errors")
    # Exclusive creation protects historical evaluation CSV files.
    with output_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for episode in episodes:
            category = classify_episode(episode)
            row = {key: episode.get(key, "UNKNOWN") for key in columns[:6]}
            row.update(first_success_time_seconds=category["first_success_time_seconds"],
                       terminal_kind=category["terminal_kind"], failure_category=category["category"],
                       initial_randomized_configuration=json.dumps(episode.get("initial_randomized_configuration", "UNKNOWN"), allow_nan=False),
                       pre_reset_terminal_state=json.dumps(category["pre_reset_terminal_state"], allow_nan=False),
                       recorded_geometric_errors=json.dumps(category["recorded_geometric_errors"], allow_nan=False))
            writer.writerow(row)


def plot_distributions(records: list[dict], output_dir: Path) -> list[str]:
    """Write conventional timing/action/control plots from measured raw records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(fig, name):
        for suffix in ("svg", "png"):
            path = output_dir / f"{name}.{suffix}"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)
    timing = [row for row in records if row.get("kind") == "timing" and not row.get("warmup", False)]
    if timing:
        summary = summarize_timings(timing)
        names = sorted(summary["stages_ms"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name in names:
            values = np.sort([row["timings_ms"][name] for row in timing if name in row.get("timings_ms", {})])
            axes[0].plot(values, np.arange(1, len(values)+1)/len(values), label=name)
        axes[0].set(xlabel="Latency (ms)", ylabel="Empirical cumulative fraction", title="Measured latency (warmup excluded)")
        if names:
            axes[0].legend(fontsize=6)
        commands = [row.get("wall_command_monotonic_ns") for row in timing]
        intervals = [(b-a)/1e6 for a,b in zip(commands,commands[1:]) if a is not None and b is not None]
        axes[1].hist(intervals, bins=min(50, max(1, len(intervals))))
        axes[1].axvline(POLICY_INTERVAL_MS, color="red", linestyle="--", label="15 Hz simulated interval")
        axes[1].set(xlabel="Wall command interval (ms)", ylabel="Count", title="Actual software command cadence")
        axes[1].legend(fontsize=7)
        save(fig, "latency_and_command_intervals")
    actions = [row for row in records if row.get("kind") == "action"]
    if actions:
        summary = summarize_actions(actions)
        raw, ema = _stack(actions, "raw_mu", 6), _stack(actions, "ema", 6)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        axes[0].boxplot([raw[:,i] for i in range(6)], labels=ACTION_DIMENSIONS, showfliers=False)
        axes[0].axhline(1, color="red", linestyle="--")
        axes[0].axhline(-1, color="red", linestyle="--")
        axes[0].set(ylabel="Raw policy mean", title="Raw action distribution")
        x = np.arange(6)
        axes[1].bar(x-.18, [dim["raw_outside_unit_rate"] for dim in summary["dimensions"]], .36, label="abs(raw mu) > 1")
        axes[1].bar(x+.18, [dim["clipped_at_unit_rate"] for dim in summary["dimensions"]], .36, label="abs(clipped) >= 1")
        axes[1].set(xticks=x, xticklabels=ACTION_DIMENSIONS, ylabel="Fraction", ylim=(0,1), title="Clipping and saturation are distinct")
        axes[1].legend()
        fig.tight_layout()
        save(fig, "action_saturation")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--episodes", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", type=Path)
    parser.add_argument("--episode-csv", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for path in args.trace for line in path.read_text().splitlines() if line.strip()]
    episodes = []
    for path in args.episodes:
        value = json.loads(path.read_text())
        episodes.extend(value if isinstance(value, list) else value["episodes"])
    summary = dict(timing=summarize_timings([r for r in records if r.get("kind") == "timing"]),
                   actions=summarize_actions([r for r in records if r.get("kind") == "action"]),
                   controller=summarize_controller([r for r in records if r.get("kind") == "controller"]),
                   evaluation=summarize_episodes(episodes),
                   source_files=[str(path) for path in args.trace+args.episodes])
    if args.plots:
        summary["plots"] = plot_distributions(records, args.plots)
    if args.episode_csv:
        write_episode_csv(episodes, args.episode_csv)
    atomic_write_json(args.output, summary)


if __name__ == "__main__":
    main()
