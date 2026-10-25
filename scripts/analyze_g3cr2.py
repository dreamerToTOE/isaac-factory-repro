"""CPU-only analysis of the G3C-R2 protocol, including an incomplete run.

No checkpoints or simulator modules are loaded. Missing evaluation milestones
remain null; historical G2 episodes are never added to a new denominator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from analyze_g2 import (finite_number, identity, read_audit,
                            read_evaluations, read_scalars, training_summary, write_json)
except ImportError:
    from scripts.analyze_g2 import (finite_number, identity, read_audit,
                                   read_evaluations, read_scalars, training_summary, write_json)


G3_EPOCHS = [100, 200, 300, 400]
G2_EPOCHS = [50, 100, 150, 200]
SEEDS = [1001, 1002, 1003]


def read_json_if_present(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def budget_summary(epochs: list[dict], num_envs=32, horizon=128, planned_epochs=400) -> dict:
    """Count only actual contiguous audit records, never an unfinished epoch."""
    ids = [r["epoch"] for r in epochs]
    contiguous = bool(ids and ids == list(range(1, len(ids) + 1)))
    for row in epochs:
        value = row.get("environment_transitions")
        if (not finite_number(value) or value != int(value) or value < 0):
            raise ValueError("epoch transition count must be a finite nonnegative integer")
        if value != num_envs * horizon:
            raise ValueError("audit epoch transitions differ from registered vector batch")
    actual = sum(int(r["environment_transitions"]) for r in epochs)
    planned = num_envs * horizon * planned_epochs
    if actual > planned:
        raise ValueError("completed transitions exceed registered budget")
    complete = contiguous and len(epochs) == planned_epochs
    return {"num_envs": num_envs, "horizon_length": horizon,
            "planned_epochs": planned_epochs, "planned_transitions": planned,
            "completed_epochs": len(epochs), "last_completed_epoch": max(ids, default=None),
            "completed_transitions": actual, "completed_budget_fraction": actual / planned,
            "remaining_planned_transitions": planned - actual,
            "audit_contiguous_from_epoch_1": contiguous,
            "registered_transition_budget_completed": complete,
            "source": "sum of actual completed audit epochs; in-flight work is excluded"}


def matched_comparison(new_evaluation: dict, old_evaluation: dict) -> dict:
    """Retain two independent evaluation populations at matched intended budgets."""
    new = {r["checkpoint_epoch"]: r for r in new_evaluation["checkpoints"]}
    old = {r["checkpoint_epoch"]: r for r in old_evaluation["checkpoints"]}
    rows = []
    for new_epoch, old_epoch in zip(G3_EPOCHS, G2_EPOCHS):
        new_row, old_row = new.get(new_epoch), old.get(old_epoch)
        new_n = new_row["aggregate"]["episodes_completed"] if new_row else 0
        old_n = old_row["aggregate"]["episodes_completed"] if old_row else 0
        rows.append({"environment_transitions": new_epoch * 32 * 128,
                     "g3cr2_checkpoint_epoch": new_epoch, "g2_checkpoint_epoch": old_epoch,
                     "g3cr2": new_row, "g2_historical": old_row,
                     "g3cr2_evaluated": bool(new_n), "g2_evaluated": bool(old_n),
                     "between_protocol_delta_assessed": bool(new_n and old_n),
                     "ever_success_rate_delta": (new_row["aggregate"]["ever_success_rate"] -
                                                  old_row["aggregate"]["ever_success_rate"]
                                                  if new_n and old_n else None),
                     "terminal_success_rate_delta": (new_row["aggregate"]["terminal_success_rate"] -
                                                      old_row["aggregate"]["terminal_success_rate"]
                                                      if new_n and old_n else None)})
    return {"schema_version": "g3cr2_matched_intended_transitions_v1", "rows": rows,
            "new_protocol": "32-env / 400-epoch seed0 baseline",
            "historical_protocol": "G2 64-env / 200-epoch seed0 baseline",
            "training_seed_replication": False,
            "pooling_rule": "No episode pooling across protocols, checkpoints or training runs.",
            "limits": ["Equal environment transitions do not equal identical PPO batches or optimizer updates.",
                       "One training seed in each protocol cannot establish seed robustness.",
                       "Missing milestones have zero observed episodes and null rates, not zero success."]}


def numerical_summary(audit: dict, epochs: list[dict]) -> dict:
    finite = [r.get("epoch_loss_arrays_finite") for r in epochs]
    skips = audit.get("amp_skips", [])
    return {"observed_epoch_count": len(epochs),
            "all_observed_epoch_loss_arrays_finite": (all(x is True for x in finite) if finite else None),
            "actual_optimizer_steps": audit.get("optimizer_steps"),
            "actor_amp_step_attempts": audit.get("actor_attempts"),
            "amp_skipped_actor_updates": len(skips), "amp_skip_events": skips,
            "instrumentation_errors": audit.get("instrumentation_errors"),
            "peak_cuda_allocated_MiB": max((r["cuda_peak_allocated_bytes"] / 2**20 for r in epochs
                                            if finite_number(r.get("cuda_peak_allocated_bytes"))), default=None),
            "peak_cuda_reserved_MiB": max((r["cuda_peak_reserved_bytes"] / 2**20 for r in epochs
                                           if finite_number(r.get("cuda_peak_reserved_bytes"))), default=None),
            "limits": ["Logged finite loss arrays do not prove all gradients or applied updates finite.",
                       "Actual Adam.step calls and GradScaler skips are observations, not inferred successful learning.",
                       "CUDA allocator peaks omit simulator/graphics memory; whole-GPU NVML is reported separately."]}


def read_supervised_run(directory: Path, issues: dict) -> tuple[dict, list[dict]]:
    files = {name: directory / name for name in (
        "guard/status.json", "guard/samples.csv", "supervisor/status.json",
        "supervisor/child_identity.json", "supervisor/exec_release.json", "supervisor/child_exit.json")}
    guard = read_json_if_present(files["guard/status.json"])
    supervisor = read_json_if_present(files["supervisor/status.json"])
    spawn = read_json_if_present(files["supervisor/child_identity.json"])
    release = read_json_if_present(files["supervisor/exec_release.json"])
    exit_record = read_json_if_present(files["supervisor/child_exit.json"])
    samples = []
    if files["guard/samples.csv"].is_file():
        with files["guard/samples.csv"].open(newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    sample = {k: float(row[k]) for k in ("elapsed_seconds", "monotonic_seconds", "used_MiB", "free_MiB")}
                    if not all(math.isfinite(x) for x in sample.values()):
                        raise ValueError("nonfinite sample")
                    samples.append(sample)
                except (KeyError, TypeError, ValueError):
                    issues["warnings"].append("Ignored malformed/nonfinite guard sample; coverage cannot be reconstructed from it.")
    intervals = [b["monotonic_seconds"] - a["monotonic_seconds"] for a, b in zip(samples, samples[1:])]
    if any(x <= 0 for x in intervals):
        issues["errors"].append("Guard sample times are not strictly increasing.")
    floor = guard.get("policy", {}).get("gap_limit_seconds", .5)
    observed_gaps = [{"after_monotonic_seconds": a["monotonic_seconds"],
                      "duration_seconds": b["monotonic_seconds"] - a["monotonic_seconds"]}
                     for a, b in zip(samples, samples[1:])
                     if b["monotonic_seconds"] - a["monotonic_seconds"] > floor]
    start, end = spawn.get("monotonic_seconds"), exit_record.get("monotonic_seconds")
    elapsed = end - start if finite_number(start) and finite_number(end) and end >= start else None
    brackets = bool(samples and finite_number(start) and finite_number(end) and
                    samples[0]["monotonic_seconds"] <= start <= end <= samples[-1]["monotonic_seconds"])
    complete = bool(brackets and not observed_gaps and guard.get("continuous_monitoring") is True)
    return {"input_identities": {name: identity(path) for name, path in files.items() if path.is_file()},
            "guard": guard, "supervisor": supervisor, "child_exit": exit_record,
            "child_raw_wait_returncode": exit_record.get("raw_wait_returncode"),
            "child_spawn_to_wait_wall_seconds": elapsed,
            "exec_release_to_wait_wall_seconds": (end - release["monotonic_seconds"]
                if finite_number(end) and finite_number(release.get("monotonic_seconds")) else None),
            "wall_duration_seconds": elapsed,
            "wall_evidence": {"start": spawn, "exec_release": release, "end": exit_record},
            "sample_count": len(samples), "observed_gaps": observed_gaps,
            "max_sample_interval_seconds": max(intervals, default=None),
            "samples_bracket_child": brackets, "sampled_coverage_complete": complete,
            "min_free_MiB": min((s["free_MiB"] for s in samples), default=None),
            "peak_total_gpu_used_MiB": max((s["used_MiB"] for s in samples), default=None),
            "extrema_scope": "observed samples only; scheduling/measurement gaps may hide extrema",
            "coverage_note": "A gap remains an evidence gap even when free memory is above the floor; sampled intervals are not continuous measurement."}, samples


def scalar_transition_points(scalars: dict, tag: str, batch_transitions: int) -> list[tuple[float, float]]:
    """Map official log coordinates to completed transitions without actor offset."""
    epoch_at_step = {p["step"]: p["value"] for p in scalars.get("tags", {}).get("info/epochs", [])
                     if p.get("finite") and finite_number(p.get("value"))}
    result = []
    for point in scalars.get("tags", {}).get(tag, []):
        if not point.get("finite") or not finite_number(point.get("value")):
            continue
        if tag.endswith("/iter"):
            x = point["step"] * batch_transitions
        elif tag in ("losses/cval_loss", "info/cval_lr"):
            x = point["step"]
        elif point["step"] in epoch_at_step:
            x = epoch_at_step[point["step"]] * batch_transitions
        else:
            continue
        result.append((x, point["value"]))
    return result


def plots(output: Path, scalars: dict, g2_scalars: dict, epochs: list[dict], memory: list[dict],
          budget: dict, comparison: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    artifacts = []

    def save(fig, name):
        for suffix in ("png", "pdf"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=170)
            artifacts.append(path.name)
        plt.close(fig)

    def curves(ax, tag):
        for source, batch, label, color in ((g2_scalars, 8192, "Historical G2 64-env seed0", "0.65"),
                                          (scalars, 4096, "G3C-R2 32-env seed0 (incomplete)", "tab:blue")):
            points = scalar_transition_points(source, tag, batch)
            if points:
                ax.plot([p[0] / 1e6 for p in points], [p[1] for p in points], label=label, color=color)
        ax.set_xlabel("Completed environment transitions (millions)")
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "No finite logged observations", transform=ax.transAxes, ha="center")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    prefix_million = budget["completed_transitions"] / 1e6
    for row, tag, title in ((0, "rewards/iter", "Raw episodic reward"),
                            (1, "shaped_rewards/iter", "Shaped reward (timeout bootstrap)")):
        for column in (0, 1):
            curves(axes[row, column], tag)
            axes[row, column].set(title=title + (": full history" if column == 0 else ": observed prefix"), ylabel="Episodic sum")
        if prefix_million:
            axes[row, 1].set_xlim(0, prefix_million * 1.04)
            prefix_values = [y for source, batch in ((scalars, 4096), (g2_scalars, 8192))
                             for x, y in scalar_transition_points(source, tag, batch)
                             if x <= budget["completed_transitions"]]
            if prefix_values:
                lo, hi = min(prefix_values), max(prefix_values)
                pad = max(1, (hi - lo) * .1)
                axes[row, 1].set_ylim(lo - pad, hi + pad)
    fig.suptitle("Training reward is not independent insertion success", fontsize=12)
    fig.tight_layout()
    save(fig, "training_reward")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, tag, title in zip(axes.flat,
        ("losses/a_loss", "losses/cval_loss", "losses/entropy", "info/kl", "info/last_lr", "losses/bounds_loss"),
        ("Actor loss", "Central value loss", "Policy entropy", "Policy KL", "Adaptive actor learning rate", "Action bounds loss")):
        curves(ax, tag)
        if prefix_million:
            ax.set_xlim(0, prefix_million * 1.04)
            prefix_values = [y for source, batch in ((scalars, 4096), (g2_scalars, 8192))
                             for x, y in scalar_transition_points(source, tag, batch)
                             if x <= budget["completed_transitions"]]
            if prefix_values:
                lo, hi = min(prefix_values), max(prefix_values)
                pad = max(abs(lo), abs(hi), 1e-10) * .1
                ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title)
    fig.suptitle("Observed transition prefix; finite logged losses do not prove finite gradients", fontsize=12)
    fig.tight_layout()
    save(fig, "training_losses")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    for key, label in (("total_fps", "PPO total"), ("env_step_fps", "Environment stepping")):
        rows = [r for r in epochs if finite_number(r.get(key))]
        if rows:
            axes[0].plot([r["epoch"] for r in rows], [r[key] for r in rows], marker=".", label=label)
    axes[0].set(xlabel="Completed PPO epoch", ylabel="Environment transitions / s",
                title="Observed new-run throughput; distinct timing denominators")
    if axes[0].lines:
        axes[0].legend(fontsize=8)
    if memory:
        pieces, piece = [], []
        for sample in memory:
            if piece and sample["monotonic_seconds"] - piece[-1]["monotonic_seconds"] > .5:
                pieces.append(piece)
                piece = []
            piece.append(sample)
        if piece:
            pieces.append(piece)
        for index, segment in enumerate(pieces):
            x = [r["elapsed_seconds"] for r in segment]
            for field, color, label in (("used_MiB", "tab:blue", "Whole-GPU used"), ("free_MiB", "tab:green", "Whole-GPU free")):
                axes[1].plot(x, [r[field] for r in segment], color=color, linewidth=.7,
                             label=label if index == 0 else None)
        axes[1].axhline(1024, color="tab:red", linestyle="--", label="1024 MiB runtime guard")
        gaps = [(b["elapsed_seconds"], b["monotonic_seconds"] - a["monotonic_seconds"])
                for a, b in zip(memory, memory[1:])
                if b["monotonic_seconds"] - a["monotonic_seconds"] > .5]
        if gaps:
            axes[1].axvline(gaps[0][0], color="tab:red", linestyle=":", linewidth=1)
            axes[1].text(.98, .96, f"Monitoring gap: {gaps[0][1]:.6f} s", transform=axes[1].transAxes,
                         ha="right", va="top", color="tab:red", fontsize=9)
        axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Guard elapsed time (s)", ylabel="MiB", title="Observed NVML samples; lines break across gaps >0.5 s")
    fig.tight_layout()
    save(fig, "throughput_memory")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, title in ((axes[0], "ever_success", "Official ever-success"),
                           (axes[1], "terminal_success", "Official terminal-success")):
        for protocol, label, color, marker in (("g2_historical", "Historical G2", "0.5", "o"),
                                               ("g3cr2", "New G3C-R2", "tab:blue", "s")):
            selected = [(r, r[protocol]["aggregate"]) for r in comparison["rows"] if
                        r.get(protocol) and r[protocol]["aggregate"]["episodes_completed"]]
            if selected:
                x = [r["environment_transitions"] / 1e6 for r, _ in selected]
                y = [a[key + "_rate"] for _, a in selected]
                ax.plot(x, y, marker=marker, color=color, label=label)
                for xx, yy, (_, a) in zip(x, y, selected):
                    ax.annotate(f"n={a['episodes_completed']}", (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        if not any(r["g3cr2_evaluated"] for r in comparison["rows"]):
            ax.text(.5, .42, "New run: no registered milestone reached\nNo evaluation denominator or success rate", transform=ax.transAxes, ha="center", color="tab:blue")
        ax.set(title=title, xlabel="Matched intended environment transitions (millions)", ylabel="Completed-episode fraction", ylim=(-.03, 1.17))
        if ax.lines:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Independent evaluation populations; missing new points are not zero", fontsize=12)
    fig.tight_layout()
    save(fig, "matched_transition_success")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].barh(["Completed", "Registered budget"], [budget["completed_transitions"], budget["planned_transitions"]], color=["tab:blue", "0.7"])
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set(xlabel="Environment transitions", title=f"Completed {budget['completed_budget_fraction']:.1%} of budget")
    for role, color in (("actor", "tab:blue"), ("central_value", "tab:orange")):
        rows = [r for r in epochs if isinstance(r.get("optimizer_steps_cumulative"), dict)]
        if rows:
            axes[1].plot([r["epoch"] for r in rows], [r["optimizer_steps_cumulative"][role] for r in rows], marker=".", label=role, color=color)
    axes[1].set(xlabel="Completed epoch", ylabel="Actual cumulative Adam.step calls", title="Observed optimizer calls")
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "completed_budget_optimizer")
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-log", "audit", "supervised-run", "eval-root", "g2-root", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tensorboard-python", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output exists; preserve prior analysis and use a fresh directory")
    issues = {"errors": [], "warnings": []}
    audit, epochs = read_audit(args.audit, issues)
    budget = budget_summary(epochs)
    monitor, memory = read_supervised_run(args.supervised_run, issues)
    scalars = read_scalars(args.train_log, args.tensorboard_python, issues)
    evaluation = read_evaluations(args.eval_root, G3_EPOCHS, SEEDS, 32, 3, issues)
    g2_evaluation = read_json_if_present(args.g2_root / "analysis/evaluation_summary.json")
    if "checkpoints" not in g2_evaluation:
        raise ValueError("historical G2 evaluation summary missing")
    g2_scalars = read_json_if_present(args.g2_root / "analysis/tensorboard_scalars.json")
    comparison = matched_comparison(evaluation, g2_evaluation)
    train = training_summary(scalars, audit, epochs, monitor, 32, 128, issues)
    engineering_complete = bool(budget["registered_transition_budget_completed"] and
                                monitor["child_raw_wait_returncode"] == 0 and
                                monitor["sampled_coverage_complete"] and
                                monitor["supervisor"].get("termination_reason") == "child_exit" and
                                monitor["guard"].get("stop_reason") == "child_wait_observed")
    summary = {"schema_version": "g3cr2_analysis_v1", "analysis_timing": "post hoc analysis; frozen launch protocol remains separate",
               "budget": budget, "training": train, "numerical_observations": numerical_summary(audit, epochs),
               "supervision": monitor, "evaluation": evaluation, "matched_transition_comparison": comparison,
               "engineering_status": "COMPLETE" if engineering_complete else "INCOMPLETE",
               "scientific_status": "DESCRIPTIVE_EVALUATION_AVAILABLE" if any(r["g3cr2_evaluated"] for r in comparison["rows"]) else "NOT_ASSESSED",
               "manifest_identity": identity(args.manifest), "analysis_script_identity": identity(Path(__file__)),
               "audit_identity": identity(args.audit), "issues": issues,
               "scientific_limits": ["32-env/400-epoch protocol is not G2 seed-only replication.",
                    "Capacity/monitoring interruption is not scientific evidence of learning failure.",
                    "Unreached checkpoint evaluations remain missing with null rates.",
                    "G2 episodes are historical context only and are not added to any G3C-R2 denominator.",
                    "No training seed1/2 robustness assessment follows from seed0 observations."]}
    args.output.mkdir(parents=True)
    summary["plots"] = plots(args.output, scalars, g2_scalars, epochs, memory, budget, comparison)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "tensorboard_scalars.json", scalars)
    write_json(args.output / "evaluation_summary.json", evaluation)
    write_json(args.output / "matched_transition_comparison.json", comparison)
    with (args.output / "evaluation_per_seed.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["protocol", "checkpoint_epoch", "transitions", "seed", "status", "episodes_requested", "episodes_completed", "ever_success_count", "ever_success_rate", "terminal_success_count", "terminal_success_rate"])
        for row in comparison["rows"]:
            for protocol in ("g3cr2", "g2_historical"):
                if not row[protocol]:
                    continue
                for seed in row[protocol]["per_seed"]:
                    writer.writerow([protocol, row[protocol]["checkpoint_epoch"], row["environment_transitions"], seed["seed"], seed["status"], seed["episodes_requested"], seed["episodes_completed"], seed["ever_success_count"], seed["ever_success_rate"], seed["terminal_success_count"], seed["terminal_success_rate"]])
    print(json.dumps({"output": str(args.output.resolve()), "budget": budget,
                      "engineering_status": summary["engineering_status"], "scientific_status": summary["scientific_status"],
                      "errors": issues["errors"], "warnings": issues["warnings"]}, allow_nan=False))
    if issues["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
