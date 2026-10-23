"""Summarize D001 bounded training observations; missing evidence cannot pass."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics


def summarize(path):
    result_path, audit_path = path / "result.json", path / "training_audit.json"
    if not result_path.exists() or not audit_path.exists():
        return {"passed": False, "missing_evidence": str(path)}
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    actors, critics = audit["actor_attempts"], audit["central_value_attempts"]
    gradients = audit["gradient_events"]
    applied = [g for g in gradients if g["phase"] == "optimizer_pre_step_post_clip"]
    groups = Counter((g["role"], g["phase"], g["finite"]) for g in gradients)
    epochs = audit["epochs"]
    losses_finite = bool(actors and critics) and all(
        s["finite"] for a in actors for s in list(a["scalars"].values()) + [a["total_objective"]]
    ) and all(c["loss"]["finite"] for c in critics)
    applied_finite = bool(applied) and all(g["finite"] for g in applied)
    console = (path / "console.txt").read_text()
    warnings = [line for line in console.splitlines() if re.search(r"warning|error|deprecated|nan|overflow", line, re.I)]
    row = {
        "result": result,
        "actual_num_envs": audit["runtime"].get("num_actors"),
        "actor_attempts": len(actors), "critic_attempts": len(critics),
        "optimizer_steps": audit["optimizer_steps"],
        "amp_skips": [{k: a[k] for k in ("attempt", "epoch", "scale_before", "scale_after")}
                      for a in actors if a["amp_skipped"]],
        "all_observed_losses_finite": losses_finite,
        "all_applied_gradients_finite": applied_finite,
        "gradient_groups": [{"role": role, "phase": phase, "finite": finite, "count": count}
                            for (role, phase, finite), count in groups.items()],
        "epochs": epochs, "checkpoint_events": audit["checkpoint_events"],
        "cuda_allocator": audit["cuda_allocator"],
        "instrumentation_errors": audit["instrumentation_errors"],
        "warning_error_lines": warnings,
        "exit_error": audit.get("exit_error"),
        "read_only_audit_overhead_included": True,
    }
    row["passed"] = (result["returncode"] == 0 and result["stop_reason"] is None
                     and not result["monitor_errors"] and not audit["instrumentation_errors"]
                     and audit.get("exit_error") is None and losses_finite and applied_finite
                     and audit["optimizer_steps"]["actor"] > 0 and audit["optimizer_steps"]["central_value"] > 0)
    if len(epochs) >= 3:
        stable = epochs[1:]
        seconds = statistics.mean(e["official_total_seconds"] for e in stable)
        row["epochs_2_3_mean_seconds"] = seconds
        row["epochs_2_3_aggregate_total_fps"] = sum(e["environment_transitions"] for e in stable) / sum(e["official_total_seconds"] for e in stable)
        row["extrapolated_200_epoch_hours"] = seconds * 200 / 3600
        row["transitions_at_200_epochs"] = audit["runtime"]["num_actors"] * audit["runtime"]["horizon_length"] * 200
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {str(n): {phase: summarize(args.logs_root / f"d001_{phase}{n}") for phase in ("train", "resume")}
            for n in (16, 32, 64)}
    summary = {"configurations": rows,
               "all_six_runs_passed": all(x["passed"] for row in rows.values() for x in row.values()),
               "timing_note": "200-epoch extrapolation uses epochs 2/3, includes synchronized read-only audit overhead, excludes cold startup and checkpoint I/O; only a short-run estimate, no convergence or long-duration memory guarantee.",
               "memory_note": "NVML v2 used excludes driver reserved memory. Total includes desktop. Sampled maxima can miss sub-100ms transients; allocator peaks cover PyTorch only.",
               "amp_note": "Fresh runs have one initial skipped overflow with scale 65536 to 32768; all applied optimizer gradients must be finite. Raw or unscaled pre-clip gradients are not all finite."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"all_six_runs_passed": summary["all_six_runs_passed"], "output": str(args.output)}))
    raise SystemExit(0 if summary["all_six_runs_passed"] else 1)


if __name__ == "__main__":
    main()
