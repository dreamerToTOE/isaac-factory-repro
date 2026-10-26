"""Durably observe one fresh G3C-R5 run and conditionally evaluate its milestones.

Run this controller in a separate systemd user service with Restart=no. It never
starts, resumes, kills or retries training. GPU telemetry is neither queried nor
required. A missing training status keeps this observer waiting; it cannot turn
a missing telemetry record into a signal to the scientific process.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

try:
    from supervision_common import atomic_json, sha, stamp
except ImportError:
    from scripts.supervision_common import atomic_json, sha, stamp


EPOCHS = (100, 200, 300, 400)
SEEDS = (1001, 1002, 1003)
TRANSITIONS_PER_EPOCH = 32 * 128
TOTAL_TRANSITIONS = 400 * TRANSITIONS_PER_EPOCH
TRAINING_SEED = 1


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_training_identity(manifest):
    """Reject a controller manifest that is not the preregistered seed-1 run.

    The completion controller never infers a training seed from checkpoint names
    or evaluator seeds.  It requires the frozen manifest to state seed 1 and a
    distinct output identity, making accidental seed-0 resume/concatenation
    impossible to pass the gate.
    """
    require(manifest.get("training_seed") == TRAINING_SEED,
            "R5 completion requires training_seed=1")
    require(manifest.get("training_num_envs", 32) == 32,
            "R5 completion requires 32 training environments")
    require(manifest.get("training_epochs", 400) == 400,
            "R5 completion requires 400 training epochs")
    require(manifest.get("training_horizon", 128) == 128,
            "R5 completion requires horizon length 128")
    require(manifest.get("automatic_resume", False) is False,
            "R5 completion forbids automatic resume")
    require(manifest.get("seed0_baseline") or manifest.get("seed0_analysis_report"),
            "R5 manifest must identify the seed-0 comparison baseline")


def verify_sources(frozen):
    require(bool(frozen), "Frozen source identities are required")
    for path, expected in frozen.items():
        require(sha(path) == expected, f"Frozen source changed: {path}")


def validate_supervisor(status):
    """A raw exit zero alone is insufficient when a supervisor sent a signal."""
    require(status.get("finalized") is True, "Supervisor did not finalize")
    require(status.get("raw_wait_returncode") == 0, "Training child wait was not zero")
    require(status.get("termination_reason") == "child_exit", "Training did not exit normally")
    require(status.get("error") is None, "Supervisor recorded an error")
    require(not status.get("stop_signals"), "Supervisor stopped the child")
    require(not status.get("received_supervisor_signals"), "Supervisor received a signal")
    require(status.get("child_signal") is None, "Training child was signalled")
    require(status.get("guard_required") is False, "Expected explicitly selected no-guard mode")


def validate_training(audit, supervisor, finalizer, console):
    validate_supervisor(supervisor)
    rows = audit.get("epochs", [])
    require([row.get("epoch") for row in rows] == list(range(1, 401)),
            "Training audit is not contiguous epochs 1..400")
    require(all(row.get("environment_transitions") == TRANSITIONS_PER_EPOCH for row in rows),
            "An epoch does not contain 32*128 transitions")
    require(all(row.get("epoch_loss_arrays_finite") is True for row in rows),
            "Observed loss arrays are missing or nonfinite")
    require(audit.get("instrumentation_errors") == [], "Training audit instrumentation errors")
    require(not any(row.get("operation") == "restore" for row in audit.get("checkpoint_events", [])),
            "Unexpected restore; this protocol requires fresh training")
    runtime = audit.get("runtime", {})
    require(runtime.get("num_actors") == 32 and runtime.get("horizon_length") == 128,
            "Actual training dimensions differ from registration")
    require(finalizer.get("status") == "complete" and finalizer.get("training_error") is None,
            "TensorBoard finalizer did not complete")
    require(bool(finalizer.get("agents")) and all(
        not record.get("writer_expected", True) or record.get("success") is True
        for record in finalizer["agents"]), "Expected TensorBoard writer did not finalize")
    require("MAX EPOCHS NUM!" in console, "Official maximum-epoch completion marker missing")
    return {"completed_epochs": 400, "completed_transitions": TOTAL_TRANSITIONS,
            "observed_epoch_loss_arrays_finite": True,
            "gradient_finiteness_measured": False,
            "automatic_resume": False}


def inspect_trusted_checkpoint(path):
    """Inspect only a newly generated local training artifact, entirely on CPU."""
    import torch
    torch.set_num_threads(1)
    state = torch.load(path, map_location="cpu", weights_only=False)

    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        if isinstance(value, float):
            return math.isfinite(value)
        # RL-Games checkpoints may contain numpy scalars/arrays.
        if type(value).__module__.split(".")[0] == "numpy":
            import numpy as np
            return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
        return True

    optimizer = state.get("optimizer", {}).get("state", {})
    steps = [int(item["step"]) for item in optimizer.values() if "step" in item]
    nonzero = sum(bool(torch.count_nonzero(item["exp_avg"])) for item in optimizer.values()
                  if "exp_avg" in item)
    return {"path": str(Path(path).resolve()), "size_bytes": Path(path).stat().st_size,
            "sha256": sha(path), "epoch": state.get("epoch"), "frame": state.get("frame"),
            "all_tensors_and_numeric_arrays_finite": finite(state),
            "optimizer_state_count": len(optimizer), "optimizer_step_min": min(steps, default=0),
            "optimizer_step_max": max(steps, default=0), "nonzero_adam_exp_avg_states": nonzero,
            "actor_model_present": "model" in state,
            "central_value_present": "assymetric_vf_nets" in state}


def validate_checkpoint(record, epoch):
    require(record.get("epoch") == epoch and record.get("frame") == epoch * TRANSITIONS_PER_EPOCH,
            f"Checkpoint epoch/frame mismatch for epoch {epoch}")
    require(record.get("all_tensors_and_numeric_arrays_finite") is True,
            f"Nonfinite checkpoint at epoch {epoch}")
    require(record.get("optimizer_step_min", 0) > 0 and record.get("nonzero_adam_exp_avg_states", 0) > 0,
            f"Missing actual optimization state at epoch {epoch}")
    require(record.get("actor_model_present") is True and record.get("central_value_present") is True,
            f"Checkpoint models missing at epoch {epoch}")


def select_checkpoint(directory, audit, epoch, *, training_cwd=None):
    """Choose the last audited milestone save, including the official final save.

    Pinned RL-Games writes a periodic epoch-400 file and then a distinct final
    epoch-400 filename with ``_rew__..._``. Both remain in the inventory. The
    chronological read-only save audit, not filesystem ordering, selects the
    final save; the special final filename is checked independently.
    """
    directory = Path(directory).resolve()
    candidates = sorted(directory.glob(f"last_Factory_ep_{epoch}_rew_*.pth"))
    by_path = {path.resolve(): path for path in candidates}
    matching = []
    for event in audit.get("checkpoint_events", []):
        if (event.get("operation") != "save" or event.get("epoch") != epoch or
                event.get("frame") != epoch * TRANSITIONS_PER_EPOCH):
            continue
        filename = Path(event.get("filename", ""))
        saved = Path(str(filename) + ".pth") if filename.suffix != ".pth" else filename
        if not saved.is_absolute():
            require(training_cwd is not None, "Relative checkpoint audit filename requires training_cwd")
            saved = Path(training_cwd) / saved
        if saved.resolve() in by_path:
            matching.append((by_path[saved.resolve()], event))
    require(bool(matching), f"No existing audited milestone checkpoint for epoch {epoch}")
    path, event = matching[-1]
    if epoch == 400:
        require(path.name.startswith("last_Factory_ep_400_rew__") and path.name.endswith("_.pth"),
                "Last epoch-400 save is not the official maximum-epoch final checkpoint")
    return path, {"basis": "last matching checkpoint save in chronological training audit",
                  "selected_save_event": event, "all_milestone_candidate_paths": [str(p) for p in candidates],
                  "all_candidates_preserved": True}


def validate_evaluation(metrics, checkpoint, epoch, seed):
    episodes = 3 if epoch == 400 else 1
    count = 32 * episodes
    require(metrics.get("status") == "complete", "Evaluation is incomplete")
    require(metrics.get("checkpoint_sha256") == checkpoint["sha256"], "Evaluation checkpoint identity differs")
    require(metrics.get("seed") == seed and metrics.get("num_envs") == 32,
            "Evaluation seed/environment count differs")
    require(metrics.get("deterministic") is True and metrics.get("auto_reset_order_verified") is True,
            "Evaluation deterministic/reset evidence missing")
    require(metrics.get("official_success_capture") == "pre_reset_factory_get_curr_successes",
            "Official pre-reset success capture missing")
    require(metrics.get("episodes_completed") == count and len(metrics.get("episodes", [])) == count,
            "Evaluation completed-episode denominator differs")
    require(metrics.get("episodes_per_env_requested") == episodes, "Evaluation episode request differs")
    ids = [(row.get("env_id"), row.get("episode_id")) for row in metrics["episodes"]]
    require(set(ids) == {(env, ep) for env in range(32) for ep in range(episodes)} and len(set(ids)) == count,
            "Evaluation identifiers are duplicate or incomplete")
    aggregate = metrics.get("aggregate", {})
    require(aggregate.get("episodes_completed") == count, "Aggregate denominator differs")
    for kind in ("ever_success", "terminal_success"):
        values = [row.get(kind) for row in metrics["episodes"]]
        require(all(type(value) is bool for value in values), "Evaluation success masks must be boolean")
        require(aggregate.get(kind + "_count") == sum(values), "Aggregate success count differs")
        rate = aggregate.get(kind + "_rate")
        require(isinstance(rate, (int, float)) and math.isclose(rate, sum(values) / count),
                "Aggregate success rate differs")
    require(all(not row["terminal_success"] or row["ever_success"] for row in metrics["episodes"]),
            "Terminal success without ever success")
    dt = metrics.get("step_dt_seconds")
    require(isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0,
            "Evaluation step duration missing/nonfinite")
    for row in metrics["episodes"]:
        require(row.get("steps") == 149 and row.get("terminal_kind") == "timeout",
                "Factory episode horizon/reset differs")
        step, elapsed = row.get("first_success_step"), row.get("first_success_time_seconds")
        if row["ever_success"]:
            require(type(step) is int and 1 <= step <= 149 and
                    isinstance(elapsed, (int, float)) and math.isclose(elapsed, step * dt),
                    "First-success timing differs from observed episode steps")
        else:
            require(step is None and elapsed is None, "Failed episode must not have first-success timing")
    return {"epoch": epoch, "seed": seed, "episodes_completed": count,
            "checkpoint_sha256": checkpoint["sha256"], "aggregate": aggregate}


def evaluation_command(manifest, checkpoint, epoch, seed, output):
    repo, scripts = Path(manifest["repository"]), Path(manifest["repository"]) / "scripts"
    episodes = 3 if epoch == 400 else 1
    prefix = manifest["evaluation_unit_prefix"] + f"-e{epoch}-s{seed}"
    return [str(manifest.get("system_python", "/usr/bin/python3")), str(manifest["launcher"]),
            "--output", str(output), "--unit-prefix", prefix,
            "--cwd", str(manifest["evaluation_cwd"]), *manifest.get("launcher_arguments", []),
            "--setenv", "FACTORY_VENV=" + str(Path(manifest["python"]).parent.parent),
            "--setenv", "ISAACSIM_PATH=" + str(manifest["isaacsim_path"]),
            "--setenv", "OMP_NUM_THREADS=4", "--wait", "--",
            "env", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1", "bash", str(scripts / "run_sim.sh"),
            str(scripts / "evaluate_checkpoint.py"), "--headless", "--device", "cuda:0",
            "--task", "Isaac-Factory-PegInsert-Direct-v0", "--checkpoint", checkpoint["path"],
            "--num_envs", "32", "--episodes", str(episodes), "--seed", str(seed),
            "--max_steps", str(160 * episodes), "--snapshot-limit", "12", "--progress-interval", "10",
            "--output", str(output / "metrics.json")]


def recorded_command(command, prefix, *, cwd=None, environment=None):
    """No shell, retries, or destructive command handling; preserve real exit."""
    prefix = Path(prefix)
    atomic_json(prefix.with_suffix(".command.json"), {"command": command, "cwd": str(cwd) if cwd else None,
                "started": stamp()}, replace=False)
    with prefix.with_suffix(".stdout.txt").open("x") as stdout, prefix.with_suffix(".stderr.txt").open("x") as stderr:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr)
    atomic_json(prefix.with_suffix(".exit.json"), {"returncode": result.returncode, "ended": stamp()}, replace=False)
    require(result.returncode == 0, f"Command failed ({result.returncode}); see {prefix}")
    return prefix.with_suffix(".stdout.txt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--inspect-checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inspect_checkpoint:
        print(json.dumps(inspect_trusted_checkpoint(args.inspect_checkpoint), allow_nan=False))
        return 0
    if not args.manifest:
        parser.error("--manifest is required")
    manifest = load(args.manifest)
    output = Path(manifest["completion_output"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "command.json", {"argv": sys.argv, "manifest": manifest,
                "manifest_path": str(args.manifest.resolve()), "manifest_sha256": sha(args.manifest),
                "started": stamp(), "automatic_training_launch_or_resume": False,
                "gpu_guard_required": False, "gpu_telemetry_dependency": False}, replace=False)
    progress = {"schema_version": "g3cr5_completion_v1", "phase": "waiting_for_training",
                "started": stamp(), "training": None, "selected_checkpoints": {}, "evaluations": []}

    def pulse():
        atomic_json(output / "progress.json", progress, durable=False)

    try:
        validate_training_identity(manifest)
        verify_sources(manifest["frozen_source_sha256"])
        training_status = Path(manifest["training_supervisor_directory"]) / "status.json"
        while not training_status.is_file():
            pulse()
            time.sleep(2)
        progress["phase"] = "validating_training"
        pulse()
        supervisor = load(training_status)
        # Reject failed supervision first; missing scientific artifacts are then
        # recorded as missing evidence rather than a reason to restart training.
        validate_supervisor(supervisor)
        verify_sources(manifest["frozen_source_sha256"])
        audit = load(manifest["training_audit"])
        finalizer = load(manifest["training_finalizer"])
        console = (Path(manifest["training_supervisor_directory"]) / "console.txt").read_text(errors="replace")
        progress["training"] = validate_training(audit, supervisor, finalizer, console)
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for name in ("FACTORY_G2_AUDIT_OUTPUT", "FACTORY_G3A_FINALIZE_TENSORBOARD", "FACTORY_G3A_FINALIZER_OUTPUT"):
            environment.pop(name, None)
        scripts = Path(manifest["repository"]) / "scripts"
        tail_stdout = recorded_command([str(manifest["python"]), str(scripts / "tensorboard_tail.py"),
            str(manifest["training_event_directory"]), "--expected-epochs", "400",
            "--transitions-per-epoch", "4096"], output / "tensorboard_tail", environment=environment)
        tail = load(tail_stdout)
        require(tail.get("status") == "complete", "TensorBoard strict tail validation incomplete")
        progress["training"]["tensorboard_tail_complete"] = True
        checkpoint_directory = Path(manifest["checkpoint_directory"])
        inventory = [{"path": str(path.resolve()), "sha256": sha(path), "size_bytes": path.stat().st_size}
                     for path in sorted(checkpoint_directory.glob("*.pth"))]
        atomic_json(output / "checkpoint_inventory.json", inventory, replace=False)
        for epoch in EPOCHS:
            checkpoint_path, selection = select_checkpoint(checkpoint_directory, audit, epoch,
                training_cwd=manifest.get("training_cwd", manifest["evaluation_cwd"]))
            inspected_stdout = recorded_command([str(manifest["python"]), str(Path(__file__).resolve()),
                "--inspect-checkpoint", str(checkpoint_path)], output / f"checkpoint_epoch{epoch}", environment=environment)
            checkpoint = load(inspected_stdout)
            validate_checkpoint(checkpoint, epoch)
            checkpoint["selection"] = selection
            progress["selected_checkpoints"][str(epoch)] = checkpoint
            pulse()
        atomic_json(output / "selected_checkpoints.json", progress["selected_checkpoints"], replace=False)
        progress["phase"] = "evaluating"
        for epoch in EPOCHS:
            checkpoint = progress["selected_checkpoints"][str(epoch)]
            for seed in SEEDS:
                verify_sources(manifest["frozen_source_sha256"])
                require(sha(checkpoint["path"]) == checkpoint["sha256"], "Selected checkpoint changed")
                evaluation_output = output / "evaluation" / f"epoch{epoch:03d}" / f"seed{seed}"
                evaluation_output.parent.mkdir(parents=True, exist_ok=True)
                command = evaluation_command(manifest, checkpoint, epoch, seed, evaluation_output)
                progress["active_evaluation"] = {"epoch": epoch, "seed": seed, "command": command}
                pulse()
                recorded_command(command, output / f"evaluation_epoch{epoch}_seed{seed}",
                                 cwd=manifest["evaluation_cwd"], environment=environment)
                evaluation_supervisor = load(evaluation_output / "supervisor" / "status.json")
                validate_supervisor(evaluation_supervisor)
                result = validate_evaluation(load(evaluation_output / "metrics.json"), checkpoint, epoch, seed)
                result["metrics"] = str(evaluation_output / "metrics.json")
                progress["evaluations"].append(result)
                pulse()
        verify_sources(manifest["frozen_source_sha256"])
        progress.pop("active_evaluation", None)
        progress["phase"] = "training_and_evaluation_complete"
        returncode = 0
    except BaseException as error:
        progress["phase"] = "incomplete"
        progress["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        returncode = 1
    progress["ended"] = stamp()
    progress["controller_exit_code"] = returncode
    atomic_json(output / "status.json", progress, replace=False)
    pulse()
    print(json.dumps({"phase": progress["phase"], "output": str(output), "returncode": returncode}), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
