#!/usr/bin/env python3
"""[ENGINEERING] Explicit G3C-R5 seed-1 launcher without a GPU guard.

This is the user-authorized supervision mode for the independent 32-environment
training-seed-1 experiment.  The unchanged durable supervisor owns the device
lock, child wait, console, and exit evidence.  No NVML/GPU guard is started or
required: missing telemetry can never terminate the child.  CUDA's own error
handling, the official task, PPO implementation, and checkpoint writer remain
in the child command.

Training invocations must explicitly select ``--training-seed 1`` and use the
registered 32-env/400-epoch configuration.  Evaluation invocations may use the
same launcher with ``--launch-kind evaluation``; their checkpoint is read-only
and their evaluation seed is distinct from the training seed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from supervision_common import atomic_json, read_json, sha, stamp

POLICY = "user-authorized-no-gpu-guard"
TRAINING_SEED = 1
NUM_ENVS = 32
EPOCHS = 400
HORIZON_LENGTH = 128
TOTAL_TRANSITIONS = NUM_ENVS * EPOCHS * HORIZON_LENGTH


def status_exit_code(status):
    """Map only a finalized child result to the launcher result."""
    code = status.get("raw_wait_returncode")
    if status.get("error") is not None:
        return 124
    if status.get("termination_reason") == "child_exit" and type(code) is int and code >= 0:
        return code if code <= 255 else 124
    if status.get("termination_reason") == "child_signal" and type(code) is int and code < 0:
        return min(128 - code, 255)
    return 124


def unit_state(unit):
    """Diagnostic only; a dead service does not fabricate a child result."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=LoadState",
             "--property=ActiveState", "--property=SubState", "--property=Result",
             "--property=ExecMainCode", "--property=ExecMainStatus"],
            capture_output=True, text=True, timeout=5,
        )
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        return {"returncode": result.returncode, "properties": values,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": {"type": type(exc).__name__, "message": str(exc)}}


def wait_for_status(output, unit, timeout):
    """Wait on durable status only; no GPU/guard state is consulted."""
    status_path = output / "supervisor/status.json"
    until = time.monotonic() + timeout + 60
    next_probe = time.monotonic() + 2
    dead_since = None
    last_probe = None
    while time.monotonic() < until:
        status = read_json(status_path)
        if status and status.get("finalized") is True:
            code = status_exit_code(status)
            atomic_json(output / "wait_result.json", {
                "schema_version": 1, "status_source": str(status_path),
                "raw_wait_returncode": status.get("raw_wait_returncode"),
                "termination_reason": status.get("termination_reason"),
                "supervisor_error": status.get("error"), "launcher_exit_code": code,
                "monitoring_policy": POLICY, **stamp(),
            }, replace=False)
            return code
        now = time.monotonic()
        if now >= next_probe:
            last_probe = unit_state(unit)
            properties = last_probe.get("properties", {})
            dead = (properties.get("ActiveState") in ("inactive", "failed")
                    or properties.get("LoadState") == "not-found")
            dead_since = (dead_since or now) if dead else None
            if dead_since is not None and now - dead_since >= 2:
                break
            next_probe = now + 1
        time.sleep(.2)
    atomic_json(output / "wait_result.json", {
        "schema_version": 1, "status_source": None, "raw_wait_returncode": None,
        "termination_reason": "supervisor_status_unavailable",
        "launcher_exit_code": 124, "last_service_probe": last_probe,
        "monitoring_policy": POLICY, **stamp(),
    }, replace=False)
    return 124


def _value_after(command, option):
    prefix = option + "="
    for item in command:
        if str(item).startswith(prefix):
            return str(item)[len(prefix):]
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _validate_training_command(command, parser):
    """Reject accidental seed-0/resume launches before systemd submission."""
    seed = _value_after(command, "--seed")
    if seed is None or seed != str(TRAINING_SEED):
        parser.error("training command must contain --seed 1")
    envs = _value_after(command, "--num_envs")
    if envs is None or envs != str(NUM_ENVS):
        parser.error("training command must contain --num_envs 32")
    epochs = _value_after(command, "--max_iterations")
    if epochs is None:
        epochs = _value_after(command, "--max_epochs")
    if epochs != str(EPOCHS):
        parser.error("training command must select 400 epochs")
    if "Isaac-Factory-PegInsert-Direct-v0" not in command:
        parser.error("training command must select the official Factory PegInsert task")
    lower = [str(item).lower() for item in command]
    forbidden = ("--checkpoint", "--load", "--resume", "--resume_path", "--restore")
    if any(item in forbidden or "last_factory" in item or "checkpoint" in item and item.startswith("--")
           for item in lower):
        parser.error("seed-1 training must start from scratch and cannot restore a checkpoint")


def _kind(command, requested, training_seed=None):
    if requested != "auto":
        return requested
    names = {Path(str(item)).name for item in command}
    if "evaluate_checkpoint.py" in names:
        return "evaluation"
    # The explicit training-seed selector is itself sufficient to make an
    # invocation a training launch when a test or wrapper uses a generic child
    # command instead of the upstream train.py path.
    return "training" if ("train.py" in names or training_seed is not None
                          or "--max_iterations" in command or "--max_epochs" in command) else "evaluation"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitoring-policy", required=True, choices=[POLICY])
    parser.add_argument("--training-seed", type=int,
                        help="Required value 1 for a fresh training invocation")
    parser.add_argument("--launch-kind", choices=("auto", "training", "evaluation"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit-prefix", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path,
                        default=Path(f"/run/user/{os.getuid()}/isaac-factory-gpu0.lock"))
    parser.add_argument("--setenv", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=43200)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not all(c.isascii() and (c.isalnum() or c == "-") for c in args.unit_prefix):
        parser.error("Command and ASCII alphanumeric/hyphen unit prefix required")
    if not 0 < args.timeout <= 43200 or not args.cwd.is_dir():
        parser.error("Timeout in (0,43200] and an existing cwd required")
    from durable_supervisor import SAFE_ENV
    if any("=" not in item or item.split("=", 1)[0] not in SAFE_ENV for item in args.setenv):
        parser.error("Only documented safe environment keys supported")
    kind = _kind(command, args.launch_kind, args.training_seed)
    if kind == "training":
        if args.training_seed != TRAINING_SEED:
            parser.error("--training-seed 1 is required for G3C-R5 training")
        _validate_training_command(command, parser)
    elif args.training_seed is not None and args.training_seed != TRAINING_SEED:
        parser.error("--training-seed, when supplied, must be 1")
    args.output = args.output.resolve()
    args.cwd = args.cwd.resolve()
    args.lock_file = args.lock_file.resolve()
    scripts = Path(__file__).resolve().parent
    # Fresh creation prevents overwrite, resume, or concatenation of evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    unit = args.unit_prefix + "-supervisor"
    supervisor_command = [
        sys.executable, str(scripts / "durable_supervisor.py"),
        "--output", str(args.output / "supervisor"), "--lock-file", str(args.lock_file),
        "--timeout", str(args.timeout), "--", *command,
    ]
    service_command = [
        "systemd-run", "--user", "--unit=" + unit,
        "--property=WorkingDirectory=" + str(args.cwd), "--property=Restart=no",
        "--property=KillMode=control-group", "--property=TimeoutStopSec=10",
        *["--setenv=" + value for value in args.setenv], *supervisor_command,
    ]
    plan = {
        "schema_version": 1, "engineering": True, "created": stamp(),
        "experiment": "G3C-R5-independent-training-seed-1",
        "launch_kind": kind, "monitoring_policy": args.monitoring_policy,
        "authorization": "G3C-R5 user explicitly continues the G3C-R4 authorized no-GPU-guard policy for the independent seed-1 run, retaining the previously accepted CUDA OOM, runtime failure, desktop slowdown, and restart risks.",
        "gpu_guard_enabled": False, "gpu_telemetry_required": False,
        "gpu_telemetry_gaps_terminate_child": False, "continuous_gpu_coverage_claimed": False,
        "normal_cuda_error_handling": "unchanged", "automatic_retry": False,
        "automatic_resume": False, "fresh_start": kind == "training",
        "checkpoint_restore_requested": False if kind == "training" else None,
        "training_seed": TRAINING_SEED if kind == "training" else None,
        "evaluation_seeds_are_not_training_seeds": True,
        "scientific_configuration": {"num_envs": NUM_ENVS, "epochs": EPOCHS,
                                     "horizon_length": HORIZON_LENGTH,
                                     "planned_transitions": TOTAL_TRANSITIONS},
        "training_command": command, "cwd": str(args.cwd), "safe_environment": args.setenv,
        "timeout_seconds": args.timeout, "lock_file": str(args.lock_file),
        "units": {"supervisor": unit}, "unit_commands": {"supervisor": service_command},
        "source_sha256": {str(scripts / name): sha(scripts / name) for name in (
            "supervision_common.py", "supervised_exec.py", "durable_supervisor.py", "g3cr5_launch.py")},
    }
    atomic_json(args.output / "launch.json", plan, replace=False)
    try:
        result = subprocess.run(service_command, capture_output=True, text=True)
        submission = {"role": "supervisor", "returncode": result.returncode,
                      "stdout": result.stdout, "stderr": result.stderr, **stamp()}
        submission_code = result.returncode
    except OSError as exc:
        submission = {"role": "supervisor", "returncode": None,
                      "error": {"type": type(exc).__name__, "message": str(exc)}, **stamp()}
        submission_code = 127
    atomic_json(args.output / "supervisor_submission.json", submission, replace=False)
    if submission_code:
        return submission_code
    print(str(args.output), flush=True)
    return wait_for_status(args.output, unit, args.timeout) if args.wait else 0


if __name__ == "__main__":
    raise SystemExit(main())
