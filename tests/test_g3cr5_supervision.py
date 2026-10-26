"""CPU-only contract tests for the explicit G3C-R5 seed-1 launcher."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.supervision_common import read_json

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts/g3cr5_launch.py"
PY = sys.executable
POLICY = "user-authorized-no-gpu-guard"


def _fake_systemd(tmp_path, monkeypatch):
    tools = tmp_path / "bin"
    tools.mkdir()
    record = tmp_path / "submitted.json"
    fake = tools / "systemd-run"
    fake.write_text(f'''#!{PY}
import json,os,subprocess,sys
from pathlib import Path
record=Path({str(record)!r})
old=json.loads(record.read_text()) if record.exists() else []
old.append(sys.argv[1:]); record.write_text(json.dumps(old))
env=dict(os.environ); cwd=None
for item in sys.argv[1:]:
    if item.startswith('--setenv='):
        key,value=item[len('--setenv='):].split('=',1); env[key]=value
    if item.startswith('--property=WorkingDirectory='):
        cwd=item.split('=',2)[2]
first=next(i for i,item in enumerate(sys.argv[1:],1) if not item.startswith('--'))
raise SystemExit(subprocess.run(sys.argv[first:],cwd=cwd,env=env).returncode)
''')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + ":" + str(Path(sys.executable).parent))
    return record


def _command(tmp_path, code="print('ok')", *, seed=1, training_seed=1,
             checkpoint=None, wait=True, kind="training", extra=()):
    args = [PY, str(LAUNCH), "--monitoring-policy", POLICY,
            "--output", str(tmp_path / "run"), "--unit-prefix", "g3cr5-cpu-test",
            "--cwd", str(ROOT), "--lock-file", str(tmp_path / "lock"), "--timeout", "10",
            "--launch-kind", kind]
    if training_seed is not None:
        args += ["--training-seed", str(training_seed)]
    if wait:
        args += ["--wait"]
    args += ["--", PY, "-c", code]
    if kind == "training":
        args += ["--seed", str(seed), "--num_envs", "32", "--max_iterations", "400",
                 "--task", "Isaac-Factory-PegInsert-Direct-v0"]
    elif checkpoint:
        args += ["--checkpoint", str(checkpoint)]
    args += list(extra)
    return args


def test_policy_and_training_seed_are_required_before_submission(tmp_path, monkeypatch):
    record = _fake_systemd(tmp_path, monkeypatch)
    no_policy = _command(tmp_path)
    no_policy[no_policy.index("--monitoring-policy") + 1] = "invalid-policy"
    assert subprocess.run(no_policy, capture_output=True, text=True).returncode == 2
    missing_seed = _command(tmp_path, training_seed=None)
    assert subprocess.run(missing_seed, capture_output=True, text=True).returncode == 2
    assert not (tmp_path / "run").exists()
    assert not record.exists()


def test_seed_zero_and_restore_are_rejected_without_side_effects(tmp_path, monkeypatch):
    record = _fake_systemd(tmp_path, monkeypatch)
    result = subprocess.run(_command(tmp_path, seed=0), capture_output=True, text=True)
    assert result.returncode == 2 and "--seed 1" in result.stderr
    # A checkpoint option must be rejected even when supplied after the command delimiter.
    args = _command(tmp_path / "second", extra=("--checkpoint", "seed0.pth"))
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2 and "start from scratch" in result.stderr
    assert not record.exists()


def test_fresh_seed1_plan_has_no_guard_and_preserves_exit(tmp_path, monkeypatch):
    record = _fake_systemd(tmp_path, monkeypatch)
    result = subprocess.run(_command(tmp_path, "print('seed1 child')"),
                            capture_output=True, text=True, timeout=20)
    plan = read_json(tmp_path / "run/launch.json")
    status = read_json(tmp_path / "run/supervisor/status.json")
    assert result.returncode == 0 and status["raw_wait_returncode"] == 0
    assert plan["experiment"] == "G3C-R5-independent-training-seed-1"
    assert plan["training_seed"] == 1 and plan["fresh_start"] is True
    assert plan["scientific_configuration"] == {"num_envs": 32, "epochs": 400,
                                                  "horizon_length": 128,
                                                  "planned_transitions": 1638400}
    assert plan["gpu_guard_enabled"] is False
    assert plan["gpu_telemetry_required"] is False
    assert plan["gpu_telemetry_gaps_terminate_child"] is False
    assert plan["normal_cuda_error_handling"] == "unchanged"
    assert "--guard-directory" not in str(plan["unit_commands"])
    assert status["guard_required"] is False and status["stop_signals"] == []
    assert "seed1 child" in (tmp_path / "run/supervisor/console.txt").read_text()
    assert len(json.loads(record.read_text())) == 1


def test_missing_gpu_telemetry_cannot_stop_child(tmp_path, monkeypatch):
    _fake_systemd(tmp_path, monkeypatch)
    result = subprocess.run(_command(tmp_path, "import time;time.sleep(.1);print('no telemetry')"),
                            capture_output=True, text=True, timeout=20)
    status = read_json(tmp_path / "run/supervisor/status.json")
    assert result.returncode == 0 and status["termination_reason"] == "child_exit"
    assert status["guard_required"] is False and status["stop_signals"] == []
    assert not (tmp_path / "run/guard").exists()


def test_cuda_error_is_recorded_once_without_retry(tmp_path, monkeypatch):
    record = _fake_systemd(tmp_path, monkeypatch)
    result = subprocess.run(_command(tmp_path, "raise RuntimeError('CUDA out of memory (CPU test)')"),
                            capture_output=True, text=True, timeout=20)
    status = read_json(tmp_path / "run/supervisor/status.json")
    console = (tmp_path / "run/supervisor/console.txt").read_text()
    assert result.returncode == 1 and status["raw_wait_returncode"] == 1
    assert status["termination_reason"] == "child_exit" and status["error"] is None
    assert "CUDA out of memory (CPU test)" in console
    assert len(json.loads(record.read_text())) == 1


def test_evaluation_kind_can_read_checkpoint_without_training_seed(tmp_path, monkeypatch):
    _fake_systemd(tmp_path, monkeypatch)
    checkpoint = tmp_path / "seed0-checkpoint.pth"
    checkpoint.write_bytes(b"trusted test placeholder")
    result = subprocess.run(_command(tmp_path, "print('evaluation')", training_seed=None,
                                     kind="evaluation", checkpoint=checkpoint),
                            capture_output=True, text=True, timeout=20)
    plan = read_json(tmp_path / "run/launch.json")
    assert result.returncode == 0 and plan["launch_kind"] == "evaluation"
    assert plan["training_seed"] is None and plan["evaluation_seeds_are_not_training_seeds"] is True


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    _fake_systemd(tmp_path, monkeypatch)
    first = subprocess.run(_command(tmp_path), capture_output=True, text=True, timeout=20)
    before = (tmp_path / "run/launch.json").read_bytes()
    second = subprocess.run(_command(tmp_path, "raise SystemExit(9)"), capture_output=True, text=True)
    assert first.returncode == 0 and second.returncode == 1
    assert (tmp_path / "run/launch.json").read_bytes() == before
