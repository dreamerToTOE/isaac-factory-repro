"""CPU evidence for the explicitly selected G3C-R4 supervision policy.

Fake systemd submission tests run the real unchanged durable supervisor and a
finite CPU child. The final lifecycle test submits only a fresh sacrificial unit.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import pytest

from scripts.supervision_common import read_json

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'scripts/g3cr4_launch.py'
PY = sys.executable
POLICY = 'user-authorized-no-gpu-guard'


def await_json(path, timeout=20):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = read_json(path)
        if value:
            return value
        time.sleep(.02)
    raise AssertionError(f'No valid JSON at {path}')


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('g3cr4_launch')


@pytest.fixture
def fake_systemd(tmp_path, monkeypatch):
    """Submission succeeds once; actual child result is in supervisor/status."""
    tools = tmp_path / 'bin'
    tools.mkdir()
    record = tmp_path / 'submitted.json'
    fake = tools / 'systemd-run'
    fake.write_text(f'''#!{PY}
import json,os,subprocess,sys
from pathlib import Path
args=sys.argv[1:]
record=Path({str(record)!r})
previous=json.loads(record.read_text()) if record.exists() else []
previous.append(args)
record.write_text(json.dumps(previous))
environment=dict(os.environ)
cwd=None
for value in args:
    if value.startswith('--setenv='):
        key,item=value[len('--setenv='):].split('=',1);environment[key]=item
    if value.startswith('--property=WorkingDirectory='):
        cwd=value.split('=',2)[2]
first=next(i for i,value in enumerate(args) if not value.startswith('--'))
subprocess.run(args[first:],cwd=cwd,env=environment)
raise SystemExit(0)
''')
    fake.chmod(0o755)
    monkeypatch.setenv('PATH', str(tools) + os.pathsep + os.environ['PATH'])
    return record


def command(tmp_path, code, *, policy=True, wait=True):
    args = [PY, str(LAUNCH), '--output', str(tmp_path / 'run'),
            '--unit-prefix', 'g3cr4-cpu-test', '--cwd', str(ROOT),
            '--lock-file', str(tmp_path / 'lock'), '--timeout', '10']
    if policy:
        args += ['--monitoring-policy', POLICY]
    if wait:
        args += ['--wait']
    return args + ['--', PY, '-c', code]


def test_explicit_policy_required_before_any_output_or_submission(tmp_path, fake_systemd):
    result = subprocess.run(command(tmp_path, 'pass', policy=False), capture_output=True, text=True)
    assert result.returncode == 2
    assert '--monitoring-policy' in result.stderr
    assert not (tmp_path / 'run').exists() and not fake_systemd.exists()


def test_policy_manifest_and_only_supervisor_are_immutable(tmp_path, fake_systemd):
    result = subprocess.run(command(tmp_path, "print('done')"), capture_output=True, text=True, timeout=20)
    plan = read_json(tmp_path / 'run/launch.json')
    status = read_json(tmp_path / 'run/supervisor/status.json')
    assert result.returncode == 0 and status['raw_wait_returncode'] == 0
    assert plan['monitoring_policy'] == POLICY
    assert 'explicitly accepts' in plan['authorization']
    assert plan['gpu_guard_enabled'] is False
    assert plan['gpu_telemetry_required'] is False
    assert plan['gpu_telemetry_gaps_terminate_child'] is False
    assert plan['continuous_gpu_coverage_claimed'] is False
    assert plan['normal_cuda_error_handling'] == 'unchanged'
    assert plan['automatic_retry'] is False and plan['automatic_resume'] is False
    assert set(plan['units']) == {'supervisor'}
    assert '--guard-directory' not in plan['unit_commands']['supervisor']
    assert not any('gpu_guard.py' in item for item in plan['unit_commands']['supervisor'])
    assert '--property=Restart=no' in plan['unit_commands']['supervisor']
    assert status['guard_required'] is False and status['guard_identity'] is None
    assert len(json.loads(fake_systemd.read_text())) == 1
    assert (tmp_path / 'run/supervisor/console.txt').read_text() == 'done\n'


def test_missing_telemetry_does_not_prevent_or_stop_child(tmp_path, fake_systemd):
    result = subprocess.run(command(tmp_path, "import time;time.sleep(.15);print('normal without telemetry')"),
                            capture_output=True, text=True, timeout=20)
    status = read_json(tmp_path / 'run/supervisor/status.json')
    assert result.returncode == 0 and status['termination_reason'] == 'child_exit'
    assert not (tmp_path / 'run/guard').exists()
    assert status['stop_signals'] == []


def test_stale_error_telemetry_is_not_read_or_used_to_signal(tmp_path, fake_systemd):
    guard = tmp_path / 'run/guard'
    code = (f"import pathlib,time,json;d=pathlib.Path({str(guard)!r});d.mkdir();"
            "(d/'heartbeat.json').write_text(json.dumps({'monotonic_seconds':0,'free_MiB':0}));"
            "(d/'status.json').write_text(json.dumps({'stop_reason':'monitoring_gap'}));"
            "time.sleep(.3);print('stale telemetry ignored')")
    result = subprocess.run(command(tmp_path, code), capture_output=True, text=True, timeout=20)
    status = read_json(tmp_path / 'run/supervisor/status.json')
    assert result.returncode == 0 and status['raw_wait_returncode'] == 0
    assert status['stop_signals'] == [] and status['termination_reason'] == 'child_exit'


def test_nonzero_child_exit_is_exact_and_never_retried(tmp_path, fake_systemd):
    result = subprocess.run(command(tmp_path, "print('single attempt');raise SystemExit(7)"),
                            capture_output=True, text=True, timeout=20)
    status = read_json(tmp_path / 'run/supervisor/status.json')
    wait = read_json(tmp_path / 'run/wait_result.json')
    assert result.returncode == 7
    assert status['child_exit_code'] == wait['raw_wait_returncode'] == 7
    assert wait['launcher_exit_code'] == 7
    assert len(json.loads(fake_systemd.read_text())) == 1
    assert (tmp_path / 'run/supervisor/console.txt').read_text().count('single attempt') == 1


def test_cuda_oom_exception_console_is_preserved_without_retry(tmp_path, fake_systemd):
    result = subprocess.run(command(tmp_path, "raise RuntimeError('CUDA out of memory. simulated CPU-only failure')"),
                            capture_output=True, text=True, timeout=20)
    console = (tmp_path / 'run/supervisor/console.txt').read_text()
    status = read_json(tmp_path / 'run/supervisor/status.json')
    assert result.returncode == status['child_exit_code'] == 1
    assert 'Traceback (most recent call last)' in console
    assert 'RuntimeError: CUDA out of memory. simulated CPU-only failure' in console
    assert status['termination_reason'] == 'child_exit' and status['error'] is None
    assert len(json.loads(fake_systemd.read_text())) == 1


def test_signal_is_not_reported_as_normal_exit(tmp_path, fake_systemd):
    code = 'import os,signal;os.kill(os.getpid(),signal.SIGTERM)'
    result = subprocess.run(command(tmp_path, code), capture_output=True, text=True, timeout=20)
    status = read_json(tmp_path / 'run/supervisor/status.json')
    assert result.returncode == 143
    assert status['raw_wait_returncode'] == -15 and status['child_signal'] == 'SIGTERM'
    assert status['child_exit_code'] is None


def test_existing_run_directory_is_never_reused(tmp_path, fake_systemd):
    first = subprocess.run(command(tmp_path, 'pass'), capture_output=True, text=True, timeout=20)
    before = (tmp_path / 'run/launch.json').read_bytes()
    second = subprocess.run(command(tmp_path, 'raise SystemExit(8)'), capture_output=True, text=True, timeout=20)
    assert first.returncode == 0 and second.returncode != 0
    assert (tmp_path / 'run/launch.json').read_bytes() == before
    assert len(json.loads(fake_systemd.read_text())) == 1


def test_exclusive_device_lock_prevents_duplicate_child(tmp_path, fake_systemd):
    import fcntl
    lock = (tmp_path / 'lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = tmp_path / 'must-not-run'
    # No wait: failed supervisor's own no-child exit is retained in submission console.
    try:
        result = subprocess.run(command(tmp_path, f"open({str(marker)!r},'w').close()", wait=False),
                                capture_output=True, text=True, timeout=20)
    finally:
        lock.close()
    assert result.returncode == 0  # submission success does not claim child success
    assert 'duplicate launch prevented' in read_json(tmp_path / 'run/supervisor_submission.json')['stderr']
    assert not marker.exists() and not (tmp_path / 'run/supervisor/status.json').exists()


def test_timeout_preserved_as_supervision_failure(tmp_path, fake_systemd):
    args = command(tmp_path, 'import time;time.sleep(5)')
    args[args.index('--timeout') + 1] = '.15'
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    status = read_json(tmp_path / 'run/supervisor/status.json')
    assert result.returncode == 124
    assert status['termination_reason'] == 'timeout' and status['raw_wait_returncode'] < 0
    assert status['stop_signals']


def test_exit_code_mapping_rejects_unfinalized_or_supervision_error(module):
    assert module.status_exit_code({'raw_wait_returncode': 0, 'termination_reason': 'timeout', 'error': None}) == 124
    assert module.status_exit_code({'raw_wait_returncode': 0, 'termination_reason': 'child_exit', 'error': {'type': 'OSError'}}) == 124
    assert module.status_exit_code({'raw_wait_returncode': None, 'termination_reason': 'supervisor_error', 'error': None}) == 124


def test_missing_status_dead_service_is_explicit_without_forged_child_status(tmp_path, module, monkeypatch):
    output = tmp_path / 'run'
    output.mkdir()
    monkeypatch.setattr(module, 'unit_state', lambda unit: {'properties': {'LoadState': 'not-found'}})
    # Move monotonic time forward per call while leaving the status reader real.
    counter = iter(range(1000, 2000))
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(counter))
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    assert module.wait_for_status(output, 'missing-unit', 10) == 124
    status = read_json(output / 'wait_result.json')
    assert status['raw_wait_returncode'] is None
    assert status['termination_reason'] == 'supervisor_status_unavailable'


def test_only_documented_environment_keys_can_be_injected(tmp_path, fake_systemd):
    args = command(tmp_path, 'pass')
    args[args.index('--'):args.index('--')] = ['--setenv', 'CUDA_LAUNCH_BLOCKING=1']
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and not (tmp_path / 'run').exists()


def test_systemd_supervisor_survives_launcher_sigkill(tmp_path):
    available = subprocess.run(['systemctl', '--user', 'show-environment'], capture_output=True)
    if available.returncode:
        pytest.skip('No user systemd manager for the finite lifecycle test')
    args = command(tmp_path, "import time;time.sleep(.8);print('survived submitting client')")
    args[args.index('--unit-prefix') + 1] = 'g3cr4-cpu-' + uuid.uuid4().hex[:10]
    launcher = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        await_json(tmp_path / 'run/supervisor/exec_release.json')
        launcher.kill()
        launcher.communicate(timeout=5)
        status = await_json(tmp_path / 'run/supervisor/status.json')
        assert status['raw_wait_returncode'] == 0 and status['termination_reason'] == 'child_exit'
        assert status['guard_required'] is False
        assert 'survived submitting client' in (tmp_path / 'run/supervisor/console.txt').read_text()
    finally:
        if launcher.poll() is None:
            launcher.terminate()
            launcher.communicate(timeout=5)
