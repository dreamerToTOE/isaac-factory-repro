#!/usr/bin/env python3
"""[ENGINEERING] Explicit G3C-R4 mode: durable supervision without a GPU guard.

The human-authorized monitoring policy is required on every invocation. GPU
telemetry is neither sampled nor used as a termination condition. systemd owns
one Restart=no supervisor; its child keeps the exclusive device lock. CUDA's
normal exceptions, stderr, checkpoints, and application finalizers are unchanged.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from supervision_common import atomic_json, read_json, sha, stamp

POLICY = 'user-authorized-no-gpu-guard'


def status_exit_code(status):
    """Return the child result only when supervision completed without an error."""
    code = status.get('raw_wait_returncode')
    if status.get('error') is not None:
        return 124
    if status.get('termination_reason') == 'child_exit' and type(code) is int and code >= 0:
        return code if code <= 255 else 124
    if status.get('termination_reason') == 'child_signal' and type(code) is int and code < 0:
        return min(128 - code, 255)
    return 124


def unit_state(unit):
    """Diagnostic only: detect a dead service that could not publish status."""
    try:
        result = subprocess.run(
            ['systemctl', '--user', 'show', unit, '--property=LoadState',
             '--property=ActiveState', '--property=SubState', '--property=Result',
             '--property=ExecMainCode', '--property=ExecMainStatus'],
            capture_output=True, text=True, timeout=5,
        )
        values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        return {'returncode': result.returncode, 'properties': values,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': {'type': type(exc).__name__, 'message': str(exc)}}


def wait_for_status(output, unit, timeout):
    """Waiting client may disappear; it never owns or signals the training child."""
    status_path = output / 'supervisor/status.json'
    until = time.monotonic() + timeout + 60
    next_probe = time.monotonic() + 2
    dead_since = None
    last_probe = None
    while time.monotonic() < until:
        status = read_json(status_path)
        if status and status.get('finalized') is True:
            code = status_exit_code(status)
            atomic_json(output / 'wait_result.json', {
                'schema_version': 1, 'status_source': str(status_path),
                'raw_wait_returncode': status.get('raw_wait_returncode'),
                'termination_reason': status.get('termination_reason'),
                'supervisor_error': status.get('error'), 'launcher_exit_code': code,
                'monitoring_policy': POLICY, **stamp(),
            }, replace=False)
            return code
        now = time.monotonic()
        if now >= next_probe:
            last_probe = unit_state(unit)
            properties = last_probe.get('properties', {})
            dead = (properties.get('ActiveState') in ('inactive', 'failed')
                    or properties.get('LoadState') == 'not-found')
            dead_since = (dead_since or now) if dead else None
            # Allow a final fsync/status publication after service state changes.
            if dead_since is not None and now - dead_since >= 2:
                break
            next_probe = now + 1
        time.sleep(.2)
    # This is a client/supervision availability result, never a fabricated child exit.
    atomic_json(output / 'wait_result.json', {
        'schema_version': 1, 'status_source': None, 'raw_wait_returncode': None,
        'termination_reason': 'supervisor_status_unavailable',
        'launcher_exit_code': 124, 'last_service_probe': last_probe,
        'monitoring_policy': POLICY, **stamp(),
    }, replace=False)
    return 124


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monitoring-policy', required=True, choices=[POLICY])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit-prefix', required=True)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('--lock-file', type=Path,
                        default=Path(f'/run/user/{os.getuid()}/isaac-factory-gpu0.lock'))
    parser.add_argument('--setenv', action='append', default=[])
    parser.add_argument('--timeout', type=float, default=43200)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or not args.unit_prefix
            or not all(c.isascii() and (c.isalnum() or c == '-') for c in args.unit_prefix)):
        parser.error('Command and ASCII alphanumeric/hyphen unit prefix required')
    if not 0 < args.timeout <= 43200 or not args.cwd.is_dir():
        parser.error('Timeout in (0,43200] and an existing cwd required')
    from durable_supervisor import SAFE_ENV
    if any('=' not in item or item.split('=', 1)[0] not in SAFE_ENV for item in args.setenv):
        parser.error('Only documented safe environment keys supported')
    args.output = args.output.resolve()
    args.cwd = args.cwd.resolve()
    args.lock_file = args.lock_file.resolve()
    scripts = Path(__file__).resolve().parent
    # Exclusive creation prevents reuse, overwrites, or accidental concatenation.
    args.output.mkdir(parents=True, exist_ok=False)
    unit = args.unit_prefix + '-supervisor'
    supervisor_command = [
        sys.executable, str(scripts / 'durable_supervisor.py'),
        '--output', str(args.output / 'supervisor'), '--lock-file', str(args.lock_file),
        '--timeout', str(args.timeout), '--', *command,
    ]
    service_command = [
        'systemd-run', '--user', '--unit=' + unit,
        '--property=WorkingDirectory=' + str(args.cwd), '--property=Restart=no',
        '--property=KillMode=control-group', '--property=TimeoutStopSec=10',
        *['--setenv=' + value for value in args.setenv], *supervisor_command,
    ]
    plan = {
        'schema_version': 1, 'engineering': True, 'created': stamp(),
        'monitoring_policy': args.monitoring_policy,
        'authorization': 'G3C-R4 user explicitly accepts CUDA OOM, training failure, desktop slowdown, and restart risk; prioritizes experiment completion over GPU monitoring availability.',
        'gpu_guard_enabled': False, 'gpu_telemetry_required': False,
        'gpu_telemetry_gaps_terminate_child': False, 'continuous_gpu_coverage_claimed': False,
        'normal_cuda_error_handling': 'unchanged', 'automatic_retry': False,
        'automatic_resume': False, 'training_command': command, 'cwd': str(args.cwd),
        'safe_environment': args.setenv, 'timeout_seconds': args.timeout,
        'lock_file': str(args.lock_file), 'units': {'supervisor': unit},
        'unit_commands': {'supervisor': service_command},
        'source_sha256': {str(scripts / name): sha(scripts / name) for name in (
            'supervision_common.py', 'supervised_exec.py', 'durable_supervisor.py', 'g3cr4_launch.py')},
    }
    atomic_json(args.output / 'launch.json', plan, replace=False)
    try:
        result = subprocess.run(service_command, capture_output=True, text=True)
        submission = {'role': 'supervisor', 'returncode': result.returncode,
                      'stdout': result.stdout, 'stderr': result.stderr, **stamp()}
        submission_code = result.returncode
    except OSError as exc:
        submission = {'role': 'supervisor', 'returncode': None,
                      'error': {'type': type(exc).__name__, 'message': str(exc)}, **stamp()}
        submission_code = 127
    atomic_json(args.output / 'supervisor_submission.json', submission, replace=False)
    if submission_code:
        return submission_code
    print(str(args.output), flush=True)
    return wait_for_status(args.output, unit, args.timeout) if args.wait else 0


if __name__ == '__main__':
    raise SystemExit(main())
