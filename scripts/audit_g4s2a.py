"""CPU regression and complete historical preservation receipt for G4-S2A.

Actor arithmetic tests are authorized here. This runner never invokes the S2A
experiment CLI or starts a physical policy episode. The existing S1B regression
includes its separately bounded, no-policy MuJoCo collector.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
START = '5b39da565b4e579aaf600739d2e8fb180222a5da'
FACTORY_PYTHON = REPO.parents[1] / 'work/envs/factory-g01/bin/python'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def baseline_records():
    """Read exact committed blobs, including byte lengths and executable bits."""
    raw = subprocess.check_output(['git', 'ls-tree', '-rz', START], cwd=REPO)
    entries = []
    for item in raw.split(b'\0'):
        if not item:
            continue
        metadata, name = item.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        if kind != b'blob' or mode not in (b'100644', b'100755', b'120000'):
            raise RuntimeError('Unsupported protected tree entry: ' + repr(item))
        entries.append((name.decode('utf-8'), mode.decode('ascii'), oid))
    batch = subprocess.run(['git', 'cat-file', '--batch'], cwd=REPO,
                           input=b''.join(oid + b'\n' for _, _, oid in entries),
                           capture_output=True, check=True).stdout
    records = {}
    cursor = 0
    for name, mode, expected_oid in entries:
        line_end = batch.index(b'\n', cursor)
        oid, kind, size_text = batch[cursor:line_end].split()
        if oid != expected_oid or kind != b'blob':
            raise RuntimeError('Unexpected git object response for ' + name)
        size = int(size_text)
        content = batch[line_end + 1:line_end + 1 + size]
        cursor = line_end + 1 + size
        if len(content) != size or batch[cursor:cursor + 1] != b'\n':
            raise RuntimeError('Truncated git object response for ' + name)
        cursor += 1
        records[name] = {'sha256': hashlib.sha256(content).hexdigest(),
                         'bytes': size, 'executable': mode == '100755',
                         'symlink': mode == '120000'}
    if cursor != len(batch):
        raise RuntimeError('Unconsumed git object response')
    return records


def snapshot(expected):
    # Reuse all existing external-source, checkpoint, original-USD, unrelated
    # checkout and pinned-Panda checks. Its older tracked-file scope is replaced
    # below by EVERY tracked file at the S1B starting commit, without exceptions.
    from scripts.audit_g4s1b import snapshot as previous_snapshot
    previous = previous_snapshot()
    previous.pop('historical_tracked_except_two_forward_docs')
    actual = {}
    mismatches = []
    for name, record in expected.items():
        path = REPO / name
        if not path.exists() and not path.is_symlink():
            mismatches.append(name + ': missing')
            continue
        is_link = path.is_symlink()
        content = os.fsencode(os.readlink(path)) if is_link else None
        value = {'sha256': hashlib.sha256(content).hexdigest() if is_link else sha(path),
                 'bytes': len(content) if is_link else path.stat().st_size,
                 'executable': False if is_link else bool(path.stat().st_mode & 0o111),
                 'symlink': is_link}
        actual[name] = value
        if value != record:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError('Protected S1B history changed: ' + repr(mismatches))
    return {'all_starting_commit_tracked_files_no_exceptions': actual, **previous}


def source_records():
    paths = sorted(REPO.glob('scripts/*g4s2a*.py')) + sorted(REPO.glob('tests/test_g4s2a*.py'))
    return {str(path.relative_to(REPO)): {'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in paths}


def run_tests(output, python, arguments, label):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    temp_root = Path(tempfile.mkdtemp(prefix='g4s2a_cpu_',
                                    dir='/dev/shm' if Path('/dev/shm').is_dir() else None))
    junit = output / (label + '.xml')
    cmd = [str(python), '-m', 'pytest', '-q', *arguments, '-p', 'no:cacheprovider',
           '--basetemp=' + str(temp_root / 'pytest'), '--junitxml=' + str(junit)]
    started = time.monotonic()
    process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
    duration = time.monotonic() - started
    (output / (label + '_stdout.txt')).write_text(stdout)
    (output / (label + '_stderr.txt')).write_text(stderr)
    with (output / (label + '.txt')).open('x') as stream:
        stream.write(stdout + stderr)
    cases = ET.parse(junit).findall('.//testcase') if junit.exists() else []
    failed = sum(case.find('failure') is not None or case.find('error') is not None for case in cases)
    skipped = sum(case.find('skipped') is not None for case in cases)
    return {'command': cmd, 'exit_code': process.returncode, 'duration_s': duration,
            'timed_out': timed_out, 'cases': len(cases), 'passed': len(cases) - failed - skipped,
            'failed': failed, 'skipped': skipped,
            'test_node_ids': [case.get('classname', '') + '::' + case.get('name', '') for case in cases],
            'environment_overrides': {key: env[key] for key in ('CUDA_VISIBLE_DEVICES',
                'PYTEST_DISABLE_PLUGIN_AUTOLOAD', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'log': str((output / (label + '.txt')).relative_to(REPO)),
            'stdout': str((output / (label + '_stdout.txt')).relative_to(REPO)),
            'stderr': str((output / (label + '_stderr.txt')).relative_to(REPO)),
            'junit': str(junit.relative_to(REPO)) if junit.exists() else None,
            'pytest_temporary_directory': str(temp_root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(REPO)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (output / 'runner_source.py.txt').write_text(Path(__file__).read_text())
    expected = baseline_records()
    before = snapshot(expected)
    write(output / 'preservation_before.json', before)
    sources_before = source_records()
    write(output / 'sources_before.json', sources_before)
    full = run_tests(output, FACTORY_PYTHON, ['tests'], 'full_cpu')
    plot = run_tests(output, '/usr/bin/python3',
                     ['tests/test_g4i2_analysis.py::test_plots_are_real_files_and_cannot_overwrite'],
                     'system_plot')
    after = snapshot(expected)
    write(output / 'preservation_after.json', after)
    sources_after = source_records()
    write(output / 'sources_after.json', sources_after)
    preserved = before == after
    sources_stable = sources_before == sources_after
    passed = preserved and sources_stable and all(
        receipt['exit_code'] == 0 and receipt['failed'] == 0 and receipt['cases'] > 0
        for receipt in (full, plot))
    artifacts = {str(path.relative_to(REPO)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(output.iterdir()) if path.is_file()}
    summary = {'task': 'G4-S2A', 'starting_commit': START,
               'generated_utc': datetime.now(timezone.utc).isoformat(),
               'regression_status': 'PASS' if passed else 'FAIL',
               'full_cpu': full, 'system_plot': plot, 'excluded_policy_cases': 0,
               'full_regression_all_existing_tests_requested': True,
               'sources_unchanged_during_regression': sources_stable,
               'preservation': {'equal_before_after': preserved,
                   'historical_tracked_protected': len(expected), 'exceptions': [],
                   'checkpoint_count_hash_only': len(before['checkpoints_hash_only']),
                   'external_source_configuration_count': len(before['frozen_sources_and_configurations']),
                   'original_USD_count': len(before['four_original_USD_hashes']),
                   'independent_pinned_Panda_file_count': len(before['independent_pinned_Panda_files']),
                   'unrelated_G3V_count': len(before['unrelated_original_checkout_g3v'])},
               'scope': {'historical_actor_arithmetic_tests_authorized': True,
                   'physical_policy_episodes_launched_by_runner': 0,
                   'smoke_experiment_CLI_invoked': False,
                   'existing_bounded_no_policy_MuJoCo_regression_allowed': True,
                   'training': False, 'Isaac_Sim': False, 'GPU_execution': False},
               'sources': sources_after, 'artifacts': artifacts,
               'duration_s': time.monotonic() - started}
    write(output / 'summary.json', summary)
    print(json.dumps({'regression_status': summary['regression_status'],
                      'full_cpu': {key: full[key] for key in ('exit_code', 'cases', 'passed', 'failed', 'skipped', 'duration_s')},
                      'system_plot': {key: plot[key] for key in ('exit_code', 'cases', 'passed', 'failed', 'skipped', 'duration_s')},
                      'preservation': summary['preservation'],
                      'sources_unchanged_during_regression': sources_stable,
                      'duration_s': summary['duration_s']}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    # Normalizing script imports avoids a scripts/ top-level module shadowing
    # the package imports used by the historical tests and preservation code.
    sys.path[:] = [str(REPO)] + [path for path in sys.path
                               if Path(path).resolve() not in (REPO, REPO / 'scripts')]
    raise SystemExit(main())
