"""Preservation and no-policy CPU regression receipt for adapted G4-S1B.

Run this after the bounded physics collector. Never imports a policy, simulator
application, training entry point, or checkpoint deserializer. Existing actor
tests are excluded and preserved; the S1A call-entry tripwire guards pytest.
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
START = '6e282557e2fd0c689932b6a572b53d987ad4c8d3'
FACTORY_PYTHON = REPO.parents[1] / 'work/envs/factory-g01/bin/python'
MUJOCO_PYTHON = Path('/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python')
FORWARD_DOCS = {'README.md', 'docs/ROADMAP_POLICY_INFERENCE_SIM2SIM.md'}
EXCLUDED = {
    'tests/test_g4i1_policy.py': 49,
    'tests/test_g4i1_edge_cpu.py': 9,
    'tests/test_g4i2_fault_contract.py': 11,
    'tests/test_g4i2_policy_probe.py': 4,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def snapshot():
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', START], cwd=REPO, text=True).splitlines()
    protected = sorted(set(names) - FORWARD_DOCS)
    changed = subprocess.check_output(['git', 'diff', '--name-only', START, '--', *protected], cwd=REPO, text=True).splitlines()
    if changed:
        raise RuntimeError('Protected history changed: ' + repr(changed))
    tracked = {name: {'sha256': sha(REPO/name), 'executable': bool((REPO/name).stat().st_mode & 0o111)} for name in protected}
    audit = json.loads((REPO/'evidence/g4_i/i0/checkpoint_audit.json').read_text())
    external = {}
    for entry in audit['source_files'].values():
        value = sha(entry['path'])
        assert value == entry['sha256'], entry['path']
        external[entry['path']] = value
    checkpoints = {}
    for label, entry in audit['checkpoints'].items():
        digest = sha(entry['path'])
        assert digest == entry['sha256'], label
        checkpoints[label] = {'path': entry['path'], 'sha256': digest, 'bytes': Path(entry['path']).stat().st_size}
        for kind in ('agent', 'env'):
            path = entry[kind+'_config_path']
            digest = sha(path)
            assert digest == entry[kind+'_config_sha256'], path
            external[path] = digest
    assert checkpoints['g2_epoch200_primary']['sha256'] == 'f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f'
    # Also protect external originals and unrelated G3V files using the archived
    # S1A receipt. Hash-only reads do not import or convert these assets.
    previous = json.loads((REPO/'evidence/g4_s/s1a/cpu_validation_20261026T_g4s1a_r2/preservation_after.json').read_text())
    other = {}
    for category in ('four_original_USD_hashes', 'unrelated_original_checkout_g3v'):
        other[category] = {path: sha(path) for path in previous[category]}
        assert other[category] == previous[category], category
    s0 = json.loads((REPO/'evidence/g4_s/s0/assets_audit.json').read_text())
    panda = s0['candidate_inventories'][0]
    independent = {}
    for entry in [panda['license'], panda['readme'], *panda['model_files'], *panda['referenced_geometry']]:
        digest = sha(entry['path'])
        assert digest == entry['sha256'], entry['path']
        independent[entry['path']] = digest
    return {'historical_tracked_except_two_forward_docs': tracked, 'frozen_sources_and_configurations': external,
            'checkpoints_hash_only': checkpoints, 'independent_pinned_Panda_files': independent, **other}


def run_tests(output, python, arguments, label):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    base = Path(tempfile.mkdtemp(prefix='g4s1b_cpu_', dir='/dev/shm' if Path('/dev/shm').is_dir() else None))/'pytest'
    junit = output/(label+'.xml')
    guard = output/(label+'_guard.json')
    cmd = [str(python), str(Path(__file__).resolve()), '--guarded-pytest', str(guard),
           '-q', *arguments, '-p', 'no:cacheprovider', '--basetemp='+str(base), '--junitxml='+str(junit)]
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    with (output/(label+'.txt')).open('x') as stream:
        stream.write(result.stdout+result.stderr)
    cases = ET.parse(junit).findall('.//testcase') if junit.exists() else []
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    record = json.loads(guard.read_text()) if guard.exists() else {}
    return {'command': cmd, 'exit_code': result.returncode, 'duration_s': time.monotonic()-start,
            'cases': len(cases), 'passed': len(cases)-failed-skipped, 'failed': failed, 'skipped': skipped,
            'guard_verified': record.get('active') is True and record.get('forbidden_calls') == [],
            'log': str((output/(label+'.txt')).relative_to(REPO)), 'junit': str(junit.relative_to(REPO)),
            'guard': str(guard.relative_to(REPO))}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--guarded-pytest':
        sys.path[:] = [str(REPO)] + [p for p in sys.path if Path(p).resolve() not in (REPO, REPO/'scripts')]
        from scripts.audit_g4s1a import guarded_pytest
        # The reviewed S1B engine subprocess is allowed to run bounded no-policy
        # physics. The parent tripwire still rejects actor/checkpoint entry.
        return guarded_pytest(sys.argv[3:], Path(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (output/'runner_source.py.txt').open('x') as stream:
        stream.write(Path(__file__).read_text())
    before = snapshot()
    write(output/'preservation_before.json', before)
    write(output/'excluded_policy_tests.json', {'unchanged_files_and_historical_case_counts': EXCLUDED,
          'total_not_run': sum(EXCLUDED.values()), 'reason': 'S1B prohibits actor execution and checkpoint loads; do not claim complete actor regression.'})
    negative_receipt = output/'tripwire_negative_control.json'
    negative_cmd = [str(FACTORY_PYTHON), str(Path(__file__).resolve()), '--guarded-pytest', str(negative_receipt), '--guard-negative-control']
    negative = subprocess.run(negative_cmd, cwd=REPO, capture_output=True, text=True, timeout=60)
    data = json.loads(negative_receipt.read_text()) if negative_receipt.exists() else {}
    negative_ok = negative.returncode == 97 and data.get('aborted_before_body') is True
    write(output/'tripwire_process.json', {'command': negative_cmd, 'exit_code': negative.returncode, 'passed': negative_ok,
          'stdout': negative.stdout, 'stderr': negative.stderr, 'dummy_only_no_actor_loaded': True})
    if not negative_ok:
        raise RuntimeError('No-policy tripwire negative control failed')
    selected = run_tests(output, FACTORY_PYTHON, ['tests', *['--ignore='+p for p in EXCLUDED]], 'selected_cpu')
    plot = run_tests(output, '/usr/bin/python3', ['tests/test_g4i2_analysis.py::test_plots_are_real_files_and_cannot_overwrite'], 'system_plot')
    after = snapshot()
    write(output/'preservation_after.json', after)
    assert before == after, 'Protected content changed during validation'
    source_files = sorted(REPO.glob('scripts/*g4s1b*.py'))+sorted(REPO.glob('tests/test_g4s1b*.py'))
    passed = all(x['exit_code'] == 0 and x['failed'] == 0 and x['cases'] > 0 and x['guard_verified'] for x in (selected, plot))
    summary = {'task': 'G4-S1B', 'protocol': 'ADAPTED_ZERO_SHOT_SIM2SIM_NO_POLICY_ENGINEERING_ONLY',
        'starting_commit': START, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'regression_status': 'PASS' if passed else 'FAIL', 'selected_cpu': selected, 'system_plot': plot,
        'excluded_policy_cases': sum(EXCLUDED.values()), 'tripwire_negative_control_passed': negative_ok,
        'preservation': {'equal_before_after': True, 'historical_tracked_protected': len(before['historical_tracked_except_two_forward_docs']),
                         'forward_only_docs_allowed': sorted(FORWARD_DOCS), 'checkpoint_count_hash_only': len(before['checkpoints_hash_only']),
                         'external_source_configuration_count': len(before['frozen_sources_and_configurations']), 'original_USD_count': len(before['four_original_USD_hashes']),
                         'independent_pinned_Panda_file_count': len(before['independent_pinned_Panda_files']),
                         'unrelated_G3V_count': len(before['unrelated_original_checkout_g3v'])},
        'sources': {str(p.relative_to(REPO)): sha(p) for p in source_files}, 'duration_s': time.monotonic()-started,
        'policy_inferences': 0, 'checkpoint_loads': 0, 'training': False, 'S2_started': False,
        'guard_scope': 'pytest parent process/current and new Python threads; known-entry tripwire, not sandbox. Reviewed MuJoCo child performs bounded physics without policy.'}
    write(output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
