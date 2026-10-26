"""Guarded CPU regression for R1; no policy calls or physics integration.

Historical tests are unchanged. Explicitly excluded tests need actor execution
or an unregistered 518-step headless collector. Scoped entry tripwires provide
additional evidence; they are not a general security sandbox.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
START = '9f521002cd1ee1694fabc1f2012a8bdd79e26434'
PYTHON = ROOT.parents[1]/'work/envs/factory-g01/bin/python'
ALLOWED_CASE_DOCS = {'docs/case_studies/SIM2SIM_PANDA_CONTROL_INSTABILITY.md', 'docs/case_studies/README.md'}
EXCLUDED_FILES = {'tests/test_g4i1_policy.py': 49, 'tests/test_g4i1_edge_cpu.py': 9,
                  'tests/test_g4i2_fault_contract.py': 11, 'tests/test_g4i2_policy_probe.py': 4}
EXCLUDED_NODES = {
    'tests/test_g4s2a_inference.py::test_real_frozen_bridge_matches_direct_once_normalized_recurrent_sequence_and_reset':
        'Would execute five actual actor forwards and load the checkpoint.',
    'tests/test_g4s1b_env.py::test_environment_collector_real_exit_and_no_policy':
        'Would execute 518 new headless physical steps outside the registered GUI-only diagnostic matrix.'}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def helpers():
    from scripts import audit_g4s2a as previous
    return previous


def expected_history():
    # Reuse the reviewed committed-blob reader with an explicit read-only Git
    # target. Restore the utility's module constant immediately afterwards.
    previous = helpers()
    original = previous.START
    try:
        previous.START = START
        expected = previous.baseline_records()
    finally:
        previous.START = original
    return {name: value for name, value in expected.items() if name not in ALLOWED_CASE_DOCS}


def sources():
    previous = helpers()
    paths = sorted(ROOT.glob('scripts/*g4s2a_r1*.py'))+sorted(ROOT.glob('tests/test_g4s2a_r1.py'))
    return {str(path.relative_to(ROOT)): {'sha256': previous.sha(path), 'bytes': path.stat().st_size} for path in paths}


def guarded_pytest(arguments, receipt):
    sys.path[:] = [str(ROOT)]+[path for path in sys.path if Path(path).resolve() not in (ROOT, ROOT/'scripts')]
    import pytest
    forbidden = []
    scientific_child_names = {'g4s1b_env.py', 'g4s2a_smoke.py', 'g4s2a_inference.py',
                              'g4s2a_r1_diagnostics.py', 'g4s2a_r1_viewer.py',
                              'g4i1_runtime.py', 'g4i2_runtime.py', 'evaluate_checkpoint.py'}
    def profile(frame, event, arg):
        blocked = False
        if event == 'call':
            filename = frame.f_code.co_filename.replace('\\', '/')
            name = frame.f_code.co_name
            blocked = (
                (filename.endswith('/scripts/g4i1_policy.py') and name in ('infer', 'from_checkpoint'))
                or (filename.endswith('/tests/test_g4i1_policy.py') and name == 'forward')
                or ('/rl_games/algos_torch/players.py' in filename and name == 'get_action')
                or (any('/rl_games/algos_torch/'+suffix in filename for suffix in ('models.py', 'network_builder.py')) and name == 'forward')
                or (filename.endswith('/torch/serialization.py') and name == 'load')
                or (filename.endswith('/scripts/g4s1b_env.py') and name == 'physics_step')
                or (filename.endswith('/mujoco/viewer.py') and name in ('launch', 'launch_passive'))
                or (filename.endswith('/scripts/g4s2a_r1_diagnostics.py') and name in ('child_run', 'coordinate'))
                or (filename.endswith('/scripts/g4s2a_r1_viewer.py') and name == 'main')
                or ('/isaacsim/' in filename and name == '__init__' and frame.f_locals.get('self', None).__class__.__name__ == 'SimulationApp'))
            if filename.endswith('/subprocess.py') and name == '__init__':
                command = frame.f_locals.get('args', [])
                if isinstance(command, (tuple, list)):
                    blocked |= any(Path(str(part)).name in scientific_child_names for part in command)
        elif event == 'c_call':
            name = getattr(arg, '__name__', '')
            filename = getattr(arg, '__module__', '') or ''
            blocked = name in ('mj_step', 'mj_step1', 'mj_step2') and 'mujoco' in filename
        else:
            return
        if blocked:
            forbidden.append({'event': event, 'module_or_file': filename, 'function': name})
            write(receipt, {'active': True, 'forbidden_calls': forbidden, 'aborted_before_body': True})
            os._exit(97)
    started = time.monotonic()
    sys.setprofile(profile); threading.setprofile(profile)
    try:
        if len(arguments) == 1 and arguments[0].startswith('--negative-'):
            kind = arguments[0].removeprefix('--negative-')
            filename, name = {
                'actor': (ROOT/'scripts/g4i1_policy.py', 'infer'),
                'physics': (ROOT/'scripts/g4s1b_env.py', 'physics_step'),
                'viewer': (ROOT/'mujoco/viewer.py', 'launch_passive')}[kind]
            exec(compile(f"def {name}():\n    raise AssertionError('forbidden body executed')\n{name}()\n", str(filename), 'exec'), {})
            raise AssertionError('Negative control did not abort')
        code = pytest.main(arguments)
    finally:
        sys.setprofile(None); threading.setprofile(None)
    write(receipt, {'active': True, 'forbidden_calls': forbidden, 'duration_s': time.monotonic()-started,
                    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'scope': 'Known Python actor/checkpoint/physics/GUI entry points, known scientific child script names and MuJoCo C integration calls in this process/new Python threads. Reviewed static collector children may compile/forward without integration. Not a general sandbox.'})
    return int(code)


def run_tests(output, python, arguments, label):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    base = Path(tempfile.mkdtemp(prefix='g4s2a_r1_cpu_', dir='/dev/shm'))/'pytest'
    junit = output/(label+'.xml'); guard = output/(label+'_guard.json')
    cmd = [str(python), str(Path(__file__).resolve()), '--guarded-pytest', str(guard), '-q', *arguments,
           '-p', 'no:cacheprovider', '--basetemp='+str(base), '--junitxml='+str(junit)]
    started = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    with (output/(label+'.txt')).open('x') as stream: stream.write(result.stdout+result.stderr)
    cases = ET.parse(junit).findall('.//testcase') if junit.exists() else []
    failed = sum(case.find('failure') is not None or case.find('error') is not None for case in cases)
    skipped = sum(case.find('skipped') is not None for case in cases)
    data = json.loads(guard.read_text()) if guard.exists() else {}
    return {'command': cmd, 'exit_code': result.returncode, 'duration_s': time.monotonic()-started,
            'cases': len(cases), 'passed': len(cases)-failed-skipped, 'failed': failed, 'skipped': skipped,
            'r1_cases': sum('test_g4s2a_r1' in case.get('classname', '') for case in cases),
            'guard_verified': data.get('active') is True and data.get('forbidden_calls') == [],
            'log': str((output/(label+'.txt')).relative_to(ROOT)), 'junit': str(junit.relative_to(ROOT)),
            'guard': str(guard.relative_to(ROOT))}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--guarded-pytest':
        return guarded_pytest(sys.argv[3:], Path(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); previous = helpers()
    (output/'runner_source.py.txt').write_text(Path(__file__).read_text())
    expected = expected_history()
    before = previous.snapshot(expected); write(output/'preservation_before.json', before)
    source_before = sources(); write(output/'sources_before.json', source_before)
    exclusions = {'file_exclusions_historical_case_counts': EXCLUDED_FILES,
                  'node_exclusions': EXCLUDED_NODES, 'excluded_count': sum(EXCLUDED_FILES.values())+len(EXCLUDED_NODES),
                  'reason': 'R1 forbids additional actor/checkpoint execution and unregistered/headless physical integration. Original tests preserved unchanged.'}
    write(output/'excluded_tests.json', exclusions)
    negatives = []
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    for kind in ('actor', 'physics', 'viewer'):
        receipt = output/('negative_'+kind+'_guard.json')
        cmd = [str(PYTHON), str(Path(__file__).resolve()), '--guarded-pytest', str(receipt), '--negative-'+kind]
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        record = json.loads(receipt.read_text()) if receipt.exists() else {}
        negatives.append({'kind': kind, 'command': cmd, 'exit_code': result.returncode,
                          'passed': result.returncode == 97 and record.get('aborted_before_body') is True,
                          'stdout': result.stdout, 'stderr': result.stderr,
                          'dummy_only_no_real_actor_physics_or_viewer': True})
    write(output/'negative_controls.json', negatives)
    if not all(record['passed'] for record in negatives): raise RuntimeError('Tripwire negative control failed')
    arguments = ['tests', *['--ignore='+name for name in EXCLUDED_FILES], *['--deselect='+name for name in EXCLUDED_NODES]]
    selected = run_tests(output, PYTHON, arguments, 'selected_cpu')
    plot = run_tests(output, '/usr/bin/python3', ['tests/test_g4i2_analysis.py::test_plots_are_real_files_and_cannot_overwrite'], 'system_plot')
    after = previous.snapshot(expected); write(output/'preservation_after.json', after)
    source_after = sources(); write(output/'sources_after.json', source_after)
    source_equal = source_before == source_after
    passed = before == after and source_equal and all(record['exit_code'] == 0 and record['failed'] == 0
              and record['cases'] > 0 and record['guard_verified'] for record in (selected, plot))
    artifacts = {str(path.relative_to(ROOT)): {'sha256': previous.sha(path), 'bytes': path.stat().st_size}
                 for path in output.iterdir() if path.is_file()}
    summary = {'task': 'G4-S2A-R1', 'starting_commit': START, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'regression_status': 'PASS' if passed else 'FAIL', 'selected_cpu': selected, 'system_plot': plot,
        'new_R1_test_cases': selected['r1_cases'], 'excluded_cases': exclusions['excluded_count'],
        'exclusions': exclusions, 'tripwire_negative_controls_passed': True,
        'sources_unchanged_during_regression': source_equal,
        'preservation': {'equal_before_after': before == after, 'protected_tracked_files': len(expected),
            'authorized_case_doc_exceptions': sorted(ALLOWED_CASE_DOCS),
            'checkpoint_count_hash_only': len(before['checkpoints_hash_only']),
            'external_source_configuration_count': len(before['frozen_sources_and_configurations']),
            'original_USD_count': len(before['four_original_USD_hashes']),
            'Panda_files': len(before['independent_pinned_Panda_files']),
            'unrelated_G3V_files': len(before['unrelated_original_checkout_g3v'])},
        'no_additional_actor_forward': True, 'no_checkpoint_deserialization': True,
        'no_new_physics_integration_or_GUI_from_regression': True,
        'scientific_effectiveness_not_inferred_from_test_PASS': True,
        'source_hashes': source_after, 'artifacts': artifacts, 'wall_s': time.monotonic()-started}
    write(output/'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('regression_status', 'new_R1_test_cases', 'excluded_cases',
          'sources_unchanged_during_regression', 'preservation', 'wall_s')}, ensure_ascii=False))
    print(json.dumps({'selected_cpu': selected, 'system_plot': plot}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.path[:] = [str(ROOT)]+[path for path in sys.path if Path(path).resolve() not in (ROOT, ROOT/'scripts')]
    raise SystemExit(main())
