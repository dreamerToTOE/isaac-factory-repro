"""Guarded CPU regression for R2-R1; no policy calls or physics integration.

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
START = '5abe4375c2bea828816a7d2f0ff04e9ee0673ff0'
PYTHON = ROOT.parents[1]/'work/envs/factory-g01/bin/python'
MUJOCO_PYTHON = Path('/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python')
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
    paths = sorted(ROOT.glob('scripts/*g4s2a_r2_r1*.py'))+sorted(ROOT.glob('tests/test_g4s2a_r2_r1.py'))
    return {str(path.relative_to(ROOT)): {'sha256': previous.sha(path), 'bytes': path.stat().st_size} for path in paths}


def historical_static_model_checks():
    """Compile/forward real 3.5.0 models; assert no integration or actor needed.

    This separate pass uses the pinned MuJoCo Python because the historical
    pytest environment has no MuJoCo. Its profile guard is installed before
    importing any of these model helpers.
    """
    import hashlib
    import numpy as np
    from scripts import g4s2a_r2_model as model
    from scripts.g4s2a_r1_viewer import apply_archived_state, load_archive, XML_SHA
    initial, _, _ = load_archive()
    identities = {}
    results = {}
    for variant, (physical_selection, controller_selection) in model.VARIANTS.items():
        pair = model.ModelPair(variant)
        env = pair.plant
        assert env.mj.__version__ == env.mj.mj_versionString() == '3.5.0'
        assert (env.model.nq, env.model.nv, env.model.nu) == (16, 15, 9)
        apply_archived_state(env, initial)
        original_state = {key: getattr(env.data, key).copy() for key in ('qpos', 'qvel', 'ctrl', 'qM', 'qacc_warmstart')}
        original_time = env.data.time
        values = pair.matrices()
        for key, value in original_state.items():
            np.testing.assert_array_equal(getattr(env.data, key), value)
        assert env.data.time == original_time
        for shadow in pair.shadows:
            np.testing.assert_array_equal(shadow.data.qpos, original_state['qpos'])
            assert shadow.data.time == 0.
        for key in ('physical_full_M', 'physical_M', 'controller_M', 'same_q_M0', 'same_q_M1'):
            np.linalg.cholesky(values[key])
            np.testing.assert_allclose(values[key], values[key].T, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(values['physical_M'], values['same_q_M'+str(physical_selection)])
        np.testing.assert_array_equal(values['controller_M'], values['same_q_M'+str(controller_selection)])
        # An independent kinetic-energy oracle checks compiled body inertia.
        expected_delta = sum(.01*np.asarray(j).T@np.asarray(j) for j in values['body_angular_jacobians'])
        actual_delta = values['same_q_M1']-values['same_q_M0']
        error = float(np.max(abs(actual_delta-expected_delta)))
        assert error < 2e-12
        assert np.max(abs(actual_delta-np.diag(np.diag(actual_delta)))) > 1e-8  # Not joint armature.
        identities[variant] = model.model_identity(env)
        results[variant] = {'plant_selection': physical_selection, 'controller_selection': controller_selection,
            'physical_M': values['physical_M'].tolist(), 'controller_M': values['controller_M'].tolist(),
            'same_qpos': values['same_qpos'].tolist(), 'delta_M_energy_oracle_max_error': error,
            'physical_M_condition': float(np.linalg.cond(values['physical_M'])),
            'controller_M_condition': float(np.linalg.cond(values['controller_M'])),
            'plant_data_unchanged': True, 'shadow_data_time_zero': True,
            'model_xml_sha256': hashlib.sha256(env.xml.encode()).hexdigest()}
    assert identities['A'] == identities['C']
    assert identities['B'] == identities['D']
    assert identities['A']['xml_sha256'] == XML_SHA
    assert identities['A']['xml_sha256'] != identities['B']['xml_sha256']
    baseline, adapted = model.make_model(False), model.make_model(True)
    changed = {adapted.model.body(name).id for name in model.ROBOT_BODIES}
    expected_inertia = baseline.model.body_inertia.copy()
    for body in changed: expected_inertia[body] += .01
    np.testing.assert_allclose(adapted.model.body_inertia, expected_inertia, rtol=0, atol=3e-16)
    for attribute in ('body_mass', 'body_ipos', 'dof_armature',
                      'actuator_forcerange', 'actuator_ctrlrange', 'jnt_range',
                      'geom_size', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity'):
        np.testing.assert_allclose(getattr(adapted.model, attribute), getattr(baseline.model, attribute), rtol=0, atol=3e-16)
    # Recompiling full inertia tensors re-diagonalizes them. The XML changes
    # exactly the three diagonal entries; principal-axis quaternions can differ
    # by roundoff. This bound is checked separately from physical parameters.
    axis_error = float(np.max(abs(adapted.model.body_iquat-baseline.model.body_iquat)))
    assert axis_error < 1e-12
    assert baseline.model.neq == adapted.model.neq
    assert all(baseline.model.eq_type[i] != baseline.mj.mjtEq.mjEQ_WELD for i in range(baseline.model.neq))
    # Same physical model, arbitrary complete q, including both fingers/free peg.
    q = initial['qpos'].copy(); q[0] += .01; q[7:9] = [.019, .021]; q[9] += .013
    before = baseline.data.qpos.copy()
    mass = model.mass_at(adapted, q)
    np.testing.assert_array_equal(adapted.data.qpos, q)
    np.testing.assert_array_equal(baseline.data.qpos, before)
    np.linalg.cholesky(mass)
    return {'mujoco_version': baseline.mj.__version__, 'variants': results,
            'eleven_body_tensor_offset_verified': True, 'mass_com_armature_geometry_force_limits_unchanged': True,
            'compiled_principal_axis_quaternion_roundoff_max': axis_error,
            'compiled_principal_axis_quaternion_tolerance': 1e-12,
            'all_plant_controller_combinations_verified': True, 'static_compiled_model_pairs': 4,
            'new_physics_steps': 0, 'actor_calls': 0, 'checkpoint_loads': 0}



def static_model_checks():
    """Validate real compiled inertias and matrices without stepping physics."""
    import numpy as np
    from scripts import g4s2a_r2_r1_model as model
    from scripts.g4s1b_env import AdaptedPegInsertEnv
    from scripts.g4s2a_r1_viewer import apply_archived_state, load_archive
    previous = historical_static_model_checks()
    initial, _, _ = load_archive()
    baseline = model.make_model('S1B')
    summary = {}
    for selection in ('P_FULL', 'P_ARM_HAND'):
        compiled = model.make_model(selection)
        ids = {compiled.model.body(name).id for name in model.SELECTIONS[selection]}
        expected = baseline.model.body_inertia.copy()
        for body_id in ids: expected[body_id] += .01
        np.testing.assert_allclose(compiled.model.body_inertia, expected, rtol=0, atol=3e-16)
        for attr in ('body_mass', 'body_ipos', 'dof_armature', 'actuator_forcerange',
                     'actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm',
                     'jnt_range', 'geom_size', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity'):
            np.testing.assert_allclose(getattr(compiled.model, attr), getattr(baseline.model, attr), rtol=0, atol=3e-16)
        for finger in ('left_finger', 'right_finger'):
            i = compiled.model.body(finger).id
            np.testing.assert_allclose(compiled.model.body_inertia[i] - baseline.model.body_inertia[i],
                                       [0.01]*3 if selection == 'P_FULL' else [0.]*3, rtol=0, atol=3e-16)
        assert compiled.model.opt.timestep == baseline.model.opt.timestep
        assert compiled.model.opt.integrator == baseline.model.opt.integrator
        np.testing.assert_array_equal(compiled.model.opt.gravity, baseline.model.opt.gravity)
        np.testing.assert_array_equal(compiled.model.geom_solref, baseline.model.geom_solref)
        np.testing.assert_array_equal(compiled.model.geom_solimp, baseline.model.geom_solimp)
        assert type(compiled).physics_step is AdaptedPegInsertEnv.physics_step
        combinations = {}
        for control in ('C0', 'C1'):
            pair = model.ModelPair(selection, control)
            apply_archived_state(pair.plant, initial)
            state = {k:getattr(pair.plant.data, k).copy() for k in ('qpos','qvel','ctrl','qM','qacc_warmstart')}
            t = pair.plant.data.time
            matrices = pair.matrices()
            for key, value in state.items(): np.testing.assert_array_equal(getattr(pair.plant.data, key), value)
            assert pair.plant.data.time == t
            for key in ('physical_full_M','C0_full_M','C1_full_M','physical_M','controller_M'):
                np.linalg.cholesky(matrices[key])
                np.testing.assert_allclose(matrices[key], matrices[key].T, rtol=0, atol=1e-14)
            np.testing.assert_allclose(matrices['physical_full_M'], matrices['C1_full_M'], rtol=2e-12, atol=2e-12)
            ix = np.ix_(pair.plant.arm_dof_ids, pair.plant.arm_dof_ids)
            np.testing.assert_array_equal(matrices['controller_M'], matrices[control+'_full_M'][ix])
            assert matrices['finger_predictor_mass_source'] == 'physical_full_M'
            assert type(pair.plant).physics_step is AdaptedPegInsertEnv.physics_step
            delta = matrices['C1_full_M'] - matrices['C0_full_M']
            oracle = np.zeros_like(delta)
            for name in model.SELECTIONS[selection]:
                jp = np.zeros((3,pair.plant.model.nv)); jr = np.zeros_like(jp)
                pair.plant.mj.mj_jacBody(pair.plant.model,pair.plant.data,jp,jr,pair.plant.model.body(name).id)
                oracle += .01*jr.T@jr
            error = float(np.max(abs(delta-oracle)))
            assert error < 2e-12
            combinations[control] = {'physical_full_M':matrices['physical_full_M'].tolist(),
                'controller_M':matrices['controller_M'].tolist(), 'same_qpos':matrices['same_qpos'].tolist(),
                'energy_oracle_max_error':error, 'plant_data_unchanged':True,
                'finger_predictor_uses_inherited_physical_plant_implementation':True}
        np.testing.assert_allclose(combinations['C0']['physical_full_M'],combinations['C1']['physical_full_M'],atol=0,rtol=0)
        assert not np.allclose(combinations['C0']['controller_M'],combinations['C1']['controller_M'],atol=1e-8,rtol=0)
        summary[selection] = combinations
    return {'previous_r2_checks':previous, 'mujoco_version':baseline.mj.__version__, 'variants':summary,
            'all_four_variants_verified':True, 'original_finger_tensor_verified':True,
            'mass_com_limits_actuators_contact_solver_unchanged':True,
            'full_mass_energy_oracle_verified':True, 'new_physics_steps':0, 'actor_calls':0, 'checkpoint_loads':0}

def guarded_pytest(arguments, receipt):
    sys.path[:] = [str(ROOT)]+[path for path in sys.path if Path(path).resolve() not in (ROOT, ROOT/'scripts')]
    if arguments != ['--static-models']:
        import pytest
    forbidden = []
    scientific_child_names = {'g4s1b_env.py', 'g4s2a_smoke.py', 'g4s2a_inference.py',
                              'g4s2a_r1_diagnostics.py', 'g4s2a_r1_viewer.py',
                              'g4s2a_r2_diagnostics.py', 'g4s2a_r2_viewer.py', 'g4s2a_r2_r1_diagnostics.py', 'g4s2a_r2_r1_viewer.py',
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
                or (filename.endswith('/scripts/g4s2a_r2_diagnostics.py') and name in ('child_run', 'coordinate', 'main', 'run_matrix', 'run_variant', 'run_preparation', 'prepare_child', 'trajectory_child'))
                or (filename.endswith('/scripts/g4s2a_r1_viewer.py') and name == 'main')
                or (filename.endswith('/scripts/g4s2a_r2_viewer.py') and name == 'main')
                or (filename.endswith('/scripts/g4s2a_r2_r1_diagnostics.py') and name in ('child_run', 'coordinate', 'main', 'run_matrix', 'run_variant', 'run_preparation', 'prepare_child', 'trajectory_child'))
                or (filename.endswith('/scripts/g4s2a_r2_r1_viewer.py') and name == 'main')
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
    static_result = None
    sys.setprofile(profile); threading.setprofile(profile)
    try:
        if len(arguments) == 1 and arguments[0].startswith('--negative-'):
            kind = arguments[0].removeprefix('--negative-')
            filename, name = {
                'actor': (ROOT/'scripts/g4i1_policy.py', 'infer'),
                'physics': (ROOT/'scripts/g4s1b_env.py', 'physics_step'),
                'viewer': (ROOT/'mujoco/viewer.py', 'launch_passive'),
                'r2_runner': (ROOT/'scripts/g4s2a_r2_diagnostics.py', 'run_variant'),
                'r2_r1_runner': (ROOT/'scripts/g4s2a_r2_r1_diagnostics.py', 'run_variant')}[kind]
            exec(compile(f"def {name}():\n    raise AssertionError('forbidden body executed')\n{name}()\n", str(filename), 'exec'), {})
            raise AssertionError('Negative control did not abort')
        if arguments == ['--static-models']:
            static_result = static_model_checks()
            code = 0
        else:
            code = pytest.main(arguments)
    finally:
        sys.setprofile(None); threading.setprofile(None)
    write(receipt, {'active': True, 'forbidden_calls': forbidden, 'duration_s': time.monotonic()-started,
                    'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'static_model_checks': static_result,
                    'scope': 'Known Python actor/checkpoint/physics/GUI entry points, known scientific child script names and MuJoCo C integration calls in this process/new Python threads. Reviewed static collector children may compile/forward without integration. Not a general sandbox.'})
    return int(code)


def run_tests(output, python, arguments, label):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    base = Path(tempfile.mkdtemp(prefix='g4s2a_r2_r1_cpu_', dir='/dev/shm'))/'pytest'
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
            'r2_r1_cases': sum('test_g4s2a_r2_r1' in case.get('classname', '') for case in cases),
            'guard_verified': data.get('active') is True and data.get('forbidden_calls') == [],
            'log': str((output/(label+'.txt')).relative_to(ROOT)), 'junit': str(junit.relative_to(ROOT)),
            'guard': str(guard.relative_to(ROOT))}


def run_static_models(output):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    guard = output/'static_models_guard.json'
    cmd = [str(MUJOCO_PYTHON), str(Path(__file__).resolve()), '--guarded-pytest', str(guard), '--static-models']
    started = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    with (output/'static_models.txt').open('x') as stream: stream.write(result.stdout+result.stderr)
    record = json.loads(guard.read_text()) if guard.exists() else {}
    checks = record.get('static_model_checks') or {}
    return {'command': cmd, 'exit_code': result.returncode, 'duration_s': time.monotonic()-started,
            'guard_verified': record.get('active') is True and record.get('forbidden_calls') == [],
            'all_four_variants_verified': checks.get('all_four_variants_verified') is True,
            'new_physics_steps': checks.get('new_physics_steps'), 'actor_calls': checks.get('actor_calls'),
            'log': str((output/'static_models.txt').relative_to(ROOT)), 'guard': str(guard.relative_to(ROOT))}


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
                  'reason': 'R2-R1 forbids additional actor/checkpoint execution and unregistered/headless physical integration. Original tests preserved unchanged.'}
    write(output/'excluded_tests.json', exclusions)
    negatives = []
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    for kind in ('actor', 'physics', 'viewer', 'r2_runner', 'r2_r1_runner'):
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
    static = run_static_models(output)
    plot = run_tests(output, '/usr/bin/python3', ['tests/test_g4i2_analysis.py::test_plots_are_real_files_and_cannot_overwrite'], 'system_plot')
    after = previous.snapshot(expected); write(output/'preservation_after.json', after)
    source_after = sources(); write(output/'sources_after.json', source_after)
    source_equal = source_before == source_after
    passed = before == after and source_equal and all(record['exit_code'] == 0 and record['failed'] == 0
              and record['cases'] > 0 and record['guard_verified'] for record in (selected, plot))
    passed = passed and static['exit_code'] == 0 and static['guard_verified'] and static['all_four_variants_verified']
    artifacts = {str(path.relative_to(ROOT)): {'sha256': previous.sha(path), 'bytes': path.stat().st_size}
                 for path in output.iterdir() if path.is_file()}
    summary = {'task': 'G4-S2A-R2-R1', 'starting_commit': START, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'regression_status': 'PASS' if passed else 'FAIL', 'passed': bool(passed), 'selected_cpu': selected, 'static_models': static, 'system_plot': plot,
        'new_R2_R1_test_cases': selected['r2_r1_cases'], 'excluded_cases': exclusions['excluded_count'],
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
    print(json.dumps({key: summary[key] for key in ('regression_status', 'new_R2_R1_test_cases', 'excluded_cases',
          'sources_unchanged_during_regression', 'preservation', 'wall_s')}, ensure_ascii=False))
    print(json.dumps({'selected_cpu': selected, 'static_models': static, 'system_plot': plot}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.path[:] = [str(ROOT)]+[path for path in sys.path if Path(path).resolve() not in (ROOT, ROOT/'scripts')]
    raise SystemExit(main())
