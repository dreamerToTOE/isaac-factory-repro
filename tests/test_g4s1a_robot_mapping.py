"""Independent numeric/static controls for G4-S1A; no policy or integration."""
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from scripts import g4s1a_robot_audit as a

ROOT = Path(__file__).resolve().parents[1]


def test_collector_refuses_existing_evidence_before_loading_model(tmp_path, monkeypatch):
    receipt = tmp_path/'historical.json'
    receipt.write_bytes(b'{"preserve":true}\n')
    monkeypatch.setattr(sys, 'argv', ['g4s1a_robot_audit.py', '--output', str(tmp_path)])
    def forbidden(*args, **kwargs):
        raise AssertionError('existing output should be rejected before loading assets')
    monkeypatch.setattr(a, 'audit', forbidden)
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        a.main()
    assert receipt.read_bytes() == b'{"preserve":true}\n'


def evidence():
    return json.loads((ROOT/'evidence/g4_s/s1a/robot_mapping.json').read_text())


def original():
    return json.loads((ROOT/'evidence/g4_s/s0/assets_audit.json').read_text())['original_usd_probe']['result']['assets'][0]


def test_principal_axis_tensor_against_independent_90_degree_rotation():
    tensor = a.inertia_tensor([1, 2, 2.5], [math.sqrt(.5), 0, 0, math.sqrt(.5)])
    np.testing.assert_allclose(tensor, np.diag([2, 1, 2.5]), atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(tensor), [1, 2, 2.5])


@pytest.mark.parametrize('q', [[0, 0, 0, 0], [1, 0, 0], [math.nan, 0, 0, 1], [1e308]*4])
def test_invalid_principal_frame_rejected(q):
    with pytest.raises(ValueError): a.quat_matrix(q)


@pytest.mark.parametrize('diagonal', [[1, -1, 1], [0, 1, 1], [1, math.inf, 1]])
def test_invalid_inertia_metadata_rejected(diagonal):
    with pytest.raises(ValueError): a.inertia_tensor(diagonal, [1, 0, 0, 0])


def test_factory_finger_opening_and_body_axis_convention_against_hand_frame():
    # Body1 joint frame inverse cancels Rx(pi), but parent-frame slide axis flips.
    q = dict.fromkeys(a.JOINT_MAP, 0.)
    zero = a.original_fk(original(), q)
    q['panda_finger_joint1'] = .02; q['panda_finger_joint2'] = .02
    opened = a.original_fk(original(), q)
    hand = zero['panda_hand']
    for name, displacement in [('panda_leftfinger', [0, .02, 0]), ('panda_rightfinger', [0, -.02, 0])]:
        delta = hand[:3, :3].T @ (opened[name][:3, 3]-zero[name][:3, 3])
        np.testing.assert_allclose(delta, displacement, atol=1e-15)
        np.testing.assert_allclose(hand[:3, :3].T @ zero[name][:3, :3], np.eye(3), atol=1e-14)


def test_missing_coordinate_is_not_guessed_from_storage_order():
    q = dict.fromkeys(a.JOINT_MAP, 0.); del q['panda_joint3']
    with pytest.raises(ValueError): a.original_fk(original(), q)


def test_static_tcp_gold_geometry_not_inferred_from_finger_mesh_center():
    q = dict.fromkeys(a.JOINT_MAP, 0.)
    fk = a.original_fk(original(), q)
    # Zero-arm chain: .333+.316+.384-.107-.112071 = .813929 metres.
    np.testing.assert_allclose(fk['panda_fingertip_centered'][:3, 3], [.088, 0, .813929], atol=3e-8)
    np.testing.assert_allclose(np.linalg.inv(fk['panda_hand']) @ fk['panda_fingertip_centered'],
        a.transform([0, 0, .11207100003957748]), atol=1e-14)


def test_all_nine_names_units_limits_addresses_recorded():
    x = evidence(); joints = x['joints']
    assert len(joints) == 9
    assert {j['source']: j['target'] for j in joints} == a.JOINT_MAP
    assert len({j['qpos_address'] for j in joints}) == 9
    assert len({j['dof_address'] for j in joints}) == 9
    assert [j['unit'] for j in joints] == ['radian']*7 + ['metre']*2
    assert max(j['max_limit_difference'] for j in joints[:7]) < 3e-7
    # Source floats are not silently rounded to the candidate's nominal limits.
    assert max(j['max_limit_difference'] for j in joints) > 1e-8


def test_candidate_finger_body_flip_is_reported_not_erased():
    x = evidence()
    for fixture in x['static_frame_checks']:
        right = next(f for f in fixture['frames'] if f['source'] == 'panda_rightfinger')
        left = next(f for f in fixture['frames'] if f['source'] == 'panda_leftfinger')
        assert abs(right['rotation_difference_rad']-math.pi) < 2e-7
        assert left['rotation_difference_rad'] < 2e-7
        assert fixture['TCP_origin_difference_m'] < 3e-8
    right = next(b for b in x['bodies'] if b['source'] == 'panda_rightfinger')
    assert right['source_CoM_expressed_in_target_body_m'][1] < -.0113999
    assert right['CoM_difference_after_frame_mapping_m'] > .025


def test_extra_body_and_unknown_sensor_inertia_are_preserved():
    x = evidence(); bodies = {b['source']: b for b in x['bodies']}
    assert len(bodies) == 13
    sensor = bodies['force_sensor']
    assert sensor['target'] is None and sensor['source_missing_authored_inertia']
    assert sensor['source_CoM_tensor_in_body_axes_kg_m2'] is None
    assert sensor['source_mass_kg'] == pytest.approx(.01)
    assert bodies['panda_fingertip_centered']['source_mass_kg'] == pytest.approx(1e-6)


def test_runtime_body_inertia_addition_is_not_armature():
    x = evidence()
    for b in x['bodies']:
        if b['source_CoM_tensor_in_body_axes_kg_m2'] is None: continue
        delta = np.array(b['source_tensor_plus_runtime_0_01I_conditional'])-np.array(b['source_CoM_tensor_in_body_axes_kg_m2'])
        np.testing.assert_allclose(delta, np.eye(3)*.01, atol=1e-17)
    assert all(j['target_armature'] == .1 for j in x['joints'])
    assert x['candidate_defaults']['all_body_gravcomp'] == [0.]*12


def test_default_actuators_cannot_be_mistaken_for_direct_torque_motors():
    acts = evidence()['candidate_actuators']
    assert len(acts) == 8
    assert all(a['bias_parameters'][1] < 0 for a in acts[:7])
    assert acts[-1]['control_range'] == [0, 255]
    assert acts[-1]['gain_parameters'][0] == pytest.approx(.01568627451)


def test_full_collision_origins_coverage_and_no_faithfulness_claim():
    x = evidence()
    assert len(x['candidate_collision_geoms']) == 23
    assert len(x['original_collision_geometry_archived_metadata']) == 11
    assert sum(g['type'] == 6 for g in x['candidate_collision_geoms']) == 10  # actual fingertip boxes
    assert all(len(g['position_in_body_m']) == 3 and len(g['quaternion_in_body_wxyz']) == 4 for g in x['candidate_collision_geoms'])
    assert 'not faithful' in x['scope']


def test_repeated_compilation_and_invalid_inputs_actual_results():
    x = evidence()
    assert x['engine_version'] == x['engine_runtime_version'] == '3.5.0'
    assert x['compilation']['same_process_recompilation_identical']
    assert x['compilation']['first_MJB_sha256'] == x['compilation']['second_MJB_sha256']
    assert len(x['invalid_asset_controls']) == 4
    assert all(c['rejected'] and c['actual_error'] for c in x['invalid_asset_controls'])
    assert x['restrictions']['physics_integration_steps'] == x['restrictions']['policy_calls'] == 0


def test_fresh_live_compilation_is_pinned_and_has_no_integration():
    interpreter = Path('/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python')
    if not interpreter.is_file():
        pytest.skip('pinned MuJoCo 3.5.0 environment unavailable')
    command = [str(interpreter), '-c', 'import json; from scripts.g4s1a_robot_audit import audit; print(json.dumps(audit()))']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60, check=True)
    x = json.loads(result.stdout)
    assert x['compilation']['nq'] == x['compilation']['nv'] == 9
    assert x['compilation']['nu'] == 8
    assert x['compilation']['first_MJB_sha256'] == evidence()['compilation']['first_MJB_sha256']
    assert x['restrictions']['physics_integration_steps'] == 0
