"""Pure CPU contract and independent analytic oracles; no integration/actor.

Simulator-shaped objects are fakes here. A separate guarded static compilation
pass in audit_g4s2a_r2 checks the real pinned model without advancing physics.
"""
from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts import g4s2a_r2_model as model
from scripts import analyze_g4s2a_r2 as independent


def source_xml():
    root = ET.Element('mujoco')
    world = ET.SubElement(root, 'worldbody')
    for index, name in enumerate(model.ROBOT_BODIES):
        body = ET.SubElement(world, 'body', name=name, pos='1 2 3')
        attrs = {'mass': str(index+1), 'pos': '.1 .2 .3'}
        if index % 2:
            attrs['fullinertia'] = '.2 .3 .4 .001 .002 .003'
        else:
            attrs['diaginertia'] = '.2 .3 .4'
        ET.SubElement(body, 'inertial', **attrs)
        ET.SubElement(body, 'geom', type='sphere', size='.03')
    extra = ET.SubElement(world, 'body', name='peg')
    ET.SubElement(extra, 'inertial', mass='.019', pos='0 0 0', diaginertia='.001 .002 .003')
    return ET.tostring(root, encoding='unicode')


def permit_fixture_xml(monkeypatch, text):
    monkeypatch.setattr(model, 'XML_SHA', hashlib.sha256(text.encode()).hexdigest())


def test_body_offset_modifies_only_eleven_rotational_tensors_including_offdiagonal_preservation(monkeypatch):
    text = source_xml(); permit_fixture_xml(monkeypatch, text)
    before, after = ET.fromstring(text), ET.fromstring(model.adapted_xml(text))
    expected = copy.deepcopy(before)
    for name in model.ROBOT_BODIES:
        inertial = expected.find(f'.//body[@name="{name}"]/inertial')
        key = 'fullinertia' if 'fullinertia' in inertial.attrib else 'diaginertia'
        values = [.21, .31, .41] + ([.001, .002, .003] if key == 'fullinertia' else [])
        actual = after.find(f'.//body[@name="{name}"]/inertial')
        np.testing.assert_allclose([float(x) for x in actual.get(key).split()], values, rtol=0, atol=6e-17)
        # Canonicalize only the expected changed attribute, then compare all
        # other source nodes/attributes, including peg, mass, CoM and geometry.
        inertial.set(key, actual.get(key))
    assert ET.tostring(after) == ET.tostring(expected)


def test_adaptation_refuses_a_changed_frozen_xml_before_parsing():
    with pytest.raises(ValueError, match='FROZEN_XML_CHANGED'):
        model.adapted_xml('<untrusted/>')


def test_adaptation_refuses_missing_mapped_body(monkeypatch):
    text = source_xml().replace('name="hand"', 'name="not_hand"')
    permit_fixture_xml(monkeypatch, text)
    with pytest.raises(ValueError, match='MISSING_MAPPED_INERTIAL:hand'):
        model.adapted_xml(text)


def test_adaptation_refuses_an_invalid_source_tensor(monkeypatch):
    text = source_xml().replace('diaginertia=".2 .3 .4"', 'diaginertia=".2 .3"', 1)
    permit_fixture_xml(monkeypatch, text)
    with pytest.raises(ValueError, match='INVALID_SOURCE_INERTIA'):
        model.adapted_xml(text)


@pytest.mark.parametrize('selector', [0, 1, None, 'P1', np.bool_(True)])
def test_model_selection_requires_an_explicit_boolean_before_compilation(selector, monkeypatch):
    monkeypatch.setattr(model, 'AdaptedPegInsertEnv', lambda: pytest.fail('must reject before compiling'))
    with pytest.raises(ValueError, match='EXPLICIT_BOOLEAN'):
        model.make_model(selector)


@pytest.mark.parametrize('variant', ['E', '', None, 0])
def test_unknown_factorial_selection_cannot_compile_any_model(variant, monkeypatch):
    monkeypatch.setattr(model, 'make_model', lambda *a: pytest.fail('must reject before compiling'))
    with pytest.raises(ValueError, match='UNKNOWN_FACTORIAL_VARIANT'):
        model.ModelPair(variant)


def test_mass_evaluation_copies_the_complete_configuration_and_only_forwards():
    calls = []
    state = SimpleNamespace(qpos=np.zeros(16), qM=np.arange(3), qvel=np.arange(15.), time=8.)
    def forward(m, d): calls.append(('forward', d.qpos.copy()))
    def full_mass(m, output, packed): output[:] = np.diag(np.arange(1., 16.))
    shadow = SimpleNamespace(model=SimpleNamespace(nq=16, nv=15), data=state, peg_qpos_adr=9,
                             mj=SimpleNamespace(mj_forward=forward, mj_fullM=full_mass))
    q = np.arange(16.)/100; q[12:] = [1., 0., 0., 0.]
    before_velocity = state.qvel.copy()
    result = model.mass_at(shadow, q)
    assert len(calls) == 1 and calls[0][0] == 'forward'
    np.testing.assert_array_equal(calls[0][1], q)
    np.testing.assert_array_equal(result, np.diag(np.arange(1., 16.)))
    np.testing.assert_array_equal(state.qvel, before_velocity)
    assert state.time == 8.
    q[0] = -500
    assert state.qpos[0] != -500


@pytest.mark.parametrize('q', [np.zeros(15), np.zeros((16, 1)), [np.nan]*16, [np.inf]*16])
def test_invalid_same_configuration_is_rejected_before_shadow_mutation(q):
    initial = np.arange(16.)
    shadow = SimpleNamespace(model=SimpleNamespace(nq=16), data=SimpleNamespace(qpos=initial.copy()),
                             mj=SimpleNamespace(mj_forward=lambda *a: pytest.fail('invalid state forwarded')))
    with pytest.raises(ValueError, match='INVALID_SAME_CONFIGURATION'):
        model.mass_at(shadow, q)
    np.testing.assert_array_equal(shadow.data.qpos, initial)


@pytest.mark.parametrize('quaternion', [[0., 0., 0., 0.], [2., 0., 0., 0.]])
def test_nonunit_shadow_free_quaternion_is_rejected_before_mutation(quaternion):
    shadow = SimpleNamespace(model=SimpleNamespace(nq=16), peg_qpos_adr=9,
        data=SimpleNamespace(qpos=np.ones(16)),
        mj=SimpleNamespace(mj_forward=lambda *a: pytest.fail('invalid quaternion forwarded')))
    q = np.zeros(16); q[12:] = quaternion
    with pytest.raises(ValueError, match='QUATERNION'):
        model.mass_at(shadow, q)
    np.testing.assert_array_equal(shadow.data.qpos, np.ones(16))


@pytest.mark.parametrize('variant,plant_id,controller_id', [('A', 0, 0), ('B', 1, 0), ('C', 0, 1), ('D', 1, 1)])
def test_all_four_mass_dependencies_are_independent_and_use_one_joint_configuration(monkeypatch, variant, plant_id, controller_id):
    ids = np.array([0, 2, 4, 6, 8, 10, 12])
    mass = [np.diag(np.arange(1., 16.)), np.diag(np.arange(21., 36.))]
    q = np.arange(16.)/20
    data = SimpleNamespace(qpos=q.copy(), qvel=np.arange(15.), qM=np.array([111.]), time=9.)
    physical_model = SimpleNamespace(nv=15, body=lambda name: SimpleNamespace(id=model.ROBOT_BODIES.index(name)))
    def full_mass(m, out, packed): out[:] = mass[plant_id]
    def jac_body(m, d, linear, angular, body): angular[:] = body+1
    pair = object.__new__(model.ModelPair)
    pair.variant = variant; pair.plant_selection = plant_id; pair.controller_selection = controller_id
    pair.plant = SimpleNamespace(model=physical_model, data=data, arm_dof_ids=ids,
                                 mj=SimpleNamespace(mj_fullM=full_mass, mj_jacBody=jac_body))
    pair.shadows = [object(), object()]
    calls = []
    def shadow_mass(shadow, configuration):
        index = pair.shadows.index(shadow); calls.append((index, configuration.copy()))
        return mass[index].copy()
    monkeypatch.setattr(model, 'mass_at', shadow_mass)
    before = copy.deepcopy(data.__dict__)
    result = pair.matrices()
    np.testing.assert_array_equal(result['physical_M'], mass[plant_id][np.ix_(ids, ids)])
    np.testing.assert_array_equal(result['controller_M'], mass[controller_id][np.ix_(ids, ids)])
    assert [index for index, _ in calls] == [0, 1]
    for _, configuration in calls: np.testing.assert_array_equal(configuration, q)
    for key in ('qpos', 'qvel', 'qM'): np.testing.assert_array_equal(getattr(data, key), before[key])
    assert data.time == before['time']
    np.testing.assert_array_equal(result['same_q_M0'], mass[0][np.ix_(ids, ids)])
    np.testing.assert_array_equal(result['same_q_M1'], mass[1][np.ix_(ids, ids)])


def clean_state():
    return {'qpos': [0.]*16, 'qvel': [0.]*15, 's2a': {
        'peg_hand_slip_m': 0., 'peg_hand_slip_rad': 0., 'arm_joint_limit_violation_rad': 0.,
        'finger_joint_limit_violation_m': 0., 'all_contact_penetration_m': 0.}}


@pytest.mark.parametrize('field,threshold,label', [
    ('peg_hand_slip_m', 'stop_slip_m', 'PEG_ESCAPE_SLIP'),
    ('peg_hand_slip_rad', 'stop_rotation_rad', 'PEG_ESCAPE_ROTATION'),
    ('arm_joint_limit_violation_rad', 'stop_arm_limit_rad', 'ARM_LIMIT'),
    ('finger_joint_limit_violation_m', 'stop_finger_limit_m', 'FINGER_LIMIT'),
    ('all_contact_penetration_m', 'stop_penetration_m', 'SEVERE_PENETRATION')])
def test_fixed_stop_thresholds_reject_strict_exceedance_and_nonfinite(field, threshold, label):
    state = clean_state(); state['s2a'][field] = model.THRESHOLDS[threshold]
    assert label not in model.stop_reasons(state, {'left': 1., 'right': 1.}, 0, [])
    state['s2a'][field] = np.nextafter(model.THRESHOLDS[threshold], np.inf)
    assert label in model.stop_reasons(state, {'left': 1., 'right': 1.}, 0, [])
    state['s2a'][field] = np.nan
    assert label in model.stop_reasons(state, {'left': 1., 'right': 1.}, 0, [])


def test_stopping_records_warning_nonfinite_velocity_and_third_contact_loss():
    state = clean_state()
    assert model.stop_reasons(state, {'left': 1., 'right': 1.}, 2, [0]) == []
    assert model.stop_reasons(state, {'left': 0., 'right': 1.}, 3, [1]) == [
        'NUMERICAL_SOLVER_WARNING', 'SUSTAINED_CONTACT_LOSS']
    state['qvel'][6] = 15.00001
    assert model.stop_reasons(state, {'left': 1., 'right': 1.}, 0, []) == ['IMPLAUSIBLE_ARM_VELOCITY']
    state['qvel'][6] = np.inf
    assert 'NONFINITE_STATE' in model.stop_reasons(state, {'left': 1., 'right': 1.}, 0, [])


@pytest.mark.parametrize('step', [0, 236, -1, 1., True])
def test_reference_cannot_extend_the_registered_horizon(step):
    with pytest.raises(ValueError, match='OUTSIDE_REGISTERED_HORIZON'):
        model.fixed_reference(step, np.zeros(3), np.array([1., 0., 0., 0.]), None)


def test_fixed_reference_switches_at_seventeen_and_copies_initial_pose():
    calls = []
    contract = SimpleNamespace(targets=lambda: calls.append('target') or {'archived_action': True})
    position, quaternion = np.array([.1, .2, .3]), np.array([1., 0., 0., 0.])
    assert model.fixed_reference(16, position, quaternion, contract) == {'archived_action': True}
    value = model.fixed_reference(17, position, quaternion, contract)
    value['position_target'][0] = 99; value['quaternion_target'][0] = 99
    assert position[0] == .1 and quaternion[0] == 1.
    final = model.fixed_reference(235, position, quaternion, contract)
    np.testing.assert_array_equal(final['position_target'], position)
    assert calls == ['target']
    assert 12*model.MAX_STEPS+2*30 == 2880


def diagonal_controller_inputs():
    return {'q7': independent.Q_REFERENCE.copy(), 'qdot7': [0., 0., 0., 0., 0., 0., .5],
        'jacobian': np.c_[np.eye(6), np.zeros(6)], 'mass_matrix': np.diag([1., 2., 3., 4., 5., 6., 7.]),
        'quaternion': [1., 0., 0., 0.], 'target_quaternion': [1., 0., 0., 0.],
        'position': [0., 0., 0.], 'target_position': [.01, -.02, .03],
        'linear_fd': [.1, -.1, .05], 'angular_fd': [0., 0., 0.]}


def test_independent_controller_matches_hand_solvable_task_and_nullspace_oracle():
    inputs = diagonal_controller_inputs()
    result = independent.reconstruct_controller(inputs)
    expected_task = np.array([-1., 0., 2., 0., 0., 0., 0.])
    expected_null = np.array([0., 0., 0., 0., 0., 0., -7*6.3246*.5])
    np.testing.assert_allclose(result['task_torque'], expected_task, atol=1e-14)
    np.testing.assert_allclose(result['null_torque'], expected_null, atol=1e-14)
    np.testing.assert_allclose(result['requested'], expected_task+expected_null, atol=1e-14)
    np.testing.assert_allclose(result['null_projector'], np.diag([0.]*6+[1.]), atol=1e-14)
    np.testing.assert_allclose(result['effective_arm_inertias_fingers_locked_kg_m2'], np.arange(1., 8.), atol=1e-14)
    assert result['null_dynamic_residual'] < 1e-14
    inputs['mass_matrix'][6, 6] = 14.
    changed = independent.reconstruct_controller(inputs)
    np.testing.assert_array_equal(changed['task_torque'], result['task_torque'])
    assert changed['null_torque'][6] == pytest.approx(2*result['null_torque'][6])


def test_independent_body_increment_has_analytic_offdiagonal_energy():
    jacobians = {name: np.zeros((3, 7)) for name in independent.BODY_NAMES}
    jacobians['hand'][2, :2] = [1., 2.]
    result = independent.body_mass_increment(jacobians)
    expected = np.zeros((7, 7)); expected[:2, :2] = [[.01, .02], [.02, .04]]
    np.testing.assert_array_equal(result, expected)
    velocity = np.array([3., 4., 0., 0., 0., 0., 0.])
    assert velocity@result@velocity == pytest.approx(.01*(3+2*4)**2)
    assert not np.array_equal(result, .01*np.eye(7))
    del jacobians['link0']
    with pytest.raises(ValueError, match='coverage'):
        independent.body_mass_increment(jacobians)


def test_independent_contact_scope_and_penetration_have_separate_assembly_denominators():
    def contact(a, b, force, distance):
        return {'geom1': a, 'geom2': b, 'normal_force_n': force, 'distance_m': distance}
    contacts = [contact('peg_collision', 'left_finger_geom_3', 2., -4e-6),
        contact('right_finger_geom_7', 'peg_collision', 3., -6e-6),
        contact('peg_collision', 'left_finger_geom_4', -9., -1e-6),
        contact('peg_collision', 'left_finger_geom_1', 100., -2e-6),
        contact('right_finger_geom_3', 'support', 200., -8e-6),
        contact('hole_segment_0', 'peg_collision', 5., -3e-6)]
    assert independent.pad_forces(contacts) == {'left': 2., 'right': 3.}
    assert independent.penetrations(contacts) == (8e-6, 3e-6)
    assert independent.penetrations([]) == (0., 0.)


def test_independent_slip_is_invariant_under_common_rigid_motion_and_detects_relative_motion():
    initial = {'tcp_position_world': [0., 0., 0.], 'tcp_quaternion_wxyz': [1., 0., 0., 0.],
               'peg_root_position': [.1, .2, .3], 'peg_quaternion_wxyz': [1., 0., 0., 0.]}
    moved = {'tcp_position_world': [3., 4., 5.],
        'tcp_quaternion_wxyz': [np.sqrt(.5), 0., 0., np.sqrt(.5)],
        'peg_root_position': [2.8, 4.1, 5.3],
        'peg_quaternion_wxyz': [np.sqrt(.5), 0., 0., np.sqrt(.5)]}
    translation, rotation = independent.slip(moved, initial)
    assert translation < 1e-14 and rotation < 3e-8
    moved['peg_root_position'][2] += .002
    assert independent.slip(moved, initial)[0] == pytest.approx(.002)


def test_independent_velocity_difference_is_interval_acceleration_and_uses_physics_clock():
    before = {'time_s': .25, 'tcp_position_world': [0., 0., 0.],
        'tcp_quaternion_wxyz': [1., 0., 0., 0.], 'qvel': [0.]*15}
    after = copy.deepcopy(before); after['time_s'] += 1/120
    after['tcp_position_world'] = [.001, 0., 0.]; after['qvel'][6] = .5
    # Deliberately incompatible post-forward solver qacc must not be used.
    after['qacc'] = [-999.]*15
    linear, angular, acceleration = independent.finite_difference(before, after)
    np.testing.assert_allclose(linear, [.12, 0., 0.], atol=1e-14)
    np.testing.assert_array_equal(angular, np.zeros(3))
    assert acceleration[6] == pytest.approx(60.)
    after['time_s'] += .01
    with pytest.raises(ValueError, match='timestamp'):
        independent.finite_difference(before, after)


def test_independent_joint_limits_use_registered_indices_and_exact_bounds():
    q = np.array([500., -.2, 500., .4, 500.])
    assert independent.limit_violation(q, [1, 3], [[-.2, .2], [0., .4]]) == 0.
    q[3] = .40001
    assert independent.limit_violation(q, [1, 3], [[-.2, .2], [0., .4]]) == pytest.approx(1e-5)
