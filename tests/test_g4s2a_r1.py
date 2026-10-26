"""CPU-only archived-data/algebra/guard tests; no actor, GUI or integration.

All simulator-shaped objects below are fakes. The inertia oracle uses explicit
kinetic-energy matrices and never compiles a MuJoCo model or advances physics.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import g4s2a_r1_viewer as viewer


@pytest.fixture(scope='module')
def archive():
    return viewer.load_archive()


def test_archived_first_eight_steps_are_one_action_and_instability_precedes_saturation(archive):
    initial, rows, action = archive
    assert len(action['control_substeps']) == 8
    assert [row['policy_step'] for row in rows[:8]] == [1]*8
    np.testing.assert_allclose(action['clipped'], [-1., 1., -1., .09105674177408218, .4455225169658661, -1.], rtol=0, atol=1e-14)
    requested = [sub['requested_arm_torque_nm'][6] for sub in action['control_substeps']]
    velocity = [row['state']['qvel'][6] for row in rows[:8]]
    actual = [row['state']['integration_actuation']['arm_actuator_force_nm'][6] for row in rows[:8]]
    np.testing.assert_allclose(velocity[:4], [.5046560985564515, -5.149692851915195, 9.126765297359182, -5.161160601229339], atol=1e-12, rtol=0)
    assert velocity[0] > 0 and velocity[1] < 0 and abs(velocity[1]) > 10*abs(velocity[0])
    assert abs(requested[1]) < 12 and actual[1] == requested[1]
    assert next(i+1 for i, force in enumerate(requested) if abs(force) > 12) == 3
    assert actual[2] == 12 and actual[3] == -12
    assert len(initial['qpos']) == 16 and len(initial['qvel']) == 15


def test_archived_task_torque_dominates_joint7_and_damping_growth_is_recorded(archive):
    _, _, action = archive
    substeps = action['control_substeps']
    assert max(abs(row['null_arm_torque_nm'][6]) for row in substeps) < 1e-14
    for row in substeps:
        np.testing.assert_array_equal(np.asarray(row['task_arm_torque_nm']) + row['null_arm_torque_nm'], row['requested_arm_torque_nm'])
        velocity = row['angular_fd_before_rad_s'][2]
        # Factory's world-z rotational Kd is 2sqrt(30), with NEGATIVE damping.
        derivative = -2*np.sqrt(30)*velocity
        if row is substeps[1]:
            assert derivative == pytest.approx(4.965572350189648)
            assert row['pd_wrench'][5] > 0
        if row is substeps[2]:
            assert derivative == pytest.approx(-58.599919150085654)
            assert row['pd_wrench'][5] < 0


def test_archived_grasp_loss_and_joint_limit_onsets_are_distinct(archive):
    _, rows, _ = archive
    forces = [viewer.contact_forces(row['state']) for row in rows]
    first_lost = next(i+1 for i, force in enumerate(forces) if min(force.values()) <= 1e-6)
    bounds = np.asarray([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                         [-3.0718, -.0698], [-2.8973, 2.8973], [-.0175, 3.7525], [-2.8973, 2.8973]])
    first_arm = next(i+1 for i, row in enumerate(rows)
                     if np.any(np.asarray(row['state']['qpos'][:7]) < bounds[:, 0]-1e-6)
                     or np.any(np.asarray(row['state']['qpos'][:7]) > bounds[:, 1]+1e-6))
    first_finger = next(i+1 for i, row in enumerate(rows)
                        if min(row['state']['qpos'][7:9]) < -1e-6 or max(row['state']['qpos'][7:9]) > .040001)
    assert (first_lost, first_finger, first_arm) == (4, 61, 76)
    assert viewer.EVENTS['grasp_loss'] == first_lost
    assert viewer.EVENTS['first_finger_limit'] == first_finger
    assert viewer.EVENTS['first_arm_limit'] == first_arm
    assert viewer.EVENTS['first_slip'] == 3


def test_archive_digest_rejects_tamper_before_loading_trajectory(tmp_path):
    payload = tmp_path/'data.json'
    payload.write_text('{"changed":true}\n')
    original = b'{"original":true}\n'
    collection = {'raw_directory': '/raw', 'artifacts': [
        {'raw_path': '/raw/data.json', 'bundle_sha256': hashlib.sha256(original).hexdigest(), 'lossless_gzip': False}]}
    (tmp_path/'collection.json').write_text(json.dumps(collection))
    with pytest.raises(ValueError, match='ARCHIVE_DIGEST_MISMATCH'):
        viewer.load_archive(tmp_path)


def fake_environment():
    calls = []
    data = SimpleNamespace(qpos=np.zeros(16), qvel=np.zeros(15), ctrl=np.zeros(9), time=0.)
    data.qpos[12] = 1.
    model = SimpleNamespace(nq=16, nv=15, body_pos=np.zeros((2, 3)), body_quat=np.tile([1., 0., 0., 0.], (2, 1)))
    fake = SimpleNamespace(model=model, data=data, arm_actuator_ids=np.arange(7), finger_actuator_ids=np.array([7, 8]),
                           hole_body_id=1, peg_qpos_adr=9,
                           mj=SimpleNamespace(mj_forward=lambda *args: calls.append('forward')))
    return fake, calls


def valid_state():
    q = np.zeros(16); q[12] = 1.
    return {'qpos': q.tolist(), 'qvel': np.zeros(15).tolist(), 'time_s': .25,
            'software_limited_arm_torque_nm': np.arange(7, dtype=float).tolist(),
            'finger_command': {'commanded_force_n': [-.2, -.21]},
            'hole_root_position': [.6, 0., .05], 's2a': {'hole_quaternion_wxyz': [1., 0., 0., 0.]}}


def test_archived_state_application_is_forward_only_and_preserves_recorded_clock():
    env, calls = fake_environment()
    state = valid_state()
    viewer.apply_archived_state(env, state)
    assert calls == ['forward']
    assert env.data.time == .25
    np.testing.assert_array_equal(env.data.qpos, state['qpos'])
    np.testing.assert_array_equal(env.data.ctrl, list(range(7)) + [-.2, -.21])
    np.testing.assert_array_equal(env.model.body_pos[1], [.6, 0., .05])


@pytest.mark.parametrize('field,value', [
    ('qpos', [0.]*15), ('qvel', [0.]*14),
    ('qpos', [float('nan')]+[0.]*15), ('qvel', [float('inf')]+[0.]*14),
    ('time_s', float('nan')), ('software_limited_arm_torque_nm', [0.]*6),
    ('software_limited_arm_torque_nm', [float('inf')]+[0.]*6),
    ('hole_root_position', [.6, float('nan'), .05]),
    ('finger_command', {'commanded_force_n': [float('nan'), 0.]}),
    ('s2a', {'hole_quaternion_wxyz': [float('nan'), 0., 0., 0.]}),
    ('s2a', {'hole_quaternion_wxyz': [2., 0., 0., 0.]}),
])
def test_invalid_archived_fields_are_rejected_before_mutation_or_forward(field, value):
    env, calls = fake_environment()
    state = valid_state(); state[field] = value
    before = copy.deepcopy(env.data.__dict__)
    position = env.model.body_pos.copy(); quaternion = env.model.body_quat.copy()
    with pytest.raises((ValueError, RuntimeError)):
        viewer.apply_archived_state(env, state)
    assert calls == []
    for key in ('qpos', 'qvel', 'ctrl'):
        np.testing.assert_array_equal(getattr(env.data, key), before[key])
    assert env.data.time == before['time']
    np.testing.assert_array_equal(env.model.body_pos, position)
    np.testing.assert_array_equal(env.model.body_quat, quaternion)


def test_contact_force_totals_use_only_original_five_pads_and_peg_contacts():
    def contact(a, b, force): return {'geom1': a, 'geom2': b, 'normal_force_n': force}
    state = {'contacts': [contact('peg_collision', 'left_finger_geom_3', 2.),
        contact('right_finger_geom_7', 'peg_collision', 3.),
        contact('peg_collision', 'left_finger_geom_6', -100.),
        contact('peg_collision', 'left_finger_geom_1', 500.),
        contact('peg_collision', 'left_finger_geom_99', 600.),
        contact('left_finger_geom_4', 'independent_support', 700.),
        contact('peg_collision', 'hand_geom_5', 800.)]}
    assert viewer.contact_forces(state) == {'left': 2., 'right': 3.}


def test_gui_gate_rejects_missing_display_even_with_wayland(monkeypatch):
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    monkeypatch.setattr(viewer.subprocess, 'run', lambda *a, **k: pytest.fail('Display probe must not run'))
    with pytest.raises(RuntimeError, match='DISPLAY missing'):
        viewer.require_gui_environment()


@pytest.mark.parametrize('backend', ['egl', 'EGL', 'osmesa'])
def test_gui_gate_rejects_offscreen_backends_before_probe(monkeypatch, backend):
    monkeypatch.setenv('DISPLAY', ':99')
    monkeypatch.setenv('MUJOCO_GL', backend)
    monkeypatch.setattr(viewer.subprocess, 'run', lambda *a, **k: pytest.fail('Offscreen is forbidden'))
    with pytest.raises(RuntimeError, match='offscreen backend'):
        viewer.require_gui_environment()


def test_gui_gate_rejects_nonmain_thread_without_graphics_calls(monkeypatch):
    errors = []
    monkeypatch.setenv('DISPLAY', ':99')
    monkeypatch.delenv('MUJOCO_GL', raising=False)
    monkeypatch.setattr(viewer.subprocess, 'run', lambda *a, **k: pytest.fail('Graphics probe must not run'))
    def worker():
        try: viewer.require_gui_environment()
        except Exception as exc: errors.append(exc)
    thread = threading.Thread(target=worker); thread.start(); thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(errors) == 1 and 'GUI_MAIN_THREAD_REQUIRED' in str(errors[0])


def test_gui_gate_performs_real_session_probe_but_unit_test_opens_no_window(monkeypatch):
    calls = []
    monkeypatch.setenv('DISPLAY', ':99')
    monkeypatch.delenv('MUJOCO_GL', raising=False)
    monkeypatch.setattr(viewer.subprocess, 'run', lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    viewer.require_gui_environment()
    assert calls[0][0] == ['xdpyinfo']
    assert calls[0][1]['check'] is True


def test_playback_controls_navigate_only_recorded_frames_and_pause_at_end():
    controller = viewer.PlaybackController(nsteps=8)
    controller.step(); assert controller.index == 1 and not controller.playing
    controller.jump(viewer.EVENTS['first_slip']); assert controller.index == 3
    controller.toggle(); assert controller.playing
    assert controller.tick(10.) == 3
    assert controller.tick(10.+1.1/(120*.1)) == 4
    controller.cycle_speed(); assert controller.speed == .25
    controller.cycle_speed(); assert controller.speed == 1.
    controller.tick(20.); controller.tick(20.1)
    assert controller.index == 8 and not controller.playing
    controller.jump(0); assert controller.index == 0 and not controller.playing
    controller.cycle_speed(); assert controller.speed == .1


@pytest.mark.parametrize('bad_index', [-1, 9, 1.5])
def test_playback_rejects_out_of_range_frame_without_state_change(bad_index):
    controller = viewer.PlaybackController(nsteps=8)
    controller.jump(3)
    with pytest.raises(ValueError): controller.jump(bad_index)
    assert controller.index == 3 and not controller.playing


@pytest.mark.parametrize('bad_clock', [float('nan'), float('inf'), 9.])
def test_playback_clock_is_finite_and_monotonic(bad_clock):
    controller = viewer.PlaybackController(nsteps=8)
    controller.toggle(); controller.tick(10.)
    with pytest.raises(ValueError): controller.tick(bad_clock)
    assert controller.index == 0


def test_passive_viewer_close_joins_only_supplied_owned_render_threads():
    calls = []
    renderer = SimpleNamespace(name='owned-renderer', join=lambda wait: calls.append(('join', wait)), is_alive=lambda: False)
    fake_viewer = SimpleNamespace(close=lambda: calls.append(('close',)))
    receipt = viewer.close_and_join(fake_viewer, [renderer], timeout_s=1)
    assert calls[0] == ('close',) and calls[1][0] == 'join'
    assert 0 <= calls[1][1] <= 1
    assert receipt['remaining_render_threads'] == []


def test_passive_viewer_close_does_not_claim_success_with_live_render_thread():
    renderer = SimpleNamespace(name='not-exited', join=lambda wait: None, is_alive=lambda: True)
    fake_viewer = SimpleNamespace(close=lambda: None)
    with pytest.raises(RuntimeError, match='GUI_CLOSE_RENDER_THREAD_TIMEOUT'):
        viewer.close_and_join(fake_viewer, [renderer], timeout_s=0)


def test_body_rotational_inertia_pullback_is_not_joint_armature(monkeypatch):
    from scripts.g4s2a_r1_dynamics_audit import delta_mass_from_body_inertia
    jacobians = {1: np.zeros((3, 7))}
    jacobians[1][2, :2] = 1.
    def jac_body(model, data, linear, angular, body): angular[:] = jacobians[body]
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(mj_jacBody=jac_body))
    model = SimpleNamespace(nv=7, body=lambda name: SimpleNamespace(id=1))
    delta = delta_mass_from_body_inertia(model, object(), np.arange(7), ['bodyA'])
    expected = np.zeros((7, 7)); expected[:2, :2] = .01
    np.testing.assert_allclose(delta, expected, atol=1e-15, rtol=0)
    assert not np.array_equal(delta, .01*np.eye(7))
    velocity = np.array([.2, -.3, .4, 0., 0., 0., 0.])
    assert .5*velocity@delta@velocity == pytest.approx(.5*.01*np.dot(jacobians[1]@velocity, jacobians[1]@velocity))
    assert delta[0, 1] != 0 and delta[2, 2] == 0


def test_isotropic_body_inertia_offset_is_invariant_to_body_frame_rotation(monkeypatch):
    from scripts.g4s2a_r1_dynamics_audit import delta_mass_from_body_inertia
    base = np.zeros((3, 7))
    base[:, :3] = [[1., 0., 1.], [0., 2., -1.], [1., 1., 0.]]
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    active = [base]
    def jac_body(model, data, linear, angular, body): angular[:] = active[0]
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(mj_jacBody=jac_body))
    model = SimpleNamespace(nv=7, body=lambda name: SimpleNamespace(id=1))
    before = delta_mass_from_body_inertia(model, object(), np.arange(7), ['body'])
    active[0] = rotation@base
    after = delta_mass_from_body_inertia(model, object(), np.arange(7), ['body'])
    np.testing.assert_allclose(before, after, atol=1e-15, rtol=0)


def test_integration_budget_rejects_unregistered_duplicate_and_thirteenth_variant():
    from scripts.g4s2a_r1_diagnostics import IntegrationBudget
    budget = IntegrationBudget(max_variants=12, max_steps=16, max_wall_s=600)
    with pytest.raises((ValueError, RuntimeError)): budget.before_step('never-registered')
    budget.begin('case0')
    with pytest.raises((ValueError, RuntimeError)): budget.begin('case0')
    for index in range(1, 12): budget.begin(f'case{index}')
    with pytest.raises((ValueError, RuntimeError)): budget.begin('case12')


def test_integration_budget_rejects_seventeenth_step_before_any_extra_work():
    from scripts.g4s2a_r1_diagnostics import IntegrationBudget
    budget = IntegrationBudget(max_variants=12, max_steps=16, max_wall_s=600)
    budget.begin('bounded')
    for _ in range(16):
        budget.before_step('bounded'); budget.record_step(.01)
    with pytest.raises((ValueError, RuntimeError)): budget.before_step('bounded')


@pytest.mark.parametrize('elapsed', [float('nan'), float('inf'), -.1])
def test_integration_budget_rejects_invalid_timing_instead_of_weakening_limit(elapsed):
    from scripts.g4s2a_r1_diagnostics import IntegrationBudget
    budget = IntegrationBudget(max_variants=12, max_steps=16, max_wall_s=600)
    budget.begin('bounded'); budget.before_step('bounded')
    with pytest.raises((ValueError, RuntimeError)): budget.record_step(elapsed)


def test_integration_budget_stops_after_ten_minutes_of_recorded_physics():
    from scripts.g4s2a_r1_diagnostics import IntegrationBudget
    budget = IntegrationBudget(max_variants=12, max_steps=16, max_wall_s=600)
    budget.begin('bounded'); budget.before_step('bounded'); budget.record_step(600)
    with pytest.raises((ValueError, RuntimeError)): budget.before_step('bounded')


@pytest.mark.parametrize('override', [{'max_variants': 13}, {'max_steps': 17}, {'max_wall_s': 601}])
def test_integration_budget_constructor_cannot_widen_authorized_scope(override):
    from scripts.g4s2a_r1_diagnostics import IntegrationBudget
    limits = {'max_variants': 12, 'max_steps': 16, 'max_wall_s': 600}
    limits.update(override)
    with pytest.raises((ValueError, RuntimeError)): IntegrationBudget(**limits)
