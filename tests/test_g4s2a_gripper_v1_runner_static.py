"""Runner preflight: loaded reset, observation and command wiring, no physics."""

import numpy as np
import pytest
pytest.importorskip("mujoco")

from scripts.g4s2a_gripper_v1_diagnostics import actuation, initial, placement
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv


@pytest.mark.parametrize('name', ['I2', 'I3'])
def test_archived_preclosure_placement_and_actuation_without_step(name, monkeypatch):
    env = NativeGripperEnv()
    def forbidden(*args, **kwargs):
        raise AssertionError('NO_PHYSICAL_INTEGRATION_IN_STATIC_TEST')
    monkeypatch.setattr(env.mj, 'mj_step', forbidden)
    archived = initial(env, name)
    placed = placement(env)
    transmission = actuation(env)
    np.testing.assert_array_equal(env.data.qpos, archived['qpos'])
    np.testing.assert_array_equal(env.data.qvel, np.zeros(env.model.nv))
    assert float(env.data.time) == 0
    assert placed['valid'] and placed['independent_peg'] and placed['entirely_above_opening']
    assert not placed['forbidden_initial_contacts']
    np.testing.assert_allclose(placed['finger_openings_m'], [.00499125, .00499125], atol=1e-15, rtol=0)
    assert transmission['tendon_length_m'] == pytest.approx(.0099825, abs=1e-15)
    assert transmission['tendon_velocity_m_s'] == 0
    assert transmission['control_m'] == 0
    assert transmission['actuator_force_n'] == pytest.approx(-37.434375, abs=1e-10)
    moment = np.zeros(env.model.nv)
    moment[env.finger_dof_ids] = 1
    np.testing.assert_array_equal(transmission['moment_arm_by_dof'], moment)
    np.testing.assert_allclose(transmission['finger_generalized_force_n'], [-37.434375]*2, atol=1e-10, rtol=0)


def test_native_step_wiring_with_stub_no_integration(monkeypatch):
    env = NativeGripperEnv()
    previous_q = env.data.qpos.copy()
    previous_time = float(env.data.time)
    called = []
    def stub(model, data):
        called.append({'ctrl': data.ctrl.copy(), 'time': float(data.time)})
        # Deliberately do not call the original mj_step or advance state.
    monkeypatch.setattr(env.mj, 'mj_step', stub)
    torques = np.arange(7, dtype=float)
    state = env.physics_step(torques, finger_target=.013, record=False)
    assert len(called) == 1
    np.testing.assert_array_equal(called[0]['ctrl'][env.arm_actuator_ids], torques)
    assert called[0]['ctrl'][env.finger_actuator_ids[0]] == pytest.approx(.026, abs=1e-15)
    assert state['finger_command']['target_opening_per_finger_m'] == .013
    assert state['finger_command']['target_tendon_length_m'] == .026
    assert called[0]['time'] == previous_time == state['time_s']
    np.testing.assert_array_equal(env.data.qpos, previous_q)
