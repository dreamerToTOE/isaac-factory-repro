"""State guard negative controls: no MuJoCo integration or policy execution."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
from scripts.g4s2a_gripper_v1_diagnostics import initial
from scripts.g4s2a_gripper_v1_r1_state import (snapshot, compare, guard, authorized_ctrl,
    copy_display_state, model_fingerprint, IsolatedViewer, StateIntegrityError)

@pytest.fixture
def env():
    env = NativeGripperEnv()
    initial(env, 'I3')
    return env

@pytest.mark.parametrize('field,index,delta', [
    ('qpos', 0, 1e-6), ('qpos', 7, 1e-6), ('qpos', 9, 1e-6),
    ('qpos', 12, 1e-6), ('qvel', 0, 1e-6), ('time', None, 1e-6),
    ('qfrc_applied', 0, 1e-6), ('qacc_warmstart', 0, 1e-6),
    ('ctrl', 0, 1e-6), ('xfrc_applied', (1, 0), 1e-6),
])
def test_rejects_tampering_before_step(env, field, index, delta):
    expected = snapshot(mujoco, env.model, env.data)
    if index is None:
        setattr(env.data, field, getattr(env.data, field) + delta)
    else:
        getattr(env.data, field)[index] += delta
    calls = []
    def integration(): calls.append('called')
    with pytest.raises(StateIntegrityError) as error:
        guard(expected, mujoco, env.model, env.data, 'immediately_before_mj_step')
        integration()
    assert calls == []
    assert field in {item['field'] for item in error.value.mismatches}
    assert error.value.expected == expected
    assert error.value.actual != expected


def test_controls_authorized_without_admitting_qpos_tampering(env):
    initial_state = snapshot(mujoco, env.model, env.data)
    controls = env.data.ctrl.copy(); controls[-1] = .005
    expected = authorized_ctrl(mujoco, env.model, initial_state, controls)
    env.data.ctrl[:] = controls
    guard(expected, mujoco, env.model, env.data, 'control_update')
    env.data.qpos[0] += 1e-6
    with pytest.raises(StateIntegrityError):
        guard(expected, mujoco, env.model, env.data, 'before_step')


def test_independent_copy_is_one_way_and_forward_does_not_integrate(env):
    viewer = IsolatedViewer(env, 'NO_PHYSICS')
    before = snapshot(mujoco, env.model, env.data)
    copy_display_state(mujoco, env.model, env.data, viewer.model, viewer.data)
    assert not compare(before, snapshot(mujoco, env.model, env.data))
    np.testing.assert_array_equal(viewer.data.qpos, env.data.qpos)
    viewer.data.qpos[0] += .1
    viewer.data.qfrc_applied[0] = 1
    assert not compare(before, snapshot(mujoco, env.model, env.data))
    copy_display_state(mujoco, env.model, env.data, viewer.model, viewer.data)
    np.testing.assert_array_equal(viewer.data.qpos, env.data.qpos)
    np.testing.assert_array_equal(viewer.data.qfrc_applied, env.data.qfrc_applied)


def test_same_instances_and_unequal_models_are_rejected(env):
    viewer = IsolatedViewer(env, 'NO_PHYSICS')
    with pytest.raises(ValueError, match='DISTINCT'):
        copy_display_state(mujoco, env.model, env.data, env.model, env.data)
    viewer.model.body_mass[1] += .01
    assert model_fingerprint(viewer.model) != model_fingerprint(env.model)
    with pytest.raises(ValueError, match='FINGERPRINT'):
        copy_display_state(mujoco, env.model, env.data, viewer.model, viewer.data)


def test_new_experiment_id_does_not_reset_previous_data(env):
    before = snapshot(mujoco, env.model, env.data)
    second = NativeGripperEnv(); initial(second, 'I2')
    assert not compare(before, snapshot(mujoco, env.model, env.data))
    assert not np.array_equal(env.data.qpos, second.data.qpos)


def test_bitwise_signed_zero_is_not_silently_accepted(env):
    expected = snapshot(mujoco, env.model, env.data)
    env.data.qvel[0] = -0.0
    with pytest.raises(StateIntegrityError):
        guard(expected, mujoco, env.model, env.data, 'signed_zero')
