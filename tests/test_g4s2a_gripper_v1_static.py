"""Independent mutation tests for the pre-integration engineering audit."""
import pytest

pytest.importorskip('mujoco')
from scripts import audit_g4s2a_gripper_v1_static as audit_module
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv


def test_static_transmission_and_saturation_independent_oracle():
    result = audit_module.audit()
    assert result['status'] == 'PASS'
    assert result['physics_steps'] == 0
    assert result['finite_difference_dlength_dq'] == pytest.approx([1., 1.], abs=1e-9)
    cases = result['force_tests']
    assert [x['actuator_force_n'] for x in cases] == pytest.approx([0., 3.75, 3.4905, -40., 40.])
    assert cases[3]['joint_generalized_actuator_force_n'] == pytest.approx([-40., -40.])
    assert cases[4]['joint_generalized_actuator_force_n'] == pytest.approx([40., 40.])
    assert [result['initials'][i]['seed'] for i in ('I2', 'I3')] == [2101, 2102]


@pytest.mark.parametrize(('mutation', 'message'), [
    ('force', 'force_range'),
    ('geometry', 'geom_friction'),
    ('arm', 'arm_torque_1.actuator_forcerange'),
    ('gain', 'gain_kp'),
])
def test_static_audit_rejects_mutations(monkeypatch, mutation, message):
    env = NativeGripperEnv()
    if mutation == 'force': env.model.actuator_forcerange[env.model.actuator('gripper_v1_position').id] = [-41, 41]
    if mutation == 'geometry': env.model.geom_friction[env.model.geom('peg_collision').id, 0] += .01
    if mutation == 'arm': env.model.actuator_forcerange[env.model.actuator('arm_torque_1').id] = [-88, 88]
    if mutation == 'gain': env.model.actuator_gainprm[env.model.actuator('gripper_v1_position').id, 0] += 1
    monkeypatch.setattr(audit_module, 'NativeGripperEnv', lambda: env)
    with pytest.raises(ValueError, match=message): audit_module.audit()
