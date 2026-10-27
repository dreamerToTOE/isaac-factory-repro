"""Static-only native gripper checks; no mj_step or PPO calls."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest
pytest.importorskip("mujoco")

from scripts.g4s1b_env import AdaptedPegInsertEnv
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv, native_xml, tendon_target
from scripts.g4s2a_r2_model import adapted_xml, make_model


@pytest.fixture(scope='module')
def native():
    return NativeGripperEnv()


@pytest.fixture(scope='module')
def baseline():
    return make_model(True)


def test_exact_xml_delta():
    frozen = AdaptedPegInsertEnv().xml
    old = ET.fromstring(adapted_xml(frozen))
    new = ET.fromstring(native_xml(frozen))
    # Remove only the declared replacement mechanisms, then compare every
    # element of the remaining physical scene and seven arm actuators.
    old.remove(old.find('equality'))
    new.remove(new.find('tendon'))
    for name in ('finger_pd_1', 'finger_pd_2'):
        arm = old.find('actuator')
        arm.remove(arm.find(f"motor[@name='{name}']"))
    arm = new.find('actuator')
    arm.remove(arm.find("position[@name='gripper_v1_position']"))
    assert ET.tostring(old) == ET.tostring(new)


def test_compiled_unchanged_physics_and_arm(native, baseline):
    m, old = native.model, baseline.model
    assert (m.nq, m.nv, m.nu, m.ntendon, m.neq) == (16, 15, 8, 1, 0)
    assert m.opt.timestep == old.opt.timestep == 1/120
    assert m.opt.integrator == old.opt.integrator
    for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                 'body_gravcomp', 'geom_type', 'geom_size', 'geom_pos',
                 'geom_quat', 'geom_friction', 'geom_solref', 'geom_solimp',
                 'geom_contype', 'geom_conaffinity', 'jnt_range', 'jnt_type',
                 'dof_armature', 'dof_damping', 'dof_frictionloss'):
        np.testing.assert_array_equal(getattr(m, name), getattr(old, name), err_msg=name)
    for name in ('actuator_forcerange', 'actuator_ctrlrange', 'actuator_gear',
                 'actuator_trntype', 'actuator_trnid'):
        np.testing.assert_array_equal(getattr(m, name)[:7], getattr(old, name)[:7], err_msg=name)
    assert int(m.jnt_type[native.peg_joint_id]) == int(native.mj.mjtJoint.mjJNT_FREE)
    assert np.all(m.body_mass[1:] > 0)
    dynamic = m.body_mass > 0
    assert np.all(m.body_inertia[dynamic] > 0)
    for name in ('left_finger', 'right_finger'):
        body = m.body(name).id
        assert any(m.geom_contype[i] and m.geom_conaffinity[i]
                   for i in range(m.ngeom) if m.geom_bodyid[i] == body)
    assert m.geom('peg_collision').contype != 0 and m.geom('peg_collision').conaffinity != 0


def test_tendon_coordinate_moment_arms_and_asymmetry(native):
    native.reset(peg_pos=[1, 1, 1], finger_opening=.02)
    data, model, mj = native.data, native.model, native.mj
    data.qpos[native.finger_qpos_ids] = [.013, .021]
    data.qvel[native.finger_dof_ids] = [.02, -.01]
    mj.mj_forward(model, data)
    assert data.ten_length[native.tendon_id] == pytest.approx(.034, abs=1e-14)
    assert data.ten_velocity[native.tendon_id] == pytest.approx(.01, abs=1e-14)
    index = int(native.finger_actuator_ids[0]); start = int(data.moment_rowadr[index]); count = int(data.moment_rownnz[index])
    assert count == 2
    np.testing.assert_array_equal(data.moment_colind[start:start+count], native.finger_dof_ids)
    np.testing.assert_allclose(data.actuator_moment[start:start+count], [1, 1], atol=1e-14, rtol=0)
    # Independent finite difference of tendon length with respect to each q.
    delta = 1e-7
    length = float(data.ten_length[native.tendon_id])
    for qid in native.finger_qpos_ids:
        data.qpos[qid] += delta
        mj.mj_forward(model, data)
        assert (data.ten_length[native.tendon_id] - length) / delta == pytest.approx(1, abs=1e-9)
        data.qpos[qid] -= delta
        mj.mj_forward(model, data)
    assert data.qpos[native.finger_qpos_ids[0]] != data.qpos[native.finger_qpos_ids[1]]


@pytest.mark.parametrize('q,v,target', [
    ((.012, .013), (.001, .002), .013),
    ((.02, .02), (0., 0.), 0.),
    ((0., 0.), (0., 0.), .04),
    ((.021, .009), (.005, -.003), .015),
])
def test_force_mapping_and_clipping(native, q, v, target):
    native.reset(peg_pos=[1, 1, 1], finger_opening=.02)
    native.data.qpos[native.finger_qpos_ids] = q
    native.data.qvel[native.finger_dof_ids] = v
    native.data.ctrl[native.finger_actuator_ids[0]] = tendon_target(target)
    native.mj.mj_forward(native.model, native.data)
    force = np.clip(3750 * (2*target - sum(q)) - 86.5 * sum(v), -40, 40)
    assert native.data.actuator_force[native.finger_actuator_ids[0]] == pytest.approx(force, abs=1e-11)
    np.testing.assert_allclose(native.data.qfrc_actuator[native.finger_dof_ids], [force, force], atol=1e-11, rtol=0)
    assert max(abs(native.data.qfrc_actuator[native.finger_dof_ids])) <= 40


@pytest.mark.parametrize('value,expected', [(0, 0), (.02, .04), (.04, .08)])
def test_target_units(value, expected):
    assert tendon_target(value) == pytest.approx(expected)


@pytest.mark.parametrize('value', [-.0001, .0401, float('nan'), float('inf')])
def test_invalid_target_rejected(value):
    with pytest.raises(ValueError):
        tendon_target(value)


def test_generated_xml_is_deterministic():
    original = AdaptedPegInsertEnv().xml
    assert native_xml(original) == native_xml(original)
    with pytest.raises(ValueError):
        native_xml(original + ' ')
