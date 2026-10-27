"""Versioned, native MuJoCo tendon gripper; no actor and no implicit probes.

The original S1B environment remains unchanged.  P_FULL body inertia is
retained, while the two finger motors and finger equality are replaced by one
fixed sum tendon and one limited native position actuator.  A tendon is a
transmission, not a kinematic equality: unequal finger positions are allowed.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import numpy as np

from scripts.g4s1b_env import AdaptedPegInsertEnv
from scripts.g4s2a_r2_model import adapted_xml

TENDON_NAME = 'gripper_v1_sum'
ACTUATOR_NAME = 'gripper_v1_position'
KP = 3750.0
KV = 86.5
FORCE_LIMIT_N = 40.0


def tendon_target(finger_target: float) -> float:
    """Translate the frozen per-finger opening into the sum coordinate."""
    if not math.isfinite(finger_target) or not 0 <= finger_target <= .04:
        raise ValueError('INVALID_FINGER_TARGET')
    return 2.0 * float(finger_target)


def native_xml(frozen_s1b_xml: str) -> str:
    """Construct the only candidate; adapted_xml authenticates frozen input."""
    root = ET.fromstring(adapted_xml(frozen_s1b_xml))
    equality = root.find('equality')
    if equality is None or len(equality) != 1:
        raise ValueError('UNEXPECTED_FINGER_EQUALITY')
    eq = equality[0]
    if eq.tag != 'joint' or eq.get('joint1') != 'finger_joint1' or eq.get('joint2') != 'finger_joint2':
        raise ValueError('UNEXPECTED_FINGER_EQUALITY')
    equality.remove(eq)
    root.remove(equality)
    actuators = root.find('actuator')
    if actuators is None:
        raise ValueError('MISSING_ACTUATORS')
    for name in ('finger_pd_1', 'finger_pd_2'):
        motor = actuators.find(f"motor[@name='{name}']")
        if motor is None:
            raise ValueError('MISSING_FINGER_MOTOR:' + name)
        actuators.remove(motor)
    if root.find('tendon') is not None:
        raise ValueError('UNEXPECTED_TENDON')
    tendon = ET.SubElement(root, 'tendon')
    fixed = ET.SubElement(tendon, 'fixed', name=TENDON_NAME)
    for name in ('finger_joint1', 'finger_joint2'):
        ET.SubElement(fixed, 'joint', joint=name, coef='1')
    ET.SubElement(actuators, 'position', name=ACTUATOR_NAME, tendon=TENDON_NAME,
                  kp=str(KP), kv=str(KV), ctrlrange='0 .08', ctrllimited='true',
                  forcerange='-40 40', forcelimited='true')
    return ET.tostring(root, encoding='unicode')


class NativeGripperEnv(AdaptedPegInsertEnv):
    """P_FULL plant with native sum-tendon drive and unchanged arm control.

    finger_target is the existing per-finger opening semantic; ctrl is twice
    this value in tendon coordinates.  Logging distinguishes the native
    actuator output from each generalized finger force and contact forces.
    """
    def __init__(self):
        super().__init__()
        original = self.model
        self.xml = native_xml(self.xml)
        self.model = self.mj.MjModel.from_xml_string(self.xml)
        if (self.model.nq, self.model.nv, self.model.nu) != (original.nq, original.nv, 8):
            raise ValueError('UNEXPECTED_MODEL_DIMENSIONS')
        # Names and qpos/dof ordering must remain unchanged; original attributes
        # are intentionally preserved only after this verification.
        for kind, names in ((self.mj.mjtObj.mjOBJ_JOINT, [f'joint{i}' for i in range(1, 8)] + ['finger_joint1', 'finger_joint2', 'peg_free']),
                            (self.mj.mjtObj.mjOBJ_BODY, ['hand', 'left_finger', 'right_finger', 'peg', 'hole']),
                            (self.mj.mjtObj.mjOBJ_SITE, ['factory_tcp'])):
            for name in names:
                if self.mj.mj_name2id(original, kind, name) != self.mj.mj_name2id(self.model, kind, name):
                    raise ValueError('MODEL_ORDER_CHANGED:' + name)
        self.finger_actuator_ids = np.array([self.model.actuator(ACTUATOR_NAME).id])
        self.tendon_id = self.model.tendon(TENDON_NAME).id
        self.data = self.mj.MjData(self.model)
        self.reset()

    def snapshot(self):
        state = super().snapshot()
        # The frozen base constructor calls reset/snapshot before replacement.
        if not hasattr(self, 'tendon_id'):
            return state
        state.update({
            'gripper_tendon_length_m': float(self.data.ten_length[self.tendon_id]),
            'gripper_tendon_velocity_m_s': float(self.data.ten_velocity[self.tendon_id]),
            'gripper_actuator_control_m': float(self.data.ctrl[self.finger_actuator_ids[0]]),
            'gripper_actuator_force_n': float(self.data.actuator_force[self.finger_actuator_ids[0]]),
            'finger_generalized_actuator_force_n': self.data.qfrc_actuator[self.finger_dof_ids].tolist(),
        })
        return state

    def physics_step(self, arm_torque=None, finger_target=0., record=True):
        if arm_torque is not None:
            self.apply_arm_torque(arm_torque)
        target = tendon_target(finger_target)
        self.data.ctrl[self.finger_actuator_ids[0]] = target
        self.last_finger_command = {'mode': 'native_sum_tendon_position',
                                    'target_opening_per_finger_m': float(finger_target),
                                    'target_tendon_length_m': target}
        warnings_before = np.array([x.number for x in self.data.warning])
        before_time = float(self.data.time)
        self.mj.mj_step(self.model, self.data)
        integrated = {
            'stage': 'mj_step_solver_actuation_before_wrapper_mj_forward',
            'state_time_s': before_time, 'step_end_time_s': float(self.data.time),
            'arm_actuator_force_nm': self.data.actuator_force[self.arm_actuator_ids].tolist(),
            'arm_qfrc_actuator_nm': self.data.qfrc_actuator[self.arm_dof_ids].tolist(),
            'gripper_actuator_force_n': float(self.data.actuator_force[self.finger_actuator_ids[0]]),
            'finger_generalized_actuator_force_n': self.data.qfrc_actuator[self.finger_dof_ids].tolist(),
        }
        self.mj.mj_forward(self.model, self.data)
        warning_delta = np.array([x.number for x in self.data.warning]) - warnings_before
        if np.any(warning_delta):
            raise RuntimeError(f'MUJOCO_WARNING_COUNTS_{warning_delta.tolist()}')
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('NONFINITE_SIMULATION_STATE')
        result = self.snapshot()
        result['integration_actuation'] = integrated
        result['snapshot_actuation_stage'] = 'poststep_mj_forward_recomputed_at_time_s'
        if record:
            self.log.append(result)
        return result
