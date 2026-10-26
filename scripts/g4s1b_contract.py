# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved. SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Adapted Sim2Sim CPU observation/action/controller contract; no policy actor.

The equations follow pinned Isaac Lab Factory v2.1.1, while the numerical
controller uses NumPy float64 for MuJoCo's double state. It is not bitwise PhysX
parity. No gains, damping regularization, gravity or Coriolis feedforward are
added. The environment must supply the mean finger-body-origin Jacobian and
the seven-arm-DOF principal mass matrix, not silently substitute a TCP Jacobian.
"""
from __future__ import annotations

import math
import numpy as np

PHYSICS_DT = 1.0 / 120.0
DECIMATION = 8
KP = np.array([100., 100., 100., 30., 30., 30.])
KD = 2.0 * np.sqrt(KP)
DEFAULT_KP = KP
DEFAULT_KD = KD
DEFAULT_Q = np.array([-1.3003, -.4015, 1.1791, -2.1493, .4001, 1.9425, .4754])
for _frozen_parameter in (KP, KD, DEFAULT_Q):
    _frozen_parameter.setflags(write=False)


def finite_array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite shape {shape}')
    return a.copy()


def unit_quat(value):
    q = finite_array(value, (4,), 'world wxyz quaternion')
    if abs(float(np.linalg.norm(q)) - 1.0) > 1e-6:
        raise ValueError('quaternion must be unit; no silent normalization or sign canonicalization')
    return q


def quat_mul(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def quat_conjugate(q):
    return np.asarray(q) * np.array([1., -1., -1., -1.])


def quat_from_euler(roll, pitch, yaw):
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    return np.array([cy*cr*cp+sy*sr*sp, cy*sr*cp-sy*cr*sp,
                     cy*cr*sp+sy*sr*cp, sy*cr*cp-cy*sr*sp])


def source_euler(q):
    """Isaac Sim extrinsic XYZ formulas, including [0,2pi) wrapping."""
    w, x, y, z = unit_quat(q)
    roll = math.atan2(2*(w*x+y*z), w*w-x*x-y*y+z*z)
    sinp = 2*(w*y-z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2*(w*z+x*y), w*w+x*x-y*y-z*z)
    return np.mod([roll, pitch, yaw], 2*math.pi)


def axis_angle(q):
    """Pinned axis_angle_from_quat, also defined at the source zero delta."""
    q = finite_array(q, (4,), 'difference quaternion')
    q *= 1.0 - 2.0 * (q[0] < 0.0)
    half = math.atan2(float(np.linalg.norm(q[1:])), float(q[0]))
    angle = 2*half
    denominator = math.sin(half)/angle if abs(angle) > 1e-6 else .5-angle*angle/48
    return q[1:] / denominator


def rotate(q, v):
    q = unit_quat(q)
    v = finite_array(v, (3,), 'local vector')
    return quat_mul(quat_mul(q, np.r_[0., v]), quat_conjugate(q))[1:]


def cartesian_targets(position, quaternion, action_cache, fixed_action_frame):
    """Recompute on every physics substep with current pose; do not integrate."""
    p = finite_array(position, (3,), 'TCP position')
    q = unit_quat(quaternion)
    a = finite_array(action_cache, (6,), 'EMA action cache')
    b = finite_array(fixed_action_frame, (3,), 'reset action frame')
    position_target = b + np.clip(p + .02*a[:3] - b, -.05, .05)
    rotvec = .097*a[3:]
    angle = float(np.linalg.norm(rotvec))
    dq = np.r_[math.cos(angle/2), rotvec/angle*math.sin(angle/2)] if angle > 1e-6 else np.array([1., 0., 0., 0.])
    pre_constraint = quat_mul(dq, q)
    yaw = source_euler(pre_constraint)[2]
    target_q = quat_from_euler(3.14159, 0.0, yaw)
    return {'position_target': position_target, 'quaternion_target': target_q,
            'quaternion_pre_constraint': pre_constraint, 'world_rotvec_increment': rotvec,
            'gripper_position_target': 0.0}


def torque_command(q7, qdot7, position, quaternion, linear_fd, angular_fd,
                   jacobian, mass_matrix, target_position, target_quaternion):
    """Return requested and source software-limited arm torque separately.

    Reject invalid/singular numerical states; do not substitute zero torque,
    pseudoinverses or damping. The extra condition gate rejects numerically
    unresolved inversions (>1/eps condition number), never changes the equation.
    Finger position PD belongs to the explicit MuJoCo actuator adaptation.
    """
    q = finite_array(q7, (7,), 'arm q')
    qdot = finite_array(qdot7, (7,), 'raw arm qdot')
    p = finite_array(position, (3,), 'TCP position')
    quat = unit_quat(quaternion)
    v = finite_array(linear_fd, (3,), 'world linear FD')
    w = finite_array(angular_fd, (3,), 'world angular FD')
    j = finite_array(jacobian, (6, 7), 'mean finger-origin Jacobian')
    m = finite_array(mass_matrix, (7, 7), 'arm mass matrix')
    target = finite_array(target_position, (3,), 'target position')
    tq = unit_quat(target_quaternion)
    if not np.allclose(m, m.T, atol=1e-11, rtol=1e-11):
        raise ValueError('arm mass matrix is not symmetric')
    try:
        np.linalg.cholesky(m)
        if np.linalg.cond(m) >= 1/np.finfo(float).eps:
            raise np.linalg.LinAlgError('numerically singular arm mass matrix')
        mi = np.linalg.inv(m)
        task_inverse = j @ mi @ j.T
        if np.linalg.matrix_rank(j) < 6 or np.linalg.cond(task_inverse) >= 1/np.finfo(float).eps:
            raise np.linalg.LinAlgError('singular or numerically unresolved task inertia')
        task_mass = np.linalg.inv(task_inverse)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f'controller inversion rejected: {exc}') from exc
    tq = tq if np.dot(tq, quat) >= 0 else -tq
    quat_inverse = quat_conjugate(quat) / np.dot(quat, quat)
    pose_error = np.r_[target-p, axis_angle(quat_mul(tq, quat_inverse))]
    wrench = KP*pose_error - KD*np.r_[v, w]
    task_torque = j.T @ wrench
    j_hash = task_mass @ j @ mi
    distance = (DEFAULT_Q-q+math.pi) % (2*math.pi)-math.pi
    u_null = m @ (10*distance-6.3246*qdot)
    projector = np.eye(7)-j.T @ j_hash
    null_torque = projector @ u_null
    requested = task_torque+null_torque
    if not np.isfinite(requested).all():
        raise ValueError('non-finite torque result')
    return {'requested': requested, 'software_limited': np.clip(requested, -100., 100.),
            'wrench': wrench, 'pose_error': pose_error, 'task_torque': task_torque,
            'null_torque': null_torque, 'null_projector': projector,
            'mass_condition': float(np.linalg.cond(m)),
            'task_inverse_condition': float(np.linalg.cond(task_inverse))}


class PhysicsContract:
    """Single environment physics-rate FD/reference/EMA state; no LSTM actor.

    Reset caches the noisy fixed *tip* reference (root + R*[0,0,.025]).
    Caller owns the separate frozen actor h/c reset, when later authorized.
    New timestamps must advance by exactly one physics dt; a repeated timestamp
    leaves FD untouched, as the source timestamp gate does within decimation.
    """
    def __init__(self):
        self.initialized = False
        self.policy_updates = 0

    def reset(self, position, quaternion_wxyz, fixed_root_pos, fixed_root_quat,
              observation_noise=None, timestamp=0.0):
        self.position = finite_array(position, (3,), 'reset TCP position')
        self.quaternion = unit_quat(quaternion_wxyz)
        root = finite_array(fixed_root_pos, (3,), 'reset fixed root')
        noise = np.zeros(3) if observation_noise is None else finite_array(observation_noise, (3,), 'episode observation noise')
        self.fixed_tip = root + rotate(fixed_root_quat, np.array([0., 0., .025]))
        self.observation_noise = noise.copy()
        self.fixed_action_frame = self.fixed_tip + noise
        self.action_cache = np.zeros(6)
        self.action_cache[:3] = (self.position-self.fixed_action_frame)/.05
        rel = quat_mul(quat_from_euler(-math.pi, 0., 0.), self.quaternion)
        yaw = source_euler(rel)[2]
        if yaw > math.pi/2:
            yaw -= 2*math.pi
        if yaw < -math.pi:
            yaw += 2*math.pi
        self.action_cache[5] = (yaw+math.pi)/(3*math.pi/2)*2-1
        if not math.isfinite(timestamp):
            raise ValueError('invalid reset timestamp')
        self.timestamp = float(timestamp)
        self.linear_fd = np.zeros(3)
        self.angular_fd = np.zeros(3)
        self.policy_updates = 0
        self.initialized = True
        return self.observation()

    def _ready(self):
        if not self.initialized:
            raise ValueError('reset the contract before use')

    def policy_update(self, raw6):
        self._ready()
        clipped = np.clip(finite_array(raw6, (6,), 'raw six-dimensional action'), -1., 1.)
        self.action_cache = .2*clipped + .8*self.action_cache
        self.policy_updates += 1
        return self.action_cache.copy()

    def update_physics(self, position, quaternion_wxyz, timestamp):
        self._ready()
        p = finite_array(position, (3,), 'TCP position')
        q = unit_quat(quaternion_wxyz)
        if not math.isfinite(timestamp):
            raise ValueError('invalid physics timestamp')
        delta_t = timestamp-self.timestamp
        if abs(delta_t) <= 1e-13:
            if not np.array_equal(p, self.position) or not np.array_equal(q, self.quaternion):
                raise ValueError('pose changed without a new physics timestamp')
            return self.linear_fd.copy(), self.angular_fd.copy()
        if not math.isclose(delta_t, PHYSICS_DT, abs_tol=1e-12, rel_tol=1e-10):
            raise ValueError('physics timestamp must advance by one frozen 1/120 s step')
        self.linear_fd = (p-self.position)/PHYSICS_DT
        dq = quat_mul(q, quat_conjugate(self.quaternion))
        dq *= np.sign(dq[0])  # exact pinned sign(0)=0 behavior is deliberate.
        self.angular_fd = axis_angle(dq)/PHYSICS_DT
        self.position, self.quaternion = p, q
        self.timestamp = float(timestamp)
        return self.linear_fd.copy(), self.angular_fd.copy()

    def targets(self, position=None, quaternion=None):
        self._ready()
        return cartesian_targets(self.position if position is None else position,
                                 self.quaternion if quaternion is None else quaternion,
                                 self.action_cache, self.fixed_action_frame)

    def observation(self):
        self._ready()
        # Source raw observation is FP32; subtraction/addition happen in FP32.
        relative = self.position.astype(np.float32) - (self.fixed_tip.astype(np.float32)+self.observation_noise.astype(np.float32))
        result = np.r_[relative, self.quaternion, self.linear_fd, self.angular_fd, self.action_cache].astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('FP32 observation overflow')
        return result


def normalize_observation(raw19, frozen_mean19, frozen_variance19):
    """Frozen RMS interface only. Statistics must be provided; no asset loading."""
    obs = finite_array(raw19, (19,), 'raw observation').astype(np.float32)
    mean = finite_array(frozen_mean19, (19,), 'frozen RMS mean').astype(np.float32)
    var = finite_array(frozen_variance19, (19,), 'frozen RMS variance').astype(np.float32)
    if np.any(var < 0) or not np.isfinite(obs).all() or not np.isfinite(mean).all() or not np.isfinite(var).all():
        raise ValueError('invalid FP32 frozen RMS inputs')
    return np.clip((obs-mean)/np.sqrt(var+np.float32(1e-5)), -5., 5.).astype(np.float32)


def original_success(held_root, fixed_root):
    """Unmodified PegInsert geometric predicate; not physical contact evidence."""
    held = finite_array(held_root, (3,), 'held root').astype(np.float32)
    fixed = finite_array(fixed_root, (3,), 'fixed root').astype(np.float32)
    if not np.isfinite(held).all() or not np.isfinite(fixed).all():
        raise ValueError('FP32 success state overflow')
    delta = held-fixed
    return bool(np.linalg.norm(delta[:2]) < np.float32(.0025)
                and delta[2] < np.float32(.025*.04))


class ReferenceRunner:
    """Explicit no-policy action bridge; caller supplies bounded reference inputs.

    This is not an actor runner. ``step_reference_action`` makes one EMA update,
    eight physical steps, and returns the pre-reset final state/observation.
    At source timeout (149 decisions), a subsequent explicit reset is required;
    this prevents silently evaluating a new episode's state as terminal state.
    The source auto-reset wrapper behavior is intentionally not provided here.
    """
    def __init__(self, environment):
        self.env = environment
        self.contract = PhysicsContract()
        self.initialized = False
        self.lstm_reset_pending = True  # Metadata only; no actor/h/c is loaded.

    def reset(self, seed=0, observation_noise=None, **physical_reset):
        self.env.reset(seed=seed, **physical_reset)
        p, q = self.env.tcp_state()
        noise = (self.env.rng.normal(0., .001, 3) if observation_noise is None
                 else finite_array(observation_noise, (3,), 'specified episode noise'))
        self.noise_mode = 'seeded_sigma_.001m_Gaussian_once_per_episode' if observation_noise is None else 'explicitly_specified_noise_diagnostic'
        obs = self.contract.reset(p, q, self.env.data.xpos[self.env.hole_body_id],
                                  self.env.data.xquat[self.env.hole_body_id], noise,
                                  timestamp=float(self.env.data.time))
        self.episode_steps = 0
        self.episode_start_time = float(self.env.data.time)
        self.reset_required = False
        self.ever_success = False
        self.first_success_step = None
        self.last_fault = None
        self.lstm_reset_pending = True
        self.initialized = True
        return {'observation': obs, 'ema_cache': self.contract.action_cache.copy(),
                'episode_noise_m': noise.copy(), 'noise_mode': self.noise_mode,
                'lstm_reset_pending': True, 'state': self.env.snapshot()}

    def step_reference_action(self, raw6):
        if not self.initialized or self.reset_required:
            raise ValueError('explicit episode reset is required before another reference step')
        try:
            return self._step_reference_action(raw6)
        except Exception as exc:
            # A failed step may already have updated EMA or some physical
            # substeps. Never let a caller unknowingly continue that episode.
            self.reset_required = True
            self.last_fault = {'type': type(exc).__name__, 'message': str(exc),
                               'simulation_time_s': float(self.env.data.time)}
            raise

    def _step_reference_action(self, raw6):
        self.contract.policy_update(raw6)
        substeps = []
        for _ in range(DECIMATION):
            p, q = self.env.tcp_state()
            self.contract.update_physics(p, q, float(self.env.data.time))
            target = self.contract.targets()
            control = torque_command(self.env.data.qpos[self.env.arm_qpos_ids],
                                     self.env.data.qvel[self.env.arm_dof_ids], p, q,
                                     self.contract.linear_fd, self.contract.angular_fd,
                                     self.env.controller_jacobian(), self.env.arm_mass_matrix(),
                                     target['position_target'], target['quaternion_target'])
            time_before = float(self.env.data.time)
            physical_state = self.env.physics_step(control['requested'], finger_target=0.0)
            substeps.append({'time_before_s': time_before,
                             'position_before_m': p.tolist(), 'quaternion_before_wxyz': q.tolist(),
                             'linear_fd_before_m_s': self.contract.linear_fd.tolist(),
                             'angular_fd_before_rad_s': self.contract.angular_fd.tolist(),
                             'target_position_m': target['position_target'].tolist(),
                             'target_quaternion_wxyz': target['quaternion_target'].tolist(),
                             'requested_arm_torque_nm': control['requested'].tolist(),
                             'software_limited_arm_torque_nm': control['software_limited'].tolist(),
                             'task_arm_torque_nm': control['task_torque'].tolist(),
                             'null_arm_torque_nm': control['null_torque'].tolist(),
                             'pd_wrench': control['wrench'].tolist(),
                             'mass_condition': control['mass_condition'],
                             'task_inverse_condition': control['task_inverse_condition'],
                             'physical_state_after': physical_state})
        # Source _get_dones refreshes the final new physics state before success,
        # reset and returned observation. Never evaluate post-reset peg pose.
        p, q = self.env.tcp_state()
        self.contract.update_physics(p, q, float(self.env.data.time))
        self.episode_steps += 1
        success = original_success(self.env.data.xpos[self.env.peg_body_id],
                                   self.env.data.xpos[self.env.hole_body_id])
        self.ever_success |= success
        if success and self.first_success_step is None:
            self.first_success_step = self.episode_steps
        timeout = self.episode_steps >= 149
        self.reset_required = timeout
        return {'protocol': 'NO_POLICY_EXPLICIT_REFERENCE_ACTIONS',
                'observation': self.contract.observation(), 'ema_cache': self.contract.action_cache.copy(),
                'episode_steps': self.episode_steps,
                'episode_simulation_time_s': float(self.env.data.time)-self.episode_start_time,
                'official_geometric_success': success, 'ever_geometric_success': self.ever_success,
                'first_geometric_success_step': self.first_success_step,
                'first_geometric_success_time_s': None if self.first_success_step is None else self.first_success_step*DECIMATION*PHYSICS_DT,
                'terminated': timeout, 'truncated': timeout, 'reset_required': timeout,
                'state_is_pre_reset': True, 'lstm_reset_pending': self.lstm_reset_pending,
                'physical_insertion_diagnostic': self.env.insertion_diagnostics(),
                'substeps': substeps}
