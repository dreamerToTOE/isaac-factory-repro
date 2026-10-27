#!/usr/bin/env python3
"""Independent arithmetic review of saved R2 traces; no simulator or policy.

Only NumPy and standard-library readers are used. In particular this module
does not import the experiment runner, frozen controller, MuJoCo, or PyTorch.
Reconstruction uses saved preintegration inputs and a separate linear-solve
formulation of the source controller. A successful arithmetic review is not a
stability certificate or an insertion-success claim.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 120.0
KP = np.array([100., 100., 100., 30., 30., 30.])
KD = 2.0 * np.sqrt(KP)
Q_REFERENCE = np.array([-1.3003, -.4015, 1.1791, -2.1493, .4001, 1.9425, .4754])
ARM_FORCE_LIMITS = np.array([87., 87., 87., 87., 12., 12., 12.])
BODY_NAMES = [f'link{i}' for i in range(8)] + ['hand', 'left_finger', 'right_finger']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def trace(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite(value, shape=None):
    result = np.asarray(value, dtype=float)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError('Invalid finite numerical input or shape')
    return result


def multiply(a, b):
    """Hamilton product, wxyz; independent component implementation."""
    a, b = finite(a, (4,)), finite(b, (4,))
    return np.r_[a[0]*b[0]-np.dot(a[1:], b[1:]),
                 a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])]


def inverse_quaternion(quaternion):
    q = finite(quaternion, (4,))
    norm2 = float(np.dot(q, q))
    if abs(norm2-1.) > 2.1e-6:
        raise ValueError('Non-unit quaternion; reviewer never silently normalizes')
    return q*np.array([1., -1., -1., -1.])/norm2


def rotation_vector(quaternion):
    """Source shortest-arc logarithm, including its small-angle expansion."""
    q = finite(quaternion, (4,)).copy()
    if q[0] < 0:
        q = -q
    half = math.atan2(float(np.linalg.norm(q[1:])), float(q[0]))
    angle = 2*half
    scale = math.sin(half)/angle if abs(angle) > 1e-6 else .5-angle*angle/48
    return q[1:]/scale


def rotation_matrix(quaternion):
    q = finite(quaternion, (4,))
    inverse_quaternion(q)
    w = q[0]
    v = q[1:]
    cross = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3)+2*w*cross+2*(cross@cross)


def relative_peg(state):
    hand_rotation = rotation_matrix(state['tcp_quaternion_wxyz'])
    peg_rotation = rotation_matrix(state['peg_quaternion_wxyz'])
    offset = finite(state['peg_root_position'], (3,))-finite(state['tcp_position_world'], (3,))
    return hand_rotation.T@offset, hand_rotation.T@peg_rotation


def slip(state, initial):
    now_position, now_rotation = relative_peg(state)
    initial_position, initial_rotation = relative_peg(initial)
    cosine = (np.trace(now_rotation.T@initial_rotation)-1)/2
    return float(np.linalg.norm(now_position-initial_position)), float(math.acos(np.clip(cosine, -1., 1.)))


def pad_forces(contacts):
    """Positive normal force only, exact frozen fingertip pad geometry scope."""
    result = {'left': 0., 'right': 0.}
    for contact in contacts:
        pair = {contact['geom1'], contact['geom2']}
        if 'peg_collision' not in pair:
            continue
        for side in result:
            if pair.intersection({f'{side}_finger_geom_{i}' for i in range(3, 8)}):
                result[side] += max(0., float(contact['normal_force_n']))
    return result


def penetrations(contacts):
    all_contact = max((max(0., -float(c['distance_m'])) for c in contacts), default=0.)
    assembly = [c for c in contacts if 'peg_collision' in (c['geom1'], c['geom2'])
                and any(g.startswith('hole_') for g in (c['geom1'], c['geom2']))]
    return all_contact, max((max(0., -float(c['distance_m'])) for c in assembly), default=0.)


def limit_violation(qpos, qpos_ids, limits):
    q = finite(qpos)[np.asarray(qpos_ids, dtype=int)]
    bounds = finite(limits, (len(q), 2))
    return max(0., float(np.max(bounds[:, 0]-q)), float(np.max(q-bounds[:, 1])))


def reconstruct_controller(inputs):
    """Source arithmetic via solves; no call to production torque_command.

    M enters only the dynamically consistent nullspace projection and drive.
    J.T(Kp e-Kd v) has no M dependency at a fixed state/reference/FD velocity.
    """
    q, velocity = finite(inputs['q7'], (7,)), finite(inputs['qdot7'], (7,))
    jacobian, mass = finite(inputs['jacobian'], (6, 7)), finite(inputs['mass_matrix'], (7, 7))
    np.linalg.cholesky(mass)
    quaternion = finite(inputs['quaternion'], (4,))
    target = finite(inputs['target_quaternion'], (4,))
    if np.dot(target, quaternion) < 0:
        target = -target
    error = np.r_[finite(inputs['target_position'], (3,))-finite(inputs['position'], (3,)),
                  rotation_vector(multiply(target, inverse_quaternion(quaternion)))]
    wrench = KP*error-KD*np.r_[finite(inputs['linear_fd'], (3,)), finite(inputs['angular_fd'], (3,))]
    task = jacobian.T@wrench
    mass_inverse_jt = np.linalg.solve(mass, jacobian.T)
    task_inverse = jacobian@mass_inverse_jt
    # M is symmetric, so solve(M,J.T).T = J M^-1.
    dynamic_inverse = np.linalg.solve(task_inverse, mass_inverse_jt.T)
    projector = np.eye(7)-jacobian.T@dynamic_inverse
    wrapped_error = (Q_REFERENCE-q+np.pi) % (2*np.pi)-np.pi
    drive = mass@(10.*wrapped_error-6.3246*velocity)
    null = projector@drive
    requested = task+null
    eigenvalues = np.linalg.eigvalsh(mass)
    return {'pose_error': error, 'wrench': wrench, 'task_torque': task, 'null_torque': null,
            'requested': requested, 'software_limited': np.clip(requested, -100., 100.),
            'null_projector': projector, 'mass_condition': float(eigenvalues[-1]/eigenvalues[0]),
            'task_inverse_condition': float(np.linalg.cond(task_inverse)),
            'null_dynamic_residual': float(np.max(np.abs(jacobian@np.linalg.solve(mass, projector)))),
            'effective_arm_inertias_fingers_locked_kg_m2': 1/np.diag(np.linalg.solve(mass, np.eye(7)))}


def finite_difference(previous, state):
    delta = float(state['time_s'])-float(previous['time_s'])
    if not math.isclose(delta, DT, abs_tol=2e-12, rel_tol=1e-10):
        raise ValueError('Nonuniform physics-step timestamp')
    linear = (finite(state['tcp_position_world'])-finite(previous['tcp_position_world']))/DT
    # Source uses conjugation, not general inverse, in its FD timestamp path.
    old = finite(previous['tcp_quaternion_wxyz'], (4,))*[1., -1., -1., -1.]
    difference = multiply(state['tcp_quaternion_wxyz'], old)
    difference *= np.sign(difference[0])
    angular = rotation_vector(difference)/DT
    acceleration = (finite(state['qvel'])-finite(previous['qvel']))/DT
    return linear, angular, acceleration


def maximum_error(actual, expected):
    a, b = finite(actual), finite(expected)
    if a.shape != b.shape:
        raise ValueError(f'Compared shapes differ: {a.shape} versus {b.shape}')
    return float(np.max(np.abs(a-b))) if a.size else 0.


def body_mass_increment(body_angular_jacobians, offset=.01):
    """Independent kinetic-energy pullback; no simulator mass-matrix routine."""
    if set(body_angular_jacobians) != set(BODY_NAMES):
        raise ValueError('Mapped body angular Jacobian coverage differs from 11-body scope')
    result = np.zeros((7, 7))
    for jacobian in body_angular_jacobians.values():
        j = finite(jacobian, (3, 7))
        result += offset*(j.T@j)
    return result


def pad_penetration(contacts):
    scoped = [contact for contact in contacts if 'peg_collision' in (contact['geom1'], contact['geom2'])
              and any(geometry in {f'{side}_finger_geom_{i}' for side in ('left', 'right') for i in range(3, 8)}
                      for geometry in (contact['geom1'], contact['geom2']))]
    return max((max(0., -float(contact['distance_m'])) for contact in scoped), default=0.)


def inspect_inertias(baseline, adapted):
    """Compare body tensors in body axes, allowing quaternion-basis equivalence."""
    if baseline['body_names'] != adapted['body_names']:
        raise ValueError('Body ordering changed across inertia models')
    rows = []
    for index, name in enumerate(baseline['body_names']):
        if name not in BODY_NAMES:
            continue
        before_values = finite(baseline['body_inertia_kg_m2'][index], (3,))
        after_values = finite(adapted['body_inertia_kg_m2'][index], (3,))
        r0 = rotation_matrix(baseline['body_iquat_wxyz'][index])
        r1 = rotation_matrix(adapted['body_iquat_wxyz'][index])
        tensor0, tensor1 = r0@np.diag(before_values)@r0.T, r1@np.diag(after_values)@r1.T
        error = float(np.max(np.abs(tensor1-tensor0-.01*np.eye(3))))
        rows.append({'body': name, 'principal_inertia_before_kg_m2': before_values.tolist(),
                     'principal_inertia_after_kg_m2': after_values.tolist(),
                     'positive_definite_before': bool(min(before_values) > 0),
                     'positive_definite_after': bool(min(after_values) > 0),
                     'isotropic_offset_tensor_error_kg_m2': error})
    if len(rows) != 11:
        raise ValueError('Missing mapped robot body')
    fixed = ('body_mass_kg', 'body_ipos_m', 'dof_armature', 'force_limits', 'arm_dof_ids',
             'arm_qpos_ids', 'finger_dof_ids', 'finger_qpos_ids', 'arm_limits', 'finger_limits')
    changes = {key: maximum_error(baseline[key], adapted[key]) for key in fixed}
    if any(changes.values()) or max(row['isotropic_offset_tensor_error_kg_m2'] for row in rows) > 2e-13:
        raise ValueError('Inertia intervention changed an unrelated mapped parameter')
    if not all(row['positive_definite_before'] and row['positive_definite_after'] for row in rows):
        raise ValueError('Non-positive robot inertia')
    return {'body_count': 11, 'bodies': rows, 'unchanged_parameter_max_errors': changes,
            'original_PhysX_runtime_mass_matrix': 'UNKNOWN', 'original_PhysX_applied_actuator_forces': 'UNKNOWN',
            'unmapped_original_bodies': baseline.get('unmapped_source_bodies', ['force_sensor', 'fingertip_centered'])}


def independently_reconstruct_rows(identity, rows):
    """Raw-trace review. Caller binds the saved runner schema to these fields.

    Required identity: initial_snapshot, model, variant, thresholds. Required row:
    controller_input, controller, matrices, state, step, episode_time_s,
    pad_force_n. This contract deliberately has no summary-only fallback.
    """
    model = identity['model']
    initial = identity['initial_snapshot']
    variant = identity['variant']
    if isinstance(variant, dict):
        variant = variant['id']
    p_selection, c_selection = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1), 'D': (1, 1)}[variant]
    arm, fingers = model['arm_dof_ids'], model['finger_dof_ids']
    arm_q, finger_q = model['arm_qpos_ids'], model['finger_qpos_ids']
    force_ranges = finite(model['force_limits'])[model['arm_actuator_ids']]
    if maximum_error(force_ranges, np.c_[-ARM_FORCE_LIMITS, ARM_FORCE_LIMITS]) != 0:
        raise ValueError('Original arm force limits changed')
    errors = {}
    def compare(name, actual, expected):
        errors[name] = max(errors.get(name, 0.), maximum_error(actual, expected))
    previous, previous_pre = initial, None
    metrics = []
    for sequence, row in enumerate(rows, 1):
        if row['step'] != sequence:
            raise ValueError('Raw physics steps are not contiguous from one')
        state, inputs, control, matrices = row['state'], row['controller_input'], row['controller'], row['matrices']
        finite(state['qpos']); finite(state['qvel'])
        compare('physics_clock_s', row['episode_time_s'], sequence*DT)
        compare('pre_q7_rad', inputs['q7'], finite(previous['qpos'])[arm_q])
        compare('pre_qdot7_rad_s', inputs['qdot7'], finite(previous['qvel'])[arm])
        compare('pre_tcp_position_m', inputs['position'], previous['tcp_position_world'])
        compare('pre_tcp_quaternion', inputs['quaternion'], previous['tcp_quaternion_wxyz'])
        compare('same_qpos', matrices['same_qpos'], previous['qpos'])
        m0, m1 = finite(matrices['same_q_M0'], (7, 7)), finite(matrices['same_q_M1'], (7, 7))
        physical, controller_mass = finite(matrices['physical_M'], (7, 7)), finite(matrices['controller_M'], (7, 7))
        for mass in (m0, m1, physical, controller_mass):
            np.linalg.cholesky(mass)
            compare('mass_symmetry', mass, mass.T)
        compare('physical_mass_selection', physical, (m0, m1)[p_selection])
        compare('controller_mass_selection', controller_mass, (m0, m1)[c_selection])
        compare('injected_controller_mass', inputs['mass_matrix'], controller_mass)
        angular = matrices['body_angular_jacobians']
        if isinstance(angular, list):
            if len(angular) != 11:
                raise ValueError('Body Jacobian list does not cover all mapped bodies')
            angular = dict(zip(BODY_NAMES, angular))
        compare('body_inertia_pullback', m1-m0, body_mass_increment(angular))
        if sequence == 1:
            compare('initial_fd_linear_m_s', inputs['linear_fd'], np.zeros(3))
            compare('initial_fd_angular_rad_s', inputs['angular_fd'], np.zeros(3))
        else:
            linear, angular_fd, _ = finite_difference(previous_pre, previous)
            compare('fd_linear_m_s', inputs['linear_fd'], linear)
            compare('fd_angular_rad_s', inputs['angular_fd'], angular_fd)
        independent = reconstruct_controller(inputs)
        for key in ('pose_error', 'wrench', 'task_torque', 'null_torque', 'requested', 'software_limited', 'null_projector'):
            compare('controller_'+key, control[key], independent[key])
        compare('software_limited_snapshot_nm', state['software_limited_arm_torque_nm'], independent['software_limited'])
        actual = finite(state['integration_actuation']['arm_actuator_force_nm'], (7,))
        expected_force = np.clip(independent['software_limited'], force_ranges[:, 0], force_ranges[:, 1])
        compare('actuator_force_clipping_nm', actual, expected_force)
        compare('generalized_unit_gear_actuator_force_nm', state['integration_actuation']['arm_qfrc_actuator_nm'], actual)
        # Preserved S1B finger adaptation depends on the PHYSICAL full mass
        # matrix, including all arm/finger/free-peg DOFs, independently of C.
        full_mass = finite(matrices['physical_full_M'])
        np.linalg.cholesky(full_mass)
        finger_inverse = np.linalg.solve(full_mass, np.eye(len(full_mass)))[np.ix_(fingers, fingers)]
        finger_position, finger_velocity = finite(previous['qpos'])[finger_q], finite(previous['qvel'])[fingers]
        raw_finger_pd = -7500.*finger_position-173.*finger_velocity
        predicted_force = np.linalg.solve(np.eye(2)+DT*(173.+DT*7500.)*finger_inverse,
                                          -7500.*finger_position-(173.+DT*7500.)*finger_velocity)
        finger_force = np.clip(predicted_force, -40., 40.)
        if state['finger_command']['mode'] != 'step-implicit-pd' or state['finger_command']['target_opening_per_finger_m'] != 0:
            raise ValueError('Original implicit finger actuation semantics changed')
        compare('finger_raw_pd_n', state['finger_command']['raw_pd_force_n'], raw_finger_pd)
        compare('finger_commanded_n', state['finger_command']['commanded_force_n'], finger_force)
        compare('finger_integrated_force_n', state['integration_actuation']['finger_actuator_force_n'], finger_force)
        position_slip, orientation_slip = slip(state, initial)
        forces = pad_forces(state['contacts'])
        all_penetration, assembly_penetration = penetrations(state['contacts'])
        pad_depth = pad_penetration(state['contacts'])
        arm_limit = limit_violation(state['qpos'], arm_q, model['arm_limits'])
        finger_limit = limit_violation(state['qpos'], finger_q, model['finger_limits'])
        compare('slip_translation_m', state['s2a']['peg_hand_slip_m'], position_slip)
        compare('slip_rotation_rad', state['s2a']['peg_hand_slip_rad'], orientation_slip)
        compare('pad_force_n', [row['pad_force_n'][key] for key in ('left', 'right')], [forces[key] for key in ('left', 'right')])
        compare('arm_limit_rad', state['s2a']['arm_joint_limit_violation_rad'], arm_limit)
        compare('finger_limit_m', state['s2a']['finger_joint_limit_violation_m'], finger_limit)
        compare('all_penetration_m', state['s2a']['all_contact_penetration_m'], all_penetration)
        compare('assembly_penetration_m', state['s2a']['peg_hole_penetration_m'], assembly_penetration)
        _, _, interval_acceleration = finite_difference(previous, state)
        if 'qvel_interval_acceleration' not in row:
            raise ValueError('Missing saved interval acceleration')
        compare('interval_acceleration', row['qvel_interval_acceleration'], interval_acceleration)
        for key in ('qpos', 'qvel', 'tcp_position_world', 'tcp_quaternion_wxyz'):
            compare('pre_state_'+key, row['pre_state'][key], previous[key])
        physical_effective = 1/np.diag(np.linalg.solve(physical, np.eye(7)))
        metrics.append({'step': sequence, 'time_s': sequence*DT,
            'max_arm_velocity_rad_s': float(max(abs(finite(state['qvel'])[arm]))),
            'joint7_velocity_rad_s': float(state['qvel'][arm[6]]),
            'joint7_interval_acceleration_rad_s2': float(interval_acceleration[arm[6]]),
            'max_arm_interval_acceleration_rad_s2': float(max(abs(interval_acceleration[arm]))),
            'translation_slip_m': position_slip, 'rotation_slip_rad': orientation_slip,
            'left_pad_force_n': forces['left'], 'right_pad_force_n': forces['right'],
            'bilateral_positive_contact': min(forces.values()) > identity['thresholds']['positive_pad_force_n'],
            'all_penetration_m': all_penetration, 'pad_penetration_m': pad_depth, 'assembly_penetration_m': assembly_penetration,
            'arm_limit_violation_rad': arm_limit, 'finger_limit_violation_m': finger_limit,
            'requested_max_abs_torque_nm': float(max(abs(independent['requested']))),
            'max_abs_task_torque_nm': float(max(abs(independent['task_torque']))),
            'max_abs_null_torque_nm': float(max(abs(independent['null_torque']))),
            'max_abs_actual_motor_force_nm': float(max(abs(actual))),
            'max_abs_finger_command_n': float(max(abs(finger_force))),
            'finger_command_force_n': finger_force.tolist(),
            'joint7_task_torque_nm': float(independent['task_torque'][6]),
            'joint7_null_torque_nm': float(independent['null_torque'][6]),
            'joint7_requested_torque_nm': float(independent['requested'][6]),
            'joint7_actual_motor_force_nm': float(actual[6]),
            'cartesian_position_error_norm_m': float(np.linalg.norm(independent['pose_error'][:3])),
            'cartesian_orientation_error_norm_rad': float(np.linalg.norm(independent['pose_error'][3:])),
            'fd_linear_speed_m_s': float(np.linalg.norm(inputs['linear_fd'])),
            'fd_angular_speed_rad_s': float(np.linalg.norm(inputs['angular_fd'])),
            'actuator_saturation_fraction': float(np.mean(abs(independent['software_limited']) > ARM_FORCE_LIMITS+1e-12)),
            'actual_motor_force_at_limit_fraction': float(np.mean(abs(actual) >= ARM_FORCE_LIMITS-1e-9)),
            'source_software_clipping_fraction': float(np.mean(abs(independent['requested']) > 100.+1e-12)),
            'physical_mass_condition': float(np.linalg.cond(physical)), 'controller_mass_condition': float(np.linalg.cond(controller_mass)),
            'physical_joint7_effective_inertia_kg_m2': float(physical_effective[6]),
            'controller_joint7_effective_inertia_kg_m2': float(independent['effective_arm_inertias_fingers_locked_kg_m2'][6]),
            'null_dynamic_residual': independent['null_dynamic_residual']})
        previous_pre, previous = previous, state
    return metrics, errors


def aggregate(metrics):
    if not metrics:
        raise ValueError('No physics results for admitted trajectory')
    fields = ['max_arm_velocity_rad_s', 'max_arm_interval_acceleration_rad_s2', 'translation_slip_m',
              'rotation_slip_rad', 'all_penetration_m', 'pad_penetration_m', 'assembly_penetration_m',
              'arm_limit_violation_rad', 'finger_limit_violation_m', 'requested_max_abs_torque_nm',
              'max_abs_task_torque_nm', 'max_abs_null_torque_nm', 'physical_mass_condition',
              'controller_mass_condition', 'null_dynamic_residual', 'max_abs_actual_motor_force_nm',
              'max_abs_finger_command_n', 'cartesian_position_error_norm_m',
              'cartesian_orientation_error_norm_rad', 'fd_linear_speed_m_s', 'fd_angular_speed_rad_s']
    result = {'steps': len(metrics), 'simulation_duration_s': metrics[-1]['time_s']}
    result.update({'max_'+key if not key.startswith('max_') else key: max(row[key] for row in metrics) for key in fields})
    result['first_contact_loss_step'] = next((row['step'] for row in metrics if not row['bilateral_positive_contact']), None)
    result['bilateral_positive_contact_fraction'] = sum(row['bilateral_positive_contact'] for row in metrics)/len(metrics)
    result['actuator_saturation_fraction'] = float(np.mean([row['actuator_saturation_fraction'] for row in metrics]))
    result['actual_motor_force_at_limit_fraction'] = float(np.mean([row['actual_motor_force_at_limit_fraction'] for row in metrics]))
    result['source_software_clipping_fraction'] = float(np.mean([row['source_software_clipping_fraction'] for row in metrics]))
    for side in ('left', 'right'):
        values = [row[side+'_pad_force_n'] for row in metrics]
        for name, operation in [('min', min), ('max', max), ('mean', np.mean)]:
            result[side+'_pad_force_n_'+name] = float(operation(values))
    result['first_slip_over_1mm_step'] = next((row['step'] for row in metrics if row['translation_slip_m'] > .001), None)
    result['initial_physical_joint7_effective_inertia_kg_m2'] = metrics[0]['physical_joint7_effective_inertia_kg_m2']
    result['initial_controller_joint7_effective_inertia_kg_m2'] = metrics[0]['controller_joint7_effective_inertia_kg_m2']
    return result


def stability_gates(metrics, thresholds, horizon):
    x = aggregate(metrics)
    dynamic = (x['steps'] == horizon and x['max_arm_velocity_rad_s'] <= thresholds['accept_arm_velocity_rad_s']
               and x['max_arm_limit_violation_rad'] <= thresholds['accept_arm_limit_rad']
               and x['max_finger_limit_violation_m'] <= thresholds['accept_finger_limit_m'])
    grasp = (x['steps'] == horizon and x['first_contact_loss_step'] is None
             and x['max_translation_slip_m'] < thresholds['accept_slip_m']
             and x['max_rotation_slip_rad'] <= thresholds['accept_rotation_rad'])
    contact = (x['max_pad_penetration_m'] <= thresholds['accept_pad_penetration_m']
               and x['max_assembly_penetration_m'] <= thresholds['accept_assembly_penetration_m'])
    return {'robot_dynamical_stability': dynamic, 'sustained_grasp_stability': grasp,
            'contact_physical_quality': contact, 'all_stability_gates': dynamic and grasp and contact,
            'scope': 'Measured physics-step states over the registered no-policy horizon only; no continuous-time or task-success certificate.'}


def euler_quaternion(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll/2), math.sin(roll/2), math.cos(pitch/2), math.sin(pitch/2), math.cos(yaw/2), math.sin(yaw/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def wrapped_euler_yaw(q):
    w, x, y, z = finite(q, (4,))
    return math.atan2(2*(w*z+x*y), w*w+x*x-y*y-z*z) % (2*math.pi)


def inspect_reference_schedule(identity, initial_record, registry, rows):
    """Independently rebuild the two archived EMA updates and frozen-pose hold."""
    state = identity['initial_snapshot']
    hole_q = initial_record['state'].get('s2a', {}).get('hole_quaternion_wxyz', [1., 0., 0., 0.])
    # Registered S1B hole is identity-oriented. Require it rather than silently
    # assume that missing general orientation metadata has been reconstructed.
    if maximum_error(hole_q, [1., 0., 0., 0.]) > 1e-12:
        raise ValueError('Unexpected initial hole quaternion')
    frame = finite(state['hole_root_position'])+rotation_matrix(hole_q)@[0., 0., .025]+finite(initial_record['episode_noise_m'])
    cache = np.zeros(6)
    cache[:3] = (finite(state['tcp_position_world'])-frame)/.05
    relative_quaternion = multiply(euler_quaternion(-math.pi, 0., 0.), state['tcp_quaternion_wxyz'])
    yaw = wrapped_euler_yaw(relative_quaternion)
    if yaw > math.pi/2:
        yaw -= 2*math.pi
    if yaw < -math.pi:
        yaw += 2*math.pi
    cache[5] = (yaw+math.pi)/(3*math.pi/2)*2-1
    errors = {'initial_ema': maximum_error(cache, identity['initial_ema']), 'target_position_m': 0., 'target_quaternion': 0.}
    actions = registry['input_schedule']['archived_clipped_actions']
    if len(actions) != 2:
        raise ValueError('Unexpected diagnostic action count')
    for row in rows:
        step, inputs = row['step'], row['controller_input']
        if step in (1, 9):
            cache = .8*cache+.2*np.clip(finite(actions[(step-1)//8], (6,)), -1., 1.)
        if step <= 16:
            position = frame+np.clip(finite(inputs['position'])+.02*cache[:3]-frame, -.05, .05)
            rotation = .097*cache[3:]
            angle = float(np.linalg.norm(rotation))
            increment = np.r_[math.cos(angle/2), rotation*math.sin(angle/2)/angle] if angle > 1e-6 else np.array([1., 0., 0., 0.])
            yaw = wrapped_euler_yaw(multiply(increment, inputs['quaternion']))
            quaternion = euler_quaternion(3.14159, 0., yaw)
        else:
            position, quaternion = finite(state['tcp_position_world']), finite(state['tcp_quaternion_wxyz'])
        errors['target_position_m'] = max(errors['target_position_m'], maximum_error(position, inputs['target_position']))
        errors['target_quaternion'] = max(errors['target_quaternion'], maximum_error(quaternion, inputs['target_quaternion']))
    return errors


def inspect_admission(directory, item, model):
    """Recheck saved physical closing states; archived I1 has historical basis."""
    admission = read(directory/'admission.json')
    if admission['admission'] != 'PASS':
        return {'initial_id': item['id'], 'status': 'RECORDED_ADMISSION_FAILURE',
                'errors': admission.get('admission_errors', []), 'not_resampled': True}
    initial_record = read(directory/'initial_state.json')
    state = initial_record['state']
    force = pad_forces(state['contacts'])
    arm_limit = limit_violation(state['qpos'], model['arm_qpos_ids'], model['arm_limits'])
    finger_limit = limit_violation(state['qpos'], model['finger_qpos_ids'], model['finger_limits'])
    hole_rotation = rotation_matrix([1., 0., 0., 0.])
    axis = hole_rotation.T@rotation_matrix(state['peg_quaternion_wxyz'])[:, 2]
    bottom = hole_rotation.T@(finite(state['peg_root_position'])-finite(state['hole_root_position']))
    top = bottom+.05*axis
    cylinder_z_min = float(min(bottom[2], top[2])-.003993*math.sqrt(max(0., 1-axis[2]**2)))
    present = min(force.values()) > 1e-6 and arm_limit <= 1e-6 and finger_limit <= 1e-6 and cylinder_z_min >= .025
    result = {'initial_id': item['id'], 'present_forward_bilateral_force_n': force,
              'present_arm_limit_rad': arm_limit, 'present_finger_limit_m': finger_limit,
              'entire_cylinder_z_min_in_hole_frame_m': cylinder_z_min,
              'present_state_passed': bool(present), 'initial_state_sha256': digest(directory/'initial_state.json')}
    closure_path = directory/'closure.jsonl'
    if item['kind'] == 'archived_admitted':
        result.update({'status': 'PRESENT_STATE_RECHECKED_HISTORICAL_ADMISSION_PRESERVED',
            'new_preparation_steps': 0, 'historical_ten_step_grasp_window': 'Not reconstructed; original archived admission SHA remains the evidence basis.',
            'historical_admission_sha256': admission['historical_admission_sha256']})
        return result
    closing = trace(closure_path)
    if len(closing) != 30 or [row['closure_step'] for row in closing] != list(range(1, 31)):
        raise ValueError('Physical preparation is not one registered 30-step attempt')
    last_ten = closing[-10:]
    relative = [relative_peg(row['state']) for row in last_ten]
    translation = max(float(np.linalg.norm(pair[0]-relative[-1][0])) for pair in relative)
    rotation = max(float(math.acos(np.clip((np.trace(pair[1].T@relative[-1][1])-1)/2, -1., 1.))) for pair in relative)
    bilateral = all(min(pad_forces(row['state']['contacts']).values()) > 1e-6 for row in last_ten)
    interference = False
    for row in last_ten:
        for contact in row['state']['contacts']:
            pair = {contact['geom1'], contact['geom2']}
            pads = {f'{side}_finger_geom_{i}' for side in ('left', 'right') for i in range(3, 8)}
            if 'peg_collision' in pair and not pair.intersection(pads):
                interference = True
            if 'peg_collision' not in pair and contact['distance_m'] < -1e-6:
                interference = True
    last = closing[-1]['state']
    final_error = max(maximum_error(last[key], state[key]) for key in ('qpos', 'qvel'))
    stable = bilateral and translation <= .001 and rotation <= .05 and not interference and final_error == 0
    result.update({'status': 'INDEPENDENT_30_STEP_PREPARATION_REVIEW', 'new_preparation_steps': 30,
        'closing_trace_sha256': digest(closure_path), 'last10_bilateral_contact': bool(bilateral),
        'last10_translation_drift_m': translation, 'last10_rotation_drift_rad': rotation,
        'last10_nonpad_or_non_grasp_interference': interference, 'saved_final_state_error': final_error,
        'drift_vs_runner_abs_error_m': abs(translation-admission['stability_m']),
        'rotation_drift_vs_runner_abs_error_rad': abs(rotation-admission['angular_stability_rad']),
        'independently_admitted': bool(present and stable)})
    return result


def paired_factorial(results, records):
    by_initial = {}
    for initial_id in sorted({row['initial_id'] for row in results}):
        group = {row['variant']: row for row in results if row['initial_id'] == initial_id}
        if set(group) != set('ABCD'):
            by_initial[initial_id] = {'status': 'INCOMPLETE_FACTORIAL', 'available_variants': sorted(group)}
            continue
        initial_states = [group[variant]['initial_snapshot'] for variant in 'ABCD']
        identical = all(maximum_error(initial_states[0][key], state[key]) == 0 for state in initial_states[1:] for key in ('qpos', 'qvel'))
        if not identical:
            raise ValueError('Matched initial physical states differ across factorial variants')
        length = min(group[variant]['metrics']['steps'] for variant in 'ABCD')
        prefix = {variant: aggregate(records[initial_id+'_'+variant][:length]) for variant in 'ABCD'}
        pair_effects = []
        for changed, baseline, label in [('B', 'A', 'P1 at C0'), ('D', 'C', 'P1 at C1'), ('C', 'A', 'C1 at P0'), ('D', 'B', 'C1 at P1')]:
            before, after = prefix[baseline], prefix[changed]
            item = {'contrast': label, 'baseline': baseline, 'intervention': changed, 'common_prefix_steps': length,
                    'same_initial_state': True, 'comparison_scope': 'Deterministic matched-state recorded common prefix; not independent repeated trials.'}
            for key in ('max_translation_slip_m', 'max_arm_velocity_rad_s', 'max_rotation_slip_rad', 'max_all_penetration_m'):
                item[key+'_baseline'] = before[key]
                item[key+'_intervention'] = after[key]
                item[key+'_relative_reduction'] = None if before[key] == 0 else 1-after[key]/before[key]
            pair_effects.append(item)
        interactions = {key: prefix['D'][key]-prefix['B'][key]-prefix['C'][key]+prefix['A'][key]
                        for key in ('max_translation_slip_m', 'max_arm_velocity_rad_s', 'max_rotation_slip_rad')}
        by_initial[initial_id] = {'status': 'COMPLETE_FACTORIAL', 'identical_initial_qpos_qvel': identical,
            'common_prefix_steps': length, 'common_prefix_metrics': prefix, 'paired_effects': pair_effects,
            'descriptive_factorial_interaction_D_minus_B_minus_C_plus_A': interactions,
            'interaction_limitation': 'Maxima are nonlinear outcomes; this descriptive contrast alone does not establish a unique dynamical mechanism.',
            'first_step_matched_state': {variant: {key: records[initial_id+'_'+variant][0][key] for key in (
                'joint7_task_torque_nm', 'joint7_null_torque_nm', 'joint7_requested_torque_nm',
                'joint7_actual_motor_force_nm', 'joint7_velocity_rad_s', 'joint7_interval_acceleration_rad_s2',
                'physical_joint7_effective_inertia_kg_m2', 'controller_joint7_effective_inertia_kg_m2',
                'finger_command_force_n')} for variant in 'ABCD'},
            'full_registered_horizon_outcomes': {variant: group[variant]['gates'] for variant in 'ABCD'}}
    return by_initial


def late_grasp_witnesses(run_id, rows, metrics, thresholds):
    if len(rows) <= 16:
        return None
    definitions = [('first_pad_contact_loss', lambda row: not row['bilateral_positive_contact']),
                   ('first_pad_quality_depth_exceedance', lambda row: row['pad_penetration_m'] > thresholds['accept_pad_penetration_m']),
                   ('first_rotation_acceptance_exceedance', lambda row: row['rotation_slip_rad'] > thresholds['accept_rotation_rad']),
                   ('first_severe_depth_exceedance', lambda row: row['all_penetration_m'] > thresholds['stop_penetration_m'])]
    events = {}
    selected = set()
    for label, predicate in definitions:
        match = next((row for row in metrics[16:] if predicate(row)), None)
        events[label] = None if match is None else {'step': match['step'], 'time_s': match['time_s']}
        if match is not None:
            selected.update(range(max(1, match['step']-1), min(len(rows), match['step']+1)+1))
    if not selected:
        return {'run_id': run_id, 'late_failure_observed': False, 'recorded_steps': len(rows), 'events': events}
    witnesses = []
    for step in sorted(selected):
        state, metric = rows[step-1]['state'], metrics[step-1]
        deepest = min(state['contacts'], key=lambda contact: contact['distance_m'], default=None)
        witnesses.append({'step': step, 'time_s': step*DT, 'derived_metrics': metric,
            'qpos': state['qpos'], 'qvel': state['qvel'], 'deepest_contact': deepest,
            'finger_command': state['finger_command'], 'integration_actuation': state['integration_actuation'],
            'stop_reasons': rows[step-1]['stop_reasons']})
    return {'run_id': run_id, 'late_failure_observed': True, 'recorded_steps': len(rows),
            'events': events, 'witnesses': witnesses,
            'interpretation': 'Saved-step contact loss and subsequent penetration are observed; full original contact impulses and a unique late-failure mechanism are not established. High normal contact force is distinct from bounded actuator force.'}


def causal_review(results, factorial, late_reviews):
    by_variant = {variant: [row for row in results if row['variant'] == variant] for variant in 'ABCD'}
    sustained = {variant: [row['initial_id'] for row in group if row['gates']['all_stability_gates']]
                 for variant, group in by_variant.items()}
    complete_factorial = len(factorial) == 3 and all(group['status'] == 'COMPLETE_FACTORIAL' for group in factorial.values())
    plant_prefix_improvement = complete_factorial and all(
        contrast['max_arm_velocity_rad_s_intervention'] < contrast['max_arm_velocity_rad_s_baseline']
        for group in factorial.values() for contrast in group['paired_effects'] if contrast['contrast'].startswith('P1'))
    c_alone_failed = len(by_variant['C']) == 3 and all(row['termination'] == 'DIAGNOSTIC_EARLY_STOP' for row in by_variant['C'])
    d_failed = [row['run_id'] for row in by_variant['D'] if not row['gates']['all_stability_gates']]
    supported_candidate = len(sustained['D']) == 3
    candidate = 'CANDIDATE_SUPPORTED' if supported_candidate else ('CANDIDATE_PARTIAL' if sustained['D'] else 'CANDIDATE_BLOCKED')
    return {'scope': 'Registered deterministic no-policy factorial cases only; no unique global root-cause claim.',
        'full_horizon_all_gate_pass_initials_by_variant': sustained,
        'registered_initial_denominator': 3,
        'classifications': [
            {'claim': 'Physical P1 intervention suppresses the initial arm oscillation with either C0 or C1 in the registered cases.',
             'status': 'VERIFIED_LOCAL_EFFECT' if plant_prefix_improvement else 'INCONCLUSIVE', 'evidence': 'Matched-state common-prefix P1/C0−P0/C0 and P1/C1−P0/C1 contrasts at each initial; physical/controller M selections and source arithmetic independently verified.',
             'limitation': 'P1 includes the preserved physical-full-M-dependent finger-force response; no isolated contact/finger/inertia submechanism or global stability theorem.'},
            {'claim': 'Changing only the arm-controller mass matrix to C1 is sufficient to eliminate the observed instability.',
             'status': 'CONTRADICTED' if c_alone_failed else 'INCONCLUSIVE', 'evidence': 'P0+C1 raw guard-stop outcomes retained independently for each admitted initial condition.'},
            {'claim': 'Both P1 and C1 are required for initial oscillation suppression.', 'status': 'CONTRADICTED' if plant_prefix_improvement and sustained['B'] else 'INCONCLUSIVE',
             'evidence': 'P1+C0 common-prefix initial oscillation and complete-horizon gate outcomes are retained separately.'},
            {'claim': 'Low target body inertia interacting with task-space PD/FD is an important initial instability mechanism.',
             'status': 'SUPPORTED_MECHANISM' if plant_prefix_improvement and c_alone_failed else 'INCONCLUSIVE', 'evidence': 'At fixed initial state the task torque has no controller-M dependence, while physical P1 changes effective joint inertia; inspect C1-only and P1 paired outcomes.',
             'limitation': 'Nonlinear contacts, implicit finger prediction and discrete feedback remain coupled; original PhysX runtime M/forces UNKNOWN.'},
            {'claim': 'The versioned P1+C1 candidate sustains acceptable grasp over the entire registered horizon at all admitted poses.',
             'status': 'CONTRADICTED' if d_failed else ('VERIFIED_LOCAL_EFFECT' if supported_candidate else 'INCONCLUSIVE'),
             'evidence': {'failed_registered_D_trajectories': d_failed, 'complete_horizon_D_passes': sustained['D']}},
            {'claim': 'The controller-M intervention has no meaningful effect on any later dynamics or failure timing.',
             'status': 'INCONCLUSIVE', 'evidence': 'B and D have different nullspace torques and late-failure times. No repetition, contact-isolation intervention or preregistered equivalence bound establishes absence of effect.'},
            {'claim': 'The globally unique cause of complete Sim2Sim policy failure has been established.',
             'status': 'INCONCLUSIVE', 'evidence': 'No new PPO inference, no task-level policy validation, finite poses/horizons, late grasp failure, incomplete original body correspondence and unknown original engine runtime dynamics.'}],
        'factorial_common_prefix_steps': {key: value.get('common_prefix_steps') for key, value in factorial.items()},
        'late_grasp_reviews': late_reviews, 'root_cause_status': 'ROOT_CAUSE_PENDING',
        'solution_status': 'SOLUTION_PENDING', 'policy_validation_status': 'VALIDATION_PENDING',
        'engineering_candidate_recommendation': candidate,
        'R3_recommendation': ('Separately authorized limited frozen-PPO gate may be considered; no policy execution in R2.' if supported_candidate else
            'Do not begin policy validation on the basis of the current results; resolve the registered grasp/contact failures in a separately authorized no-policy gate.')}


def plot_traces(output, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    initial_ids = sorted({name.split('_')[0] for name in records})
    if not initial_ids:
        return []
    fig, axes = plt.subplots(len(initial_ids), 2, figsize=(12, 3.5*len(initial_ids)), squeeze=False, constrained_layout=True)
    colors = {'A': '#b63b39', 'B': '#177b54', 'C': '#d68d21', 'D': '#3268ae'}
    for row_index, initial_id in enumerate(initial_ids):
        for variant in 'ABCD':
            data = records.get(initial_id+'_'+variant)
            if not data:
                continue
            time_values = [row['time_s'] for row in data]
            axes[row_index, 0].plot(time_values, [row['joint7_velocity_rad_s'] for row in data], label=variant, color=colors[variant])
            axes[row_index, 1].semilogy(time_values, [max(1e-6, row['translation_slip_m']*1000) for row in data], label=variant, color=colors[variant])
        axes[row_index, 0].set_ylabel(initial_id+' J7 velocity [rad/s]')
        axes[row_index, 1].set_ylabel(initial_id+' peg-hand slip [mm, log]')
        axes[row_index, 1].axhline(1, color='grey', linestyle=':', label='1 mm gate')
        for axis in axes[row_index]:
            axis.axvline(16*DT, color='grey', linewidth=.7, linestyle='--')
            axis.grid(True, alpha=.25)
            axis.legend(ncol=5, fontsize=8)
            axis.set_xlabel('Registered diagnostic simulation time [s]')
    fig.suptitle('NO_POLICY_DIAGNOSTIC_ONLY: P0/P1 × C0/C1, matched initial states\nLines end at recorded guard stops; each pose has its own baseline. No insertion-success interpretation.')
    path = output/'factorial_traces.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path.name]


def analyze(source, output, make_plots=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    registry, summary = read(source/'preregistered_matrix.json'), read(source/'summary.json')
    if registry['scope'] != 'NO_POLICY_DIAGNOSTIC_ONLY' or registry['maximum_new_integration_steps'] != 2880 or registry['steps_per_trajectory'] != 235:
        raise ValueError('Unexpected registered experiment scope or budget')
    schedule_hash = hashlib.sha256(json.dumps(registry['input_schedule'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if schedule_hash != registry['input_schedule_sha256']:
        raise ValueError('Preregistered input schedule canonical hash is invalid')
    inertia = inspect_inertias(registry['models']['P0'], registry['models']['P1'])
    write(output/'inertia_tensor_review.json', inertia)
    thresholds, horizon = registry['thresholds'], registry['steps_per_trajectory']
    results, records, hashes, initial_statuses, admission_reviews, late_reviews = [], {}, {}, [], [], []
    for initial in registry['initials']:
        initial_id = initial['id']
        admission_path = source/'initials'/initial_id/'admission.json'
        prep_path = source/'initials'/initial_id/'result.json'
        if not admission_path.exists() or not prep_path.exists():
            initial_statuses.append({'initial_id': initial_id, 'status': 'NOT_EXECUTED_OR_INCOMPLETE'})
            continue
        admission, preparation = read(admission_path), read(prep_path)
        admission_reviews.append(inspect_admission(source/'initials'/initial_id, initial, registry['models']['P0']))
        initial_statuses.append({'initial_id': initial_id, 'diagnostic_seed': initial['seed'],
            'admission': admission['admission'], 'admission_errors': admission.get('admission_errors', []),
            'preparation_steps': preparation['preparation_steps'], 'preparation_runtime_failure': preparation['failure']})
        hashes[str(admission_path.relative_to(ROOT))] = digest(admission_path)
        if admission['admission'] != 'PASS':
            for variant in 'ABCD':
                skipped = source/(initial_id+'_'+variant+'_skipped.json')
                if not skipped.exists() or read(skipped)['status'] != 'SKIPPED_ADMISSION':
                    raise ValueError('Failed-admission initial silently dropped from registered denominator')
                hashes[str(skipped.relative_to(ROOT))] = digest(skipped)
            continue
        initial_path = source/'initials'/initial_id/'initial_state.json'
        initial_record = read(initial_path)
        for variant in 'ABCD':
            run_id = initial_id+'_'+variant
            directory = source/'trajectories'/run_id
            identity_path, result_path, trajectory_path = directory/'identity.json', directory/'result.json', directory/'physics.jsonl.gz'
            if not all(path.exists() for path in (identity_path, result_path, trajectory_path)):
                raise ValueError('Admitted initial missing complete trajectory evidence: '+run_id)
            identity, result = read(identity_path), read(result_path)
            if identity['variant'] != variant or identity['run_id'] != run_id:
                raise ValueError('Trajectory identity does not match matrix')
            if identity['initial_state_sha256'] != digest(initial_path) or identity['input_schedule_sha256'] != registry['input_schedule_sha256']:
                raise ValueError('Initial state or action schedule hash mismatch')
            if result['trace_sha256'] != digest(trajectory_path):
                raise ValueError('Full raw trajectory hash mismatch')
            events = trace(trajectory_path)
            requests = [row for row in events if row['event'] == 'integration_request']
            rows = [row for row in events if row['event'] == 'physics_result']
            if len(events) != 2*len(rows) or len(requests) != len(rows):
                raise ValueError('Unpaired integration or exception requires explicit numerical review: '+run_id)
            if any(row['event'] != ('integration_request' if index % 2 == 0 else 'physics_result') for index, row in enumerate(events)):
                raise ValueError('Integration request/result ordering invalid')
            if [row['step'] for row in requests] != list(range(1, len(rows)+1)):
                raise ValueError('Integration request sequence invalid')
            if len(rows) != result['completed_steps'] or len(rows) > horizon:
                raise ValueError('Registered trajectory step accounting mismatch')
            bound_identity = {**identity, 'model': identity['models']['plant'], 'thresholds': thresholds}
            metric_rows, errors = independently_reconstruct_rows(bound_identity, rows)
            late_review = late_grasp_witnesses(run_id, rows, metric_rows, thresholds)
            if late_review is not None:
                late_reviews.append(late_review)
            schedule_errors = inspect_reference_schedule(identity, initial_record, registry, rows)
            tolerances = {key: 1e-8 if key.startswith('controller_') or key == 'interval_acceleration' else 1e-9 for key in errors}
            tolerances['slip_rotation_rad'] = 1e-7
            failures = {key: {'maximum_error': value, 'tolerance': tolerances[key]} for key, value in errors.items() if value > tolerances[key]}
            failures.update({'schedule_'+key: {'maximum_error': value, 'tolerance': 1e-10} for key, value in schedule_errors.items() if value > 1e-10})
            metrics = aggregate(metric_rows)
            summaries = {'max_slip_m': metrics['max_translation_slip_m'], 'max_rotation_slip_rad': metrics['max_rotation_slip_rad'],
                'max_arm_qvel_rad_s': metrics['max_arm_velocity_rad_s'], 'max_penetration_m': metrics['max_all_penetration_m'],
                'bilateral_positive_contact_fraction': metrics['bilateral_positive_contact_fraction']}
            summary_errors = {key: abs(result[key]-value) for key, value in summaries.items()}
            if max(summary_errors.values()) > 1e-7 or result['first_contact_loss_step'] != metrics['first_contact_loss_step']:
                failures['runner_summary_reconstruction'] = summary_errors
            if not rows:
                failures['empty_trajectory'] = True
            warning_count = sum(sum(row['warning_delta']) for row in rows)
            gates = stability_gates(metric_rows, thresholds, horizon)
            gates['interface_correctness'] = not failures and len(rows) == result['completed_steps'] and warning_count == 0 and result['runtime_failure'] is None
            gates['all_stability_gates'] &= gates['interface_correctness']
            receipt_path = source/(run_id+'_process.json')
            receipt = read(receipt_path)
            if receipt['exact_child_exit_code'] != 0:
                gates['interface_correctness'] = gates['all_stability_gates'] = False
            row = {'run_id': run_id, 'initial_id': initial_id, 'variant': variant,
                'plant': identity['plant_selection'], 'controller': identity['controller_selection'],
                'metrics': metrics, 'gates': gates, 'arithmetic_max_errors': errors, 'arithmetic_tolerances': tolerances,
                'schedule_max_errors': schedule_errors, 'review_failures': failures, 'warning_count': warning_count,
                'termination': result['termination'], 'early_stop_reasons': result['early_stop_reasons'],
                'exact_child_exit_code': receipt['exact_child_exit_code'], 'physics_wall_s': result['physics_wall_s'],
                'viewer_sync_wall_s': result['viewer_sync_wall_s'], 'wall_s': result['wall_s'],
                'gui_and_logging_overhead_s': result['gui_and_logging_overhead_s'],
                'initial_snapshot': identity['initial_snapshot']}
            results.append(row)
            records[run_id] = metric_rows
            for path in (identity_path, result_path, trajectory_path, receipt_path):
                hashes[str(path.relative_to(ROOT))] = digest(path)
    factorial = paired_factorial(results, records)
    causal = causal_review(results, factorial, late_reviews)
    write(output/'causal_and_late_grasp_review.json', causal)
    write(output/'initial_admission_review.json', admission_reviews)
    write(output/'factorial_effects.json', factorial)
    with gzip.open(output/'independent_step_metrics.jsonl.gz', 'xt') as stream:
        for run_id, rows in records.items():
            for row in rows:
                stream.write(json.dumps({'run_id': run_id, **row}, allow_nan=False)+'\n')
    plot_files = plot_traces(output, records) if make_plots else []
    public_results = [{key: value for key, value in row.items() if key != 'initial_snapshot'} for row in results]
    with (output/'trajectory_metrics.csv').open('x', newline='') as stream:
        flat = [{'run_id': row['run_id'], **row['metrics'], **{key: value for key, value in row['gates'].items() if key != 'scope'}} for row in results]
        if flat:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
    admission_review_passed = all(row['status'] == 'RECORDED_ADMISSION_FAILURE' or (
        row['present_state_passed'] and row.get('independently_admitted', True)) for row in admission_reviews)
    review = {'task': 'G4-S2A-R2', 'scope': 'INDEPENDENT_NUMPY_RECONSTRUCTION_OF_RAW_NO_POLICY_TRACES',
        'new_physics_steps': 0, 'new_policy_calls': 0, 'model_compilations': 0,
        'registered_initial_count': 3, 'registered_trajectory_count': 12,
        'admitted_initial_count': sum(row.get('admission') == 'PASS' for row in initial_statuses),
        'reviewed_trajectory_count': len(results), 'reviewed_physics_results': sum(len(rows) for rows in records.values()),
        'initials': initial_statuses, 'trajectories': public_results, 'factorial_effects': factorial,
        'independent_initial_admission_review': admission_reviews,
        'causal_classifications': causal['classifications'],
        'candidate_recommendation': causal['engineering_candidate_recommendation'],
        'independent_admission_review_passed': bool(admission_review_passed),
        'numerical_review_passed': bool(results) and admission_review_passed and all(not row['review_failures'] and row['gates']['interface_correctness'] for row in results),
        'complete_registered_accounting': summary['completed_accounting'],
        'experiment_execution_gate_passed': summary['execution_gate_passed'],
        'source_hashes': hashes, 'analysis_script_sha256': digest(__file__), 'plots': plot_files,
        'runtime_s': time.monotonic()-started, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'independent_method': 'Read every raw input/state/contact. Solve source controller independently; rederive quaternion error/FD/EMA/reference schedule; verify isotropic body tensors and kinetic-energy pullback ΔM=Σ0.01 Jw.T Jw; compare physical/controller mass selections at saved same q; reconstruct clipped motor force, slip, contacts, penetration and joint limits.',
        'timing_semantics': {'interval_acceleration': '(post_qvel−pre_qvel)/(1/120); physical interval increment, includes contacts/constraints.',
            'solver_qacc_and_contacts': 'Saved qacc_post_forward and state contacts are recomputed at post-step state; not original integration-stage impulse or exact interval acceleration.',
            'actuator_force': 'Saved integration_actuation motor force precedes mj_forward; compared with source software and motor limits.',
            'saturation_fraction': 'Fraction of seven arm components per saved control step requiring additional motor clipping after source ±100Nm software clipping; actual-at-limit and source-software-clipped fractions are separately reported.',
            'finger_force': 'Independently solve preserved backward-Euler free-motion prediction using pre-state physical full-M inverse finger block, then clip±40N; not original PhysX drive equivalence.',
            'viewer_overhead': 'GUI rendering/sync/sleep/logging wall overhead does not alter physical dt.'},
        'limitations': ['Finite registered poses and horizons do not prove global dynamics or insertion success.',
            'Early guard failures are retained outcomes; unequal full horizons are not treated as equal-exposure improvements. Factorial contrasts use the common recorded prefix within each initial.',
            'No cross-pose improvement denominator or pooled 64-env/32-env policy statistic is calculated.',
            'P1 also changes the physical full-M dependency of the preserved implicit finger-force adaptation; arm controller C selection does not isolate that submechanism.',
            'Contact is verified only at saved physics-step states; no continuous-time bilateral contact certificate.',
            'Original PhysX runtime M, cooked inertias and applied actuator forces remain UNKNOWN.']}
    write(output/'numerical_review.json', review)
    write(output/'manifest.json', {'files': {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()},
                                 'new_physics_steps': 0, 'new_policy_calls': 0, 'model_compilations': 0})
    print(json.dumps({'numerical_review_passed': review['numerical_review_passed'], 'reviewed_trajectories': len(results),
                      'reviewed_physics_results': review['reviewed_physics_results'], 'runtime_s': review['runtime_s'], 'output': str(output)}))
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    review = analyze(args.source, args.output, make_plots=not args.no_plots)
    return 0 if review['numerical_review_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
