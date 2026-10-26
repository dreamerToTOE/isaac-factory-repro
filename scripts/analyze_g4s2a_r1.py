#!/usr/bin/env python3
"""Offline archived S2A substep analysis; no actor or physics imports/calls.

Reads the committed, losslessly compressed S2A traces. Acceleration is an
interval velocity-difference estimate, not an archived solver qacc. Full
historical Jacobians and mass matrices are unavailable and remain UNKNOWN.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
START = '9f521002cd1ee1694fabc1f2012a8bdd79e26434'
DT = 1 / 120
KP = np.array([100., 100., 100., 30., 30., 30.])
KD = 2 * np.sqrt(KP)
PACKAGE = ROOT / 'evidence/g4_s/s2a/run_20261026T140338'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def conjugate(a):
    return np.asarray(a) * [1., -1., -1., -1.]


def axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=float).copy()
    if q[0] < 0:
        q *= -1
    half = math.atan2(float(np.linalg.norm(q[1:])), float(q[0]))
    angle = 2*half
    denominator = math.sin(half)/angle if abs(angle) > 1e-6 else .5-angle*angle/48
    return q[1:] / denominator


def orientation_error(target, current):
    target, current = np.asarray(target), np.asarray(current)
    if np.dot(target, current) < 0:
        target = -target
    return axis_angle(multiply(target, conjugate(current)/np.dot(current, current)))


def pad_forces(state):
    forces = {'left_finger': 0., 'right_finger': 0.}
    for contact in state['contacts']:
        if 'peg_collision' not in (contact['geom1'], contact['geom2']):
            continue
        other = contact['geom2'] if contact['geom1'] == 'peg_collision' else contact['geom1']
        for side in forces:
            if other in {f'{side}_geom_{k}' for k in range(3, 8)}:
                forces[side] += max(0., contact['normal_force_n'])
    return forces


def flatten(record):
    result = {}
    for key, value in record.items():
        if isinstance(value, list) and all(isinstance(x, (int, float, bool)) for x in value):
            result.update({f'{key}_{i+1}': x for i, x in enumerate(value)})
        elif isinstance(value, (int, float, bool, str)) or value is None:
            result[key] = value
    return result


def write_csv(path, records):
    rows = [flatten(record) for record in records]
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(policy, physics, initial):
    controls = [substep for decision in policy for substep in decision['control_substeps']]
    assert len(controls) == len(physics) == 1192
    previous = initial
    output = []
    for index, (control, physical) in enumerate(zip(controls, physics)):
        state = physical['state']
        assert physical['physics_step'] == index+1
        assert state == control['physical_state_after']
        p = np.asarray(control['position_before_m'])
        q = np.asarray(control['quaternion_before_wxyz'])
        target_p = np.asarray(control['target_position_m'])
        target_q = np.asarray(control['target_quaternion_wxyz'])
        position_error = target_p-p
        rotation_error = orientation_error(target_q, q)
        velocity = np.r_[control['linear_fd_before_m_s'], control['angular_fd_before_rad_s']]
        proportional = KP*np.r_[position_error, rotation_error]
        derivative = -KD*velocity
        measured_wrench = np.asarray(control['pd_wrench'])
        requested = np.asarray(control['requested_arm_torque_nm'])
        task = np.asarray(control['task_arm_torque_nm'])
        null = np.asarray(control['null_arm_torque_nm'])
        qvel_before = np.asarray(previous['qvel'][:7])
        qvel_after = np.asarray(state['qvel'][:7])
        forces = pad_forces(state)
        delta_q = multiply(q, conjugate(previous['tcp_quaternion_wxyz']))
        # The first state is a deliberate reset-FD-zero event; all subsequent
        # controller FD values refer to the just-completed physical interval.
        if index == 0:
            expected_fd = np.zeros(6)
            quaternion_dot = 1.
        else:
            prior_p = np.asarray(physics[index-2]['state']['tcp_position_world']) if index > 1 else np.asarray(initial['tcp_position_world'])
            prior_q = np.asarray(physics[index-2]['state']['tcp_quaternion_wxyz']) if index > 1 else np.asarray(initial['tcp_quaternion_wxyz'])
            delta_q = multiply(q, conjugate(prior_q))
            delta_q *= np.sign(delta_q[0])
            expected_fd = np.r_[(p-prior_p)/DT, axis_angle(delta_q)/DT]
            quaternion_dot = float(np.dot(q, prior_q))
        item = {
            'physics_step': index+1, 'policy_step': index//8+1, 'policy_substep': index%8+1,
            'episode_time_before_s': index*DT, 'episode_time_after_s': (index+1)*DT,
            'absolute_time_before_s': control['time_before_s'], 'absolute_time_after_s': state['time_s'],
            'arm_qpos_before_rad': previous['qpos'][:7], 'arm_qpos_after_rad': state['qpos'][:7],
            'arm_qvel_before_rad_s': qvel_before.tolist(), 'arm_qvel_after_rad_s': qvel_after.tolist(),
            'arm_acceleration_interval_estimate_rad_s2': ((qvel_after-qvel_before)/DT).tolist(),
            'tcp_position_before_m': p.tolist(), 'tcp_quaternion_before_wxyz': q.tolist(),
            'tcp_position_after_m': state['tcp_position_world'], 'tcp_quaternion_after_wxyz': state['tcp_quaternion_wxyz'],
            'target_position_m': target_p.tolist(), 'target_quaternion_wxyz': target_q.tolist(),
            'cartesian_position_error_m': position_error.tolist(), 'cartesian_orientation_error_world_rad': rotation_error.tolist(),
            'feedback_world_velocity': velocity.tolist(), 'independent_feedback_world_velocity': expected_fd.tolist(),
            'pd_proportional_wrench': proportional.tolist(), 'pd_derivative_wrench': derivative.tolist(),
            'recorded_pd_wrench': measured_wrench.tolist(),
            'requested_arm_torque_nm': requested.tolist(), 'software_limited_arm_torque_nm': control['software_limited_arm_torque_nm'],
            'integration_actuator_force_nm': state['integration_actuation']['arm_actuator_force_nm'],
            'integration_generalized_actuator_contribution_nm': state['integration_actuation']['arm_qfrc_actuator_nm'],
            'post_forward_actuator_force_nm': state['actuator_force_nm'],
            'post_forward_generalized_actuator_contribution_nm': state['generalized_actuator_contribution_nm'],
            'task_arm_torque_nm': task.tolist(), 'null_arm_torque_nm': null.tolist(),
            'left_pad_normal_force_n': forces['left_finger'], 'right_pad_normal_force_n': forces['right_finger'],
            'bilateral_positive_pad_contact': all(force > 1e-6 for force in forces.values()),
            'finger_qpos_m': state['qpos'][7:9], 'finger_qvel_m_s': state['qvel'][7:9],
            'finger_integration_actuator_force_n': state['integration_actuation']['finger_actuator_force_n'],
            'finger_raw_PD_force_n': state['finger_command']['raw_pd_force_n'],
            'peg_hand_translation_slip_m': state['s2a']['peg_hand_slip_m'],
            'peg_hand_orientation_slip_rad': state['s2a']['peg_hand_slip_rad'],
            'arm_limit_violation_rad': state['s2a']['arm_joint_limit_violation_rad'],
            'finger_limit_violation_m': state['s2a']['finger_joint_limit_violation_m'],
            'all_contact_penetration_m': state['s2a']['all_contact_penetration_m'],
            'peg_hole_penetration_m': state['s2a']['peg_hole_penetration_m'],
            'contact_count': len(state['contacts']), 'contacts': state['contacts'],
            'recorded_mass_condition': control['mass_condition'],
            'recorded_task_inverse_condition': control['task_inverse_condition'],
            'quaternion_dot_with_prior_physics_pose': quaternion_dot,
            'quaternion_norm_before': float(np.linalg.norm(q)),
            'independent_pd_wrench_max_abs_error': float(max(abs(proportional+derivative-measured_wrench))),
            'independent_FD_max_abs_error': float(max(abs(expected_fd-velocity))),
            'task_plus_null_max_abs_error_nm': float(max(abs(task+null-requested))),
            'recorded_J_matrix': 'UNKNOWN_NOT_ARCHIVED', 'recorded_M_matrix': 'UNKNOWN_NOT_ARCHIVED',
            'solver_qacc': 'UNKNOWN_NOT_ARCHIVED', 'constraint_generalized_force': 'UNKNOWN_NOT_ARCHIVED',
        }
        output.append(item)
        previous = state
    return output


def plots(output, records):
    import matplotlib
    matplotlib.use('Agg')  # Scientific plots only; this is not a viewer fallback.
    import matplotlib.pyplot as plt
    early = records[:8]
    steps = np.arange(1, 9)
    figure, axes = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
    axes[0, 0].plot(steps, [x['arm_qvel_after_rad_s'][6] for x in early], 'o-')
    axes[0, 0].set_ylabel('Joint 7 velocity [rad/s]')
    axes[0, 1].plot(steps, [x['arm_acceleration_interval_estimate_rad_s2'][6] for x in early], 'o-')
    axes[0, 1].set_ylabel('Joint 7 interval acceleration [rad/s²]')
    for key, label in [('requested_arm_torque_nm', 'requested'), ('software_limited_arm_torque_nm', 'software'),
                       ('integration_actuator_force_nm', 'integration motor'), ('null_arm_torque_nm', 'null')]:
        axes[1, 0].plot(steps, [x[key][6] for x in early], 'o-', label=label)
    axes[1, 0].axhline(12, color='gray', linestyle='--'); axes[1, 0].axhline(-12, color='gray', linestyle='--')
    axes[1, 0].set_ylabel('Joint 7 torque [Nm]'); axes[1, 0].legend(fontsize=8)
    for key, label in [('pd_proportional_wrench', 'P world-z'), ('pd_derivative_wrench', 'D world-z'), ('recorded_pd_wrench', 'P+D world-z')]:
        axes[1, 1].plot(steps, [x[key][5] for x in early], 'o-', label=label)
    axes[1, 1].set_ylabel('World-z angular wrench [Nm]'); axes[1, 1].legend(fontsize=8)
    axes[2, 0].plot(steps, [x['left_pad_normal_force_n'] for x in early], 'o-', label='left')
    axes[2, 0].plot(steps, [x['right_pad_normal_force_n'] for x in early], 'o-', label='right')
    axes[2, 0].set_ylabel('Recorded pad normal force [N]'); axes[2, 0].legend()
    axes[2, 1].plot(steps, [x['peg_hand_translation_slip_m']*1000 for x in early], 'o-')
    axes[2, 1].axhline(1, color='gray', linestyle='--'); axes[2, 1].set_ylabel('Peg/hand translation slip [mm]')
    axes[3, 0].plot(steps, [x['feedback_world_velocity'][5] for x in early], 'o-')
    axes[3, 0].set_ylabel('Controller FD angular world-z [rad/s]')
    axes[3, 1].plot(steps, [np.linalg.norm(x['cartesian_orientation_error_world_rad']) for x in early], 'o-')
    axes[3, 1].set_ylabel('Target orientation error magnitude [rad]')
    for axis in axes.ravel():
        axis.set_xlabel('Archived physics substep (one unchanged PPO/EMA action)')
        axis.set_xticks(steps); axis.grid(alpha=.25)
    figure.suptitle('Archived S2A: first eight physics steps — no new integration or actor calls')
    figure.savefig(output/'first_eight_substeps.png', dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    times = [x['episode_time_after_s'] for x in records]
    axes[0].plot(times, [max(abs(np.asarray(x['arm_qvel_after_rad_s']))) for x in records]); axes[0].set_ylabel('Max arm |velocity| [rad/s]')
    axes[1].plot(times, [x['peg_hand_translation_slip_m'] for x in records]); axes[1].set_ylabel('Peg/hand slip [m]')
    axes[2].plot(times, [1000*x['all_contact_penetration_m'] for x in records]); axes[2].set_ylabel('Max contact penetration [mm]')
    for axis in axes:
        for step, label in [(3, 'slip'), (4, 'pad loss'), (61, 'finger limit'), (76, 'arm limit'), (685, 'peak support/hand')]:
            axis.axvline(step/120, color='gray', alpha=.35)
        axis.grid(alpha=.25)
    axes[2].set_xlabel('Archived episode time [s], excludes 0.25 s grasp preparation')
    figure.suptitle('Archived S2A failure chronology; sampled evidence, not a safety certificate')
    figure.savefig(output/'full_episode_chronology.png', dpi=160)
    plt.close(figure)


def analyze(output):
    output.mkdir(parents=True, exist_ok=False)
    bundle = read_json(PACKAGE/'collection.json')
    for artifact in bundle['artifacts']:
        path = ROOT/artifact['bundle_path']
        assert sha(path) == artifact['bundle_sha256'], str(path)
        content = gzip.decompress(path.read_bytes()) if artifact['lossless_gzip'] else path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact['raw_sha256'], str(path)
    policy = read_lines(PACKAGE/'seed_1001/policy.jsonl.gz')
    physics = read_lines(PACKAGE/'seed_1001/physics.jsonl.gz')
    closure = read_lines(PACKAGE/'seed_1001/closure.jsonl.gz')
    initial = read_json(PACKAGE/'seed_1001/interface_reset.json')['initial']['state']
    records = reconstruct(policy, physics, initial)
    early = records[:8]
    write_json(output/'first_eight_steps.json', {'source': 'ARCHIVED_STATE_MEASUREMENTS_AND_OFFLINE_ALGEBRA', 'steps': early})
    write_csv(output/'first_eight_steps.csv', early)
    full_keys = ('physics_step', 'policy_step', 'episode_time_after_s',
                 'left_pad_normal_force_n', 'right_pad_normal_force_n',
                 'bilateral_positive_pad_contact', 'peg_hand_translation_slip_m',
                 'peg_hand_orientation_slip_rad', 'arm_limit_violation_rad',
                 'finger_limit_violation_m', 'all_contact_penetration_m',
                 'peg_hole_penetration_m', 'contact_count')
    full_rows = []
    for item in records:
        compact = {key: item[key] for key in full_keys}
        compact.update({'joint7_qvel_rad_s': item['arm_qvel_after_rad_s'][6],
                        'joint7_requested_nm': item['requested_arm_torque_nm'][6],
                        'joint7_actual_nm': item['integration_actuator_force_nm'][6],
                        'joint7_task_nm': item['task_arm_torque_nm'][6],
                        'joint7_null_nm': item['null_arm_torque_nm'][6],
                        'maximum_arm_abs_qvel_rad_s': float(max(abs(np.asarray(item['arm_qvel_after_rad_s']))))})
        full_rows.append(compact)
    write_csv(output/'full_episode_substeps.csv', full_rows)
    plots(output, records)
    initial_forces = pad_forces(closure[-1]['state'])
    first_saturation = next(x for x in records if abs(x['requested_arm_torque_nm'][6]) > 12)
    first_bilateral_loss = next(x for x in records if not x['bilateral_positive_pad_contact'])
    first_slip = next(x for x in records if x['peg_hand_translation_slip_m'] > .001)
    first_finger = next(x for x in records if x['finger_limit_violation_m'] > 1e-6)
    first_arm = next(x for x in records if x['arm_limit_violation_rad'] > 1e-6)
    source_paths = [Path(__file__), ROOT/'scripts/g4s1b_contract.py', ROOT/'scripts/g4s1b_env.py',
        ROOT/'scripts/g4s2a_smoke.py', PACKAGE/'preregistered_protocol.json', PACKAGE/'collection.json',
        PACKAGE/'seed_1001/policy.jsonl.gz', PACKAGE/'seed_1001/physics.jsonl.gz', PACKAGE/'seed_1001/closure.jsonl.gz',
        PACKAGE/'seed_1001/interface_reset.json',
        ROOT.parents[1]/'work/upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py',
        ROOT.parents[1]/'work/upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py']
    artifact_paths = [path for path in output.iterdir() if path.is_file()]
    summary = {'task': 'G4-S2A-R1', 'analysis': 'OFFLINE_ARCHIVED_SUBSTEP_RECONSTRUCTION',
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'expected_start_commit': START,
        'HEAD_at_analysis': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'new_actor_calls': 0, 'checkpoint_loads': 0, 'new_physics_steps': 0, 'GUI_opened_by_analyzer': False,
        'archived_counts': {'policy_steps': len(policy), 'physics_steps': len(physics), 'closure_steps': len(closure)},
        'acceleration_definition': '(post-step qvel minus previous post-step/reset qvel)/(1/120); interval velocity increment, not saved solver qacc; includes coupled dynamics and contacts.',
        'actuator_stages': {'integration': 'Saved mj_step preintegration force at time_before', 'post_forward': 'Recomputed actuation after mj_forward at final state; recorded separately'},
        'first_action': {'raw_mu': policy[0]['mu'], 'clipped': policy[0]['clipped'], 'ema': policy[0]['ema'],
            'single_policy_decision_covers_all_first_eight_substeps': True,
            'interpretation': 'Torque reversals across steps1–8 occur without another PPO inference or EMA update. Per-substep pose, FD/J/M and target reconstruction are active.'},
        'first_eight_quantities': {'joint7_qvel_after_rad_s': [x['arm_qvel_after_rad_s'][6] for x in early],
            'joint7_acceleration_interval_rad_s2': [x['arm_acceleration_interval_estimate_rad_s2'][6] for x in early],
            'joint7_requested_nm': [x['requested_arm_torque_nm'][6] for x in early],
            'joint7_software_nm': [x['software_limited_arm_torque_nm'][6] for x in early],
            'joint7_integration_nm': [x['integration_actuator_force_nm'][6] for x in early],
            'joint7_task_nm': [x['task_arm_torque_nm'][6] for x in early],
            'joint7_null_nm': [x['null_arm_torque_nm'][6] for x in early],
            'world_angular_z_FD_rad_s': [x['feedback_world_velocity'][5] for x in early],
            'world_angular_z_P_Nm': [x['pd_proportional_wrench'][5] for x in early],
            'world_angular_z_D_Nm': [x['pd_derivative_wrench'][5] for x in early],
            'left_pad_force_n': [x['left_pad_normal_force_n'] for x in early],
            'right_pad_force_n': [x['right_pad_normal_force_n'] for x in early],
            'translation_slip_m': [x['peg_hand_translation_slip_m'] for x in early],
            'orientation_slip_rad': [x['peg_hand_orientation_slip_rad'] for x in early]},
        'checks': {'archived_bundle_and_uncompressed_hashes_match': True,
            'pd_wrench_independent_max_error_first8': max(x['independent_pd_wrench_max_abs_error'] for x in early),
            'FD_independent_max_error_first8': max(x['independent_FD_max_abs_error'] for x in early),
            'task_plus_null_max_error_first8_nm': max(x['task_plus_null_max_abs_error_nm'] for x in early),
            'quaternion_min_consecutive_dot_first8': min(x['quaternion_dot_with_prior_physics_pose'] for x in early),
            'quaternion_max_norm_error_first8': max(abs(x['quaternion_norm_before']-1) for x in early),
            'joint7_null_max_abs_first8_nm': max(abs(x['null_arm_torque_nm'][6]) for x in early),
            'joint7_task_max_abs_first8_nm': max(abs(x['task_arm_torque_nm'][6]) for x in early),
            'initial_mass_condition_recorded': early[0]['recorded_mass_condition'],
            'initial_task_inverse_condition_recorded': early[0]['recorded_task_inverse_condition']},
        'onsets': {key: {'physics_step': value['physics_step'], 'episode_time_s': value['episode_time_after_s']}
            for key, value in [('joint7_requested_above_actual12Nm_limit', first_saturation),
                ('peg_hand_slip_over1mm', first_slip), ('loss_of_bilateral_positive_pad_force', first_bilateral_loss),
                ('finger_limit_over1e_6m', first_finger), ('arm_limit_over1e_6rad', first_arm)]},
        'grasp_before_policy': {'last_closure_pad_normal_force_n': initial_forces,
            'finger_actuator_n': initial['finger_actuator_force_n'],
            'note': 'Normal force and actuator force are distinct physical quantities. Positive bilateral soft contact does not establish acceleration tolerance.'},
        'source_controller_review': {'PD': 'target-current pose error; negative world FD damping; same signs as frozen Factory source',
            'feedback_timestamp': 'First FD explicitly reset zero; later substep feedback uses immediately preceding physics interval, not 8dt',
            'target': 'Rebuilt from current centered TCP each physics step; therefore this is not a fixed world Cartesian target',
            'Jacobian': 'Source and port use mean left/right finger body-origin world geometric Jacobian; centered TCP hand-local z=.112071m is a distinct point',
            'nullspace': 'Saved tau_null uses dynamic projection from J/M; its full historical inputs were not logged',
            'inertia': 'Original source adds .01I to each robot body inertia, not .01 joint armature; target S1B Menagerie inertias differ. Static/controlled comparison is separate evidence.'},
        'hypotheses_from_archived_data_only': {
            'H1_inertia_mismatch': {'status': 'INCONCLUSIVE', 'reason': 'Different source body inertia rule is known; historical per-step full M/J absent, no isolated inertial experiment in this analyzer'},
            'H2_saturation_acceleration': {'status': 'SUPPORTED_BUT_NOT_PROVEN', 'reason': 'Rapid J7 velocity reversal and acceleration estimate precede grasp loss; first >12Nm request is step3, simultaneous with >1mm slip; saturation alone is not established as unique cause'},
            'H3_discrete_PD': {'status': 'SUPPORTED_BUT_NOT_PROVEN', 'reason': 'P/D independently reconstruct saved wrench; world angular FD damping magnitude rises and reverses torque within one fixed PPO action. Correct negative damping sign does not guarantee discrete-time stability.'},
            'H4_J_nullspace': {'status': 'INCONCLUSIVE', 'reason': 'Initial J7 null contribution is tiny; J7 task torque dominates. Coupling from other joints and full J/M projection cannot be isolated from archived outputs.'},
            'H5_gripper_contact': {'status': 'SUPPORTED_BUT_NOT_PROVEN', 'reason': 'Initial bilateral forces positive; lost during arm transient. Need no-arm and finger/contact diagnostic comparisons.'},
            'H6_reset': {'status': 'INCONCLUSIVE', 'reason': 'Initial state passes declared admission but only one adapted reset; acceleration robustness not tested by admission.'},
            'H7_quaternion_FD': {'status': 'CONTRADICTED', 'scope': 'Early quaternion sign jump, missing 1/120 FD update, or wrong PD damping sign as direct explanation of first-eight reversal',
                'reason': 'Positive consecutive quaternion dots, unit norms and independent FD/P-D checks. Broader cross-engine frame/dynamics mismatch remains unresolved.'}},
        'UNKNOWN': ['Historical full J and M matrices', 'Historical solver qacc', 'Historical qfrc_constraint and contact warm starts',
                    'Independent reconstruction of J-transpose wrench and dynamic nullspace from full historical inputs',
                    'Original PhysX actually applied joint torque', 'Unique root cause and validated solution'],
        'root_cause_status': 'ROOT_CAUSE_PENDING', 'solution_status': 'SOLUTION_PENDING',
        'source_hashes': {str(path): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in source_paths},
        'artifacts': {str(path.relative_to(ROOT)): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in artifact_paths}}
    write_json(output/'analysis.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.output.resolve())
    print(json.dumps({'output': str(args.output), 'counts': summary['archived_counts'],
                      'checks': summary['checks'], 'onsets': summary['onsets'],
                      'new_actor_calls': 0, 'new_physics_steps': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
