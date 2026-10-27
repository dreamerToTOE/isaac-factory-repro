#!/usr/bin/env python3
"""Once-only visible, preregistered R2-R1 diagnostics; no policy or preparation.

Use recorded R2 initial states. This module never changes the S1B plant or the
finger predictor. Execution requires a separate CPU receipt and creates an
exclusive coordinator attempt; failed runs are never resumed or retried.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.g4s2a_r2_r1_model import (RUNS, SELECTIONS, ModelPair, make_model,
    model_identity, MAX_STEPS, THRESHOLDS, digest, fixed_reference, stop_reasons)
from scripts.g4s2a_r2_diagnostics import Visible, restore_initial, integration_state
from scripts.g4s1b_contract import PhysicsContract, torque_command
from scripts.g4s2a_smoke import jsonable, physical_measurements, peg_relative, finite_tree
from scripts.g4s2a_r1_viewer import contact_forces

R2 = ROOT / 'evidence/g4_s/s2a_r2/matrix_v2'
SOURCES = [
    'scripts/g4s2a_r2_r1_model.py', 'scripts/g4s2a_r2_r1_diagnostics.py',
    'scripts/g4s2a_r2_model.py', 'scripts/g4s2a_r2_diagnostics.py',
    'scripts/g4s1b_env.py', 'scripts/g4s1b_contract.py', 'scripts/g4s2a_smoke.py',
    'scripts/g4s2a_r1_viewer.py', 'scripts/capture_g4s2a_r1_comparison.py',
]
BUDGET = MAX_STEPS * len(RUNS)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def safe_value(value):
    """Keep non-finite failure evidence valid JSON without hiding the value."""
    value = jsonable(value)
    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return 'NaN' if math.isnan(v) else ('+Infinity' if v > 0 else '-Infinity')
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        return v
    return clean(value)

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(safe_value(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def line(f, value):
    f.write(json.dumps(safe_value(value), allow_nan=False, separators=(',', ':')) + '\n')
    f.flush()
    os.fsync(f.fileno())

def identity(env):
    return model_identity(env)

def inputs():
    paths = {'R2_registry': R2 / 'preregistered_matrix.json'}
    paths.update({f'{name}_initial_state': R2 / 'initials' / name / 'initial_state.json'
                  for name in ('I1', 'I2', 'I3')})
    return {name: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for name, path in paths.items()}

def preregister(output, forensic_receipt):
    receipt = read(forensic_receipt)
    if not receipt.get('passed'):
        raise ValueError('FORENSIC_PRECONDITION_NOT_PASSED')
    previous = read(R2 / 'preregistered_matrix.json')
    schedule = previous['input_schedule']
    if digest(schedule) != previous['input_schedule_sha256']:
        raise ValueError('R2_INPUT_SCHEDULE_HASH_MISMATCH')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    models = {}
    for name in ('S1B', 'P_FULL', 'P_ARM_HAND'):
        env = make_model(name)
        models[name] = identity(env)
        (output / (name + '_diagnostic.xml')).write_text(env.xml)
    registry = {
        'task': 'G4-S2A-R2-R1', 'scope': 'NO_POLICY_DIAGNOSTIC_ONLY',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'starting_commit': '5abe4375c2bea828816a7d2f0ff04e9ee0673ff0',
        'source_sha256': {name: sha(ROOT / name) for name in SOURCES},
        'historical_inputs': inputs(), 'forensics_receipt_path': str(Path(forensic_receipt).resolve()),
        'forensics_receipt_sha256': sha(forensic_receipt),
        'matrix_order': [run[0] for run in RUNS],
        'runs': [{'id': r, 'initial_id': i, 'physical': p, 'controller': c,
                  'passive_arm': passive, 'pool': 'negative_control' if passive else 'primary'}
                 for r, i, p, c, passive in RUNS],
        'steps_per_trajectory': MAX_STEPS, 'maximum_new_integration_steps': BUDGET,
        'maximum_wall_s': 900, 'no_retry': True, 'thresholds': THRESHOLDS,
        'input_schedule': schedule, 'input_schedule_sha256': digest(schedule),
        'models': models, 'mapped_rotational_inertia_bodies': SELECTIONS,
        'model_semantics': 'C0=S1B M at same full q; C1=selected physical plant M at same full q; C affects arm controller only. Frozen finger step-implicit predictor reads physical M.',
        'initial_semantics': 'Original R2 I1/I2/I3 recorded integration states; no new preparation and no resampling.',
        'physics_and_contact_timing': 'Frozen integration actuator evidence and solver/generalized-force arrays observed immediately after mj_step; separate contact force, solver and qacc recomputed post-step by mj_forward. Neither reconstructs historical integration impulses.',
        'policy_calls': 0, 'cpu_receipt_required': True,
    }
    write(output / 'preregistered_matrix.json', registry)
    return 0

def verify_registry(path):
    registry = read(path)
    expected_runs = [{'id': r, 'initial_id': i, 'physical': p, 'controller': c,
                      'passive_arm': passive, 'pool': 'negative_control' if passive else 'primary'}
                     for r, i, p, c, passive in RUNS]
    original_schedule = read(R2 / 'preregistered_matrix.json')['input_schedule']
    if (registry['task'] != 'G4-S2A-R2-R1' or registry['matrix_order'] != [r[0] for r in RUNS]
            or registry['runs'] != expected_runs
            or registry['steps_per_trajectory'] != MAX_STEPS
            or registry['maximum_new_integration_steps'] != BUDGET
            or registry['maximum_wall_s'] != 900 or registry['no_retry'] is not True
            or registry['policy_calls'] != 0 or registry['cpu_receipt_required'] is not True
            or set(registry['source_sha256']) != set(SOURCES)
            or registry['thresholds'] != THRESHOLDS
            or registry['input_schedule'] != original_schedule
            or registry['input_schedule_sha256'] != digest(registry['input_schedule'])
            or registry['historical_inputs'] != inputs()
            or sha(registry['forensics_receipt_path']) != registry['forensics_receipt_sha256']
            or read(registry['forensics_receipt_path']).get('passed') is not True):
        raise ValueError('REGISTRY_OR_HISTORICAL_INPUT_CHANGED')
    for name, expected in registry['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('SOURCE_CHANGED_AFTER_REGISTRATION:' + name)
    return registry

def solver_and_contacts(env):
    data, model = env.data, env.model
    contacts = []
    for index, c in enumerate(data.contact):
        force = np.zeros(6)
        env.mj.mj_contactForce(model, data, index, force)
        contacts.append({'index': index, 'geom1_id': int(c.geom1), 'geom2_id': int(c.geom2),
                         'geom1': model.geom(int(c.geom1)).name, 'geom2': model.geom(int(c.geom2)).name,
                         'distance_m': float(c.dist), 'position_world_m': c.pos.copy(),
                         'frame_row_major': c.frame.copy(), 'contact_force_local': force,
                         'efc_address': int(c.efc_address)})
    n = data.nefc
    return {'stage': 'post_step_mj_forward_not_integration_impulses', 'contacts': contacts,
            'nefc': int(n), 'efc_type': data.efc_type[:n].copy(),
            'efc_id': data.efc_id[:n].copy(), 'efc_force': data.efc_force[:n].copy(),
            'efc_pos': data.efc_pos[:n].copy(), 'efc_vel': data.efc_vel[:n].copy(),
            'qfrc_constraint': data.qfrc_constraint.copy(), 'qfrc_actuator': data.qfrc_actuator.copy(),
            'qacc': data.qacc.copy(), 'qacc_warmstart': data.qacc_warmstart.copy()}

def reject_external_perturbation(env):
    # Viewer synchronization can write GUI perturbations into MjData. Check
    # after synchronization and again at the immediate integration boundary.
    if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied):
        raise RuntimeError('GUI_EXTERNAL_PERTURBATION_REJECTED')

def observed_physics_step(env, arm_torque):
    """Observe one existing mj_step before S1B's post-step mj_forward.

    The module attribute is restored even on failure. The child is single-threaded
    for Python physics; the native viewer thread does not call this Python wrapper.
    No extra integration or changes to the frozen controller are introduced.
    """
    mj = env.mj
    original = mj.mj_step
    observed = []
    def capture(model, data):
        if observed:
            raise RuntimeError('MULTIPLE_MJ_STEP_CALLS_REJECTED')
        original(model, data)
        stage = solver_and_contacts(env)
        stage['stage'] = 'immediately_after_mj_step_before_wrapper_mj_forward'
        stage['timing_caveat'] = ('MuJoCo step-computation solver data, not independently reconstructed '
                                  'historical impulses; compare with explicit post-forward data')
        stage['actuator_force'] = data.actuator_force.copy()
        stage['ctrl'] = data.ctrl.copy()
        stage['time_s'] = float(data.time)
        observed.append(stage)
    mj.mj_step = capture
    try:
        state = env.physics_step(arm_torque, finger_target=0., record=False)
    finally:
        mj.mj_step = original
    if len(observed) != 1:
        raise RuntimeError('EXACTLY_ONE_MJ_STEP_REQUIRED')
    force = observed[0]['actuator_force']
    integrated = state['integration_actuation']
    if not (np.allclose(force[env.arm_actuator_ids], integrated['arm_actuator_force_nm'], atol=1e-12, rtol=0)
            and np.allclose(force[env.finger_actuator_ids], integrated['finger_actuator_force_n'], atol=1e-12, rtol=0)
            and np.allclose(observed[0]['qfrc_actuator'][env.arm_dof_ids], integrated['arm_qfrc_actuator_nm'], atol=1e-12, rtol=0)):
        raise RuntimeError('INTEGRATION_OBSERVER_MISMATCH')
    return state, observed[0]

def trajectory_child(run_id, manifest, output):
    reg = verify_registry(manifest)
    selected = next((x for x in RUNS if x[0] == run_id), None)
    if selected is None:
        raise ValueError('UNKNOWN_REGISTERED_RUN')
    _, initial_id, physical, controller, passive = selected
    parent = Path(manifest).resolve().parent
    if not (parent / 'coordinator_attempt.json').is_file():
        raise ValueError('COORDINATOR_ADMISSION_REQUIRED')
    if Path(output).resolve() != parent / 'trajectories' / run_id:
        raise ValueError('ONCE_ONLY_REGISTERED_OUTPUT_REQUIRED')
    initial_path = R2 / 'initials' / initial_id / 'initial_state.json'
    initial = read(initial_path)
    pair = ModelPair(physical, controller)
    env = pair.plant
    restore_initial(env, initial)
    reference = peg_relative(env)
    p0, q0 = env.tcp_state()
    contract = PhysicsContract()
    contract.reset(p0, q0, env.data.xpos[env.hole_body_id], env.data.xquat[env.hole_body_id],
                   initial['episode_noise_m'], timestamp=float(env.data.time))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    models = {'plant': identity(env), 'M0': identity(pair.shadows['C0']), 'M1': identity(pair.shadows['C1'])}
    write(output / 'identity.json', {
        'task': 'G4-S2A-R2-R1', 'scope': 'NO_POLICY_DIAGNOSTIC_ONLY', 'run_id': run_id,
        'initial_id': initial_id, 'variant': 'PASSIVE_ARM' if passive else ('B' if controller == 'C0' else 'D'),
        'physical_selection': physical, 'controller_selection': controller, 'passive_arm': passive,
        'models': models, 'initial_snapshot': env.snapshot(), 'integration_state': integration_state(env),
        'initial_state_sha256': sha(initial_path), 'input_schedule_sha256': reg['input_schedule_sha256'],
        'source_sha256': reg['source_sha256'], 'initial_ema': contract.action_cache,
    })
    rows = []
    stops = []
    failure = None
    physics_wall = 0.
    visual = None
    losses = 0
    started = time.monotonic()
    start_sim = float(env.data.time)
    attempts = 0
    trace_path = output / 'physics.jsonl'
    try:
        with Visible(env, run_id + ' ' + physical + '+' + controller) as visual:
            write(output / 'gui_admission.json', {'status': 'GUI_OPEN_VISIBLE', 'window': visual.info, 'integration_started': False})
            visual.capture(output / 'initial.png', 0, env.snapshot(), models['plant']['xml_sha256'])
            with trace_path.open('x', encoding='utf-8') as stream:
                for step in range(1, MAX_STEPS + 1):
                    visual.sync(step - 1)
                    reject_external_perturbation(env)
                    if time.monotonic() - started >= 900:
                        raise TimeoutError('DIAGNOSTIC_CHILD_WALL_BUDGET')
                    if step in (1, 9):
                        contract.policy_update(reg['input_schedule']['archived_clipped_actions'][(step - 1) // 8])
                    p, q = env.tcp_state()
                    contract.update_physics(p, q, float(env.data.time))
                    target = fixed_reference(step, p0, q0, contract)
                    matrices = pair.matrices()
                    controller_input = {'q7': env.data.qpos[env.arm_qpos_ids].copy(),
                        'qdot7': env.data.qvel[env.arm_dof_ids].copy(), 'position': p, 'quaternion': q,
                        'linear_fd': contract.linear_fd.copy(), 'angular_fd': contract.angular_fd.copy(),
                        'jacobian': env.controller_jacobian(), 'mass_matrix': matrices['controller_M'],
                        'target_position': target['position_target'], 'target_quaternion': target['quaternion_target']}
                    control = torque_command(**controller_input)
                    selected_torque = np.zeros(7) if passive else control['requested']
                    before = env.snapshot()
                    vbefore = env.data.qvel.copy()
                    warnings = np.array([w.number for w in env.data.warning])
                    reject_external_perturbation(env)
                    line(stream, {'event': 'integration_request', 'run_id': run_id, 'step': step,
                         'time_before_s': float(env.data.time), 'pre_state': before, 'matrices': matrices,
                         'controller_input': controller_input, 'target': target, 'controller': control,
                         'selected_arm_torque_nm': selected_torque, 'pre_forward_solver': solver_and_contacts(env)})
                    attempts += 1
                    t = time.perf_counter()
                    try:
                        reject_external_perturbation(env)
                        state, integration_stage = observed_physics_step(env, selected_torque)
                    except Exception:
                        line(stream, {'event': 'integration_exception', 'step': step, 'raw_state': env.snapshot(),
                             'traceback': traceback.format_exc(), 'warning_counts': [w.number for w in env.data.warning]})
                        raise
                    elapsed = time.perf_counter() - t
                    physics_wall += elapsed
                    state['s2a'] = physical_measurements(env, reference)
                    forces = contact_forces(state)
                    losses = losses + 1 if min(forces.values()) <= THRESHOLDS['positive_pad_force_n'] else 0
                    wc = (np.array([w.number for w in env.data.warning]) - warnings).tolist()
                    stops = stop_reasons(state, forces, losses, wc)
                    if not finite_tree(state):
                        stops.append('NONFINITE_MEASUREMENT')
                    actual = np.asarray(state['integration_actuation']['arm_actuator_force_nm'])
                    expected = np.zeros(7) if passive else np.clip(control['software_limited'], [-87]*4+[-12]*3, [87]*4+[12]*3)
                    if not np.allclose(actual, expected, atol=1e-10, rtol=0):
                        stops.append('INVALID_ACTUATOR_BEHAVIOR')
                    row = {'event': 'physics_result', 'run_id': run_id, 'step': step,
                           'episode_time_s': float(env.data.time)-start_sim, 'physics_wall_s': elapsed,
                           'pre_state': before, 'matrices': matrices, 'controller_input': controller_input,
                           'target': target, 'controller': control, 'selected_arm_torque_nm': selected_torque,
                           'state': state, 'pad_force_n': forces,
                           'qvel_interval_acceleration': (env.data.qvel-vbefore)/(1/120),
                           'qacc_post_forward': env.data.qacc.copy(),
                           'qfrc_constraint_post_forward': env.data.qfrc_constraint.copy(),
                           'qacc_warmstart': env.data.qacc_warmstart.copy(),
                           'integration_stage': integration_stage,
                           'post_forward_solver': solver_and_contacts(env), 'warning_delta': wc, 'stop_reasons': stops}
                    line(stream, row)
                    rows.append(row)
                    visual.sync(step, state, extra='Fixed initial TCP after step 16; archived actions; NO POLICY')
                    time.sleep(.025)
                    if stops or step in (16, MAX_STEPS) or (initial_id == 'I3' and step in (175, 176, 177, 214, 215, 216)):
                        visual.capture(output / f'physics_{step:03d}.png', step, state, models['plant']['xml_sha256'])
                    if stops:
                        break
            visual.sync(len(rows), rows[-1]['state'] if rows else env.snapshot(),
                        extra='STOP ' + ','.join(stops) if stops else 'REGISTERED HORIZON COMPLETE')
            time.sleep(1.)
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    result = {'task': 'G4-S2A-R2-R1', 'scope': 'NO_POLICY_DIAGNOSTIC_ONLY', 'run_id': run_id,
        'initial_id': initial_id, 'physical_selection': physical, 'controller_selection': controller,
        'passive_arm': passive, 'registered_steps': MAX_STEPS, 'integration_requests': attempts,
        'completed_steps': len(rows), 'simulation_duration_s': float(env.data.time)-start_sim,
        'physics_wall_s': physics_wall, 'wall_s': time.monotonic()-started,
        'viewer_sync_wall_s': 0 if visual is None else visual.sync_wall,
        'viewer_sync_count': 0 if visual is None else visual.sync_count,
        'graceful_close': None if visual is None else visual.receipt,
        'runtime_failure': failure, 'early_stop_reasons': stops, 'policy_inferences': 0,
        'max_slip_m': max((r['state']['s2a']['peg_hand_slip_m'] for r in rows), default=0.),
        'max_rotation_slip_rad': max((r['state']['s2a']['peg_hand_slip_rad'] for r in rows), default=0.),
        'max_arm_qvel_rad_s': max((max(abs(np.asarray(r['state']['qvel'])[:7])) for r in rows), default=0.),
        'first_contact_loss_step': next((r['step'] for r in rows if min(r['pad_force_n'].values())<=1e-6), None),
        'max_penetration_m': max((r['state']['s2a']['all_contact_penetration_m'] for r in rows), default=0.),
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'trace_sha256': sha(trace_path) if trace_path.exists() else None,
        'termination': 'RUNTIME_FAILURE' if failure else ('DIAGNOSTIC_EARLY_STOP' if stops else 'REGISTERED_HORIZON_COMPLETE')}
    write(output / 'result.json', result)
    return 0 if failure is None else 2

def coordinate(output, cpu_receipt):
    output = Path(output).resolve()
    manifest = output / 'preregistered_matrix.json'
    registry = verify_registry(manifest)
    cpu = read(cpu_receipt)
    if not cpu.get('passed'):
        raise ValueError('CPU_PRELAUNCH_GATE_NOT_PASSED')
    write(output / 'coordinator_attempt.json', {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'cpu_receipt': str(Path(cpu_receipt).resolve()),
        'cpu_sha256': sha(cpu_receipt), 'policy_calls': 0, 'retry': False})
    started = time.monotonic()
    receipts = []
    for run_id in registry['matrix_order']:
        remaining = 900 - (time.monotonic() - started)
        if remaining <= 0:
            break
        directory = output / 'trajectories' / run_id
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--trajectory', run_id,
               '--manifest', str(manifest), '--output', str(directory)]
        t = time.monotonic()
        timed_out = False
        with (output / (run_id + '_console.txt')).open('x') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                child.kill()
                code = child.wait()
        record = {'id': run_id, 'command': cmd, 'exact_child_exit_code': code,
                  'supervisor_timeout': timed_out, 'wall_s': time.monotonic()-t,
                  'result': read(directory / 'result.json') if (directory / 'result.json').exists() else None}
        write(output / (run_id + '_process.json'), record)
        receipts.append(record)
        print(json.dumps({'id': run_id, 'exit': code, 'result': record['result']}), flush=True)
        if code != 0:
            break
    total = sum(r['result']['integration_requests'] for r in receipts if r['result'])
    summary = {'task': 'G4-S2A-R2-R1', 'receipts': receipts, 'new_integration_requests': total,
        'maximum_steps': BUDGET, 'wall_s': time.monotonic()-started, 'maximum_wall_s': 900,
        'policy_inferences': 0, 'registered_trajectories': len(RUNS),
        'completed_accounting': len(receipts) == len(RUNS),
        'execution_gate_passed': len(receipts) == len(RUNS) and all(r['exact_child_exit_code'] == 0 for r in receipts)
            and total <= BUDGET and time.monotonic()-started <= 900,
        'unstarted_runs': registry['matrix_order'][len(receipts):]}
    write(output / 'summary.json', summary)
    return 0 if summary['execution_gate_passed'] else 2

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregister', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--trajectory')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--cpu-receipt', type=Path)
    parser.add_argument('--forensic-receipt', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.preregister:
        if args.forensic_receipt is None:
            parser.error('--forensic-receipt is required')
        return preregister(args.output, args.forensic_receipt)
    if args.trajectory:
        return trajectory_child(args.trajectory, args.manifest, args.output)
    if args.run:
        return coordinate(args.output, args.cpu_receipt)
    parser.error('Explicit preregistration or once-only execution operation required')

if __name__ == '__main__':
    raise SystemExit(main())
