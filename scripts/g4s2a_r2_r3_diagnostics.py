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
from scripts.g4s2a_r2_r1_model import (SELECTIONS, ModelPair, make_model,
    model_identity, MAX_STEPS, THRESHOLDS, digest, fixed_reference, stop_reasons)
from scripts.g4s2a_r2_diagnostics import Visible, restore_initial, integration_state
from scripts.g4s1b_contract import PhysicsContract, torque_command
from scripts.g4s2a_smoke import jsonable, physical_measurements, peg_relative, finite_tree
from scripts.g4s2a_r1_viewer import contact_forces

from scripts.g4s2a_r2_r2_supervisor import ChildSupervisor, DurableEventLog

RUNS = (('BASE_I3_ACTIVE', 'I3', 'P_FULL', 'C0', False),
        ('BASE_I3_PASSIVE', 'I3', 'P_FULL', 'C0', True),
        ('BASE_I2_ACTIVE', 'I2', 'P_FULL', 'C0', False))
R2 = ROOT / 'evidence/g4_s/s2a_r2/matrix_v2'
SOURCES = [
    'scripts/g4s2a_r2_r1_model.py', 'scripts/g4s2a_r2_r3_diagnostics.py',
    'scripts/g4s2a_r2_r2_supervisor.py',
    'scripts/g4s2a_r2_model.py', 'scripts/g4s2a_r2_diagnostics.py',
    'scripts/g4s1b_env.py', 'scripts/g4s1b_contract.py', 'scripts/g4s2a_smoke.py',
    'scripts/g4s2a_r1_viewer.py', 'scripts/capture_g4s2a_r1_comparison.py',
]
BUDGET = 1300

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
    for name in ('S1B', 'P_FULL'):
        env = make_model(name)
        models[name] = identity(env)
        (output / (name + '_diagnostic.xml')).write_text(env.xml)
    registry = {
        'task': 'G4-S2A-R2-R3', 'scope': 'NO_POLICY_DIAGNOSTIC_ONLY',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'starting_commit': '155bf275a1b4e70592f7b1bd0da3fe3fcff1e1e8',
        'source_sha256': {name: sha(ROOT / name) for name in SOURCES},
        'historical_inputs': inputs(), 'forensics_receipt_path': str(Path(forensic_receipt).resolve()),
        'forensics_receipt_sha256': sha(forensic_receipt),
        'matrix_order': [run[0] for run in RUNS],
        'runs': [{'id': r, 'initial_id': i, 'physical': p, 'controller': c,
                  'passive_arm': passive, 'pool': 'baseline'}
                 for r, i, p, c, passive in RUNS],
        'steps_per_trajectory': MAX_STEPS, 'maximum_new_integration_steps': BUDGET,
        'maximum_wall_s': 900, 'no_retry': True, 'thresholds': THRESHOLDS,
        'input_schedule': schedule, 'input_schedule_sha256': digest(schedule),
        'models': models, 'mapped_rotational_inertia_bodies': SELECTIONS,
        'model_semantics': 'C0=S1B M at same full q; C1=selected physical plant M at same full q; C affects arm controller only. Frozen finger step-implicit predictor reads physical M.',
        'initial_semantics': 'Original R2 I1/I2/I3 recorded integration states; no new preparation and no resampling.',
        'physics_and_contact_timing': 'Solver forces and constraint arrays captured immediately after mj_step before wrapper mj_forward. qpos/qvel are end-step state, not solver input; forces are not impulses. Post-forward arrays are separate.',
        'policy_calls': 0, 'cpu_receipt_required': True,
    }
    write(output / 'preregistered_matrix.json', registry)
    return 0

def verify_registry(path):
    registry = read(path)
    expected_runs = [{'id': r, 'initial_id': i, 'physical': p, 'controller': c,
                      'passive_arm': passive, 'pool': 'baseline'}
                     for r, i, p, c, passive in RUNS]
    original_schedule = read(R2 / 'preregistered_matrix.json')['input_schedule']
    if (registry['task'] != 'G4-S2A-R2-R3' or registry['matrix_order'] != [r[0] for r in RUNS]
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
    """Snapshot active constraint arrays without advancing or forwarding MjData.

    Contacts and solver forces correspond to the computation stage represented
    by the supplied data, while qpos/qvel after mj_step are end-of-step state.
    Contact efc_address is a row offset, not the contact array index. Row counts
    are checked against the actual EFC type/id array, including elliptic dim.
    """
    data, model, mj = env.data, env.model, env.mj
    n = int(data.nefc)
    typ = np.asarray(data.efc_type[:n]); ids = np.asarray(data.efc_id[:n])
    contacts = []
    for index, c in enumerate(data.contact):
        force = np.zeros(6)
        mj.mj_contactForce(model, data, index, force)
        address, dim = int(c.efc_address), int(c.dim)
        rows = [j for j in range(n) if int(ids[j]) == index and int(typ[j]) in (5, 6, 7)]
        if address >= 0:
            if not rows or rows[0] != address or rows != list(range(address, address + len(rows))):
                raise RuntimeError('CONTACT_EFC_ADDRESS_OR_ID_MISMATCH')
            kind = int(typ[address])
            expected = 1 if kind == 5 else (2 * (dim - 1) if kind == 6 else dim)
            if len(rows) != expected or any(int(typ[j]) != kind for j in rows):
                raise RuntimeError('CONTACT_EFC_ROW_COUNT_MISMATCH')
        elif rows:
            raise RuntimeError('EXCLUDED_CONTACT_HAS_ACTIVE_ROWS')
        contacts.append({'index': index, 'geom1_id': int(c.geom1), 'geom2_id': int(c.geom2),
            'geom1': model.geom(int(c.geom1)).name, 'geom2': model.geom(int(c.geom2)).name,
            'distance_m': float(c.dist), 'position_world_m': c.pos.copy(),
            'frame_row_major': c.frame.copy(), 'normal_world': c.frame[:3].copy(),
            'contact_force_local': force, 'efc_address': address, 'efc_rows': rows,
            'dim': dim, 'exclude': int(c.exclude)})
    if mj.mj_isSparse(model):
        jac = np.zeros((n, model.nv))
        for i in range(n):
            start = int(data.efc_J_rowadr[i]); count = int(data.efc_J_rownnz[i])
            jac[i, data.efc_J_colind[start:start+count]] = data.efc_J[start:start+count]
    else:
        jac = np.asarray(data.efc_J[:n * model.nv]).reshape(n, model.nv).copy()
    counts = np.asarray(data.solver_niter).copy()
    stride = len(data.solver) // len(counts)
    solver = []
    for island in range(int(data.nisland)):
        count = int(counts[island])
        if count > stride:
            raise RuntimeError('SOLVER_ITERATION_COUNT_OUT_OF_RANGE')
        entries = []
        for i in range(count):
            v = data.solver[island * stride + i]
            entries.append({key: getattr(v, key) for key in
                ('improvement', 'gradient', 'lineslope', 'nactive', 'nchange', 'neval', 'nupdate')})
        solver.append({'island': island, 'iteration_count': count, 'iterations': entries})
    return {'stage': 'post_step_mj_forward', 'contacts': contacts, 'ncon': int(data.ncon),
        'nefc': n, 'efc_type': typ.copy(), 'efc_id': ids.copy(), 'efc_force': data.efc_force[:n].copy(),
        'efc_pos': data.efc_pos[:n].copy(), 'efc_vel': data.efc_vel[:n].copy(),
        'efc_aref': data.efc_aref[:n].copy(), 'efc_state': data.efc_state[:n].copy(),
        'efc_J_dense': jac, 'efc_J_originally_sparse': bool(mj.mj_isSparse(model)),
        'equality_rows': [j for j in range(n) if int(typ[j]) == 0],
        'contact_rows': [j for j in range(n) if int(typ[j]) in (5, 6, 7)],
        'qfrc_constraint': data.qfrc_constraint.copy(), 'qfrc_actuator': data.qfrc_actuator.copy(),
        'qacc': data.qacc.copy(), 'qacc_warmstart': data.qacc_warmstart.copy(),
        'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(), 'time_s': float(data.time),
        'solver_niter': counts, 'solver_nnz': data.solver_nnz.copy(), 'nisland': int(data.nisland),
        'solver_iterations': solver, 'cone': int(model.opt.cone), 'timestep_s': float(model.opt.timestep)}

def reject_external_perturbation(env):
    # Viewer synchronization can write GUI perturbations into MjData. Check
    # after synchronization and again at the immediate integration boundary.
    if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied):
        raise RuntimeError('GUI_EXTERNAL_PERTURBATION_REJECTED')

def observed_physics_step(env, arm_torque, ledger, step):
    """Persist the actual step boundary; no extra forward or integration."""
    mj = env.mj
    original = mj.mj_step
    observed = []
    def capture(model, data):
        if observed:
            raise RuntimeError('MULTIPLE_MJ_STEP_CALLS_REJECTED')
        reject_external_perturbation(env)
        start_sim = float(data.time)
        ledger.record_request(step, simulation_time_s=start_sim, request_wall_time_ns=time.time_ns())
        start_wall_ns = time.time_ns()
        try:
            original(model, data)
        except BaseException as exc:
            ledger.record_exception(step, exc)
            raise
        end_wall_ns = time.time_ns()
        stage = solver_and_contacts(env)
        stage.update({'stage': 'immediately_after_mj_step_before_wrapper_mj_forward',
            'timing_caveat': 'qpos/qvel are end state; solver arrays and actuator force reflect mj_step computation, not an impulse',
            'simulation_start_time_s': start_sim, 'simulation_end_time_s': float(data.time),
            'wall_start_time_ns': start_wall_ns, 'wall_end_time_ns': end_wall_ns,
            'actuator_force': data.actuator_force.copy(), 'ctrl': data.ctrl.copy()})
        ledger.record_result(step, integration_stage=safe_value(stage))
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
    if Path(output).resolve() != parent / 'trajectories' / run_id / 'output':
        raise ValueError('ONCE_ONLY_REGISTERED_OUTPUT_REQUIRED')
    initial_path = R2 / 'initials' / initial_id / 'initial_state.json'
    initial = read(initial_path)
    pair = ModelPair(physical, controller)
    env = pair.plant
    if env.mj.__version__ != '3.5.0':
        raise ValueError('MUJOCO_VERSION_MISMATCH')
    if identity(env) != reg['models']['P_FULL']:
        raise ValueError('REGISTERED_PLANT_IDENTITY_MISMATCH')
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
        'task': 'G4-S2A-R2-R3', 'scope': 'NO_POLICY_DIAGNOSTIC_ONLY', 'run_id': run_id,
        'initial_id': initial_id, 'variant': 'PASSIVE_ARM' if passive else ('B' if controller == 'C0' else 'D'),
        'physical_selection': physical, 'controller_selection': controller, 'passive_arm': passive,
        'models': models, 'initial_snapshot': env.snapshot(), 'integration_state': integration_state(env),
        'initial_state_sha256': sha(initial_path), 'input_schedule_sha256': reg['input_schedule_sha256'],
        'mujoco_version': env.mj.__version__, 'cone': int(env.model.opt.cone),
        'timestep_s': float(env.model.opt.timestep), 'arm_dof_ids': env.arm_dof_ids,
        'finger_dof_ids': env.finger_dof_ids, 'finger_qpos_ids': env.finger_qpos_ids,
        'finger_actuator_ids': env.finger_actuator_ids, 'actuator_forcerange': env.model.actuator_forcerange.copy(),
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
    event_dir = os.environ.get('FACTORY_DURABLE_EVENT_DIR')
    if not event_dir or Path(event_dir).resolve() != output.parent / 'events':
        raise ValueError('EXACT_DURABLE_EVENT_DIRECTORY_REQUIRED')
    ledger = DurableEventLog(event_dir, run_id=run_id)
    try:
        with Visible(env, initial_id + '_' + run_id + ' ' + physical + '+' + controller) as visual:
            write(output / 'gui_admission.json', {'status': 'GUI_OPEN_VISIBLE', 'window': visual.info, 'integration_started': False, 'DISPLAY': os.environ.get('DISPLAY'), 'XAUTHORITY': os.environ.get('XAUTHORITY'), 'mujoco_version': env.mj.__version__})
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
                        state, integration_stage = observed_physics_step(env, selected_torque, ledger, step)
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
    result = {'task': 'G4-S2A-R2-R3', 'scope': 'NO_POLICY_DIAGNOSTIC_ONLY', 'run_id': run_id,
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
    if not read(cpu_receipt).get('passed'):
        raise ValueError('CPU_PRELAUNCH_GATE_NOT_PASSED')
    write(output / 'coordinator_attempt.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
        'cpu_receipt': str(Path(cpu_receipt).resolve()), 'cpu_sha256': sha(cpu_receipt),
        'policy_calls': 0, 'retry': False})
    started = time.monotonic()
    receipts = []
    for run_id in registry['matrix_order']:
        remaining = 900 - (time.monotonic() - started)
        if remaining <= 0:
            break
        directory = output / 'trajectories' / run_id
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--trajectory', run_id,
               '--manifest', str(manifest), '--output', str(directory / 'output')]
        supervisor = ChildSupervisor(directory, cmd, cwd=ROOT,
            env={'FACTORY_DURABLE_EVENT_DIR': str(directory / 'events')}, timeout_s=remaining)
        receipt = supervisor.run()
        receipts.append(receipt)
        if receipt['raw_wait_returncode'] != 0 or receipt['timed_out']:
            break
    result_paths = [output / 'trajectories' / name / 'output/result.json' for name in registry['matrix_order']]
    results = [read(path) for path in result_paths if path.exists()]
    total = sum(item['integration_requests'] for item in results)
    summary = {'task': 'G4-S2A-R2-R3', 'receipts': receipts,
        'new_integration_requests_known_from_results': total, 'maximum_steps': BUDGET,
        'wall_s': time.monotonic()-started, 'maximum_wall_s': 900, 'policy_inferences': 0,
        'registered_trajectories': len(RUNS), 'completed_accounting': len(results) == len(RUNS),
        'execution_gate_passed': len(results) == len(RUNS) and len(receipts) == len(RUNS)
            and all(r['raw_wait_returncode'] == 0 and not r['timed_out'] for r in receipts)
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
