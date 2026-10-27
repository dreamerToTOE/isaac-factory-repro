#!/usr/bin/env python3
"""Independent, offline numerical verification of R2-R3 recorded solver traces.

No MuJoCo imports, simulation, rendering or policy calls. Constraint addresses
are reviewed against scalar EFC rows, never against contact-list indices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_g4s2a_r2 as core
from scripts import analyze_g4s2a_r2_r1 as r1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_row_count(dim, cone):
    """MuJoCo scalar contact constraints: 1 frictionless, dim elliptic,
    2*(dim-1) pyramidal. Cone is model opt.cone, 0 pyramidal, 1 elliptic.
    """
    if dim not in (1, 3, 4, 6) or cone not in (0, 1):
        raise ValueError('invalid contact dimension or cone')
    return 1 if dim == 1 else dim if cone == 1 else 2 * (dim - 1)


def review_solver(solver, cone):
    """Verify typed EFC ownership and decomposition at integration stage.

    EFC contact id is the contact-list index. The address is a scalar EFC
    offset; excluded contacts must not own EFC rows. Equality force entries
    are generalized constraint-space forces, not impulses or normal forces.
    """
    nefc = int(solver['nefc'])
    fields = ('efc_type', 'efc_id', 'efc_force', 'efc_vel', 'efc_aref', 'efc_pos', 'efc_state')
    arrays = {}
    for key in fields:
        a = np.asarray(solver[key])
        if a.shape != (nefc,) or not np.all(np.isfinite(a)):
            raise ValueError(f'{key}: invalid shape or non-finite data')
        arrays[key] = a
    types = arrays['efc_type'].astype(int)
    ids = arrays['efc_id'].astype(int)
    if not np.array_equal(types, arrays['efc_type']) or not np.array_equal(ids, arrays['efc_id']):
        raise ValueError('non-integral EFC type or ID')
    if np.any((types < 0) | (types > 7)):
        raise ValueError('unknown EFC type')
    contacts = solver['contacts']
    if len(contacts) != int(solver['ncon']):
        raise ValueError('contact count mismatch')
    claimed = set()
    attributed = []
    for index, c in enumerate(contacts):
        if int(c['index']) != index:
            raise ValueError('contact list index mismatch')
        dim, addr = int(c['dim']), int(c['efc_address'])
        if addr < 0:
            if any((types >= 5) & (ids == index)):
                raise ValueError('excluded contact owns scalar rows')
            attributed.append({'contact_index': index, 'address': addr, 'row_count': 0, 'constraint_force': []})
            continue
        count = contact_row_count(dim, cone)
        indices = list(range(addr, addr + count))
        if addr + count > nefc or claimed.intersection(indices):
            raise ValueError('invalid or overlapping contact EFC addresses')
        expected_type = 5 if dim == 1 else 7 if cone == 1 else 6
        if any(types[j] != expected_type or ids[j] != index for j in indices):
            raise ValueError('contact EFC type/ID mismatch')
        if set(np.flatnonzero((types >= 5) & (ids == index)).tolist()) != set(indices):
            raise ValueError('noncontiguous or additional contact rows')
        claimed.update(indices)
        if cone == 1 and 'contact_force_local' in c:
            actual = np.asarray(c['contact_force_local'],dtype=float)
            if actual.shape != (6,) or np.max(np.abs(actual[:dim] - arrays['efc_force'][indices])) > 1e-8:
                raise ValueError('elliptic local contact force differs from EFC rows')
        attributed.append({'contact_index': index, 'address': addr, 'row_count': count,
                           'constraint_force': arrays['efc_force'][indices].tolist()})
    if claimed != set(np.flatnonzero(types >= 5).tolist()):
        raise ValueError('orphan contact EFC rows')
    equality = np.flatnonzero(types == 0)
    out = {'nefc': nefc, 'ncon': len(contacts), 'contact_scalar_rows': len(claimed),
           'equality_scalar_rows': len(equality), 'other_scalar_rows': nefc-len(claimed)-len(equality),
           'contacts': attributed,
           'equality_rows': [{'row': int(j), 'equality_id': int(ids[j]), 'constraint_force': float(arrays['efc_force'][j])} for j in equality],
           'generalized_force_decomposition': None}
    if 'efc_J_dense' in solver:
        J = np.asarray(solver['efc_J_dense'], dtype=float)
        qfrc = np.asarray(solver['qfrc_constraint'], dtype=float)
        if J.shape != (nefc, len(qfrc)) or not np.isfinite(J).all():
            raise ValueError('invalid dense constraint Jacobian')
        force = arrays['efc_force'].astype(float)
        parts = {}
        for label, mask in [('equality', types == 0), ('contact', types >= 5), ('other', (types > 0) & (types < 5))]:
            parts[label] = (J[mask].T @ force[mask]).tolist()
        reconstructed = J.T @ force
        error = float(np.max(np.abs(reconstructed - qfrc))) if qfrc.size else 0.0
        if error > 1e-8:
            raise ValueError(f'constraint generalized force residual {error}')
        out['generalized_force_decomposition'] = {'components': parts, 'max_abs_residual': error}
    return out


def review_accounting(events, results):
    """Pair recorded operations in order; an unmatched last request is UNKNOWN."""
    expected, pending = 1, None
    count = 0
    for position, e in enumerate(events, 1):
        if 'sequence' in e and e['sequence'] != position:
            raise ValueError('noncontiguous durable event sequence')
        name = e.get('event')
        if name == 'integration_request':
            if pending is not None or e['step'] != expected:
                raise ValueError('duplicate or out-of-order request')
            pending = expected
        elif name == 'physics_result':
            if pending is None or e['step'] != pending:
                raise ValueError('result without matching request')
            count += 1
            expected += 1
            pending = None
    if count != results:
        raise ValueError('supervisor result count differs from physics trace')
    return {'requests': count + (pending is not None), 'results': count,
            'unresolved_request_step': pending, 'complete': pending is None}


def run(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    registry = json.loads((source / 'preregistered_matrix.json').read_text())
    historical = json.loads((ROOT / 'evidence/g4_s/s2a_r2/matrix_v2/preregistered_matrix.json').read_text())
    if registry['input_schedule'] != historical['input_schedule'] or registry['thresholds'] != historical['thresholds']:
        raise ValueError('historical schedule or thresholds changed')
    schedule = hashlib.sha256(json.dumps(registry['input_schedule'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if schedule != registry['input_schedule_sha256']:
        raise ValueError('schedule digest mismatch')
    expected = ['BASE_I3_ACTIVE', 'BASE_I3_PASSIVE', 'BASE_I2_ACTIVE']
    if registry['matrix_order'] != expected or registry['steps_per_trajectory'] != 235 or registry['maximum_new_integration_steps'] != 1300:
        raise ValueError('registered experiment scope mismatch')
    reports, recorded_steps, errors = {}, 0, {}
    for item in registry['runs']:
        rid = item['id']
        directory = source / 'trajectories' / rid
        path = directory / 'output'
        required = [path/'identity.json', path/'result.json', path/'physics.jsonl', directory/'events/events.jsonl', directory/'supervisor_status.json']
        if not all(p.exists() for p in required):
            reports[rid] = {'status': 'INCOMPLETE', 'missing': [str(p) for p in required if not p.exists()]}
            errors[rid] = ['missing files']
            continue
        try:
            identity = json.loads((path/'identity.json').read_text())
            result = json.loads((path/'result.json').read_text())
            process = json.loads((directory/'supervisor_status.json').read_text())
            events = core.trace(directory/'events/events.jsonl')
            trace = core.trace(path/'physics.jsonl')
            rows = [r for r in trace if r['event'] == 'physics_result']
            req = [r for r in trace if r['event'] == 'integration_request']
            recorded_steps += len(rows)
            if result['trace_sha256'] != digest(path/'physics.jsonl'):
                raise ValueError('trace digest mismatch')
            if identity['run_id'] != rid or any(identity[k] != item[v] for k,v in [('initial_id','initial_id'),('physical_selection','physical'),('controller_selection','controller'),('passive_arm','passive_arm')]):
                raise ValueError('experiment identity mismatch')
            if identity['models']['plant'] != registry['models']['P_FULL'] or identity['models']['M0'] != registry['models']['S1B'] or identity['models']['M1'] != registry['models']['P_FULL']:
                raise ValueError('model identity mismatch')
            initpath = ROOT/registry['historical_inputs'][item['initial_id']+'_initial_state']['path']
            initial = json.loads(initpath.read_text())
            if digest(initpath) != identity['initial_state_sha256'] or identity['input_schedule_sha256'] != schedule:
                raise ValueError('historical input digest mismatch')
            for key in ('qpos','qvel','time_s','hole_root_position'):
                if core.maximum_error(identity['initial_snapshot'][key], initial['state'][key]) > 1e-12:
                    raise ValueError('initial state mismatch '+key)
            accounting = review_accounting(events, len(rows))
            if not events or events[-1].get('event') != 'child_terminal' or not events[-1].get('terminal'):
                raise ValueError('missing terminal event')
            if events[-1].get('raw_wait_returncode') != process.get('raw_wait_returncode') or events[-1].get('timed_out') != process.get('timed_out'):
                raise ValueError('terminal event differs from process status')
            if len(req) != len(rows) or len(rows) != result['completed_steps'] or result['integration_requests'] != len(req):
                raise ValueError('request/result mismatch in trajectory')
            for request, row in zip(req,rows):
                if request['step'] != row['step']:
                    raise ValueError('trajectory request/result step mismatch')
                for key in ('run_id','pre_state','matrices','target','controller_input','controller','selected_arm_torque_nm'):
                    if request[key] != row[key]: raise ValueError('trajectory request/result mismatch '+key)
            durable = [event for event in events if event['event'] == 'physics_result']
            if len(durable) != len(rows): raise ValueError('durable solver result count mismatch')
            normalized = []
            stage_metrics = []
            for row,event in zip(rows,durable):
                stage = row['integration_stage']
                if stage != event['integration_stage'] or event['step'] != row['step']:
                    raise ValueError('durable integration data differs from trajectory')
                if stage['stage'] != 'immediately_after_mj_step_before_wrapper_mj_forward' or row['post_forward_solver']['stage'] != 'post_step_mj_forward':
                    raise ValueError('solver timing stage incorrect')
                if abs(stage['simulation_end_time_s']-stage['simulation_start_time_s']-identity['timestep_s']) > 1e-11:
                    raise ValueError('simulation step time mismatch')
                if stage['wall_end_time_ns'] < stage['wall_start_time_ns']:
                    raise ValueError('wall time ordering mismatch')
                checked = review_solver(stage, identity['cone'])
                if checked['generalized_force_decomposition'] is None:
                    raise ValueError('missing generalized force decomposition')
                # Sanity check geometry metadata without inferring exact contact surface points.
                for contact in stage['contacts']:
                    frame = np.asarray(contact['frame_row_major'], dtype=float).reshape(3,3)
                    if not np.allclose(frame @ frame.T, np.eye(3), atol=1e-8) or not np.allclose(frame[0],contact['normal_world'], atol=1e-12):
                        raise ValueError('invalid contact frame')
                    expected_rows = list(range(contact['efc_address'], contact['efc_address']+contact_row_count(contact['dim'],identity['cone']))) if contact['efc_address'] >= 0 else []
                    if contact['efc_rows'] != expected_rows:
                        raise ValueError('advertised EFC row indices mismatch')
                normalized.append({**row, 'post_forward_solver': {**row['post_forward_solver'], 'stage': 'post_step_mj_forward_not_integration_impulses'}})
                stage_metrics.append({'step': row['step'], 'solver': checked,
                    'contacts': [{'index':c['index'],'geom1':c['geom1'],'geom2':c['geom2'],'distance_m':c['distance_m'],
                                  'normal_force_n':c['contact_force_local'][0], 'efc_address':c['efc_address'], 'efc_rows':c['efc_rows']} for c in stage['contacts']],
                    'finger_qacc': np.asarray(stage['qacc'])[identity['finger_dof_ids']].tolist(),
                    'finger_qfrc_constraint': np.asarray(stage['qfrc_constraint'])[identity['finger_dof_ids']].tolist(),
                    'solver_niter':stage['solver_niter'], 'solver_iterations':stage['solver_iterations']})
            values, arithmetic = r1.review_rows(identity, normalized, registry['thresholds'])
            schedule_errors = core.inspect_reference_schedule(identity, initial, registry, rows)
            for key,value in arithmetic.items():
                tolerance = 1e-7 if key=='slip_rotation' else 1e-8 if key.startswith('controller_') or key=='interval_acceleration' else 1e-9
                if value > tolerance: raise ValueError(f'arithmetic {key} error {value}')
            if max(schedule_errors.values(), default=0.) > 1e-10: raise ValueError('reference schedule mismatch')
            summary = r1.aggregate(values,registry['thresholds'],235)
            if summary['first_contact_loss_step'] != result['first_contact_loss_step']:
                raise ValueError('contact loss mismatch')
            if process['retry'] or process['raw_wait_returncode'] != 0 or process['timed_out'] or result['runtime_failure'] is not None or not accounting['complete']:
                raise ValueError('incomplete process accounting')
            if any(any(r['warning_delta']) for r in rows): raise ValueError('numerical warning')
            thresholds = registry['thresholds']
            def valid(v):
                return (v['bilateral_positive_contact'] and v['translation_slip_m'] < thresholds['accept_slip_m']
                        and v['rotation_slip_rad'] <= thresholds['accept_rotation_rad']
                        and v['max_arm_velocity_rad_s'] <= thresholds['accept_arm_velocity_rad_s']
                        and v['arm_limit_violation_rad'] <= thresholds['accept_arm_limit_rad']
                        and v['finger_limit_violation_m'] <= thresholds['accept_finger_limit_m']
                        and v['pad_penetration_m'] <= thresholds['accept_pad_penetration_m']
                        and v['assembly_penetration_m'] <= thresholds['accept_assembly_penetration_m'])
            def signature(v):
                return sorted((c['geom1'], c['geom2']) for c in v['contacts'])
            def row_signature(v):
                return sorted((c['geom1'],c['geom2'],len(c['efc_rows'])) for c in v['contacts'] if c['efc_rows'])
            changes = [stage_metrics[i]['step'] for i in range(1,len(stage_metrics)) if signature(stage_metrics[i]) != signature(stage_metrics[i-1])]
            row_changes = [stage_metrics[i]['step'] for i in range(1,len(stage_metrics)) if row_signature(stage_metrics[i]) != row_signature(stage_metrics[i-1])]
            positive_loss = next((x['step'] for x in values if not x['bilateral_positive_contact']), None)
            chronology = {'last_step_meeting_all_physical_gates':next((x['step'] for x in reversed(values) if valid(x)), None),
                'first_physical_gate_failure':next((x['step'] for x in values if not valid(x)),None),
                'contact_manifold_change_steps':changes, 'contact_constraint_row_change_steps':row_changes,
                'first_post_forward_positive_pad_force_loss':positive_loss,
                'first_rotation_over_acceptance':next((x['step'] for x in values if x['rotation_slip_rad'] > thresholds['accept_rotation_rad']),None),
                'first_severe_penetration':next((x['step'] for x in values if x['all_penetration_m'] > thresholds['stop_penetration_m']),None),
                'largest_absolute_finger_interval_acceleration':max(((max(abs(a) for a in x['finger_interval_acceleration_m_s2']), x['step']) for x in values), default=None),
                'abnormal_acceleration_first_step':'INCONCLUSIVE: no preregistered abnormal-acceleration threshold',
                'interpretation':'Changes are set changes between adjacent recorded integration-stage contact snapshots, not proof of causation.'}
            reference = {'BASE_I3_ACTIVE': ROOT/'evidence/g4_s/s2a_r2/matrix_v2/trajectories/I3_B/physics.jsonl.gz',
                         'BASE_I3_PASSIVE': ROOT/'evidence/g4_s/s2a_r2_r1/matrix_v2/trajectories/I3_FULL_PASSIVE_ARM/physics.jsonl',
                         'BASE_I2_ACTIVE': ROOT/'evidence/g4_s/s2a_r2/matrix_v2/trajectories/I2_B/physics.jsonl.gz'}[rid]
            old_rows = [r for r in core.trace(reference) if r['event']=='physics_result']
            common = min(len(rows),len(old_rows))
            qpos_error = [core.maximum_error(rows[i]['state']['qpos'],old_rows[i]['state']['qpos']) for i in range(common)]
            qvel_error = [core.maximum_error(rows[i]['state']['qvel'],old_rows[i]['state']['qvel']) for i in range(common)]
            comparison = {'reference_path':str(reference.relative_to(ROOT)), 'reference_sha256':digest(reference),
                          'common_steps':common, 'reference_steps':len(old_rows),
                          'max_qpos_error':max(qpos_error,default=0.), 'max_qvel_error':max(qvel_error,default=0.),
                          'first_difference_above_1e-10':next((i+1 for i in range(common) if max(qpos_error[i],qvel_error[i])>1e-10),None),
                          'interpretation':'Recorded state comparison only; no claim of bitwise identity beyond the common prefix.'}
            reports[rid] = {'status':'REVIEWED', 'physical_pass':summary['all_stability_gates'], 'metrics':summary, 'chronology':chronology,
                'historical_comparison':comparison,
                'per_step_physical_metrics':values,
                'termination':result['termination'], 'early_stop_reasons':result['early_stop_reasons'],
                'accounting':accounting, 'child_exit_status':process['raw_wait_returncode'],
                'arithmetic_max_errors':arithmetic, 'schedule_errors':schedule_errors,
                'integration_stage_metrics':stage_metrics,
                'trace_sha256':digest(path/'physics.jsonl')}
        except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
            errors[rid] = [str(exc)]
            reports[rid] = {'status':'REVIEW_FAILED', 'error':str(exc)}
    report = {'task':'G4-S2A-R2-R3', 'method':'Independent offline NumPy reconstruction; no simulation, model compilation, GUI or policy.',
              'numerical_review_passed':len(reports)==3 and not errors, 'errors':errors, 'runs':reports,
              'reviewed_physics_rows':recorded_steps, 'no_physics_steps':True, 'no_policy_calls':True,
              'stage_semantics':'Solver arrays from mj_step; qpos/qvel are end state. Constraint forces are not impulses.',
              'historical_unresolved_request':'R2-R1 I3_FULL_C1 remains UNKNOWN; no fabricated result.',
              'script_sha256':digest(__file__), 'runtime_s':time.monotonic()-started}
    (output/'numerical_review.json').write_text(json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
    print(json.dumps({'numerical_review_passed':report['numerical_review_passed'],'reviewed_physics_rows':recorded_steps,'errors':errors}))
    return 0 if report['numerical_review_passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.exit(run(args.source, args.output))


if __name__ == '__main__':
    main()
