#!/usr/bin/env python3
"""Independent offline review of native-gripper traces. Never runs physics or a policy.

Integration-stage forces are solver forces, not impulses. Positions/velocities
in that stage are the end-of-step state; actuator force is evaluated from the
recorded *pre-step* configuration and velocity. Post-forward contacts are a
different phase and must not be attributed to the preceding solve.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_g4s2a_r2 as reference
from scripts.analyze_g4s2a_r2_r3 import review_solver, review_accounting

DT = 1. / 120.
T = {'positive_pad_force_n': 1e-6, 'accept_slip_m': .001,
     'accept_rotation_rad': .05, 'accept_arm_velocity_rad_s': 3.,
     'accept_arm_limit_rad': 1e-6, 'accept_finger_limit_m': 1e-6,
     'accept_pad_penetration_m': 50e-6, 'accept_assembly_penetration_m': 10e-6,
     'stop_slip_m': .005, 'stop_rotation_rad': .2,
     'stop_arm_velocity_rad_s': 15., 'stop_arm_limit_rad': 1e-5,
     'stop_finger_limit_m': 1e-4, 'stop_penetration_m': .001,
     'stop_consecutive_contact_loss': 3}


def finite(a, shape=None):
    x = np.asarray(a, dtype=float)
    if (shape is not None and x.shape != shape) or not np.isfinite(x).all():
        raise ValueError('NONFINITE_OR_INVALID_SHAPE')
    return x


def close(a, b, label, tol=1e-8):
    a, b = finite(a), finite(b)
    if a.shape != b.shape or (a.size and np.max(np.abs(a-b)) > tol):
        raise ValueError(f'{label}_MISMATCH')


def verify_transmission(qpos, qvel, ctrl, actuator_force, generalized_force,
                        finger_qpos_ids, finger_dof_ids, moment, tendon_length,
                        tendon_velocity, kp=3750., kv=86.5, force_limit=40.):
    """Independent virtual-work and affine actuator test for unit sum tendon.

    Input qpos/qvel and tendon state must be from immediately before mj_step.
    The generalized force is the integration-stage value. Assumes no other
    actuator acts on either finger; the static audit must verify that premise.
    """
    q, v = finite(qpos), finite(qvel)
    qi, vi = np.asarray(finger_qpos_ids, dtype=int), np.asarray(finger_dof_ids, dtype=int)
    if qi.shape != (2,) or vi.shape != (2,) or len(set(qi)) != 2 or len(set(vi)) != 2:
        raise ValueError('INVALID_FINGER_ADDRESSES')
    m = finite(moment, (len(v),))
    expected_moment = np.zeros(len(v)); expected_moment[vi] = 1.
    close(m, expected_moment, 'MOMENT_ARMS', 1e-12)
    close(tendon_length, np.sum(q[qi]), 'TENDON_LENGTH', 1e-12)
    close(tendon_velocity, np.sum(v[vi]), 'TENDON_VELOCITY', 1e-12)
    if not math.isfinite(float(ctrl)) or not 0. <= ctrl <= .08:
        raise ValueError('CONTROL_OUTSIDE_TENDON_RANGE')
    if kp != 3750. or kv != 86.5 or force_limit != 40.:
        raise ValueError('ACTUATOR_CONFIGURATION_CHANGED')
    raw = kp*(float(ctrl)-float(tendon_length))-kv*float(tendon_velocity)
    expected = float(np.clip(raw, -force_limit, force_limit))
    close(actuator_force, expected, 'ACTUATOR_FORCE', 1e-7)
    forces = finite(generalized_force, (len(v),))
    close(forces[vi], expected*m[vi], 'GENERALIZED_FINGER_FORCE', 1e-7)
    return {'control_m': float(ctrl), 'tendon_length_m': float(tendon_length),
            'tendon_velocity_m_s': float(tendon_velocity), 'unclipped_output_n': raw,
            'clipped_output_n': expected, 'finger_generalized_n': forces[vi].tolist(),
            'moment_arms': m[vi].tolist(), 'clipped': abs(raw) > force_limit}


def summarize_preparation(metrics):
    """Classify full transient and final window independently of runner summary."""
    if len(metrics) != 30:
        # Short but properly stopped preparations are physical failures, not
        # successful or complete 30-step preparations. Other completeness
        # questions are handled separately by durable process accounting.
        return {'reset_admission_status': 'FAIL',
                'preparation_contact_quality_status': 'FAIL',
                'reason': 'fewer than 30 recorded steps'}
    final = metrics[-10:]
    final_pos = finite(final[-1]['relative_position_m'], (3,))
    final_rot = finite(final[-1]['relative_rotation'], (3, 3))
    displacement = max(float(np.linalg.norm(finite(x['relative_position_m'], (3,))-final_pos)) for x in final)
    rotation = max(float(math.acos(np.clip((np.trace(finite(x['relative_rotation'], (3,3)).T @ final_rot)-1)/2,-1,1))) for x in final)
    no_limits = all(x['arm_limit_violation_rad'] <= T['accept_arm_limit_rad'] and
                    x['finger_limit_violation_m'] <= T['accept_finger_limit_m'] for x in metrics)
    no_warnings = all(not any(x['warning_delta']) for x in metrics)
    contact = all(x['bilateral'] for x in final)
    no_final_interference = all(not x['interference'] for x in final)
    admission = contact and displacement < T['accept_slip_m'] and rotation <= T['accept_rotation_rad'] and no_limits and no_final_interference
    peak = max(x['pad_penetration_m'] for x in metrics)
    peak_final = max(x['pad_penetration_m'] for x in final)
    assembly = max(x['assembly_penetration_m'] for x in metrics)
    quality = (admission and no_warnings and peak <= T['accept_pad_penetration_m']
               and assembly <= T['accept_assembly_penetration_m']
               and max(x['arm_speed_rad_s'] for x in metrics) <= T['accept_arm_velocity_rad_s'])
    return {'reset_admission_status': 'PASS' if admission else 'FAIL',
            'preparation_contact_quality_status': 'PASS' if quality else 'FAIL',
            'pad_penetration_max_m': peak, 'pad_penetration_final10_max_m': peak_final,
            'assembly_penetration_max_m': assembly, 'bilateral_final10': contact,
            'final10_translation_stability_m': displacement,
            'final10_rotation_stability_rad': rotation,
            'joint_limits_pass': no_limits, 'numerical_warnings_absent': no_warnings,
            'final_window_interference_absent': no_final_interference}


def raw_contact_metrics(state):
    contacts = state['contacts']
    forces = reference.pad_forces(contacts)
    all_depth, assembly = reference.penetrations(contacts)
    pad = reference.pad_penetration(contacts)
    interference = []
    for c in contacts:
        pair = (c['geom1'], c['geom2'])
        if 'peg_collision' in pair:
            other = pair[0] if pair[1] == 'peg_collision' else pair[1]
            if not any(other == f'{side}_finger_geom_{i}' for side in ('left', 'right') for i in range(3, 8)):
                interference.append(':'.join(pair))
        elif float(c['distance_m']) < -1e-6:
            interference.append(':'.join(pair))
    return forces, all_depth, assembly, pad, interference


def step_stop_reasons(metric, loss_count):
    stops = []
    if any(metric['warning_delta']): stops.append('NUMERICAL_SOLVER_WARNING')
    checks = ((metric['translation_slip_m'], 'stop_slip_m', 'PEG_ESCAPE_SLIP'),
              (metric['rotation_slip_rad'], 'stop_rotation_rad', 'PEG_ESCAPE_ROTATION'),
              (metric['arm_limit_violation_rad'], 'stop_arm_limit_rad', 'ARM_LIMIT'),
              (metric['finger_limit_violation_m'], 'stop_finger_limit_m', 'FINGER_LIMIT'),
              (metric['all_penetration_m'], 'stop_penetration_m', 'SEVERE_PENETRATION'))
    stops += [label for value, threshold, label in checks if value > T[threshold]]
    if metric['arm_speed_rad_s'] > T['stop_arm_velocity_rad_s']: stops.append('IMPLAUSIBLE_ARM_VELOCITY')
    if loss_count >= T['stop_consecutive_contact_loss']: stops.append('SUSTAINED_CONTACT_LOSS')
    return stops


def diagnose(metrics):
    if not metrics:
        return {'status':'INCOMPLETE', 'steps':0}
    valid = (len(metrics) == 235 and all(x['bilateral'] and
             x['translation_slip_m'] < T['accept_slip_m'] and x['rotation_slip_rad'] <= T['accept_rotation_rad'] and
             x['arm_speed_rad_s'] <= T['accept_arm_velocity_rad_s'] and
             x['arm_limit_violation_rad'] <= T['accept_arm_limit_rad'] and
             x['finger_limit_violation_m'] <= T['accept_finger_limit_m'] and
             x['pad_penetration_m'] <= T['accept_pad_penetration_m'] and
             x['assembly_penetration_m'] <= T['accept_assembly_penetration_m'] and
             not any(x['warning_delta']) for x in metrics))
    return {'status': 'PASS' if valid else 'FAIL', 'steps':len(metrics),
            'first_contact_loss_step': next((x['step'] for x in metrics if not x['bilateral']),None),
            'maximum_pad_penetration_m': max(x['pad_penetration_m'] for x in metrics),
            'maximum_translation_slip_m': max(x['translation_slip_m'] for x in metrics),
            'maximum_rotation_slip_rad': max(x['rotation_slip_rad'] for x in metrics)}


def review_stage(stage, expected_label):
    if stage['stage'] != expected_label:
        raise ValueError('WRONG_SOLVER_STAGE')
    # An empty JSON matrix is [] and loses its second dimension, so handle the
    # zero-row system explicitly rather than pretending it is shape (0, nv).
    if int(stage['nefc']) == 0:
        if any(stage[key] for key in ('efc_type','efc_id','efc_force','efc_vel','efc_aref','efc_pos','efc_state','efc_J_dense')):
            raise ValueError('NONEMPTY_ZERO_CONSTRAINT_ARRAY')
        if any(int(c['efc_address']) >= 0 for c in stage['contacts']):
            raise ValueError('ZERO_EFC_WITH_ACTIVE_CONTACT')
        close(stage['qfrc_constraint'], np.zeros(len(stage['qfrc_constraint'])), 'ZERO_EFC_FORCE')
        return {'nefc':0,'ncon':int(stage['ncon']),'equality_scalar_rows':0}
    return review_solver(stage, int(stage['cone']))


def above_hole(state, hole_quat=(1.,0.,0.,0.)):
    """Conservative full-cylinder z envelope from world body poses, not success."""
    rot = reference.rotation_matrix(hole_quat)
    peg_rot = reference.rotation_matrix(state['peg_quaternion_wxyz'])
    rel = rot.T @ (finite(state['peg_root_position'],(3,)) - finite(state['hole_root_position'],(3,)))
    axis = rot.T @ peg_rot[:,2]
    radial_extent = .003993*math.sqrt(max(0.,1.-float(axis[2])**2))
    zmin = min(float(rel[2]),float(rel[2]+.05*axis[2]))-radial_extent
    return {'zmin_m':zmin, 'entirely_above_opening':bool(zmin>=.025),
            'tcp_above_m':bool(float(state['tcp_position_world'][2])>.075)}


def review_rows(identity, rows, prep):
    qidx = identity['finger_qpos_ids']; vidx = identity['finger_dof_ids']
    aqidx = identity['arm_qpos_ids']; avidx = identity['arm_dof_ids']
    fid = int(identity['finger_actuator_id'])
    close(identity['kp'], 3750., 'KP', 0.)
    close(identity['kv'], 86.5, 'KV', 0.)
    close(identity['force_limit_n'], 40., 'FORCE_LIMIT', 0.)
    close(identity['timestep'], DT, 'TIMESTEP', 1e-15)
    close(identity['finger_joint_range'], [[0.,.04],[0.,.04]], 'FINGER_LIMITS',1e-15)
    previous = identity['initial']; reference_state = identity['initial']
    metrics, loss_count = [], 0
    for step, row in enumerate(rows,1):
        if row['event']!='physics_result' or row['step']!=step:
            raise ValueError('INVALID_TRACE_ORDER')
        if row['phase'] != ('PREPARATION' if prep else 'DIAGNOSTIC'):
            raise ValueError('INVALID_PHASE')
        pre, state = row['pre_state'],row['state']
        for key in ('qpos','qvel','tcp_position_world','tcp_quaternion_wxyz','peg_root_position','peg_quaternion_wxyz'):
            close(pre[key], previous[key], 'DISCONTINUITY_'+key,1e-10)
        close(float(state['time_s'])-float(pre['time_s']),DT,'TIME_INCREMENT',1e-11)
        stage, post = row['integration_stage'], row['post_forward_solver']
        review_stage(stage,'immediately_after_mj_step_before_wrapper_mj_forward')
        review_stage(post,'post_step_mj_forward')
        close(stage['simulation_start_time_s'], pre['time_s'], 'START_TIME',1e-11)
        close(stage['simulation_end_time_s'], state['time_s'], 'END_TIME',1e-11)
        close(stage['qpos'], state['qpos'], 'INTEGRATED_QPOS',1e-11)
        close(stage['qvel'], state['qvel'], 'INTEGRATED_QVEL',1e-11)
        close(post['qpos'], state['qpos'], 'POST_QPOS',1e-11)
        close(post['qvel'], state['qvel'], 'POST_QVEL',1e-11)
        if len(post['contacts']) != len(state['contacts']):
            raise ValueError('POST_FORWARD_CONTACT_COUNT')
        for a,b in zip(post['contacts'],state['contacts']):
            if (a['geom1'],a['geom2'])!=(b['geom1'],b['geom2']):
                raise ValueError('POST_FORWARD_CONTACT_PAIR')
            close(a['distance_m'],b['distance_m'],'POST_CONTACT_DISTANCE',1e-12)
            close(a['contact_force_local'][0],b['normal_force_n'],'POST_CONTACT_FORCE',1e-9)
        gp = pre['gripper']; gs = stage['gripper']
        control = float(stage['ctrl'][fid]); force = float(stage['actuator_force'][fid])
        close(control,0.,'TENDON_TARGET',0.)
        close(gs['control_m'],control,'OBSERVED_CONTROL',0.)
        close(gs['actuator_force_n'],force,'OBSERVED_ACTUATOR_FORCE',1e-12)
        force_result = verify_transmission(pre['qpos'],pre['qvel'],control,force,
                 stage['qfrc_actuator'],qidx,vidx,gs['moment_arm_by_dof'],
                 gp['tendon_length_m'],gp['tendon_velocity_m_s'],
                 identity['kp'],identity['kv'],identity['force_limit_n'])
        close(gs['finger_generalized_force_n'],force_result['finger_generalized_n'],'RECORDED_FINGER_FORCE',1e-9)
        if prep:
            close(row['selected_arm_torque_nm'],[0.]*7,'PREPARATION_ARM_COMMAND',1e-12)
        # Force aggregation and geometry independently calculated from raw post-forward contacts.
        pads, all_depth, assembly, pad, interference = raw_contact_metrics(state)
        close([pads['left'],pads['right']], [row['pad_force_n']['left'],row['pad_force_n']['right']], 'PAD_FORCE',1e-9)
        s2a = state['s2a']
        trans,rotation = reference.slip(state,reference_state)
        relative,relative_rot = reference.relative_peg(state)
        close(relative,s2a['relative_peg_hand_m'],'RELATIVE_POSITION',1e-9)
        close(relative_rot,s2a['relative_peg_hand_rotation'],'RELATIVE_ROTATION',1e-9)
        if not prep:
            close(trans,s2a['peg_hand_slip_m'],'TRANSLATION_SLIP',1e-9)
            close(rotation,s2a['peg_hand_slip_rad'],'ROTATION_SLIP',1e-7)
        close(all_depth,s2a['all_contact_penetration_m'],'TOTAL_PENETRATION',1e-12)
        close(assembly,s2a['peg_hole_penetration_m'],'ASSEMBLY_PENETRATION',1e-12)
        arm_limit = reference.limit_violation(state['qpos'],aqidx,identity['arm_joint_range'])
        finger_limit = reference.limit_violation(state['qpos'],qidx,identity['finger_joint_range'])
        close(arm_limit,s2a['arm_joint_limit_violation_rad'],'ARM_LIMIT',1e-12)
        close(finger_limit,s2a['finger_joint_limit_violation_m'],'FINGER_LIMIT',1e-12)
        speed = float(np.max(np.abs(finite(state['qvel'])[np.asarray(avidx,dtype=int)])))
        bilateral = min(pads.values()) > T['positive_pad_force_n']
        loss_count = 0 if bilateral else loss_count+1
        qfinger=finite(state['qpos'])[np.asarray(qidx,dtype=int)]
        metric={'step':step,'relative_position_m':relative.tolist(), 'relative_rotation':relative_rot.tolist(),
                'translation_slip_m':trans,'rotation_slip_rad':rotation,
                'arm_limit_violation_rad':arm_limit,'finger_limit_violation_m':finger_limit,
                'arm_speed_rad_s':speed,'bilateral':bilateral,'left_pad_force_n':pads['left'],
                'right_pad_force_n':pads['right'],'all_penetration_m':all_depth,
                'pad_penetration_m':pad,'assembly_penetration_m':assembly,
                'recorded_pad_penetration_m':row['pad_penetration_m'],
                'interference':interference,'warning_delta':row['warning_delta'],
                'finger_qpos_m':qfinger.tolist(),'finger_asymmetry_m':float(qfinger[0]-qfinger[1]),
                'actuation':force_result}
        if prep:
            stops=[]
            for bad,code in ((all_depth>T['stop_penetration_m'],'SEVERE_PENETRATION'),
                (arm_limit>T['stop_arm_limit_rad'],'ARM_LIMIT'),(finger_limit>T['stop_finger_limit_m'],'FINGER_LIMIT'),
                (speed>T['stop_arm_velocity_rad_s'],'ARM_VELOCITY'),(any(row['warning_delta']),'NUMERICAL_WARNING')):
                if bad: stops.append(code)
        else:
            stops=step_stop_reasons(metric,loss_count)
        if sorted(stops)!=sorted(row['stop_reasons']):
            raise ValueError(f'EARLY_STOP_MISMATCH_AT_STEP_{step}:{stops}:{row["stop_reasons"]}')
        if stops and step!=len(rows): raise ValueError('CONTINUED_AFTER_STOP')
        metric['early_stop_reasons']=stops
        metrics.append(metric); previous=state
    return metrics


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_differences(actual, expected):
    differences={}
    for key in ('qpos','qvel','tcp_position_world','tcp_quaternion_wxyz','peg_root_position','peg_quaternion_wxyz'):
        a,b=finite(actual[key]),finite(expected[key])
        if a.shape!=b.shape: raise ValueError('PRE_CLOSURE_SHAPE_MISMATCH:'+key)
        delta=np.abs(a-b)
        maximum=float(np.max(delta)) if delta.size else 0.
        if maximum>1e-10:
            differences[key]={'max_abs_difference':maximum,'differing_indices':np.flatnonzero(delta>1e-10).tolist(),
                              'registered':b.tolist(),'actual_pre_step':a.tolist()}
    return differences


def review_xml_static(native_xml, baseline_xml):
    """Independently compare P_FULL semantics and inspect the sole transmission."""
    native, base = ET.fromstring(native_xml), ET.fromstring(baseline_xml)
    ignored = {'actuator', 'equality', 'tendon'}
    stable_native = [ET.tostring(n) for n in native if n.tag not in ignored]
    stable_base = [ET.tostring(n) for n in base if n.tag not in ignored]
    if stable_native != stable_base:
        raise ValueError('P_FULL_NON_GRIPPER_XML_CHANGED')
    na, ba = native.find('actuator'), base.find('actuator')
    if na is None or ba is None: raise ValueError('ACTUATORS_MISSING')
    for i in range(1, 8):
        name=f'arm_torque_{i}'
        a,b=na.find(f"motor[@name='{name}']"),ba.find(f"motor[@name='{name}']")
        if a is None or b is None or ET.tostring(a)!=ET.tostring(b):
            raise ValueError('ARM_ACTUATOR_CHANGED:'+name)
    if len(na)!=8: raise ValueError('UNEXPECTED_ACTUATOR_COUNT')
    tendon=native.find('tendon')
    if tendon is None or len(tendon)!=1 or tendon[0].tag!='fixed':
        raise ValueError('EXPECTED_ONE_FIXED_TENDON')
    fixed=tendon[0]
    if [(c.tag,c.get('joint'),c.get('coef')) for c in fixed] != [
            ('joint','finger_joint1','1'),('joint','finger_joint2','1')]:
        raise ValueError('INVALID_TENDON_COEFFICIENTS')
    pos=na.find("position[@name='gripper_v1_position']")
    if pos is None or pos.get('tendon')!=fixed.get('name'):
        raise ValueError('WRONG_ACTUATOR_TRANSMISSION')
    if float(pos.get('kp'))!=3750. or float(pos.get('kv'))!=86.5:
        raise ValueError('UNREGISTERED_ACTUATOR_GAINS')
    if pos.get('ctrllimited')!='true' or [float(x) for x in pos.get('ctrlrange','').split()] != [0.,.08]:
        raise ValueError('INVALID_CONTROL_LIMIT')
    if pos.get('forcelimited')!='true' or [float(x) for x in pos.get('forcerange','').split()] != [-40.,40.]:
        raise ValueError('INVALID_FORCE_LIMIT')
    equality=native.find('equality')
    if equality is not None and len(equality): raise ValueError('UNEXPECTED_EQUALITY_CONSTRAINT')
    peg=native.find(".//body[@name='peg']")
    if peg is None or peg.find("freejoint[@name='peg_free']") is None:
        raise ValueError('PEG_FREE_JOINT_MISSING')
    return {'p_full_non_gripper_xml_unchanged':True,'arm_actuators_unchanged':7,
            'physical_peg_free_joint':True,'unit_sum_tendon':True,'peg_weld_absent':True,
            'native_position_control_range_m':[0.,.08], 'force_limit_n':[-40.,40.]}


def review_run(directory, rid):
    output=directory/'output'
    required=[output/'identity.json',output/'result.json',output/'physics.jsonl',
              directory/'events/events.jsonl',directory/'supervisor_status.json']
    if rid.startswith('PREP_'): required.append(output/'preparation_gate.json')
    missing=[str(p) for p in required if not p.is_file()]
    if missing: return {'status':'INCOMPLETE','missing':missing}
    identity=json.loads((output/'identity.json').read_text())
    result=json.loads((output/'result.json').read_text())
    process=json.loads((directory/'supervisor_status.json').read_text())
    if identity['run'] != rid or result['run'] != rid: raise ValueError('RUN_ID_MISMATCH')
    events=reference.trace(directory/'events/events.jsonl')
    all_rows=reference.trace(output/'physics.jsonl')
    rows=[r for r in all_rows if r['event']=='physics_result']
    accounting=review_accounting(events,len(rows))
    durable=[e for e in events if e['event']=='physics_result']
    if len(durable)!=len(rows): raise ValueError('DURABLE_RESULT_COUNT')
    for row,event in zip(rows,durable):
        if row['step']!=event['step'] or row['integration_stage']!=event['integration_stage']:
            raise ValueError('DURABLE_INTEGRATION_MISMATCH')
    if not events or events[-1].get('event')!='child_terminal' or not events[-1].get('terminal'):
        raise ValueError('MISSING_TERMINAL_EVENT')
    if events[-1].get('raw_wait_returncode')!=process.get('raw_wait_returncode') or events[-1].get('timed_out')!=process.get('timed_out'):
        raise ValueError('TERMINAL_STATUS_MISMATCH')
    if not accounting['complete'] or process.get('retry') or process.get('timed_out') or process.get('raw_wait_returncode')!=0 or result['runtime_failure'] is not None:
        return {'status':'INCOMPLETE','accounting':accounting,'process':process,'runtime_failure':result['runtime_failure'],
                'reason':'child failure, timeout or unresolved integration; physical success is not inferred'}
    if result['trace_sha256']!=sha(output/'physics.jsonl') or result['completed_steps']!=len(rows):
        raise ValueError('RESULT_TRACE_MISMATCH')
    prep=rid.startswith('PREP_')
    if prep and rows:
        # Independent admission provenance must be checked at the actual
        # integration boundary, not only before opening/syncing the GUI.
        pre=rows[0]['pre_state']; expected=identity['initial']
        differences=initial_differences(pre,expected)
        if differences:
            return {'status':'INVALID_INITIAL_STATE','accounting':accounting,'differences':differences,
                    'reason':'actual integration input differs from registered pre-closure initial state; no physics result is attributed to that registered condition',
                    'raw_wait_returncode':process['raw_wait_returncode']}
    metrics=review_rows(identity,rows,prep)
    if not prep:
        return {'status':'COMPLETE','accounting':accounting,'independent_diagnostic':diagnose(metrics),'metrics':metrics}
    # Origin and final placement are evaluated independently from world pose.
    r2=ROOT/'evidence/g4_s/s2a_r2/matrix_v2/initials'/rid[-2:]/'admission.json'
    if identity['initial_sha256']!=sha(r2): raise ValueError('PRE_CLOSURE_IDENTITY_MISMATCH')
    original=json.loads(r2.read_text())['initial']
    for key in ('qpos','qvel','time_s','peg_root_position','peg_quaternion_wxyz','hole_root_position'):
        close(identity['initial'][key],original[key],'PRE_CLOSURE_'+key,1e-12)
    draw=json.loads((r2.parent/'reset_draw.json').read_text())
    initial_above=above_hole(identity['initial'],draw['hole_quaternion_wxyz'])
    initial_forbidden=[c for c in identity['placement']['preclosure_contacts'] if c['distance_m'] < -1e-6]
    final_above=above_hole(rows[-1]['state'], rows[-1]['state']['s2a']['hole_quaternion_wxyz']) if rows else None
    summary=summarize_preparation(metrics)
    if not initial_above['entirely_above_opening'] or not initial_above['tcp_above_m'] or initial_forbidden:
        summary['reset_admission_status']='FAIL';summary['preparation_contact_quality_status']='FAIL'
    if final_above and (not final_above['entirely_above_opening'] or not final_above['tcp_above_m']):
        summary['reset_admission_status']='FAIL';summary['preparation_contact_quality_status']='FAIL'
    runner_gate=json.loads((output/'preparation_gate.json').read_text())
    disagreement={}
    for checked,original in [('reset_admission_status','reset_admission'),('preparation_contact_quality_status','physical_quality')]:
        if summary[checked] != runner_gate[original]:
            disagreement[checked]={'independent':summary[checked], 'runner':runner_gate[original]}
    return {'status':'COMPLETE','accounting':accounting,'initial_above':initial_above,'final_above':final_above,
            'independent_peg_scope':'model structural static audit required; no physics inference',
            'independent_preparation':summary,'runner_gate_disagreements':disagreement,'metrics':metrics}


def review_matrix(matrix):
    matrix=Path(matrix)
    reg=json.loads((matrix/'preregistered_matrix.json').read_text())
    if reg['runs'] != ['PREP_I2','PREP_I3','DIAG_I2','DIAG_I3'] or reg['thresholds']!=T or reg['max_steps']!=530 or reg['max_wall_s']!=900 or reg['retry'] is not False:
        raise ValueError('PREREGISTERED_CONTRACT_MISMATCH')
    for path,h in reg['source_sha256'].items():
        if sha(ROOT/path)!=h: raise ValueError('SOURCE_HASH_CHANGED:'+path)
    for path,h in reg['historical_inputs'].items():
        if sha(ROOT/path)!=h: raise ValueError('HISTORICAL_INPUT_HASH_CHANGED:'+path)
    if sha(matrix/'native_gripper_v1.xml')!=reg['model_xml_sha256']:
        raise ValueError('MODEL_XML_HASH_CHANGED')
    static_path=Path(reg['static_receipt'])
    static=json.loads(static_path.read_text())
    if (sha(static_path)!=reg['static_receipt_sha256'] or static.get('status')!='PASS'
            or static.get('candidate_xml_sha256')!=reg['model_xml_sha256']
            or static.get('physics_steps')!=0):
        raise ValueError('STATIC_RECEIPT_CHANGED_OR_FAILED')
    baseline=ROOT/'evidence/g4_s/s2a_r2_r3/matrix_v1/P_FULL_diagnostic.xml'
    xml_review=review_xml_static((matrix/'native_gripper_v1.xml').read_text(),baseline.read_text())
    summary=json.loads((matrix/'summary.json').read_text())
    runs={}; errors={}
    for rid in reg['runs']:
        directory=matrix/'runs'/rid
        if rid in summary['unstarted'] and not directory.exists():
            runs[rid]={'status':'UNSTARTED'};continue
        try: runs[rid]=review_run(directory,rid)
        except Exception as exc:
            runs[rid]={'status':'INCOMPLETE','error':f'{type(exc).__name__}: {exc}'}
            errors[rid]=runs[rid]['error']
    prep_ok=all(runs[k].get('independent_preparation',{}).get('reset_admission_status')=='PASS' and
                runs[k].get('independent_preparation',{}).get('preparation_contact_quality_status')=='PASS'
                for k in ('PREP_I2','PREP_I3'))
    started_diag=any(runs[k]['status']!='UNSTARTED' for k in ('DIAG_I2','DIAG_I3'))
    if started_diag and not prep_ok: errors['stage2']='diagnostic started without two independently verified preparation passes'
    total=sum(runs[k].get('accounting',{}).get('results',0) for k in reg['runs'])
    if total!=summary['completed_integration_steps'] or total>530: errors['accounting']='matrix integration count mismatch'
    if summary['wall_s']>900: errors['wall']='coordinator exceeded registered wall budget'
    return {'status':'INCOMPLETE' if errors or any(v['status'] in ('INCOMPLETE','INVALID_INITIAL_STATE') for v in runs.values()) else 'COMPLETE',
            'errors':errors,'new_physics_steps':total,'stage2_authorized_by_independent_gates':prep_ok,'xml_review':xml_review,
            'runs':runs,'note':'offline numerical review; no simulation, no policy; force is not an impulse'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('matrix',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=review_matrix(args.matrix)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'errors':result['errors'],'new_physics_steps':result['new_physics_steps']}))
    return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__': raise SystemExit(main())
