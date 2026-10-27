#!/usr/bin/env python3
"""Independent read-only numerical audit; no MuJoCo stepping and no actor.

All contact measures are post-forward observations; solver-stage forces are
separately audited and never interpreted as impulses.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.analyze_g4s2a_gripper_v1 import (finite, close, T, DT, verify_transmission,
    raw_contact_metrics, review_stage, step_stop_reasons, diagnose, above_hole,
    initial_differences, review_xml_static)
from scripts import analyze_g4s2a_r2 as reference
from scripts.analyze_g4s2a_r2_r3 import review_accounting

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
        if prep:
            close(control, 2. * float(row['gripper_schedule']['target_m']), 'TENDON_TARGET', 1e-15)
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
                'finger_step_displacement_m':(qfinger-finite(pre['qpos'])[np.asarray(qidx,dtype=int)]).tolist(),
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


def same_bits(a, b, label):
    """Independent exact comparison of all persisted integration fields."""
    if set(a) != set(b):
        raise ValueError(label + '_FIELDS')
    for key in sorted(a):
        if key == 'model_fingerprint':
            if a[key] != b[key]: raise ValueError(label + '_MODEL')
            continue
        left, right = finite(a[key]), finite(b[key])
        if left.shape != right.shape or not np.array_equal(left.view(np.uint64), right.view(np.uint64)):
            raise ValueError(label + '_' + key)


def review_schedule(rows, initial_contacts):
    """Recreate the preregistered controller from prior measured contacts.

    This deliberately does not import or call the runner's reference generator.
    The first unilateral positive contact freezes the reference permanently.
    """
    phase='FREE_SPACE_APPROACH'; target=.00499125; acq=hold=0
    force=reference.pad_forces(initial_contacts)
    first_contact=first_bilateral=first_hold=complete=None
    previous_target=target
    for index, row in enumerate(rows, 1):
        record=row['gripper_schedule']; left=float(force['left']); right=float(force['right'])
        if min(left,right)<0 or not math.isfinite(left+right): raise ValueError('INVALID_PAD_FORCE')
        lb=left>1e-6; rb=right>1e-6; both=lb and rb
        before=phase
        if phase=='FREE_SPACE_APPROACH':
            if lb or rb:
                phase='CONTACT_ACQUISITION'; acq=int(both)
                if first_contact is None: first_contact=index
            else: target=max(0.,target-10e-6)
        elif phase=='CONTACT_ACQUISITION':
            acq=acq+1 if both else 0
            if acq>=10: phase='GRASP_HOLD';hold=0
        elif phase=='GRASP_HOLD':
            if both:
                hold+=1
                if hold>=10: phase='COMPLETE'
            else: phase='CONTACT_ACQUISITION';acq=hold=0
        else: raise ValueError('COMMAND_AFTER_COMPLETE')
        if both and first_bilateral is None: first_bilateral=index
        if phase=='GRASP_HOLD' and first_hold is None: first_hold=index
        if phase=='COMPLETE': complete=index
        required={'phase_before':before,'phase_after':phase,'bilateral_observed':both,
                  'bilateral_streak':acq,'hold_streak':hold,'command_number':index,
                  'complete':phase=='COMPLETE'}
        for key, value in required.items():
            if record[key]!=value: raise ValueError(f'SCHEDULE_{key}_AT_{index}')
        for key,value in (('observed_left_force_n',left),('observed_right_force_n',right),
                          ('positive_contact_threshold_n',1e-6),('target_m',target),
                          ('target_delta_m',target-previous_target),
                          ('target_rate_m_s',(target-previous_target)/DT),('tendon_target_m',2*target)):
            close(record[key],value,'SCHEDULE_'+key,1e-12)
        if abs(target-previous_target)>10e-6+1e-15: raise ValueError('TARGET_RATE_EXCEEDED')
        previous_target=target
        force=reference.pad_forces(row['state']['contacts'])
    return {'first_observed_contact_command':first_contact,
            'first_bilateral_observation_command':first_bilateral,
            'hold_entry_command':first_hold,'complete_command':complete,
            'final_phase':phase,'final_target_m':target,'max_target_delta_m':10e-6,
            'commands':len(rows)}


_STATE_COMPONENTS = ('time','qpos','qvel','act','history','qacc_warmstart','ctrl',
                     'qfrc_applied','xfrc_applied','eq_active','mocap_pos','mocap_quat',
                     'userdata','plugin')

def check_vector(state):
    # MuJoCo 3.5 mjSTATE_INTEGRATION is 16383: all 14 components,
    # in increasing bit order. Reconstruct independently from recorded fields.
    vector=np.concatenate([finite(state[k]).reshape(-1) for k in _STATE_COMPONENTS])
    same_bits({'vector':vector.tolist()},{'vector':state['integration_state']},'STATE_VECTOR')

def review_integrity(identity, rows):
    expected=identity['initial_integrity']
    if expected['model_fingerprint']!=identity['model_fingerprint']: raise ValueError('INITIAL_MODEL_FINGERPRINT')
    previous_sync=-1
    for row in rows:
        n=row['step']
        same_bits(expected,row['expected_before_control'],'CONTINUITY_EXPECTED_'+str(n))
        same_bits(row['expected_before_control'],row['actual_before_control'],'PRE_CONTROL_'+str(n))
        same_bits(row['expected_before_step'],row['actual_before_step'],'PRE_STEP_'+str(n))
        # Only ctrl is authorized to differ between these two states; the full
        # integration vector includes ctrl and is cross-checked separately below.
        before=row['expected_before_control']; after=row['expected_before_step']
        for snap in (before, after, row['actual_after_forward']): check_vector(snap)
        close(before['qpos'],row['pre_state']['qpos'],'PRE_CONTROL_QPOS',0.)
        close(before['qvel'],row['pre_state']['qvel'],'PRE_CONTROL_QVEL',0.)
        close(before['time'],[row['pre_state']['time_s']],'PRE_CONTROL_TIME',0.)
        close(after['ctrl'],row['integration_stage']['ctrl'],'AUTHORIZED_STAGE_CTRL',0.)
        ctrl=finite(after['ctrl'])
        close(ctrl[int(identity['finger_actuator_id'])],2.*row['finger_target_m'],'FINGER_COMMAND',1e-15)
        close(ctrl[:7],np.clip(finite(row['selected_arm_torque_nm']),-100.,100.),'ARM_COMMAND',0.)
        for k in before:
            if k not in ('ctrl','integration_state'):
                same_bits({k:before[k]},{k:after[k]},'UNAUTHORIZED_CONTROL_CHANGE_'+str(n))
        if 'actual_after_forward' not in row: raise ValueError('MISSING_FULL_POST_FORWARD_STATE')
        expected=row['actual_after_forward']
        close(expected['qpos'],row['state']['qpos'],'POST_FORWARD_QPOS',0.)
        close(expected['qvel'],row['state']['qvel'],'POST_FORWARD_QVEL',0.)
        close(expected['time'],[row['state']['time_s']],'POST_FORWARD_TIME',0.)
        sync=row['viewer_sync_count']
        if not isinstance(sync,int) or sync < previous_sync: raise ValueError('VIEWER_SYNC_COUNT')
        previous_sync=sync
    return {'status':'PASS','steps_checked':len(rows),'last_viewer_sync_count':previous_sync,
            'limitations':'Derived state is not equated across solver/forward stages.'}


def summarize_preparation_r1(metrics, schedule):
    """Physical gates, independent of runner classification."""
    if len(metrics)<10 or schedule['final_phase']!='COMPLETE':
        return {'reset_admission_status':'FAIL','preparation_contact_quality_status':'FAIL',
                'reason':'Incomplete acquisition or fewer than ten held steps.'}
    final=metrics[-10:]; p=finite(final[-1]['relative_position_m'],(3,)); rot=finite(final[-1]['relative_rotation'],(3,3))
    displacement=max(float(np.linalg.norm(finite(x['relative_position_m'],(3,))-p)) for x in final)
    angle=max(float(math.acos(np.clip((np.trace(finite(x['relative_rotation'],(3,3)).T@rot)-1)/2,-1,1))) for x in final)
    limits=all(x['arm_limit_violation_rad']<=1e-6 and x['finger_limit_violation_m']<=1e-6 for x in metrics)
    bilateral=all(x['bilateral'] for x in final)
    admission=bilateral and displacement<.001 and angle<=.05 and limits and all(not x['interference'] for x in final)
    maximum_pad=max(x['pad_penetration_m'] for x in metrics)
    maximum_assembly=max(x['assembly_penetration_m'] for x in metrics)
    quality=(admission and maximum_pad<=50e-6 and maximum_assembly<=10e-6 and
             max(x['arm_speed_rad_s'] for x in metrics)<=3. and
             all(not any(x['warning_delta']) for x in metrics))
    return {'reset_admission_status':'PASS' if admission else 'FAIL',
            'preparation_contact_quality_status':'PASS' if quality else 'FAIL',
            'bilateral_final10':bilateral,'final10_translation_stability_m':displacement,
            'final10_rotation_stability_rad':angle,'max_pad_penetration_m':maximum_pad,
            'max_assembly_penetration_m':maximum_assembly,'joint_limits_pass':limits}


def aggregate(metrics):
    if not metrics: return {'steps':0}
    streak=longest=0
    for x in metrics:
        streak=streak+1 if x['bilateral'] else 0
        longest=max(longest,streak)
    return {'steps':len(metrics),
        'first_left_positive_step':next((x['step'] for x in metrics if x['left_pad_force_n']>1e-6),None),
        'first_right_positive_step':next((x['step'] for x in metrics if x['right_pad_force_n']>1e-6),None),
        'first_bilateral_positive_step':next((x['step'] for x in metrics if x['bilateral']),None),
        'total_bilateral_steps':sum(x['bilateral'] for x in metrics),
        'longest_bilateral_streak_steps':longest,
        'max_abs_finger_step_displacement_m':max(abs(y) for x in metrics for y in x['finger_step_displacement_m']),
        'max_abs_native_actuator_force_n':max(abs(x['actuation']['clipped_output_n']) for x in metrics),
        'max_pad_penetration_m':max(x['pad_penetration_m'] for x in metrics),
        'max_assembly_penetration_m':max(x['assembly_penetration_m'] for x in metrics),
        'last_step_early_stop_reasons':metrics[-1]['early_stop_reasons']}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def review_run(directory, rid):
    output=directory/'output'
    required=[output/'identity.json',output/'result.json',output/'physics.jsonl',
              directory/'events/events.jsonl',directory/'supervisor_status.json',output/'gui_admission.json']
    if rid.startswith('PREP_'): required.append(output/'preparation_gate.json')
    missing=[str(p) for p in required if not p.is_file()]
    if missing: return {'status':'INCOMPLETE','missing':missing}
    identity=read(output/'identity.json');result=read(output/'result.json');process=read(directory/'supervisor_status.json')
    if identity['run']!=rid or result['run']!=rid: raise ValueError('RUN_ID_MISMATCH')
    gui=read(output/'gui_admission.json')
    if (gui['status']!='GUI_OPEN_VISIBLE' or not gui['distinct_model'] or not gui['distinct_data'] or
        gui['physics_model_fingerprint']!=gui['viewer_model_fingerprint'] or
        gui['physics_model_fingerprint']!=identity['model_fingerprint']): raise ValueError('GUI_ISOLATION_EVIDENCE')
    events=reference.trace(directory/'events/events.jsonl')
    rows=[r for r in reference.trace(output/'physics.jsonl') if r['event']=='physics_result']
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
        return {'status':'INCOMPLETE','accounting':accounting,'process':process,'runtime_failure':result['runtime_failure']}
    if result['trace_sha256']!=sha(output/'physics.jsonl') or result['completed_steps']!=len(rows): raise ValueError('RESULT_TRACE_MISMATCH')
    prep=rid.startswith('PREP_');name='I2' if '_I2_' in rid else 'I3'
    matrix=directory.parent.parent
    if prep:
        registered=read(matrix/('registered_initial_'+name+'.json'))
        same_bits(registered,identity['initial_integrity'],'PREREGISTERED_FULL_STATE')
        r2=ROOT/'evidence/g4_s/s2a_r2/matrix_v2/initials'/name/'admission.json'
        if identity['initial_sha256']!=sha(r2): raise ValueError('PRE_CLOSURE_IDENTITY_MISMATCH')
        original=read(r2)['initial']
        for key in ('qpos','qvel','time_s'):
            close(identity['initial'][key],original[key],'REGISTERED_'+key,0.)
        same_bits({'qpos':identity['initial_integrity']['qpos'],'qvel':identity['initial_integrity']['qvel'],'time':identity['initial_integrity']['time']},
                  {'qpos':original['qpos'],'qvel':original['qvel'],'time':[original['time_s']]},'REGISTERED_STATE')
        if rows and initial_differences(rows[0]['pre_state'],original):
            return {'status':'INVALID_INITIAL_STATE','accounting':accounting,'differences':initial_differences(rows[0]['pre_state'],original)}
    else:
        prepared=read(matrix/'runs'/('PREP_'+name+'_RATE_LIMITED')/'output/prepared_state.json')
        same_bits(prepared['integrity_after_forward'],identity['initial_integrity'],'PREPARED_DIAGNOSTIC_STATE')
    integrity=review_integrity(identity,rows)
    metrics=review_rows(identity,rows,prep)
    if not prep:
        return {'status':'COMPLETE','accounting':accounting,'integrity':integrity,
                'independent_diagnostic':diagnose(metrics),'aggregate':aggregate(metrics),'metrics':metrics}
    schedule=review_schedule(rows,identity['initial']['contacts'])
    summary=summarize_preparation_r1(metrics,schedule)
    draw=read(r2.parent/'reset_draw.json')
    initial_above=above_hole(identity['initial'],draw['hole_quaternion_wxyz'])
    final_above=above_hole(rows[-1]['state'],rows[-1]['state']['s2a']['hole_quaternion_wxyz']) if rows else None
    initial_forbidden=[c for c in identity['placement']['preclosure_contacts'] if c['distance_m'] < -1e-6]
    if not initial_above['entirely_above_opening'] or not initial_above['tcp_above_m'] or initial_forbidden or (final_above and (not final_above['entirely_above_opening'] or not final_above['tcp_above_m'])) or (metrics and metrics[-1]['early_stop_reasons']):
        summary['reset_admission_status']='FAIL';summary['preparation_contact_quality_status']='FAIL'
    gate=read(output/'preparation_gate.json')
    disagreement={}
    for key,original in [('reset_admission_status','reset_admission'),('preparation_contact_quality_status','physical_quality')]:
        if summary[key]!=gate[original]: disagreement[key]={'independent':summary[key],'runner':gate[original]}
    return {'status':'COMPLETE','accounting':accounting,'integrity':integrity,'schedule':schedule,
            'initial_above':initial_above,'final_above':final_above,'independent_preparation':summary,
            'runner_gate_disagreements':disagreement,'aggregate':aggregate(metrics),'metrics':metrics}


def review_matrix(matrix):
    matrix=Path(matrix);reg=read(matrix/'preregistered_matrix.json');summary=read(matrix/'summary.json')
    expected=['PREP_I2_RATE_LIMITED','PREP_I3_RATE_LIMITED','DIAG_I2_NATIVE','DIAG_I3_NATIVE']
    if reg['runs']!=expected or reg['thresholds']!=T or reg['max_steps']!=950 or reg['max_wall_s']!=900 or reg['retry'] is not False:
        raise ValueError('REGISTRATION_CONTRACT_MISMATCH')
    fixed_prep={'initial_per_finger_m':.00499125,'max_target_delta_m':10e-6,
        'target_rate_m_s':10e-6/(1./120.),'max_steps':240,'contact_n_gt':1e-6,
        'contact_rule':'freeze target permanently on first positive pad contact',
        'acquisition_bilateral_observations':10,'hold_additional_bilateral_observations':10,
        'reset_final_bilateral_observations':10,'full_transient_pad_penetration_m_le':50e-6,
        'assembly_penetration_m_le':10e-6,'final10_relative_translation_m_lt':.001,
        'final10_relative_rotation_rad_le':.05,'arm_speed_rad_s_le':3.,'joint_limits_le':1e-6}
    if reg['preparation']!=fixed_prep or reg['diagnostic_steps']!=235 or reg['policy_calls']!=0:
        raise ValueError('PREPARATION_PROTOCOL_CHANGED')
    if reg['diagnostic_schedule']!=read(ROOT/'evidence/g4_s/s2a_r2/matrix_v2/preregistered_matrix.json')['input_schedule']:
        raise ValueError('DIAGNOSTIC_SCHEDULE_CHANGED')
    if (reg['kp'],reg['kv'],reg['force_limit_n'])!=(3750.,86.5,40.): raise ValueError('ACTUATOR_CHANGED')
    if sha(reg['preflight'])!=reg['preflight_sha256'] or read(reg['preflight']).get('passed') is not True:
        raise ValueError('PREFLIGHT_RECEIPT_CHANGED')
    if sha(ROOT/'evidence/g4_s/gripper_v1_r1/static_v1/geometry.json')!=reg['geometry_sha256']:
        raise ValueError('GEOMETRY_EVIDENCE_CHANGED')
    for name,h in reg['registered_initial_sha256'].items():
        if sha(matrix/('registered_initial_'+name+'.json'))!=h: raise ValueError('REGISTERED_INITIAL_HASH_CHANGED:'+name)
    if set(reg['registered_initial_sha256'])!={'I2','I3'}: raise ValueError('MISSING_REGISTERED_INITIAL')
    for path,h in reg['sources'].items():
        if sha(ROOT/path)!=h: raise ValueError('SOURCE_HASH_CHANGED:'+path)
    for path,h in reg['historical_inputs'].items():
        if sha(ROOT/path)!=h: raise ValueError('HISTORICAL_INPUT_HASH_CHANGED:'+path)
    if sha(matrix/'native_gripper.xml')!=reg['model_xml_sha256']: raise ValueError('MODEL_XML_HASH_CHANGED')
    prior=read(ROOT/'evidence/g4_s/gripper_v1/matrix_v1/preregistered_matrix.json')
    if reg['model_xml_sha256']!=prior['model_xml_sha256']: raise ValueError('NATIVE_GRIPPER_ARCHITECTURE_CHANGED')
    baseline=ROOT/'evidence/g4_s/s2a_r2_r3/matrix_v1/P_FULL_diagnostic.xml'
    xml_review=review_xml_static((matrix/'native_gripper.xml').read_text(),baseline.read_text())
    runs={};errors={}
    for rid in expected:
        directory=matrix/'runs'/rid
        if rid in summary['unstarted'] and not directory.exists(): runs[rid]={'status':'UNSTARTED'};continue
        try:runs[rid]=review_run(directory,rid)
        except Exception as exc:
            runs[rid]={'status':'INCOMPLETE','error':f'{type(exc).__name__}: {exc}'};errors[rid]=runs[rid]['error']
        if runs[rid].get('runner_gate_disagreements'): errors[rid+'_gate']='independent and runner gates disagree'
    prep_ok=all(runs[k].get('independent_preparation',{}).get('reset_admission_status')=='PASS' and
                runs[k].get('independent_preparation',{}).get('preparation_contact_quality_status')=='PASS' for k in expected[:2])
    if any(runs[k]['status']!='UNSTARTED' for k in expected[2:]) and not prep_ok: errors['stage2']='diagnostic started before both independent preparation passes'
    total=sum(runs[k].get('accounting',{}).get('results',0) for k in expected)
    if total!=summary['completed_integration_steps'] or total>950: errors['accounting']='step count or budget mismatch'
    if summary['wall_s']>900: errors['wall']='coordinator exceeded registered wall budget'
    return {'status':'INCOMPLETE' if errors or any(v['status'] in ('INCOMPLETE','INVALID_INITIAL_STATE') for v in runs.values()) else 'COMPLETE',
            'errors':errors,'new_physics_steps':total,'stage2_authorized_by_independent_gates':prep_ok,
            'xml_review':xml_review,'runs':runs,'note':'Read only; no MuJoCo integration or policy; force is not an impulse.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('matrix',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=review_matrix(args.matrix)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'errors':result['errors'],'new_physics_steps':result['new_physics_steps']}))
    return 0 if result['status']=='COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
