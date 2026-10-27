#!/usr/bin/env python3
"""Independent NumPy review of R2-R1 recorded data; no simulator or policy.

Uses existing R2-independent quaternion/controller utilities, not runner code.
No integration, rendering, model compile, or actor inference.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts import analyze_g4s2a_r2 as core

BODY_NAMES=core.BODY_NAMES
DT=core.DT

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f: json.dump(x,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n')
def worst(a,b): return core.maximum_error(a,b)
def inertia_review(models):
 baseline=models['S1B']; out={}
 for label in ('P_FULL','P_ARM_HAND'):
  adapted=models[label]; changes=[]
  if baseline['body_names']!=adapted['body_names']:raise ValueError('body ordering changed')
  selected=set(BODY_NAMES if label=='P_FULL' else BODY_NAMES[:9])
  for name in BODY_NAMES:
   i=baseline['body_names'].index(name)
   b=np.asarray(baseline['body_inertia_kg_m2'][i]);a=np.asarray(adapted['body_inertia_kg_m2'][i])
   Rb=core.rotation_matrix(baseline['body_iquat_wxyz'][i]);Ra=core.rotation_matrix(adapted['body_iquat_wxyz'][i])
   err=worst(Ra@np.diag(a)@Ra.T-Rb@np.diag(b)@Rb.T,(.01 if name in selected else 0.)*np.eye(3))
   if err>2e-13 or min(a)<=0 or min(b)<=0:raise ValueError('incorrect inertia: '+label+' '+name)
   changes.append({'body':name,'principal_before_kg_m2':b.tolist(),'principal_after_kg_m2':a.tolist(),'tensor_delta_error_kg_m2':err,'offset_applied':name in selected})
  fixed=('body_mass_kg','body_ipos_m','dof_armature','force_limits','arm_dof_ids','arm_qpos_ids','finger_dof_ids','finger_qpos_ids','arm_limits','finger_limits')
  errors={k:worst(baseline[k],adapted[k]) for k in fixed}
  if max(errors.values())>0:raise ValueError('unexpected model parameter change')
  out[label]={'selected_body_count':len(selected),'bodies':changes,'unchanged_parameter_max_errors':errors}
 return out

def review_rows(identity,rows,thresholds):
 model=identity['models']['plant'];initial=identity['initial_snapshot'];previous=initial;previous_pre=None
 arm=model['arm_dof_ids'];finger=model['finger_dof_ids'];arm_q=model['arm_qpos_ids'];finger_q=model['finger_qpos_ids']
 ranges=core.finite(model['force_limits'])[model['arm_actuator_ids']]
 if worst(ranges,np.c_[-core.ARM_FORCE_LIMITS,core.ARM_FORCE_LIMITS])!=0:raise ValueError('arm force limits changed')
 errors={};metrics=[];losses=0
 def cmp(key,a,b):errors[key]=max(errors.get(key,0.),worst(a,b))
 for step,row in enumerate(rows,1):
  if row['step']!=step:raise ValueError('step sequence')
  s=row['state'];u=row['controller_input'];M=row['matrices'];control=row['controller']
  cmp('time',row['episode_time_s'],step*DT);cmp('pre_q',u['q7'],np.asarray(previous['qpos'])[arm_q]);cmp('pre_v',u['qdot7'],np.asarray(previous['qvel'])[arm])
  cmp('pre_position',u['position'],previous['tcp_position_world']);cmp('pre_quaternion',u['quaternion'],previous['tcp_quaternion_wxyz']);cmp('same_qpos',M['same_qpos'],previous['qpos'])
  for k in ('qpos','qvel','tcp_position_world','tcp_quaternion_wxyz'):cmp('pre_state_'+k,row['pre_state'][k],previous[k])
  full=core.finite(M['physical_full_M']);f0=core.finite(M['C0_full_M']);f1=core.finite(M['C1_full_M'])
  for matrix in (full,f0,f1):np.linalg.cholesky(matrix);cmp('full_mass_symmetry',matrix,matrix.T)
  cmp('plant_full_vs_selected_C1',full,f1);m0=np.asarray(M['same_q_M0']);m1=np.asarray(M['same_q_M1']);pm=np.asarray(M['physical_M']);cm=np.asarray(M['controller_M']);ix=np.ix_(arm,arm)
  cmp('M0_extract',m0,f0[ix]);cmp('M1_extract',m1,f1[ix]);cmp('plant_arm',pm,full[ix]);cmp('selected_controller',cm,(f0 if identity['controller_selection']=='C0' else f1)[ix]);cmp('controller_injected',u['mass_matrix'],cm)
  angular=dict(zip(M['body_angular_jacobian_names'],M['body_angular_jacobians']));selected=BODY_NAMES if identity['physical_selection']=='P_FULL' else BODY_NAMES[:9]
  if set(angular)!=set(BODY_NAMES):raise ValueError('missing body Jacobians')
  delta=sum((.01*np.asarray(angular[name]).T@np.asarray(angular[name]) for name in selected),np.zeros((7,7)))
  cmp('inertia_body_pullback',m1-m0,delta)
  if M['finger_predictor_mass_source']!='physical_full_M':raise ValueError('finger matrix source changed')
  if step==1:cmp('fd_linear',u['linear_fd'],np.zeros(3));cmp('fd_angular',u['angular_fd'],np.zeros(3))
  else:
   lin,ang,_=core.finite_difference(previous_pre,previous);cmp('fd_linear',u['linear_fd'],lin);cmp('fd_angular',u['angular_fd'],ang)
  calc=core.reconstruct_controller(u)
  for k in ('pose_error','wrench','task_torque','null_torque','requested','software_limited','null_projector'):cmp('controller_'+k,control[k],calc[k])
  cmd=np.zeros(7) if identity['passive_arm'] else calc['requested'];cmp('selected_torque',row['selected_arm_torque_nm'],cmd)
  expected=np.zeros(7) if identity['passive_arm'] else np.clip(calc['software_limited'],ranges[:,0],ranges[:,1]);act=s['integration_actuation'];cmp('integrated_arm_actuator',act['arm_actuator_force_nm'],expected);cmp('integrated_arm_generalized',act['arm_qfrc_actuator_nm'],expected)
  fi=np.linalg.solve(full,np.eye(len(full)))[np.ix_(finger,finger)];q=np.asarray(previous['qpos'])[finger_q];v=np.asarray(previous['qvel'])[finger]
  raw=-7500*q-173*v;force=np.clip(np.linalg.solve(np.eye(2)+DT*(173+DT*7500)*fi,-7500*q-(173+DT*7500)*v),-40,40)
  if s['finger_command']['mode']!='step-implicit-pd' or s['finger_command']['target_opening_per_finger_m']!=0:raise ValueError('finger controller altered')
  cmp('finger_raw_pd',s['finger_command']['raw_pd_force_n'],raw);cmp('finger_command',s['finger_command']['commanded_force_n'],force);cmp('integrated_finger_force',act['finger_actuator_force_n'],force)
  trans,rot=core.slip(s,initial);pads=core.pad_forces(s['contacts']);all_depth,assembly=core.penetrations(s['contacts']);pad_depth=core.pad_penetration(s['contacts'])
  arm_limit=core.limit_violation(s['qpos'],arm_q,model['arm_limits']);finger_limit=core.limit_violation(s['qpos'],finger_q,model['finger_limits']);speed=float(max(abs(np.asarray(s['qvel'])[arm])))
  cmp('slip_translation',s['s2a']['peg_hand_slip_m'],trans);cmp('slip_rotation',s['s2a']['peg_hand_slip_rad'],rot);cmp('pad_forces',list(row['pad_force_n'].values()),[pads['left'],pads['right']]);cmp('all_depth',s['s2a']['all_contact_penetration_m'],all_depth);cmp('assembly_depth',s['s2a']['peg_hole_penetration_m'],assembly)
  cmp('arm_limit',s['s2a']['arm_joint_limit_violation_rad'],arm_limit);cmp('finger_limit',s['s2a']['finger_joint_limit_violation_m'],finger_limit)
  _,_,acc=core.finite_difference(previous,s);cmp('interval_acceleration',row['qvel_interval_acceleration'],acc)
  integration=row['integration_stage']
  if integration['stage']!='immediately_after_mj_step_before_wrapper_mj_forward':raise ValueError('unexpected integration observer timing')
  cmp('integration_time',integration['time_s'],s['time_s'])
  cmp('integration_arm_motor',np.asarray(integration['actuator_force'])[model['arm_actuator_ids']],expected)
  cmp('integration_arm_generalized',np.asarray(integration['qfrc_actuator'])[arm],expected)
  cmp('integration_finger_motor',np.asarray(integration['actuator_force'])[model['finger_actuator_ids']],force)
  cmp('integration_finger_generalized',np.asarray(integration['qfrc_actuator'])[finger],force)
  solver=row['post_forward_solver'];cmp('post_qacc',solver['qacc'],row['qacc_post_forward']);cmp('post_qfrc_constraint',solver['qfrc_constraint'],row['qfrc_constraint_post_forward']);cmp('post_qacc_warmstart',solver['qacc_warmstart'],row['qacc_warmstart'])
  if solver['stage']!='post_step_mj_forward_not_integration_impulses':raise ValueError('unexpected solver timing')
  for c,d in zip(s['contacts'],solver['contacts']):
   if (c['geom1'],c['geom2'])!=(d['geom1'],d['geom2']):raise ValueError('postforward pair mismatch')
   cmp('solver_normal_force',c['normal_force_n'],d['contact_force_local'][0]);cmp('solver_contact_distance',c['distance_m'],d['distance_m'])
  if len(s['contacts'])!=len(solver['contacts']):raise ValueError('contact count mismatch')
  positive=min(pads.values())>thresholds['positive_pad_force_n'];losses=0 if positive else losses+1
  stops=[]
  if any(row['warning_delta']):stops.append('NUMERICAL_SOLVER_WARNING')
  for val,key,label in ((trans,'stop_slip_m','PEG_ESCAPE_SLIP'),(rot,'stop_rotation_rad','PEG_ESCAPE_ROTATION'),(arm_limit,'stop_arm_limit_rad','ARM_LIMIT'),(finger_limit,'stop_finger_limit_m','FINGER_LIMIT'),(all_depth,'stop_penetration_m','SEVERE_PENETRATION')):
   if val>thresholds[key]:stops.append(label)
  if speed>thresholds['stop_arm_velocity_rad_s']:stops.append('IMPLAUSIBLE_ARM_VELOCITY')
  if losses>=thresholds['stop_consecutive_contact_loss']:stops.append('SUSTAINED_CONTACT_LOSS')
  if stops!=row['stop_reasons']:raise ValueError(f'stop mismatch step {step}: {stops} {row["stop_reasons"]}')
  if stops and step!=len(rows):raise ValueError('trajectory continued after stop')
  types={str(k):solver['efc_type'].count(k) for k in set(solver['efc_type'])}
  metrics.append({'step':step,'time_s':step*DT,'max_arm_velocity_rad_s':speed,'translation_slip_m':trans,'rotation_slip_rad':rot,'left_pad_force_n':pads['left'],'right_pad_force_n':pads['right'],'bilateral_positive_contact':positive,
   'all_penetration_m':all_depth,'pad_penetration_m':pad_depth,'assembly_penetration_m':assembly,'arm_limit_violation_rad':arm_limit,'finger_limit_violation_m':finger_limit,
   'finger_qpos_m':np.asarray(s['qpos'])[finger_q].tolist(),'finger_qvel_m_s':np.asarray(s['qvel'])[finger].tolist(),'finger_raw_pd_n':raw.tolist(),'finger_integration_force_n':force.tolist(),
   'finger_interval_acceleration_m_s2':acc[finger].tolist(),'finger_post_forward_constraint_n':np.asarray(solver['qfrc_constraint'])[finger].tolist(),'finger_post_forward_qacc':np.asarray(solver['qacc'])[finger].tolist(),
   'finger_inverse_physical_mass':fi.tolist(),'controller_max_requested_nm':float(max(abs(calc['requested']))),'actual_max_arm_force_nm':float(max(abs(np.asarray(act['arm_actuator_force_nm'])))),
   'contact_pairs':[{'geom1':c['geom1'],'geom2':c['geom2'],'distance_m':c['distance_m'],'normal_force_n':c['normal_force_n'],'position_world_m':c['position_world_m']} for c in s['contacts']],
   'integration_stage_finger_qfrc_actuator_n':np.asarray(integration['qfrc_actuator'])[finger].tolist(),'integration_stage_finger_qfrc_constraint_n':np.asarray(integration['qfrc_constraint'])[finger].tolist(),'integration_stage_qacc':np.asarray(integration['qacc'])[finger].tolist(),'integration_stage_nefc':integration['nefc'],'integration_stage_efc_type_counts':{str(k):integration['efc_type'].count(k) for k in set(integration['efc_type'])},'post_forward_nefc':solver['nefc'],'post_forward_efc_type_counts':types,'stop_reasons':stops})
  previous_pre,previous=previous,s
 return metrics,errors

def aggregate(rows,t,horizon):
 if not rows:return {'steps':0,'all_stability_gates':False}
 keys=('max_arm_velocity_rad_s','translation_slip_m','rotation_slip_rad','all_penetration_m','pad_penetration_m','assembly_penetration_m','arm_limit_violation_rad','finger_limit_violation_m','controller_max_requested_nm','actual_max_arm_force_nm')
 out={'steps':len(rows),'duration_s':rows[-1]['time_s']};out.update({'max_'+k:max(x[k] for x in rows) for k in keys});out['first_contact_loss_step']=next((x['step'] for x in rows if not x['bilateral_positive_contact']),None)
 out['min_pad_force_n']={side:min(x[side+'_pad_force_n'] for x in rows) for side in ('left','right')}
 out['all_stability_gates']=bool(len(rows)==horizon and out['first_contact_loss_step'] is None and out['max_translation_slip_m']<t['accept_slip_m'] and out['max_rotation_slip_rad']<=t['accept_rotation_rad'] and out['max_max_arm_velocity_rad_s']<=t['accept_arm_velocity_rad_s'] and out['max_arm_limit_violation_rad']<=t['accept_arm_limit_rad'] and out['max_finger_limit_violation_m']<=t['accept_finger_limit_m'] and out['max_pad_penetration_m']<=t['accept_pad_penetration_m'] and out['max_assembly_penetration_m']<=t['accept_assembly_penetration_m'])
 return out

def historical_metrics(i,c,limit,thresholds):
 p=ROOT/f'evidence/g4_s/s2a_r2/matrix_v2/trajectories/{i}_{"B" if c=="C0" else "D"}/physics.jsonl.gz';rows=[r for r in core.trace(p) if r['event']=='physics_result'][:limit]
 out=[]
 for r in rows:
  s=r['state'];forces=core.pad_forces(s['contacts']);a,b=core.penetrations(s['contacts']);out.append({'step':r['step'],'time_s':r['step']*DT,'max_arm_velocity_rad_s':max(abs(np.asarray(s['qvel'])[:7])),'translation_slip_m':s['s2a']['peg_hand_slip_m'],'rotation_slip_rad':s['s2a']['peg_hand_slip_rad'],'left_pad_force_n':forces['left'],'right_pad_force_n':forces['right'],'bilateral_positive_contact':min(forces.values())>thresholds['positive_pad_force_n'],'all_penetration_m':a,'pad_penetration_m':core.pad_penetration(s['contacts']),'assembly_penetration_m':b,'arm_limit_violation_rad':s['s2a']['arm_joint_limit_violation_rad'],'finger_limit_violation_m':s['s2a']['finger_joint_limit_violation_m'],'controller_max_requested_nm':max(abs(np.asarray(s['requested_arm_torque_nm']))),'actual_max_arm_force_nm':max(abs(np.asarray(s['integration_actuation']['arm_actuator_force_nm'])))})
 return out,p

def run(source,out):
 source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();registry=read(source/'preregistered_matrix.json');summary=read(source/'summary.json')
 if registry['steps_per_trajectory']!=235 or registry['maximum_new_integration_steps']!=2115 or len(registry['runs'])!=9:raise ValueError('scope/budget mismatch')
 schedule=hashlib.sha256(json.dumps(registry['input_schedule'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 if schedule!=registry['input_schedule_sha256']:raise ValueError('schedule hash mismatch')
 historical_registry_path=ROOT/'evidence/g4_s/s2a_r2/matrix_v2/preregistered_matrix.json'
 historical_registry=read(historical_registry_path)
 if registry['input_schedule']!=historical_registry['input_schedule'] or registry['input_schedule_sha256']!=historical_registry['input_schedule_sha256'] or registry['thresholds']!=historical_registry['thresholds']:
  raise ValueError('registered schedule or thresholds differ from frozen R2')
 tensors=inertia_review(registry['models']);write(out/'inertia_review.json',tensors)
 thresholds=registry['thresholds'];metrics={};results={};hashes={};failures={}
 for item in registry['runs']:
  rid=item['id'];d=source/'trajectories'/rid
  if not all((d/f).exists() for f in ('identity.json','result.json','physics.jsonl')):results[rid]={'status':'MISSING_OR_INCOMPLETE'};continue
  identity=read(d/'identity.json');result=read(d/'result.json');events=core.trace(d/'physics.jsonl');rows=[e for e in events if e['event']=='physics_result'];requests=[e for e in events if e['event']=='integration_request']
  if result['trace_sha256']!=sha(d/'physics.jsonl') or identity['input_schedule_sha256']!=schedule or identity['run_id']!=rid:raise ValueError('trace identity mismatch '+rid)
  initial_path=ROOT/registry['historical_inputs'][item['initial_id']+'_initial_state']['path'];initial=read(initial_path)
  if identity['initial_state_sha256']!=sha(initial_path):raise ValueError('initial digest mismatch')
  expected_identity={'initial_id':item['initial_id'],'physical_selection':item['physical'],'controller_selection':item['controller'],'passive_arm':item['passive_arm']}
  if any(identity.get(k)!=v for k,v in expected_identity.items()):raise ValueError('registered experiment identity mismatch '+rid)
  expected_models={'plant':registry['models'][item['physical']],'M0':registry['models']['S1B'],'M1':registry['models'][item['physical']]}
  if identity['models']!=expected_models:raise ValueError('registered full model identity mismatch '+rid)
  for key in ('qpos','qvel','time_s','hole_root_position'):
   if worst(identity['initial_snapshot'][key],initial['state'][key])>1e-12:raise ValueError('initial physical state mismatch '+rid+' '+key)
  if len(requests)!=len(rows) or len(events)!=2*len(rows) or len(rows)!=result['completed_steps'] or len(rows)>235:raise ValueError('unpaired requests or step mismatch '+rid)
  if any(e['event']!=('integration_request' if n%2==0 else 'physics_result') for n,e in enumerate(events)):raise ValueError('event order')
  for req,row in zip(requests,rows):
   if req['step']!=row['step']:raise ValueError('request/result step mismatch')
   for key in ('run_id','pre_state','matrices','target','controller_input','controller','selected_arm_torque_nm'):
    if req[key]!=row[key]:raise ValueError('request/result mismatch '+rid+' '+key)
   if abs(req['time_before_s']-row['pre_state']['time_s'])>1e-12:raise ValueError('request timestamp mismatch')
  values,errors=review_rows(identity,rows,thresholds);schedule_errors=core.inspect_reference_schedule(identity,initial,registry,rows)
  tolerance={key:(1e-7 if key=='slip_rotation' else 1e-8 if key.startswith('controller_') or key=='interval_acceleration' else 1e-9) for key in errors}
  bad={k:{'error':v,'tolerance':tolerance[k]} for k,v in errors.items() if v>tolerance[k]};bad.update({'schedule_'+k:{'error':v,'tolerance':1e-10} for k,v in schedule_errors.items() if v>1e-10})
  result_metrics=aggregate(values,thresholds,235);warning=sum(sum(r['warning_delta']) for r in rows);process=read(source/(rid+'_process.json'))
  if process['exact_child_exit_code']!=0 or result['runtime_failure'] is not None or warning:bad['runtime_or_warning']={'warnings':warning,'runtime_failure':result['runtime_failure'],'exit':process['exact_child_exit_code']}
  if result['first_contact_loss_step']!=result_metrics.get('first_contact_loss_step'):bad['first_contact_loss_mismatch']=True
  full_ok=result_metrics['all_stability_gates'] and not bad and result['termination']=='REGISTERED_HORIZON_COMPLETE'
  result_metrics['all_stability_gates']=full_ok;metrics[rid]=values;failures[rid]=bad
  results[rid]={'initial_id':item['initial_id'],'physical':item['physical'],'controller':item['controller'],'passive_arm':item['passive_arm'],'metrics':result_metrics,'termination':result['termination'],'arithmetic_max_errors':errors,'schedule_max_errors':schedule_errors,'failures':bad,'warning_count':warning,'exact_child_exit_code':process['exact_child_exit_code'],'physics_wall_s':result['physics_wall_s']}
  for path in (d/'identity.json',d/'result.json',d/'physics.jsonl',source/(rid+'_process.json'),initial_path):hashes[str(path.relative_to(ROOT))]=sha(path)
 comparisons=[]
 for initial_id in ('I1','I2','I3'):
  for controller in ('C0','C1'):
   new=f'{initial_id}_ARM_HAND_{controller}'
   if new not in metrics:continue
   if initial_id=='I3':
    old=f'I3_FULL_{controller}'
    if old not in metrics:continue
    a=metrics[old];kind='fresh P_FULL';oldid=old
   else:a,p=historical_metrics(initial_id,controller,len(metrics[new]),thresholds);hashes[str(p.relative_to(ROOT))]=sha(p);kind='historical R2 P_FULL';oldid=f'{initial_id}_{"B" if controller=="C0" else "D"}'
   b=metrics[new];length=min(len(a),len(b));comparisons.append({'initial':initial_id,'controller':controller,'reference_id':oldid,'reference_kind':kind,'intervention_id':new,'common_prefix_steps':length,'reference':aggregate(a[:length],thresholds,235),'intervention':aggregate(b[:length],thresholds,235),'scope':'Matched recorded initial state and action schedule; common prefix only. One deterministic trajectory per factor; no causal uniqueness.'})
 with gzip.open(out/'step_metrics.jsonl.gz','xt') as f:
  for rid,rows in metrics.items():
   for row in rows:f.write(json.dumps({'run_id':rid,**row},allow_nan=False,separators=(',',':'))+'\n')
 report={'task':'G4-S2A-R2-R1','method':'Independent NumPy reconstruction of all recorded trajectories, controller, physical finger mass solve, matching schedules and gate/stop arithmetic. No MuJoCo imports, physics, policy, or rendering.',
 'numerical_review_passed':len(results)==9 and all(rid in metrics and not failures[rid] for rid in results),'complete_registered_accounting':summary['completed_accounting'],'reviewed_trajectories':len(metrics),'reviewed_steps':sum(map(len,metrics.values())),
 'trajectories':results,'matched_common_prefix_comparisons':comparisons,'passive_arm_interpretation':'Separate negative control; never pooled in primary matched comparisons.',
 'timing_limitations':'All post_forward_solver qacc, constraint/contact forces and efc data recomputed poststep; no previous integration impulses. Integration actuation from preintegration is separately recorded. Equality/contact efc types and forces are archived for examination without attributing total qfrc_constraint solely to any one constraint.',
 'limits':'Finite trajectories, no policy success, no unique global root cause. An early stop cannot pass the 235-step horizon gate. Same-time comparisons use common prefixes.',
 'input_hashes':hashes,'analysis_script_sha256':sha(__file__),'r2_primitives_sha256':sha(ROOT/'scripts/analyze_g4s2a_r2.py'),'registry_sha256':sha(source/'preregistered_matrix.json'),'runtime_s':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 write(out/'numerical_review.json',report);write(out/'manifest.json',{'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},'new_integration_steps':0,'policy_calls':0,'model_compilations':0})
 print(json.dumps({'numerical_review_passed':report['numerical_review_passed'],'reviewed_trajectories':report['reviewed_trajectories'],'reviewed_steps':report['reviewed_steps']}));return 0 if report['numerical_review_passed'] else 1
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();sys.exit(run(args.source,args.output))
