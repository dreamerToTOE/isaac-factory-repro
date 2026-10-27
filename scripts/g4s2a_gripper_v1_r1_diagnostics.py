#!/usr/bin/env python3
"""Registered state-isolated rate-limited gripper recovery. No PPO, no retries."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv, KP, KV, FORCE_LIMIT_N
from scripts.g4s2a_gripper_v1_diagnostics import initial, placement, actuation, data_paths
from scripts.g4s2a_gripper_v1_r1_schedule import PreparationState, advance, MAX_TARGET_DELTA_M, MAX_TARGET_RATE_M_S, MAX_PREPARATION_STEPS, INITIAL_OPENING_M
from scripts.g4s2a_gripper_v1_r1_state import snapshot, guard, authorized_ctrl, model_fingerprint, StateIntegrityError, IsolatedViewer
from scripts.g4s2a_r2_model import THRESHOLDS, stop_reasons, fixed_reference, make_model, mass_at, model_identity
from scripts.g4s2a_r2_r3_diagnostics import solver_and_contacts, safe_value
from scripts.g4s2a_r2_r2_supervisor import ChildSupervisor, DurableEventLog
from scripts.g4s2a_r2_diagnostics import integration_state
from scripts.g4s1b_contract import PhysicsContract, torque_command
from scripts.g4s2a_smoke import physical_measurements, peg_relative, rotation_distance, finger_forces, independent_peg, finite_tree
from scripts.g4s2a_r1_viewer import contact_forces, window_info
R2=ROOT/'evidence/g4_s/s2a_r2/matrix_v2'
RUNS=['PREP_I2_RATE_LIMITED','PREP_I3_RATE_LIMITED','DIAG_I2_NATIVE','DIAG_I3_NATIVE']
SOURCES=['scripts/g4s2a_gripper_v1_r1_diagnostics.py','scripts/g4s2a_gripper_v1_r1_state.py','scripts/g4s2a_gripper_v1_r1_schedule.py','scripts/g4s2a_gripper_v1_model.py','scripts/g4s2a_gripper_v1_diagnostics.py','scripts/g4s2a_r2_r3_diagnostics.py','scripts/g4s2a_r2_r2_supervisor.py','scripts/g4s2a_r2_diagnostics.py','scripts/g4s2a_r2_model.py','scripts/g4s1b_env.py','scripts/g4s1b_contract.py','scripts/g4s2a_smoke.py','scripts/g4s2a_r1_viewer.py','scripts/capture_g4s2a_r1_comparison.py','scripts/audit_g4s2a_gripper_v1_r1_geometry.py','scripts/g4s2a_gripper_v1_r1_preflight.py','scripts/g4s2a_gripper_v1_r1_shared_forensics.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(safe_value(v),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def line(f,v):
 f.write(json.dumps(safe_value(v),separators=(',',':'),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def register(out,preflight):
 receipt=read(preflight)
 if receipt.get('passed') is not True:raise ValueError('PREFLIGHT_NOT_PASSED')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);env=NativeGripperEnv();(out/'native_gripper.xml').write_text(env.xml)
 registered_initial={}
 for name in ('I2','I3'):
  registered_env=NativeGripperEnv();initial(registered_env,name);fp=out/('registered_initial_'+name+'.json');write(fp,snapshot(registered_env.mj,registered_env.model,registered_env.data));registered_initial[name]=sha(fp)
 old=read(ROOT/'evidence/g4_s/gripper_v1/matrix_v1/preregistered_matrix.json')
 if sha(out/'native_gripper.xml')!=old['model_xml_sha256']:raise ValueError('NATIVE_MODEL_CHANGED')
 geom=ROOT/'evidence/g4_s/gripper_v1_r1/static_v1/geometry.json'
 r={'task':'G4-S2A-GRIPPER-V1-R1','start_commit':'d5290698fe20c52b1d8c539ce9e4e7ba8561466a','created_utc':datetime.now(timezone.utc).isoformat(),
 'model_xml_sha256':sha(out/'native_gripper.xml'),'model_fingerprint':model_fingerprint(env.model),'sources':{p:sha(ROOT/p) for p in SOURCES},'historical_inputs':data_paths(),
 'registered_initial_sha256':registered_initial,'preflight':str(Path(preflight).resolve()),'preflight_sha256':sha(preflight),'geometry_sha256':sha(geom),'runs':RUNS,'max_steps':950,'max_wall_s':900,'retry':False,
 'preparation':{'initial_per_finger_m':INITIAL_OPENING_M,'max_target_delta_m':MAX_TARGET_DELTA_M,'target_rate_m_s':MAX_TARGET_RATE_M_S,'max_steps':MAX_PREPARATION_STEPS,
 'contact_n_gt':1e-6,'contact_rule':'freeze target permanently on first positive pad contact','acquisition_bilateral_observations':10,'hold_additional_bilateral_observations':10,
 'reset_final_bilateral_observations':10,'full_transient_pad_penetration_m_le':50e-6,'assembly_penetration_m_le':10e-6,
 'final10_relative_translation_m_lt':.001,'final10_relative_rotation_rad_le':.05,'arm_speed_rad_s_le':3.,'joint_limits_le':1e-6},
 'thresholds':THRESHOLDS,'diagnostic_steps':235,'diagnostic_schedule':read(R2/'preregistered_matrix.json')['input_schedule'],
 'kp':KP,'kv':KV,'force_limit_n':FORCE_LIMIT_N,'tendon_mapping':'L=q1+q2; ctrl=2*q_target',
 'gui':'separate MjModel/MjData; one-way physics snapshot to viewer; no viewer writeback','policy_calls':0}
 write(out/'preregistered_matrix.json',r)
def verify(path):
 r=read(path);out=Path(path).parent
 if r['runs']!=RUNS or r['max_steps']!=950 or r['max_wall_s']!=900 or r['retry'] is not False or r['thresholds']!=THRESHOLDS:raise ValueError('REGISTERED_PLAN_CHANGED')
 if r['sources']!={p:sha(ROOT/p) for p in SOURCES} or r['historical_inputs']!=data_paths() or r['model_xml_sha256']!=sha(out/'native_gripper.xml'):raise ValueError('REGISTERED_SOURCES_CHANGED')
 if sha(r['preflight'])!=r['preflight_sha256'] or read(r['preflight']).get('passed') is not True:raise ValueError('PREFLIGHT_CHANGED')
 if sha(ROOT/'evidence/g4_s/gripper_v1_r1/static_v1/geometry.json')!=r['geometry_sha256']:raise ValueError('GEOMETRY_EVIDENCE_CHANGED')
 for name,h in r['registered_initial_sha256'].items():
  if sha(out/('registered_initial_'+name+'.json'))!=h:raise ValueError('REGISTERED_INITIAL_SNAPSHOT_CHANGED')
 return r

def one_step(env,torque,finger_target,ledger,step,expected):
 """Guard immediately at actual mj_step boundary after explicit ctrl update."""
 mj=env.mj;original=mj.mj_step;captured=[];ctrl=env.data.ctrl.copy();ctrl[env.arm_actuator_ids]=np.clip(torque,-100,100);ctrl[env.finger_actuator_ids[0]]=2*finger_target
 expected_step=authorized_ctrl(mj,env.model,expected,ctrl)
 def observe(model,data):
  if captured:raise RuntimeError('MULTIPLE_INTEGRATIONS')
  actual_step=guard(expected_step,mj,model,data,'immediately_before_mj_step')
  start=float(data.time);ledger.record_request(step,simulation_time_s=start,integrity_expected=expected_step,integrity_actual=actual_step)
  try:original(model,data)
  except BaseException as e:ledger.record_exception(step,e);raise
  stage=solver_and_contacts(env);stage.update({'stage':'immediately_after_mj_step_before_wrapper_mj_forward','simulation_start_time_s':start,'simulation_end_time_s':float(data.time),'actuator_force':data.actuator_force.copy(),'ctrl':data.ctrl.copy(),'gripper':actuation(env)})
  ledger.record_result(step,integration_stage=safe_value(stage));captured.append((stage,actual_step))
 mj.mj_step=observe
 try:state=env.physics_step(torque,finger_target=finger_target,record=False)
 finally:mj.mj_step=original
 if len(captured)!=1:raise RuntimeError('ONE_INTEGRATION_REQUIRED')
 return state,captured[0][0],expected_step,captured[0][1]

def preparation_gate(env,rows,place,schedule,stops):
 if not rows:return {'reset_admission':'FAIL','physical_quality':'FAIL','errors':['NO_COMPLETED_STEPS']}
 tail=rows[-10:];rel=[np.asarray(r['state']['s2a']['relative_peg_hand_m']) for r in tail];rot=[np.asarray(r['state']['s2a']['relative_peg_hand_rotation']) for r in tail]
 drift=max(float(np.linalg.norm(x-rel[-1])) for x in rel);angular=max(rotation_distance(x,rot[-1]) for x in rot)
 pads=max(r['pad_penetration_m'] for r in rows);final_pads=max(r['pad_penetration_m'] for r in tail);assembly=max(r['state']['s2a']['peg_hole_penetration_m'] for r in rows)
 bilateral=len(tail)==10 and all(min(r['pad_force_n'].values())>1e-6 for r in tail)
 joint=max(max(r['state']['s2a']['arm_joint_limit_violation_rad'],r['state']['s2a']['finger_joint_limit_violation_m']) for r in rows)
 velocity=max(max(abs(np.asarray(r['state']['qvel'])[env.arm_dof_ids])) for r in rows)
 interference=sorted(set(s for row in tail for s in row['interference_bodies']));above=bool(env.insertion_diagnostics()['entirely_above_opening'] and env.tcp_state()[0][2]>.075)
 reset=[]
 for bad,code in [(schedule.phase!='COMPLETE','HOLD_NOT_COMPLETED'),(not place['valid'],'INVALID_INITIAL_PLACEMENT'),(not bilateral,'BILATERAL_LAST10'),(drift>=.001,'TRANSLATION_STABILITY'),(angular>.05,'ROTATION_STABILITY'),(joint>1e-6,'JOINT_LIMIT'),(bool(interference),'INTERFERENCE'),(not above,'NOT_ABOVE_BORE'),(bool(stops),'EARLY_STOP')]:
  if bad:reset.append(code)
 quality=[]
 for bad,code in [(bool(reset),'RESET_NOT_ADMITTED'),(pads>50e-6,'TRANSIENT_PAD_PENETRATION'),(final_pads>50e-6,'FINAL10_PAD_PENETRATION'),(assembly>10e-6,'ASSEMBLY_PENETRATION'),(velocity>3,'ARM_VELOCITY'),(any(any(r['warning_delta']) for r in rows),'NUMERICAL_WARNING'),(not all(finite_tree(r) for r in rows),'NONFINITE')]:
  if bad:quality.append(code)
 return {'reset_admission':'FAIL' if reset else 'PASS','physical_quality':'FAIL' if quality else 'PASS','reset_errors':reset,'quality_errors':quality,
 'bilateral_last10':bilateral,'transient_max_pad_penetration_m':pads,'final10_max_pad_penetration_m':final_pads,'max_assembly_penetration_m':assembly,
 'final10_stability_m':drift,'final10_angular_stability_rad':angular,'max_joint_limit_violation':joint,'max_arm_velocity_rad_s':velocity,'interference_bodies':interference,'final_above_hole':above,'schedule_phase':schedule.phase}

def capture(visual,path,step,state,identity):
 visual.capture(path,step,state,identity);write(path.with_suffix('.json'),{'run':visual.label,'step':step,'mode':'LIVE_PHYSICS_STATE_DISPLAY','new_physics_steps_for_capture':0,'model_xml_sha256':identity,'window':window_info(),'sha256':sha(path),'requires_image_qa':True})

def child(run,manifest,out):
 reg=verify(manifest);prep=run.startswith('PREP_');name='I2' if '_I2_' in run else 'I3';parent=Path(manifest).resolve().parent
 if run not in RUNS or Path(out).resolve()!=parent/'runs'/run/'output' or not (parent/'coordinator_attempt.json').exists():raise ValueError('UNREGISTERED_CHILD')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);env=NativeGripperEnv()
 if hashlib.sha256(env.xml.encode()).hexdigest()!=reg['model_xml_sha256']:raise ValueError('CANDIDATE_MODEL_CHANGED')
 if prep:
  registered=initial(env,name);place=placement(env)
  if not place['valid'] or not np.array_equal(env.data.qpos,np.asarray(registered['qpos'])) or not np.array_equal(env.data.qvel,np.asarray(registered['qvel'])) or env.data.time!=registered['time_s']:raise ValueError('REGISTERED_INITIAL_STATE_INVALID')
  schedule=PreparationState();target_finger=schedule.target_m
 else:
  prepared=read(parent/'runs'/('PREP_'+name+'_RATE_LIMITED')/'output/prepared_state.json');draw=read(R2/'initials'/name/'reset_draw.json');env.model.body_pos[env.hole_body_id]=draw['hole_root_m'];env.model.body_quat[env.hole_body_id]=draw['hole_quaternion_wxyz'];spec=env.mj.mjtState(prepared['integration_state']['value']);env.mj.mj_setState(env.model,env.data,np.asarray(prepared['integration_state']['vector']),spec);env.mj.mj_forward(env.model,env.data);target_finger=0.0;place=None  # Preserve V1's zero finger target during diagnostics.
  ref=peg_relative(env);p0,q0=env.tcp_state();contract=PhysicsContract();contract.reset(p0,q0,env.data.xpos[env.hole_body_id],env.data.xquat[env.hole_body_id],read(R2/'initials'/name/'initial_state.json')['episode_noise_m'],timestamp=float(env.data.time));shadow=make_model(False)
 expected=snapshot(env.mj,env.model,env.data)
 registered_expected=read(parent/('registered_initial_'+name+'.json')) if prep else prepared['integrity_after_forward']
 try:guard(registered_expected,env.mj,env.model,env.data,'registered_initial_before_viewer')
 except StateIntegrityError as e:
  failure={'status':'INVALID_INITIAL_STATE','boundary':e.boundary,'mismatches':e.mismatches,'expected':e.expected,'actual':e.actual};write(out/'state_integrity_failure.json',failure);write(out/'result.json',{'run':run,'completed_steps':0,'durable_integration_requests':0,'durable_integration_results':0,'runtime_failure':failure,'policy_calls':0});return 2
 initial_snapshot=env.snapshot();initial_snapshot['s2a']=physical_measurements(env)
 identity={'run':run,'model':model_identity(env),'model_fingerprint':model_fingerprint(env.model),'mujoco_version':env.mj.__version__,'timestep':env.model.opt.timestep,'source_sha256':reg['sources'],'initial_sha256':sha(R2/'initials'/name/'admission.json'),
  'finger_qpos_ids':env.finger_qpos_ids,'finger_dof_ids':env.finger_dof_ids,'arm_qpos_ids':env.arm_qpos_ids,'arm_dof_ids':env.arm_dof_ids,
  'finger_joint_range':env.model.jnt_range[env.finger_joint_ids],'arm_joint_range':env.model.jnt_range[env.arm_joint_ids],
  'finger_actuator_id':int(env.finger_actuator_ids[0]),'kp':KP,'kv':KV,'force_limit_n':FORCE_LIMIT_N,'initial':initial_snapshot,'initial_integrity':expected,'placement':place}
 write(out/'identity.json',identity)
 path=os.environ.get('FACTORY_DURABLE_EVENT_DIR')
 if not path or Path(path).resolve()!=out.parent/'events':raise ValueError('DURABLE_PATH_MISMATCH')
 ledger=DurableEventLog(path,run_id=run);rows=[];stops=[];failure=None;visual=None;gate=None;losses=0;started=time.monotonic();seen_contact=False;seen_hold=False
 try:
  with IsolatedViewer(env,run) as visual:
   if visual.model is env.model or visual.data is env.data:raise RuntimeError('VIEWER_NOT_ISOLATED')
   guard(expected,env.mj,env.model,env.data,'after_viewer_open');write(out/'gui_admission.json',{'status':'GUI_OPEN_VISIBLE','window':visual.info,'physics_model_fingerprint':model_fingerprint(env.model),'viewer_model_fingerprint':model_fingerprint(visual.model),'distinct_model':visual.model is not env.model,'distinct_data':visual.data is not env.data,'DISPLAY':os.environ.get('DISPLAY'),'XAUTHORITY':os.environ.get('XAUTHORITY')})
   capture(visual,out/'step_000.png',0,env.snapshot(),reg['model_xml_sha256']);guard(expected,env.mj,env.model,env.data,'after_initial_screenshot')
   with (out/'physics.jsonl').open('x') as stream:
    for step in range(1,MAX_PREPARATION_STEPS+1 if prep else 236):
     visual.sync(step-1);actual_before_control=guard(expected,env.mj,env.model,env.data,'before_control_update')
     before=env.snapshot();before['gripper']=actuation(env);warnings=np.array([w.number for w in env.data.warning]);torque=np.zeros(7);control=None;target=None;schedule_record=None
     if prep:
      forces=contact_forces(before);schedule,schedule_record=advance(schedule,forces['left'],forces['right']);target_finger=schedule.target_m
     else:
      if step in (1,9):contract.policy_update(reg['diagnostic_schedule']['archived_clipped_actions'][(step-1)//8])
      p,q=env.tcp_state();contract.update_physics(p,q,float(env.data.time));target=fixed_reference(step,p0,q0,contract)
      c0=mass_at(shadow,env.data.qpos)[np.ix_(env.arm_dof_ids,env.arm_dof_ids)]
      control=torque_command(q7=env.data.qpos[env.arm_qpos_ids].copy(),qdot7=env.data.qvel[env.arm_dof_ids].copy(),position=p,quaternion=q,linear_fd=contract.linear_fd.copy(),angular_fd=contract.angular_fd.copy(),jacobian=env.controller_jacobian(),mass_matrix=c0,target_position=target['position_target'],target_quaternion=target['quaternion_target']);torque=control['requested']
     state,stage,expected_step,actual_step=one_step(env,torque,target_finger,ledger,step,expected)
     actual_after=snapshot(env.mj,env.model,env.data);state['s2a']=physical_measurements(env,None if prep else ref);forces=contact_forces(state);wc=(np.array([w.number for w in env.data.warning])-warnings).tolist()
     pads=[c for c in state['contacts'] if 'peg_collision' in (c['geom1'],c['geom2']) and any(g in {side+'_finger_geom_'+str(k) for side in ('left','right') for k in range(3,8)} for g in (c['geom1'],c['geom2']))]
     pen=max([max(0.,-c['distance_m']) for c in pads],default=0.);_,interference=finger_forces(env)
     interference += ['non_grasp_contact:'+str(c['geom1'])+':'+str(c['geom2']) for c in state['contacts'] if 'peg_collision' not in (c['geom1'],c['geom2']) and c['distance_m'] < -1e-6]
     if prep:
      m=state['s2a'];stops=[]
      for bad,code in [(m['all_contact_penetration_m']>THRESHOLDS['stop_penetration_m'],'SEVERE_PENETRATION'),(m['arm_joint_limit_violation_rad']>THRESHOLDS['stop_arm_limit_rad'],'ARM_LIMIT'),(m['finger_joint_limit_violation_m']>THRESHOLDS['stop_finger_limit_m'],'FINGER_LIMIT'),(max(abs(np.asarray(state['qvel'])[env.arm_dof_ids]))>THRESHOLDS['stop_arm_velocity_rad_s'],'ARM_VELOCITY'),(any(wc),'NUMERICAL_WARNING'),(not finite_tree(state),'NONFINITE')]:
       if bad:stops.append(code)
     else:
      losses=losses+1 if min(forces.values())<=THRESHOLDS['positive_pad_force_n'] else 0;stops=stop_reasons(state,forces,losses,wc)
     row={'event':'physics_result','phase':'PREPARATION' if prep else 'DIAGNOSTIC','step':step,'pre_state':before,'target':target,'controller':control,'selected_arm_torque_nm':torque,'finger_target_m':target_finger,'gripper_schedule':schedule_record,
      'expected_before_control':expected,'actual_before_control':actual_before_control,'expected_before_step':expected_step,'actual_before_step':actual_step,'actual_after_forward':actual_after,'viewer_sync_count':visual.sync_count,
      'state':state,'pad_force_n':forces,'pad_penetration_m':pen,'interference_bodies':interference,'warning_delta':wc,'stop_reasons':stops,'integration_stage':stage,'post_forward_solver':solver_and_contacts(env),'post_forward_gripper':actuation(env)}
     line(stream,row);rows.append(safe_value(row));expected=actual_after;visual.sync(step,state)
     firstcontact=not seen_contact and max(forces.values())>1e-6
     firsthold=prep and not seen_hold and schedule.phase in ('GRASP_HOLD','COMPLETE')
     if firstcontact:seen_contact=True
     if firsthold:seen_hold=True
     if stops or firstcontact or firsthold or step in (50,100,150,200,235,240) or (prep and schedule.phase=='COMPLETE'):capture(visual,out/f'step_{step:03d}.png',step,state,reg['model_xml_sha256'])
     guard(expected,env.mj,env.model,env.data,'after_viewer_and_optional_screenshot')
     if stops or (prep and schedule.phase=='COMPLETE'):break
   if prep:
    gate=preparation_gate(env,rows,place,schedule,stops);write(out/'preparation_gate.json',gate)
    if gate['reset_admission']=='PASS' and gate['physical_quality']=='PASS':write(out/'prepared_state.json',{'state':env.snapshot(),'integration_state':integration_state(env),'finger_target_m':target_finger,'integrity_after_forward':expected})
 except StateIntegrityError as e:
  failure={'type':type(e).__name__,'message':str(e),'boundary':e.boundary,'mismatches':e.mismatches,'expected':e.expected,'actual':e.actual,'status':'INVALID_INITIAL_STATE' if not rows else 'STATE_INTEGRITY_FAILURE'};write(out/'state_integrity_failure.json',failure)
 except Exception as e:failure={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 events=[json.loads(x) for x in (Path(path)/'events.jsonl').read_text().splitlines()] if (Path(path)/'events.jsonl').exists() else []
 result={'run':run,'completed_steps':len(rows),'durable_integration_requests':sum(x['event']=='integration_request' for x in events),'durable_integration_results':sum(x['event']=='physics_result' for x in events),'policy_calls':0,'early_stop_reasons':stops,'runtime_failure':failure,'preparation_gate':gate,'wall_s':time.monotonic()-started,'viewer_sync_count':0 if visual is None else visual.sync_count,'graceful_close':None if visual is None else visual.receipt,'trace_sha256':sha(out/'physics.jsonl') if (out/'physics.jsonl').exists() else None}
 write(out/'result.json',result)
 return 0 if failure is None else 2

def coordinate(out):
 out=Path(out).resolve();reg=verify(out/'preregistered_matrix.json');write(out/'coordinator_attempt.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'retry':False,'preflight_sha256':reg['preflight_sha256']})
 start=time.monotonic();receipts=[];unstarted=[]
 for run in RUNS:
  if run.startswith('DIAG_'):
   paths=[out/'runs'/p/'output/preparation_gate.json' for p in RUNS[:2]]
   if not all(p.exists() for p in paths) or any((g:=read(p))['reset_admission']!='PASS' or g['physical_quality']!='PASS' for p in paths):unstarted=RUNS[len(receipts):];break
  remain=900-(time.monotonic()-start)
  if remain<=0:unstarted=RUNS[len(receipts):];break
  directory=out/'runs'/run;cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',run,'--manifest',str(out/'preregistered_matrix.json'),'--output',str(directory/'output')]
  r=ChildSupervisor(directory,cmd,cwd=ROOT,env={'FACTORY_DURABLE_EVENT_DIR':str(directory/'events')},timeout_s=remain).run();receipts.append(r)
  # I2/I3 attempts are independent, including failures. No child is retried.
  if r['timed_out']:unstarted=RUNS[len(receipts):];break
  if run.startswith('DIAG_') and r['raw_wait_returncode']!=0:unstarted=RUNS[len(receipts):];break
 results=[read(out/'runs'/run/'output/result.json') for run in RUNS if (out/'runs'/run/'output/result.json').exists()]
 ledgers=[out/'runs'/run/'events/events.jsonl' for run in RUNS if (out/'runs'/run/'events/events.jsonl').exists()];ev=[json.loads(row) for path in ledgers for row in path.read_text().splitlines()];requests=sum(x['event']=='integration_request' for x in ev);completed=sum(x['event']=='physics_result' for x in ev)
 summary={'task':'G4-S2A-GRIPPER-V1-R1','receipts':receipts,'unstarted':unstarted,'completed_integration_steps':completed,'integration_requests':requests,'unresolved_requests':requests-completed,'completed_trace_rows':sum(x['completed_steps'] for x in results),'wall_s':time.monotonic()-start,'policy_calls':0,'retry':False,'max_steps':950,'max_wall_s':900,'result_runs':[x['run'] for x in results]}
 write(out/'summary.json',summary)
 return 0 if len(receipts)>=2 and summary['completed_integration_steps']<=950 and summary['wall_s']<=900 else 2

def main():
 p=argparse.ArgumentParser();p.add_argument('--preregister',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--child',choices=RUNS);p.add_argument('--manifest',type=Path);p.add_argument('--preflight',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.preregister:return register(a.output,a.preflight)
 if a.run:return coordinate(a.output)
 if a.child:return child(a.child,a.manifest,a.output)
 p.error('explicit operation required')
if __name__=='__main__':raise SystemExit(main())
