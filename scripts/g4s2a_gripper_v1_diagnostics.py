#!/usr/bin/env python3
"""Once-only, preregistered visible native-gripper preparation and diagnostics.

No actor import/call; no retry; trace integration-stage solver data before the
native environment's post-step forward. Each candidate preparation is unique.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, hashlib, resource
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv, KP, KV, FORCE_LIMIT_N
from scripts.g4s2a_r2_model import THRESHOLDS, MAX_STEPS, stop_reasons, fixed_reference, make_model, mass_at, model_identity
from scripts.g4s2a_r2_r3_diagnostics import solver_and_contacts, reject_external_perturbation, safe_value
from scripts.g4s2a_r2_diagnostics import Visible, integration_state
from scripts.g4s2a_r2_r2_supervisor import ChildSupervisor, DurableEventLog
from scripts.g4s1b_contract import PhysicsContract, torque_command
from scripts.g4s2a_smoke import physical_measurements, peg_relative, rotation_distance, finger_forces, independent_peg, finite_tree
from scripts.g4s2a_r1_viewer import contact_forces

R2=ROOT/'evidence/g4_s/s2a_r2/matrix_v2'
SOURCES=['scripts/g4s2a_gripper_v1_diagnostics.py','scripts/g4s2a_gripper_v1_model.py','scripts/g4s2a_r2_r3_diagnostics.py','scripts/g4s2a_r2_r2_supervisor.py','scripts/g4s2a_r2_diagnostics.py','scripts/g4s2a_r2_model.py','scripts/g4s1b_env.py','scripts/g4s1b_contract.py','scripts/g4s2a_smoke.py','scripts/g4s2a_r1_viewer.py','scripts/capture_g4s2a_r1_comparison.py']
RUNS=['PREP_I2','PREP_I3','DIAG_I2','DIAG_I3']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(safe_value(v),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def line(f,v):
 f.write(json.dumps(safe_value(v),separators=(',',':'),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def data_paths():
 return {str((p:=R2/'initials'/name/f).relative_to(ROOT)):sha(p) for name in ('I2','I3') for f in ('admission.json','reset_draw.json','ik.json','initial_state.json')}
def register(out,static_receipt):
 receipt=read(static_receipt)
 if receipt.get('status')!='PASS':raise ValueError('STATIC_GATE_NOT_PASSED')
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 env=NativeGripperEnv();(out/'native_gripper_v1.xml').write_text(env.xml)
 schedule=read(R2/'preregistered_matrix.json')['input_schedule']
 reg={'task':'G4-S2A-GRIPPER-V1','start_commit':'ce8815c2586e157a04e1dafb7a25d726c8dba264',
  'scope':'ONE_NATIVE_ARCHITECTURE_NO_POLICY','model_xml_sha256':sha(out/'native_gripper_v1.xml'),
  'source_sha256':{p:sha(ROOT/p) for p in SOURCES},'historical_inputs':data_paths(),
  'static_receipt':str(Path(static_receipt).resolve()),'static_receipt_sha256':sha(static_receipt),
  'runs':RUNS,'preparations':'exactly one per saved pre-closure I2/I3; up to 30 steps; no retry',
  'stage2':'only if both complete reset admission and physical preparation quality',
  'preparation_gate':{'steps':30,'bilateral_last10_n_gt':1e-6,'max_transient_pad_penetration_m':50e-6,
  'max_final10_pad_penetration_m':50e-6,'max_assembly_penetration_m':10e-6,'stability_m_lt':.001,
  'angular_stability_rad_le':.05,'max_arm_speed_rad_s_le':3.,'joint_limit_m_or_rad_le':1e-6,
  'zero_warnings':True,'no_interference':True,'above_bore':True},
  'thresholds':THRESHOLDS,'diagnostic_steps':235,'max_steps':530,'max_wall_s':900,'retry':False,
  'schedule':schedule,'position_mapping':'L=q1+q2; ctrl=2*q_target','kp':KP,'kv':KV,'force_limit_n':FORCE_LIMIT_N,
  'policy_calls':0,'integration_stage':'mj_step returns before wrapper mj_forward; forces are not impulses'}
 write(out/'preregistered_matrix.json',reg)

def verify(path):
 r=read(path)
 if r['runs']!=RUNS or r['thresholds']!=THRESHOLDS or r['max_steps']!=530 or r['max_wall_s']!=900 or r['retry'] is not False:raise ValueError('REGISTERED_CONTRACT_CHANGED')
 if r['historical_inputs']!=data_paths() or {p:sha(ROOT/p) for p in SOURCES}!=r['source_sha256']:raise ValueError('REGISTERED_INPUT_OR_SOURCE_CHANGED')
 if sha(r['static_receipt'])!=r['static_receipt_sha256'] or read(r['static_receipt']).get('status')!='PASS':raise ValueError('STATIC_RECEIPT_CHANGED')
 if sha(Path(path).parent/'native_gripper_v1.xml')!=r['model_xml_sha256']:raise ValueError('XML_CHANGED')
 return r

def initial(env,name):
 record=read(R2/'initials'/name/'admission.json')['initial']; q=np.asarray(record['qpos']);v=np.asarray(record['qvel'])
 if q.shape!=(env.model.nq,) or v.shape!=(env.model.nv,) or not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('INVALID_PRE_CLOSURE_STATE')
 if not np.allclose(q[env.finger_qpos_ids],.00499125,rtol=0,atol=1e-15) or np.max(abs(v))!=0:raise ValueError('PRE_CLOSURE_STATE_INVALID')
 env.mj.mj_resetData(env.model,env.data);env.data.qpos[:]=q;env.data.qvel[:]=v;env.data.time=0
 env.model.body_pos[env.hole_body_id]=record['hole_root_position'];env.data.ctrl[:]=0
 env.mj.mj_forward(env.model,env.data)
 return record

def actuation(env):
 d=env.data;m=env.model;aid=int(env.finger_actuator_ids[0]);tid=int(env.tendon_id)
 # Sparse actuator moments are reconstructed from the known tendon Jacobian
 # using MuJoCo's actuator_moment row indexing, not assumed joint forces.
 dense=np.zeros(m.nv)
 if d.actuator_moment.size==m.nu*m.nv:
  dense[:]=np.asarray(d.actuator_moment).reshape(m.nu,m.nv)[aid]
 else:
  start=int(d.moment_rowadr[aid]);count=int(d.moment_rownnz[aid]);dense[d.moment_colind[start:start+count]]=d.actuator_moment[start:start+count]
 return {'tendon_length_m':float(d.ten_length[tid]),'tendon_velocity_m_s':float(d.ten_velocity[tid]),
  'control_m':float(d.ctrl[aid]),'actuator_force_n':float(d.actuator_force[aid]),
  'moment_arm_by_dof':dense,'finger_generalized_force_n':d.qfrc_actuator[env.finger_dof_ids].copy()}

def one_step(env,torque,ledger,step):
 mj=env.mj;original=mj.mj_step;captured=[]
 def observe(model,data):
  if captured:raise RuntimeError('MULTIPLE_INTEGRATIONS')
  reject_external_perturbation(env);start=float(data.time)
  ledger.record_request(step,simulation_time_s=start)
  try:original(model,data)
  except BaseException as e:ledger.record_exception(step,e);raise
  stage=solver_and_contacts(env);stage.update({'stage':'immediately_after_mj_step_before_wrapper_mj_forward','simulation_start_time_s':start,'simulation_end_time_s':float(data.time),'actuator_force':data.actuator_force.copy(),'ctrl':data.ctrl.copy(),'gripper':actuation(env)})
  ledger.record_result(step,integration_stage=safe_value(stage));captured.append(stage)
 mj.mj_step=observe
 try:state=env.physics_step(torque,finger_target=0.,record=False)
 finally:mj.mj_step=original
 if len(captured)!=1:raise RuntimeError('ONE_INTEGRATION_REQUIRED')
 return state,captured[0]

def prep_gate(env,rows,placement):
 if not rows:return {'reset_admission':'FAIL','physical_quality':'FAIL','errors':['NO_STEPS']}
 tail=rows[-10:];positions=[np.asarray(r['state']['s2a']['relative_peg_hand_m']) for r in tail];rots=[np.asarray(r['state']['s2a']['relative_peg_hand_rotation']) for r in tail]
 drift=max(float(np.linalg.norm(x-positions[-1])) for x in positions);rotation=max(rotation_distance(x,rots[-1]) for x in rots)
 allpad=max(r['pad_penetration_m'] for r in rows);lastpad=max(r['pad_penetration_m'] for r in tail)
 assembly=max(r['state']['s2a']['peg_hole_penetration_m'] for r in rows)
 bilateral=len(tail)==10 and all(min(r['pad_force_n'].values())>1e-6 for r in tail)
 limits=max(max(r['state']['s2a']['arm_joint_limit_violation_rad'],r['state']['s2a']['finger_joint_limit_violation_m']) for r in rows)
 armvel=max(max(abs(np.asarray(r['state']['qvel'])[env.arm_dof_ids])) for r in rows)
 interference=sorted(set(x for r in tail for x in r['interference_bodies']))
 final_above=bool(env.insertion_diagnostics()['entirely_above_opening'] and env.tcp_state()[0][2]>.075)
 reset_errors=[]
 for bad,code in [(len(rows)!=30,'INCOMPLETE_PREPARATION'),(not placement['valid'],'INITIAL_PLACEMENT'),(not bilateral,'BILATERAL_LAST10'),(drift>=.001,'STABILITY'),(rotation>.05,'ANGULAR_STABILITY'),(limits>1e-6,'JOINT_LIMIT'),(bool(interference),'INTERFERENCE'),(not final_above,'NOT_ABOVE_BORE')]:
  if bad:reset_errors.append(code)
 quality_errors=[]
 for bad,code in [(len(rows)!=30,'INCOMPLETE_PREPARATION'),(allpad>50e-6,'TRANSIENT_PAD_PENETRATION'),(lastpad>50e-6,'FINAL10_PAD_PENETRATION'),(assembly>10e-6,'ASSEMBLY_PENETRATION'),(armvel>3,'ARM_VELOCITY'),(any(any(r['warning_delta']) for r in rows),'NUMERICAL_WARNINGS'),(not all(finite_tree(r) for r in rows),'NONFINITE'),(bool(reset_errors),'RESET_NOT_ADMITTED')]:
  if bad:quality_errors.append(code)
 return {'reset_admission':'FAIL' if reset_errors else 'PASS','physical_quality':'FAIL' if quality_errors else 'PASS',
  'reset_errors':reset_errors,'quality_errors':quality_errors,'bilateral_last10':bilateral,'transient_max_pad_penetration_m':allpad,
  'final10_max_pad_penetration_m':lastpad,'max_assembly_penetration_m':assembly,'final10_stability_m':drift,
  'final10_angular_stability_rad':rotation,'max_joint_limit_violation':limits,'max_arm_velocity_rad_s':armvel,'interference_bodies':interference,'final_above_hole':final_above}

def placement(env):
 diag=env.insertion_diagnostics();contacts=env.contacts();forbidden=[c for c in contacts if c['distance_m'] < -1e-6]
 return {'valid':bool(independent_peg(env) and diag['entirely_above_opening'] and env.tcp_state()[0][2]>.075 and not forbidden),
         'independent_peg':independent_peg(env),'entirely_above_opening':diag['entirely_above_opening'],'forbidden_initial_contacts':forbidden,
         'finger_openings_m':env.data.qpos[env.finger_qpos_ids].copy(),'preclosure_contacts':contacts}

def child(run,manifest,out):
 reg=verify(manifest);name=run[-2:];prep=run.startswith('PREP_'); parent=Path(manifest).resolve().parent
 if run not in RUNS or Path(out).resolve()!=parent/'runs'/run/'output' or not (parent/'coordinator_attempt.json').exists():raise ValueError('UNREGISTERED_CHILD')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);env=NativeGripperEnv()
 if hashlib.sha256(env.xml.encode()).hexdigest()!=reg['model_xml_sha256']:raise ValueError('PLANT_XML_MISMATCH')
 if prep: prior=initial(env,name)
 else:
  source=read(parent/'runs'/('PREP_'+name)/'output/prepared_state.json');spec=env.mj.mjtState(source['integration_state']['value'])
  env.mj.mj_setState(env.model,env.data,np.asarray(source['integration_state']['vector']),spec);env.mj.mj_forward(env.model,env.data);prior=source['state']
 place=placement(env) if prep else None
 if prep and not place['valid']:raise ValueError('PRE_CLOSURE_PLACEMENT_FAILED')
 if not prep:
  ref=peg_relative(env);p0,q0=env.tcp_state();contract=PhysicsContract();contract.reset(p0,q0,env.data.xpos[env.hole_body_id],env.data.xquat[env.hole_body_id],read(R2/'initials'/name/'initial_state.json')['episode_noise_m'],timestamp=float(env.data.time));shadow=make_model(False)
 path=os.environ.get('FACTORY_DURABLE_EVENT_DIR')
 if not path or Path(path).resolve()!=out.parent/'events':raise ValueError('DURABLE_PATH_MISMATCH')
 ledger=DurableEventLog(path,run_id=run)
 identity={'run':run,'model':model_identity(env),'mujoco_version':env.mj.__version__,'timestep':env.model.opt.timestep,
  'source_sha256':reg['source_sha256'],'initial_sha256':sha(R2/'initials'/name/'admission.json'),
  'finger_qpos_ids':env.finger_qpos_ids,'finger_dof_ids':env.finger_dof_ids,'arm_qpos_ids':env.arm_qpos_ids,'arm_dof_ids':env.arm_dof_ids,
  'finger_joint_range':env.model.jnt_range[env.finger_joint_ids],'arm_joint_range':env.model.jnt_range[env.arm_joint_ids],
  'finger_actuator_id':int(env.finger_actuator_ids[0]),'kp':KP,'kv':KV,'force_limit_n':FORCE_LIMIT_N,'initial':prior,'placement':place}
 write(out/'identity.json',identity)
 rows=[];stops=[];failure=None;visual=None;started=time.monotonic();maxstep=30 if prep else 235;losses=0
 try:
  with Visible(env,run+' NATIVE GRIPPER V1') as visual:
   with visual.viewer.lock():visual.viewer.opt.flags[env.mj.mjtVisFlag.mjVIS_CONTACTPOINT]=False
   write(out/'gui_admission.json',{'status':'GUI_OPEN_VISIBLE','window':visual.info,'mujoco_version':env.mj.__version__,'DISPLAY':os.environ.get('DISPLAY'),'XAUTHORITY':os.environ.get('XAUTHORITY')})
   visual.capture(out/'step_000.png',0,env.snapshot(),reg['model_xml_sha256'])
   with (out/'physics.jsonl').open('x') as stream:
    for step in range(1,maxstep+1):
     visual.sync(step-1);reject_external_perturbation(env)
     before=env.snapshot();before['gripper']=actuation(env);warnings=np.array([x.number for x in env.data.warning]);torque=np.zeros(7);control=None;target=None
     if not prep:
      if step in (1,9):contract.policy_update(reg['schedule']['archived_clipped_actions'][(step-1)//8])
      p,q=env.tcp_state();contract.update_physics(p,q,float(env.data.time));target=fixed_reference(step,p0,q0,contract)
      c0=mass_at(shadow,env.data.qpos)[np.ix_(env.arm_dof_ids,env.arm_dof_ids)]
      control=torque_command(q7=env.data.qpos[env.arm_qpos_ids].copy(),qdot7=env.data.qvel[env.arm_dof_ids].copy(),position=p,quaternion=q,linear_fd=contract.linear_fd.copy(),angular_fd=contract.angular_fd.copy(),jacobian=env.controller_jacobian(),mass_matrix=c0,target_position=target['position_target'],target_quaternion=target['quaternion_target']);torque=control['requested']
     try:state,integrated=one_step(env,torque,ledger,step)
     except Exception:
      line(stream,{'event':'integration_exception','step':step,'traceback':traceback.format_exc(), 'qpos':env.data.qpos.copy(),'qvel':env.data.qvel.copy(),'time_s':float(env.data.time),'warning_counts':[w.number for w in env.data.warning],'post_exception_solver':solver_and_contacts(env)});raise
     state['s2a']=physical_measurements(env,None if prep else ref);forces=contact_forces(state);wc=(np.array([x.number for x in env.data.warning])-warnings).tolist()
     pads=[c for c in state['contacts'] if 'peg_collision' in (c['geom1'],c['geom2']) and any(g in {side+'_finger_geom_'+str(k) for side in ('left','right') for k in range(3,8)} for g in (c['geom1'],c['geom2']))]
     pen=max([max(0.,-c['distance_m']) for c in pads],default=0.);_,interference=finger_forces(env)
     interference += ['non_grasp_contact:'+str(c['geom1'])+':'+str(c['geom2']) for c in state['contacts'] if 'peg_collision' not in (c['geom1'],c['geom2']) and c['distance_m'] < -1e-6]
     if prep:
      m=state['s2a'];stops=[]
      for bad,code in [(m['all_contact_penetration_m']>THRESHOLDS['stop_penetration_m'],'SEVERE_PENETRATION'),(m['arm_joint_limit_violation_rad']>THRESHOLDS['stop_arm_limit_rad'],'ARM_LIMIT'),(m['finger_joint_limit_violation_m']>THRESHOLDS['stop_finger_limit_m'],'FINGER_LIMIT'),(max(abs(np.asarray(state['qvel'])[env.arm_dof_ids]))>THRESHOLDS['stop_arm_velocity_rad_s'],'ARM_VELOCITY'),(any(wc),'NUMERICAL_WARNING'),(not finite_tree(state),'NONFINITE')]:
       if bad:stops.append(code)
     else:
      losses=losses+1 if min(forces.values())<=THRESHOLDS['positive_pad_force_n'] else 0;stops=stop_reasons(state,forces,losses,wc)
     row={'event':'physics_result','phase':'PREPARATION' if prep else 'DIAGNOSTIC','step':step,'pre_state':before,'target':target,'controller':control,'selected_arm_torque_nm':torque,'state':state,'pad_force_n':forces,'pad_penetration_m':pen,'interference_bodies':interference,'warning_delta':wc,'stop_reasons':stops,'integration_stage':integrated,'post_forward_solver':solver_and_contacts(env),'post_forward_gripper':actuation(env)}
     line(stream,row);rows.append(safe_value(row));visual.sync(step,state)
     if prep or stops or step in (16,235):visual.capture(out/f'step_{step:03d}.png',step,state,reg['model_xml_sha256'])
     if stops:break
   if prep:
    gate=prep_gate(env,rows,place);write(out/'preparation_gate.json',gate)
    if gate['reset_admission']=='PASS' and gate['physical_quality']=='PASS':write(out/'prepared_state.json',{'state':env.snapshot(),'integration_state':integration_state(env)})
   else:gate=None
 except Exception as e:failure={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 result={'run':run,'completed_steps':len(rows),'policy_calls':0,'early_stop_reasons':stops,'runtime_failure':failure,'preparation_gate':gate if failure is None else None,'wall_s':time.monotonic()-started,'graceful_close':None if visual is None else visual.receipt,'trace_sha256':sha(out/'physics.jsonl') if (out/'physics.jsonl').exists() else None}
 write(out/'result.json',result)
 return 0 if failure is None else 2

def coordinate(out,cpu):
 out=Path(out).resolve();reg=verify(out/'preregistered_matrix.json')
 if not read(cpu).get('passed'):raise ValueError('CPU_GATE_NOT_PASSED')
 write(out/'coordinator_attempt.json',{'start_utc':datetime.now(timezone.utc).isoformat(),'cpu_receipt':str(Path(cpu).resolve()),'cpu_sha256':sha(cpu),'retry':False})
 start=time.monotonic();receipts=[];unstarted=[]
 for run in RUNS:
  if run.startswith('DIAG_'):
   gates=[read(out/'runs'/p/'output/preparation_gate.json') for p in RUNS[:2]]
   if any(g['reset_admission']!='PASS' or g['physical_quality']!='PASS' for g in gates):unstarted=RUNS[len(receipts):];break
  remain=900-(time.monotonic()-start)
  if remain<=0:unstarted=RUNS[len(receipts):];break
  directory=out/'runs'/run;cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',run,'--manifest',str(out/'preregistered_matrix.json'),'--output',str(directory/'output')]
  receipt=ChildSupervisor(directory,cmd,cwd=ROOT,env={'FACTORY_DURABLE_EVENT_DIR':str(directory/'events')},timeout_s=remain).run();receipts.append(receipt)
  if receipt['raw_wait_returncode']!=0 or receipt['timed_out']:unstarted=RUNS[len(receipts):];break
 results=[read(out/'runs'/run/'output/result.json') for run in RUNS if (out/'runs'/run/'output/result.json').exists()]
 summary={'task':'G4-S2A-GRIPPER-V1','receipts':receipts,'unstarted':unstarted,'completed_integration_steps':sum(x['completed_steps'] for x in results),'wall_s':time.monotonic()-start,'policy_calls':0,'retry':False,'max_steps':530,'max_wall_s':900,'result_runs':[x['run'] for x in results]}
 write(out/'summary.json',summary)
 return 0 if len(receipts)>=2 and all(r['raw_wait_returncode']==0 for r in receipts) and summary['completed_integration_steps']<=530 and summary['wall_s']<=900 else 2

def main():
 p=argparse.ArgumentParser();p.add_argument('--preregister',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--child',choices=RUNS);p.add_argument('--manifest',type=Path);p.add_argument('--static-receipt',type=Path);p.add_argument('--cpu-receipt',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.preregister:return register(a.output,a.static_receipt)
 if a.run:return coordinate(a.output,a.cpu_receipt)
 if a.child:return child(a.child,a.manifest,a.output)
 p.error('explicit operation required')
if __name__=='__main__':raise SystemExit(main())
