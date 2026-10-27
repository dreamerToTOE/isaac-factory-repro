#!/usr/bin/env python3
"""Once-only visible R2 factorial diagnostics. NO_POLICY_DIAGNOSTIC_ONLY.

Preregister -> guarded CPU tests -> execute. Never retries a prepared pose or a
trajectory. 12*235+2*30=2880 total new integration calls, including preparation.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gzip,hashlib,json,os,resource,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s2a_r2_model import ModelPair,make_model,model_identity,VARIANTS,MAX_STEPS,THRESHOLDS,digest,fixed_reference,stop_reasons
from scripts.g4s1b_contract import PhysicsContract,torque_command
from scripts.g4s2a_smoke import jsonable,physical_measurements,peg_relative,prepare_physical_episode,finite_tree
from scripts.g4s2a_r1_viewer import load_archive,apply_archived_state,ARCHIVE,window_info,capture_desktop_window,contact_forces,require_gui_environment,close_and_join,configure_camera
from scripts.capture_g4s2a_r1_comparison import raise_own_window

INITIALS=[{'id':'I1','kind':'archived_admitted','seed':1001},
          {'id':'I2','kind':'single_physical_preparation','seed':2101},
          {'id':'I3','kind':'single_physical_preparation','seed':2102}]
SOURCES=['scripts/g4s2a_r2_diagnostics.py','scripts/g4s2a_r2_model.py',
 'scripts/g4s1b_env.py','scripts/g4s1b_contract.py','scripts/g4s2a_smoke.py',
 'scripts/g4s2a_r1_viewer.py','scripts/capture_g4s2a_r1_comparison.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 with Path(path).open('x') as f:json.dump(jsonable(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def line(f,value):f.write(json.dumps(jsonable(value),allow_nan=False)+'\n');f.flush()
def read(path):return json.loads(Path(path).read_text())

def verify_registry(path):
 reg=read(path)
 for name,h in reg['source_sha256'].items():
  if sha(ROOT/name)!=h:raise ValueError('SOURCE_CHANGED_AFTER_REGISTRATION:'+name)
 if reg['steps_per_trajectory']!=235 or reg['maximum_new_integration_steps']!=2880:raise ValueError('REGISTRY_BUDGET_CHANGED')
 if reg['thresholds']!=THRESHOLDS:raise ValueError('REGISTERED_THRESHOLDS_CHANGED')
 return reg

class Visible:
 def __init__(self,env,label):self.env=env;self.label=label;self.viewer=None;self.threads=[];self.sync_wall=0.;self.sync_count=0;self.receipt=None
 def __enter__(self):
  require_gui_environment()
  import mujoco.viewer
  before={t.ident for t in threading.enumerate()}
  self.viewer=mujoco.viewer.launch_passive(self.env.model,self.env.data,show_left_ui=False,show_right_ui=False)
  self.threads=[t for t in threading.enumerate() if t.ident not in before and '_launch_internal' in t.name]
  if not self.threads:raise RuntimeError('GUI_RENDER_THREAD_MISSING')
  with self.viewer.lock():
   self.viewer.cam.lookat[:]=[.55,0,.23];self.viewer.cam.distance=.85;self.viewer.cam.azimuth=135;self.viewer.cam.elevation=-20
   self.viewer.opt.flags[self.env.mj.mjtVisFlag.mjVIS_CONTACTPOINT]=True
  try:
   self.sync(0);time.sleep(1.0);self.info=window_info();raise_own_window(self.info['window_id']);self.sync(0);time.sleep(2.)
  except Exception:
   self.receipt=close_and_join(self.viewer,self.threads);raise
  return self
 def sync(self,step,state=None,extra=''):
  if not self.viewer.is_running():raise RuntimeError('GUI_CLOSED_BEFORE_NEXT_INTEGRATION')
  env=self.env
  if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied):raise RuntimeError('GUI_EXTERNAL_PERTURBATION_REJECTED')
  force=contact_forces(env.snapshot() if state is None else state)
  slip='' if state is None else f'Slip {state.get("s2a",{}).get("peg_hand_slip_m",0)*1000:.4f} mm'
  self.viewer.set_texts((None,env.mj.mjtGridPos.mjGRID_TOPLEFT,
   f'NO_POLICY_DIAGNOSTIC_ONLY | {self.label}\nLIVE INTEGRATION step={step}; dt=1/120 s; actor calls=0\n'
   f'Pad normal force L/R={force["left"]:.4g}/{force["right"]:.4g} N; {slip}\n'
   'Slow visible pacing DOES NOT change physical control dt\n'+extra,''))
  t=time.perf_counter();self.viewer.sync();self.sync_wall+=time.perf_counter()-t;self.sync_count+=1
 def capture(self,path,step,state,identity):
  self.sync(step,state);configure_camera(self.viewer,'closeup');info=window_info();raise_own_window(info['window_id']);self.sync(step,state)
  time.sleep(.35);capture_desktop_window(path,window_info())
  write(path.with_suffix('.json'),{'experiment_id':self.label,'initial_state_id':self.label.split('_')[0],
   'model_configuration':identity,'step':step,'simulation_time_s':step/120,
   'absolute_simulation_time_s':state['time_s'],'sha256':sha(path),
   'capture_method':'Actual Ubuntu X11 screen rectangle of own raised MuJoCo window',
   'mode':'LIVE_INTEGRATION_SCREENSHOT','obstruction':'Requires independent image QA; IsViewable alone does not prove unobstructed',
   'window':window_info()})
 def __exit__(self,*args):
  if self.viewer is not None:self.receipt=close_and_join(self.viewer,self.threads)

def integration_state(env):
 spec=env.mj.mjtState.mjSTATE_INTEGRATION
 vector=np.zeros(env.mj.mj_stateSize(env.model,spec));env.mj.mj_getState(env.model,env.data,vector,spec)
 return {'spec':'mjSTATE_INTEGRATION','value':int(spec),'vector':vector.tolist(),
  'qacc_warmstart':env.data.qacc_warmstart.tolist(),
  'scope':'Full available new-data integration state; archived S2A warmstarts UNKNOWN and not reconstructed'}

def restore_initial(env,record):
 apply_archived_state(env,record['state'])
 saved=record['integration_state'];spec=env.mj.mjtState(saved['value'])
 vector=np.asarray(saved['vector'],dtype=float)
 if vector.shape!=(env.mj.mj_stateSize(env.model,spec),) or not np.isfinite(vector).all():raise ValueError('INVALID_REGISTERED_INTEGRATION_STATE')
 env.mj.mj_setState(env.model,env.data,vector,spec);env.mj.mj_forward(env.model,env.data)
 if not np.array_equal(env.data.qpos,np.asarray(record['state']['qpos'])):raise ValueError('RESTORE_QPOS_MISMATCH')
 env.last_requested_torque=np.asarray(record['state']['requested_arm_torque_nm']).copy()
 env.last_software_torque=np.asarray(record['state']['software_limited_arm_torque_nm']).copy()
 env.last_finger_command=record['state']['finger_command'].copy()

def verify_child(manifest,output,expected):
 parent=Path(manifest).resolve().parent
 if not (parent/'coordinator_attempt.json').is_file():raise ValueError('COORDINATOR_ADMISSION_REQUIRED')
 if Path(output).resolve()!=parent/expected:raise ValueError('ONCE_ONLY_REGISTERED_OUTPUT_REQUIRED')

def prepare_child(initial_id,manifest,output):
 reg=verify_registry(manifest);item=next(x for x in INITIALS if x['id']==initial_id)
 verify_child(manifest,output,Path('initials')/initial_id)
 output=Path(output);output.mkdir(parents=True,exist_ok=False);env=make_model(False)
 physics_wall=0.;steps=0;failure=None;visual=None;result=None
 try:
  with Visible(env,initial_id+'_PREPARATION_P0') as visual:
   write(output/'gui_admission.json',{'status':'GUI_OPEN_VISIBLE','window':visual.info,'integration_started':False})
   if item['kind']=='archived_admitted':
    state,_,_=load_archive();apply_archived_state(env,state)
    # Recheck present geometry, but do not invent the historical ten-step history.
    state=env.snapshot();state['s2a']=physical_measurements(env)
    original=read(ARCHIVE/'seed_1001/reset_admission.json')
    admitted=original['admission']=='PASS' and min(contact_forces(state).values())>1e-6
    result={'admission':'PASS' if admitted else 'FAIL','historical_admission_sha256':sha(ARCHIVE/'seed_1001/reset_admission.json'),
     'present_forward_pad_force_n':contact_forces(state),'archived_solver_internal_state':'UNKNOWN_NOT_RESTORED',
     'initial':state,'final':state,'closure_steps':0,'admission_errors':[] if admitted else ['ARCHIVED_GRASP_FORWARD_INVALID']}
    noise=read(ARCHIVE/'seed_1001/interface_reset.json')['initial']['episode_noise_m']
   else:
    original_step=env.physics_step
    def bounded_preparation_step(*args,**kwargs):
     nonlocal physics_wall,steps
     if steps>=30:raise RuntimeError('PREPARATION_BUDGET')
     visual.sync(steps);t=time.perf_counter();steps+=1
     state=original_step(*args,**kwargs);physics_wall+=time.perf_counter()-t
     state['s2a']=physical_measurements(env)
     if not finite_tree(state):raise RuntimeError('NONFINITE_PREPARATION')
     if state['s2a']['all_contact_penetration_m']>.001:raise RuntimeError('SEVERE_PREPARATION_PENETRATION')
     if state['s2a']['arm_joint_limit_violation_rad']>THRESHOLDS['stop_arm_limit_rad']:raise RuntimeError('PREPARATION_ARM_LIMIT')
     if state['s2a']['finger_joint_limit_violation_m']>THRESHOLDS['stop_finger_limit_m']:raise RuntimeError('PREPARATION_FINGER_LIMIT')
     if max(abs(np.asarray(state['qvel'])[:7]))>THRESHOLDS['stop_arm_velocity_rad_s']:raise RuntimeError('PREPARATION_ARM_VELOCITY')
     visual.sync(steps,state);time.sleep(.025)
     return state
    env.physics_step=bounded_preparation_step
    result=prepare_physical_episode(env,item['seed'],output)
    noise=env.rng.normal(0.,.001,3).tolist()
   write(output/'admission.json',result)
   if result['admission']=='PASS':
    p,q=env.tcp_state();record={'id':initial_id,'diagnostic_seed':item['seed'],
     'training_or_policy_evaluation_seed':False,'state':env.snapshot(),'integration_state':integration_state(env),
     'episode_noise_m':noise,'initial_position':p,'initial_quaternion':q,
     'source_kind':item['kind'],'preparation_model':'P0; reused identically by all four variants',
     'admission_sha256':sha(output/'admission.json')}
    write(output/'initial_state.json',record)
   visual.sync(steps,env.snapshot(),extra='PREPARATION COMPLETE; independent free peg; no actor');time.sleep(2.)
 except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
 final={'id':initial_id,'preparation_steps':steps,'physics_wall_s':physics_wall,'failure':failure,
  'admission':None if result is None else result['admission'],'policy_inferences':0,
  'gui_sync_wall_s':0 if visual is None else visual.sync_wall,'graceful_close':None if visual is None else visual.receipt}
 write(output/'result.json',final)
 return 0 if failure is None else 2

def trajectory_child(run_id,manifest,output):
 reg=verify_registry(manifest);initial_id,variant=run_id.split('_');pair=ModelPair(variant);env=pair.plant
 verify_child(manifest,output,Path('trajectories')/run_id)
 parent=Path(manifest).parent;initial_path=parent/'initials'/initial_id/'initial_state.json';initial=read(initial_path)
 restore_initial(env,initial);reference=peg_relative(env);p0,q0=env.tcp_state()
 contract=PhysicsContract();contract.reset(p0,q0,env.data.xpos[env.hole_body_id],env.data.xquat[env.hole_body_id],initial['episode_noise_m'],timestamp=float(env.data.time))
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 identities={'plant':model_identity(env),'M0':model_identity(pair.shadows[0]),'M1':model_identity(pair.shadows[1])}
 write(output/'identity.json',{'task':'G4-S2A-R2','scope':'NO_POLICY_DIAGNOSTIC_ONLY','run_id':run_id,'initial_id':initial_id,
  'variant':variant,'plant_selection':pair.plant_selection,'controller_selection':pair.controller_selection,
  'models':identities,'initial_snapshot':env.snapshot(),'integration_state':integration_state(env),
  'initial_state_sha256':sha(initial_path),'input_schedule_sha256':reg['input_schedule_sha256'],
  'controller_implementation_sha256':reg['source_sha256']['scripts/g4s1b_contract.py'],
  'source_sha256':reg['source_sha256'],'initial_ema':contract.action_cache})
 rows=[];stops=[];failure=None;physics_wall=0.;visual=None;loss=0;started=time.monotonic();start_sim=float(env.data.time);attempts=0
 trace_path=output/'physics.jsonl.gz'
 try:
  with Visible(env,run_id+f' P{pair.plant_selection}+C{pair.controller_selection}') as visual:
   write(output/'gui_admission.json',{'status':'GUI_OPEN_VISIBLE','window':visual.info,'integration_started':False})
   visual.capture(output/'initial.png',0,env.snapshot(),identities['plant']['xml_sha256'])
   with gzip.open(trace_path,'xt') as stream:
    for step in range(1,MAX_STEPS+1):
     visual.sync(step-1)
     if time.monotonic()-started>=900:raise TimeoutError('DIAGNOSTIC_CHILD_WALL_BUDGET')
     if step in (1,9):contract.policy_update(reg['input_schedule']['archived_clipped_actions'][(step-1)//8])
     p,q=env.tcp_state();contract.update_physics(p,q,float(env.data.time));target=fixed_reference(step,p0,q0,contract)
     matrices=pair.matrices();inputs={'q7':env.data.qpos[env.arm_qpos_ids].copy(),'qdot7':env.data.qvel[env.arm_dof_ids].copy(),
      'position':p,'quaternion':q,'linear_fd':contract.linear_fd.copy(),'angular_fd':contract.angular_fd.copy(),
      'jacobian':env.controller_jacobian(),'mass_matrix':matrices['controller_M'],
      'target_position':target['position_target'],'target_quaternion':target['quaternion_target']}
     control=torque_command(**inputs);before=env.snapshot();vbefore=env.data.qvel.copy();warnings=[w.number for w in env.data.warning]
     line(stream,{'event':'integration_request','step':step,'time_before_s':float(env.data.time)});attempts+=1;t=time.perf_counter()
     try:state=env.physics_step(control['requested'],finger_target=0.,record=False)
     except Exception:
      # Record actual post-call data even if the frozen wrapper rejected it.
      line(stream,{'event':'integration_exception','step':step,'raw_state':env.snapshot(),
       'traceback':traceback.format_exc(),'warning_counts':[w.number for w in env.data.warning]});raise
     elapsed=time.perf_counter()-t;physics_wall+=elapsed;state['s2a']=physical_measurements(env,reference)
     forces=contact_forces(state);loss=loss+1 if min(forces.values())<=THRESHOLDS['positive_pad_force_n'] else 0
     wc=(np.array([w.number for w in env.data.warning])-warnings).tolist()
     stops=stop_reasons(state,forces,loss,wc)
     if not finite_tree(state):stops.append('NONFINITE_MEASUREMENT')
     actual=np.asarray(state['integration_actuation']['arm_actuator_force_nm']);expected=np.clip(control['software_limited'],[-87]*4+[-12]*3,[87]*4+[12]*3)
     if not np.allclose(actual,expected,atol=1e-10,rtol=0):stops.append('INVALID_ACTUATOR_BEHAVIOR')
     row={'event':'physics_result','run_id':run_id,'step':step,'episode_time_s':float(env.data.time)-start_sim,
      'physics_wall_s':elapsed,'pre_state':before,'matrices':matrices,'controller_input':inputs,'target':target,
      'controller':control,'state':state,'pad_force_n':forces,'qvel_interval_acceleration':(env.data.qvel-vbefore)/(1/120),
      'qacc_post_forward':env.data.qacc.copy(),'qfrc_constraint_post_forward':env.data.qfrc_constraint.copy(),
      'qacc_warmstart':env.data.qacc_warmstart.copy(),'warning_delta':wc,'stop_reasons':stops}
     line(stream,row);rows.append(row);visual.sync(step,state,extra='After step16: fixed INITIAL TCP target; synthetic reference, NOT PPO');time.sleep(.025)
     if step in (4,16,MAX_STEPS) or stops:visual.capture(output/f'physics_{step:03d}.png',step,state,identities['plant']['xml_sha256'])
     if stops:break
   visual.sync(len(rows),rows[-1]['state'] if rows else env.snapshot(),extra='STOP '+','.join(stops) if stops else 'REGISTERED HORIZON COMPLETE');time.sleep(2.)
 except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
 result={'task':'G4-S2A-R2','scope':'NO_POLICY_DIAGNOSTIC_ONLY','run_id':run_id,'variant':variant,'initial_id':initial_id,
  'registered_steps':MAX_STEPS,'integration_requests':attempts,'completed_steps':len(rows),
  'simulation_duration_s':float(env.data.time)-start_sim,'physics_wall_s':physics_wall,'wall_s':time.monotonic()-started,
  'viewer_sync_wall_s':0 if visual is None else visual.sync_wall,'viewer_sync_count':0 if visual is None else visual.sync_count,
  'gui_and_logging_overhead_s':time.monotonic()-started-physics_wall,
  'graceful_close':None if visual is None else visual.receipt,'runtime_failure':failure,'early_stop_reasons':stops,
  'policy_inferences':0,'max_slip_m':max((r['state']['s2a']['peg_hand_slip_m'] for r in rows),default=0.),
  'max_rotation_slip_rad':max((r['state']['s2a']['peg_hand_slip_rad'] for r in rows),default=0.),
  'max_arm_qvel_rad_s':max((max(abs(np.asarray(r['state']['qvel'])[:7])) for r in rows),default=0.),
  'bilateral_positive_contact_fraction':sum(min(r['pad_force_n'].values())>1e-6 for r in rows)/len(rows) if rows else None,
  'first_contact_loss_step':next((r['step'] for r in rows if min(r['pad_force_n'].values())<=1e-6),None),
  'max_penetration_m':max((r['state']['s2a']['all_contact_penetration_m'] for r in rows),default=0.),
  'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'trace_sha256':sha(trace_path) if trace_path.exists() else None,
  'termination':'RUNTIME_FAILURE' if failure else ('DIAGNOSTIC_EARLY_STOP' if stops else 'REGISTERED_HORIZON_COMPLETE')}
 write(output/'result.json',result);return 0 if failure is None else 2

def preregister(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 _,_,_=load_archive();actions=[json.loads(x)['clipped'] for x in gzip.open(ARCHIVE/'seed_1001/policy.jsonl.gz','rt')][:2]
 schedule={'steps_1_to_8':'First archived six-dimensional clipped action; single EMA update at1',
  'steps_9_to_16':'Second archived action; single EMA update at9',
  'steps_17_to_235':'Fixed initial admitted world TCP position and quaternion. No further EMA/action updates; same target across matched variants.',
  'archived_clipped_actions':actions,'policy_inferences':0,'reason':'Two archived excitation intervals followed by fixed-pose regulation; no synthetic PPO decisions',
  'fixed_before_new_integration':True}
 models={}
 for key,adapted in [('P0',False),('P1',True)]:
  env=make_model(adapted);models[key]=model_identity(env)
  (output/(key+'_diagnostic.xml')).write_text(env.xml)
 reg={'task':'G4-S2A-R2','scope':'NO_POLICY_DIAGNOSTIC_ONLY','generated_utc':datetime.now(timezone.utc).isoformat(),
  'starting_commit':'292ca516e4909cabe2de425b28909f790f4cf351','source_sha256':{p:sha(ROOT/p) for p in SOURCES},
  'initials':INITIALS,'variants':{k:{'plant':v[0],'arm_controller_M':v[1]} for k,v in VARIANTS.items()},
  'matrix_order':[i['id']+'_'+v for i in INITIALS for v in VARIANTS],
  'steps_per_trajectory':MAX_STEPS,'maximum_trajectory_steps':2820,'maximum_preparation_steps':60,
  'maximum_new_integration_steps':2880,'maximum_wall_s':900,'no_retry':True,
  'thresholds':THRESHOLDS,'input_schedule':schedule,'input_schedule_sha256':digest(schedule),'models':models,
  'preparation_protocol':'Reuse frozen S2A reset draw/once IK/30-step zero-arm-torque physical closure at P0; seeds2101/2102 each exactly one attempt. No resampling. Last10 bilateral>1e-6N, drift<=1mm/0.05rad, joint<=1e-6, no peg non-pad interference, peg entirely above bore. Failure removes no denominator; all four marked SKIPPED_ADMISSION.',
  'threshold_interpretation':{'slip':'1mm acceptance=17.5 nominal radial gaps; grasp retention only, never insertion quality',
   'rotation':'.05rad acceptance(~2.86deg), .2rad escape guard',
   'velocity':'3rad/s experiment plausibility gate, not manufacturer certified velocity.15rad/s abort before extended severe divergence.',
   'joint_limits':'1e-6rad/m acceptance numerical tolerance; abort1e-5rad arm /100um finger. Any acceptance exceedance fails gate.',
   'penetration':'50um pad quality and10um assembly quality separate;1mm immediate abort is severe relative7.986mm peg diameter. Never changes original10um insertion threshold.',
   'contact':'All recorded steps must have both positive pad forces>1uN for sustained-grasp PASS;3 consecutive losses abort.',
   'interface':'Complete pre/post trace, torque/actuation/model/initial/schedule hashes, exact process status, no new warning or nonfinite'},
  'finger_semantics':'Frozen step-implicit physical full-M dependency retained; P1 can affect finger command as part of plant coupling. C only selects arm/nullspace M.',
  'regression_required_before_execution':True,'official_environment_changed':False,'policy_calls':0}
 write(output/'preregistered_matrix.json',reg);return 0

def coordinate(output,cpu_receipt):
 output=Path(output).resolve();manifest=output/'preregistered_matrix.json';reg=verify_registry(manifest)
 cpu=read(cpu_receipt)
 if not cpu.get('passed'):raise ValueError('CPU_PRELAUNCH_GATE_NOT_PASSED')
 write(output/'coordinator_attempt.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
  'cpu_receipt':str(cpu_receipt),'cpu_sha256':sha(cpu_receipt),'policy_calls':0,'retry':False})
 started=time.monotonic();receipts=[];failed=False
 def launch(kind,label,directory):
  nonlocal failed
  cmd=[sys.executable,'-u',str(Path(__file__).resolve()),kind,label,'--manifest',str(manifest),'--output',str(directory)]
  t=time.monotonic()
  timed_out=False
  with (output/(label+'_console.txt')).open('x') as f:
   child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   try:code=child.wait(timeout=max(1.,900-(time.monotonic()-started)))
   except subprocess.TimeoutExpired:
    timed_out=True;child.kill();code=child.wait()
  record={'id':label,'command':cmd,'exact_child_exit_code':code,'supervisor_timeout':timed_out,'wall_s':time.monotonic()-t,
   'result':read(directory/'result.json') if (directory/'result.json').exists() else None}
  write(output/(label+'_process.json'),record);receipts.append(record)
  print(json.dumps({'id':label,'exit':code,'result':record['result']}),flush=True)
  failed|=code!=0;return record
 for initial in INITIALS:
  if failed or time.monotonic()-started>=900:break
  prep=launch('--prepare',initial['id'],output/'initials'/initial['id'])
  if failed:break
  if prep['result']['admission']!='PASS':
   for variant in VARIANTS:
    write(output/(initial['id']+'_'+variant+'_skipped.json'),{'status':'SKIPPED_ADMISSION','initial_id':initial['id'],'variant':variant,'kept_in_registered_denominator':True})
   continue
  for variant in VARIANTS:
   if failed or time.monotonic()-started>=900:break
   launch('--trajectory',initial['id']+'_'+variant,output/'trajectories'/(initial['id']+'_'+variant))
 total=sum((r['result'].get('preparation_steps',0)+r['result'].get('integration_requests',0)) for r in receipts if r['result'])
 complete=len(receipts)+len(list(output.glob('*_skipped.json')))==15
 result={'task':'G4-S2A-R2','receipts':receipts,'new_integration_requests':total,
  'maximum_steps':2880,'wall_s':time.monotonic()-started,'maximum_wall_s':900,
  'policy_inferences':0,'registered_initials':3,'registered_trajectories':12,
  'completed_accounting':complete,'process_failure':failed,
  'execution_gate_passed':complete and not failed and total<=2880 and time.monotonic()-started<=900}
 write(output/'summary.json',result);return 0 if result['execution_gate_passed'] else 2

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preregister',action='store_true');p.add_argument('--run',action='store_true')
 p.add_argument('--prepare');p.add_argument('--trajectory');p.add_argument('--manifest',type=Path);p.add_argument('--cpu-receipt',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.preregister:return preregister(a.output)
 if a.prepare:return prepare_child(a.prepare,a.manifest,a.output)
 if a.trajectory:return trajectory_child(a.trajectory,a.manifest,a.output)
 if a.run:return coordinate(a.output,a.cpu_receipt)
 p.error('Explicit preregistration or once-only execution operation required')
if __name__=='__main__':raise SystemExit(main())
