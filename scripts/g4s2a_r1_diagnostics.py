#!/usr/bin/env python3
"""Pre-registered, visible-GUI-only short diagnostics; no policy imports/calls.

Each ID can execute once, at most16 steps. The coordinator stops on any failed
child/GUI/budget gate. These are DIAGNOSTIC_ONLY, never official S1B changes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gzip,hashlib,json,math,os
from pathlib import Path
import subprocess,sys,time,threading
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s1b_env import AdaptedPegInsertEnv,EnvironmentConfig
from scripts.g4s1b_contract import PhysicsContract,torque_command
from scripts.g4s2a_r1_viewer import load_archive,apply_archived_state,window_info,capture_desktop_window,contact_forces,require_gui_environment,close_and_join,XML_SHA,ARCHIVE
from scripts.g4s2a_smoke import physical_measurements,peg_relative,jsonable

ROBOT_BODIES=[f'link{i}' for i in range(8)]+['hand','left_finger','right_finger']
VARIANTS=[
 {'id':'D01','name':'frozen_baseline','delta':{},'arm':'full'},
 {'id':'D02','name':'passive_arm','delta':{},'arm':'zero'},
 {'id':'D03','name':'fixed_initial_TCP_reference','delta':{},'arm':'fixed'},
 {'id':'D04','name':'task_torque_only','delta':{},'arm':'task'},
 {'id':'D05','name':'nullspace_torque_only','delta':{},'arm':'null'},
 {'id':'D06','name':'robot_body_rotational_inertia_plus_0p01I','delta':{'body_inertia_offset_kg_m2':.01},'arm':'full'},
 {'id':'D07','name':'robot_joint_armature_plus_0p01','delta':{'joint_armature_offset_kg_m2':.01},'arm':'full'},
 {'id':'D08','name':'arm_motor_force_limits_100Nm','delta':{'all_arm_motor_force_limits_nm':100.},'arm':'full'},
 {'id':'D09','name':'passive_arm_affine_finger_PD','delta':{'finger_actuation':'affine-pd'},'arm':'zero'},
 {'id':'D10','name':'passive_arm_contact_solref_0p02','delta':{'contact_solref':[.02,1.]},'arm':'zero'}]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 with Path(path).open('x') as f:json.dump(jsonable(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def line(stream,value):stream.write(json.dumps(jsonable(value),allow_nan=False)+'\n');stream.flush()

class IntegrationBudget:
 def __init__(self,max_variants=12,max_steps=16,max_wall_s=600):
  if not 1<=max_variants<=12 or not 1<=max_steps<=16 or not 0<max_wall_s<=600:raise ValueError('BUDGET_CANNOT_BE_WIDENED')
  self.max_variants=max_variants;self.max_steps=max_steps;self.max_wall_s=max_wall_s;self.counts={};self.wall_s=0.;self.pending=None
 def begin(self,variant_id):
  if variant_id in self.counts or len(self.counts)>=self.max_variants:raise ValueError('VARIANT_LIMIT_OR_DUPLICATE')
  self.counts[variant_id]=0
 def before_step(self,variant_id):
  if self.pending is not None or variant_id not in self.counts or self.counts[variant_id]>=self.max_steps or self.wall_s>=self.max_wall_s:raise RuntimeError('INTEGRATION_BUDGET_REJECTED')
  self.counts[variant_id]+=1;self.pending=variant_id
 def record_step(self,elapsed):
  if self.pending is None or not math.isfinite(elapsed) or elapsed<0:raise ValueError('INVALID_INTEGRATION_TIMING')
  self.wall_s+=elapsed;self.pending=None


def model_for_variant(variant):
 delta=variant['delta'];kwargs={}
 for name in ('finger_actuation','contact_solref'):
  if name in delta:kwargs[name]=tuple(delta[name]) if name=='contact_solref' else delta[name]
 env=AdaptedPegInsertEnv(EnvironmentConfig(**kwargs))
 original_xml=hashlib.sha256(env.xml.encode()).hexdigest()
 if not kwargs and original_xml!=XML_SHA:raise ValueError('FROZEN_XML_CHANGED')
 if 'body_inertia_offset_kg_m2' in delta:
  for name in ROBOT_BODIES:env.model.body_inertia[env.model.body(name).id]+=.01
 if 'joint_armature_offset_kg_m2' in delta:
  env.model.dof_armature[env.arm_dof_ids]+=.01
 if 'all_arm_motor_force_limits_nm' in delta:env.model.actuator_forcerange[env.arm_actuator_ids]=[-100.,100.]
 if any(name in delta for name in ('body_inertia_offset_kg_m2','joint_armature_offset_kg_m2')):
  # mj_setConst resets supplied data.qpos: only a disposable object is used.
  env.mj.mj_setConst(env.model,env.mj.MjData(env.model));env.data=env.mj.MjData(env.model)
 fingerprint={'generated_xml_sha256':original_xml,'config':asdict(env.config),'delta':delta,
              'body_inertia':env.model.body_inertia.tolist(),'body_mass':env.model.body_mass.tolist(),
              'armature':env.model.dof_armature.tolist(),'force_limits':env.model.actuator_forcerange.tolist(),
              'body_scope':ROBOT_BODIES if 'body_inertia_offset_kg_m2' in delta else []}
 fingerprint['model_configuration_sha256']=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
 return env,fingerprint


def child_run(variant_id,manifest_path,output):
 require_gui_environment()
 import mujoco.viewer
 registered=json.loads(Path(manifest_path).read_text());variant=next(x for x in registered['variants'] if x['id']==variant_id)
 if registered['source_sha256']!=sha(Path(__file__)):raise ValueError('DIAGNOSTIC_SOURCE_CHANGED_AFTER_REGISTRATION')
 for name,digest in registered['imported_source_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('IMPORTED_SOURCE_CHANGED_AFTER_REGISTRATION:'+name)
 if Path(output).resolve()!=Path(manifest_path).resolve().parent/variant_id:raise ValueError('VARIANT_OUTPUT_MUST_MATCH_ONCE_ONLY_REGISTRY')
 if not (Path(manifest_path).parent/'coordinator_attempt.json').exists():raise ValueError('COORDINATOR_ADMISSION_REQUIRED')
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 env,identity=model_for_variant(variant)
 initial,_,_=load_archive();apply_archived_state(env,initial)
 if env.config.finger_actuation=='affine-pd':
  env.data.ctrl[env.finger_actuator_ids]=0.;env.mj.mj_forward(env.model,env.data)
 # Arm/peg gravity and all other frozen model properties remain unchanged.
 reset=json.loads((ARCHIVE/'seed_1001/interface_reset.json').read_text())['initial']
 con=PhysicsContract();p,q=env.tcp_state();con.reset(p,q,env.data.xpos[env.hole_body_id],env.data.xquat[env.hole_body_id],reset['episode_noise_m'],timestamp=float(env.data.time))
 if not np.allclose(con.action_cache,reset['ema_cache'],atol=1e-12,rtol=0):raise ValueError('INITIAL_EMA_MISMATCH')
 actions=[json.loads(x)['clipped'] for x in gzip.open(ARCHIVE/'seed_1001/policy.jsonl.gz','rt')][:2]
 reference=peg_relative(env);fixed_p,fixed_q=env.tcp_state();budget=IntegrationBudget();budget.begin(variant_id)
 write(output/'identity.json',{'scope':'DIAGNOSTIC_ONLY','variant':variant,'model':identity,'initial_snapshot':env.snapshot(),
   'initial_provenance':'S2A admitted state qpos/qvel/ctrl; solver warmstart/contact/actuator internal variables NOT archived; regenerated forward state',
   'initial_ema':con.action_cache,'archived_action_inputs':actions,'policy_inferences':0,'source_sha256':sha(Path(__file__))})
 rows=[];failure=None;start_sim=float(env.data.time);started=time.monotonic();physics_wall=0.;viewer=None;render_threads=[];close_receipt=None
 try:
  before_threads={t.ident for t in threading.enumerate()}
  with mujoco.viewer.launch_passive(env.model,env.data,show_left_ui=False,show_right_ui=False) as viewer:
   render_threads=[t for t in threading.enumerate() if t.ident not in before_threads and '_launch_internal' in t.name]
   if not render_threads:raise RuntimeError('GUI_RENDER_THREAD_NOT_IDENTIFIED')
   with viewer.lock():
    viewer.cam.lookat[:]=[.56,0,.17];viewer.cam.distance=.65;viewer.cam.azimuth=135;viewer.cam.elevation=-22
   viewer.set_texts((None,env.mj.mjtGridPos.mjGRID_TOPLEFT,f'DIAGNOSTIC_ONLY {variant_id}: {variant["name"]}\nNEW PHYSICS: 0/16; actor calls=0\nSame archived initial qpos/qvel; solver internals UNKNOWN',''))
   viewer.sync();time.sleep(1.0);info=window_info()
   write(output/'gui_admission.json',{'status':'GUI_OPEN_VISIBLE','window':info,'policy_inferences':0,'integration_started':False})
   with (output/'physics.jsonl').open('x') as stream:
    for index in range(1,17):
     if not viewer.is_running():raise RuntimeError('VISIBLE_GUI_CLOSED_STOP_BEFORE_NEXT_INTEGRATION')
     if time.monotonic()-started>600:raise TimeoutError('DIAGNOSTIC_WALL_BUDGET')
     if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied):raise RuntimeError('GUI_EXTERNAL_PERTURBATION_REJECTED')
     if index in (1,9):con.policy_update(actions[(index-1)//8])
     p,q=env.tcp_state();con.update_physics(p,q,float(env.data.time));target=con.targets()
     if variant['arm']=='fixed':target={'position_target':fixed_p,'quaternion_target':fixed_q}
     inputs={'q7':env.data.qpos[env.arm_qpos_ids].copy(),'qdot7':env.data.qvel[env.arm_dof_ids].copy(),'position':p.copy(),'quaternion':q.copy(),
       'linear_fd':con.linear_fd.copy(),'angular_fd':con.angular_fd.copy(),'jacobian':env.controller_jacobian(),'mass_matrix':env.arm_mass_matrix(),
       'target_position':target['position_target'],'target_quaternion':target['quaternion_target']}
     control=torque_command(**inputs)
     tau={'full':control['requested'],'fixed':control['requested'],'task':control['task_torque'],'null':control['null_torque'],'zero':np.zeros(7)}[variant['arm']]
     budget.before_step(variant_id)
     line(stream,{'event':'integration_request','step':index,'time_before_s':float(env.data.time)})
     before_warning=[w.number for w in env.data.warning];clock=time.perf_counter()
     state=env.physics_step(tau,finger_target=0.,record=False)
     elapsed=time.perf_counter()-clock;budget.record_step(elapsed);physics_wall+=elapsed
     state['s2a']=physical_measurements(env,reference)
     row={'event':'physics_result','variant':variant_id,'step':index,'episode_time_s':float(env.data.time)-start_sim,
          'physics_wall_s':elapsed,'target':target,'controller_input':inputs,'controller':control,'selected_arm_torque_nm':tau,
          'qacc_post_forward':env.data.qacc.copy(),'qacc_warmstart_new_diagnostic_state':env.data.qacc_warmstart.copy(),
          'state':state,'pad_force_n':contact_forces(state),'new_warning_counts':(np.array([w.number for w in env.data.warning])-before_warning).tolist()}
     line(stream,row);rows.append(row)
     # No restore/extra forward here: visible sync cannot add dynamics calls.
     viewer.set_texts((None,env.mj.mjtGridPos.mjGRID_TOPLEFT,f'DIAGNOSTIC_ONLY {variant_id}: {variant["name"]}\nNEW PHYSICS step {index}/16; t={index/120:.6f}s; actor calls=0\n'
         f'Slip={state["s2a"]["peg_hand_slip_m"]*1000:.4f}mm; pad L/R={row["pad_force_n"]["left"]:.4g}/{row["pad_force_n"]["right"]:.4g} N\n'
         'Visible geometry: new integration; contact forces in log measured at this new state',''))
     viewer.sync();time.sleep(.07)
     if index in (4,16):
      time.sleep(.3);path=output/f'physics_{index:02d}.png';capture_desktop_window(path,window_info())
      write(output/f'physics_{index:02d}_visual.json',{'source_experiment':variant_id,'scope':'DIAGNOSTIC_ONLY','mode':'NEW_INTEGRATED_PHYSICS','physics_step':index,'simulation_time_s':index/120,'absolute_simulation_time_s':state['time_s'],'model_configuration_sha256':identity['model_configuration_sha256'],'file':str(path),'sha256':sha(path)})
 except Exception as exc:
  failure={'type':type(exc).__name__,'message':str(exc)}
 finally:
  if viewer is not None:
   try:close_receipt=close_and_join(viewer,render_threads)
   except Exception as exc:failure=failure or {'type':type(exc).__name__,'message':str(exc)}
  actual_steps=round((float(env.data.time)-start_sim)*120)
  metrics={'task':'G4-S2A-R1','scope':'DIAGNOSTIC_ONLY','variant':variant,'actual_physics_steps':actual_steps,
     'recorded_steps':len(rows),'policy_inferences':0,'physics_call_wall_s':physics_wall,'wall_s':time.monotonic()-started,
     'failure':failure,'graceful_close':close_receipt,'model_configuration_sha256':identity['model_configuration_sha256'],
     'max_slip_m':max((r['state']['s2a']['peg_hand_slip_m'] for r in rows),default=0.),
     'first_bilateral_loss_step':next((r['step'] for r in rows if min(r['pad_force_n'].values())<=1e-6),None),
     'max_arm_qvel_rad_s':max((max(abs(np.array(r['state']['qvel'][:7]))) for r in rows),default=0.),
     'max_arm_limit_rad':max((r['state']['s2a']['arm_joint_limit_violation_rad'] for r in rows),default=0.),
     'max_finger_limit_m':max((r['state']['s2a']['finger_joint_limit_violation_m'] for r in rows),default=0.),
     'max_all_penetration_m':max((r['state']['s2a']['all_contact_penetration_m'] for r in rows),default=0.)}
  write(output/'result.json',metrics)
 return 0 if failure is None and actual_steps==16 else 2


def preregister(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 record={'task':'G4-S2A-R1','generated_utc':datetime.now(timezone.utc).isoformat(),'scope':'DIAGNOSTIC_ONLY',
   'maximum_variants':12,'registered_variant_count':len(VARIANTS),'steps_per_variant':16,'maximum_registered_steps':160,
   'maximum_new_diagnostic_wall_s':600,'policy_inferences':0,'no_retry':True,'variants':VARIANTS,
   'source_sha256':sha(Path(__file__)),'frozen_environment_sha256':sha(ROOT/'scripts/g4s1b_env.py'),
   'imported_source_sha256':{name:sha(ROOT/name) for name in ('scripts/g4s2a_r1_viewer.py','scripts/g4s1b_env.py','scripts/g4s1b_contract.py','scripts/g4s2a_smoke.py')},
   'frozen_controller_sha256':sha(ROOT/'scripts/g4s1b_contract.py'),'xml_sha256':XML_SHA,
   'initial_provenance':'S2A seed1001 admitted state; exact historical warmstarts UNKNOWN',
   'action_protocol':'same first two archived clippedactions atsteps1/9, EMAreset sourceformula; noactor; fixedreference/passive/component-only deltasexplicit',
   'body_inertia_scope':ROBOT_BODIES,'armature_scope':'7 revolute arm DOFs only, kg m^2; deliberately not equivalent to body0.01I; fingers unchanged',
   'contact_D10_interpretation':'effective positive solref baseline floors .005 to2dt=.0166667; variant.02 is modest sensitivity, not4x actualcontacttimeconstant',
   'promotion_to_official_configuration':False}
 write(output/'preregistered_matrix.json',record)
 return record


def coordinate(output):
 output=Path(output);manifest=output/'preregistered_matrix.json';record=json.loads(manifest.read_text())
 if record['source_sha256']!=sha(Path(__file__)):raise ValueError('SOURCE_CHANGED_AFTER_PREREGISTRATION')
 if (output/'coordinator_attempt.json').exists():raise FileExistsError('NO_DIAGNOSTIC_RETRY')
 write(output/'coordinator_attempt.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'variants':record['variants'],'policy_inferences':0})
 started=time.monotonic();receipts=[]
 for variant in record['variants']:
  if time.monotonic()-started>=600:break
  path=output/variant['id'];console=output/(variant['id']+'_console.txt')
  cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--run-variant',variant['id'],'--manifest',str(manifest),'--output',str(path)]
  with console.open('x') as stream:
   try:child=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1.,600-(time.monotonic()-started)));code=child.returncode
   except subprocess.TimeoutExpired:code=124
  result=json.loads((path/'result.json').read_text()) if (path/'result.json').exists() else None
  receipt={'variant':variant['id'],'command':cmd,'exact_child_exit_code':code,'result':result}
  write(output/(variant['id']+'_process.json'),receipt);receipts.append(receipt)
  print(json.dumps({'variant':variant['id'],'exit':code,'steps':None if result is None else result['actual_physics_steps']}),flush=True)
  if code!=0 or result is None or result['actual_physics_steps']!=16:break
 total=sum(r['result']['actual_physics_steps'] for r in receipts if r['result'])
 summary={'task':'G4-S2A-R1','scope':'DIAGNOSTIC_ONLY','receipts':receipts,'actual_physics_steps':total,
    'completed_variants':len(receipts),'max_registered_steps':160,'actor_calls':0,'wall_s':time.monotonic()-started,
    'physics_call_wall_s':sum(r['result']['physics_call_wall_s'] for r in receipts if r['result']),
    'passed_execution_gates':len(receipts)==10 and all(r['exact_child_exit_code']==0 for r in receipts),'official_configuration_changed':False}
 write(output/'summary.json',summary)
 return 0 if summary['passed_execution_gates'] else 2


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preregister',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--run-variant');p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.preregister:preregister(a.output);return 0
 if a.run_variant:return child_run(a.run_variant,a.manifest,a.output)
 if a.run:return coordinate(a.output)
 p.error('explicit --preregister / --run / --run-variant required')
if __name__=='__main__':raise SystemExit(main())
