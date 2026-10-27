#!/usr/bin/env python3
"""Observe the original *shared-data* viewer path on disposable forensic data.

No physics integration. This new trial can establish a boundary for its own
observations only, never the unobserved historical root cause. The experiment
runs separately from the actual R1 isolated physics pipeline and admission gate.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, threading, time, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import mujoco
import mujoco.viewer
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
from scripts.g4s2a_gripper_v1_diagnostics import initial
from scripts.g4s2a_gripper_v1_r1_state import snapshot, compare
from scripts.g4s2a_r1_viewer import require_gui_environment, window_info, configure_camera, capture_desktop_window, close_and_join
from scripts.capture_g4s2a_r1_comparison import raise_own_window


def investigate(out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 original=(mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2)
 def forbidden(*_a,**_k):raise RuntimeError('FORENSIC_INTEGRATION_FORBIDDEN')
 mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
 env=NativeGripperEnv();initial(env,'I3');registered=snapshot(mujoco,env.model,env.data)
 records=[];errors=[];viewer=None;threads=[];receipt=None;shot=None
 def event(name, op):
  before=snapshot(mujoco,env.model,env.data)
  try:result=op()
  except BaseException as e:
   errors.append({'boundary':name,'error':repr(e),'traceback':traceback.format_exc()});raise
  after=snapshot(mujoco,env.model,env.data)
  records.append({'boundary':name,'before':before,'after':after,'changes':compare(before,after),'against_registered':compare(registered,after)})
  return result
 try:
  require_gui_environment()
  prethreads={t.ident for t in threading.enumerate()}
  viewer=event('VIEWER_LAUNCH_SHARED_MODEL_DATA',lambda:mujoco.viewer.launch_passive(env.model,env.data,show_left_ui=False,show_right_ui=False))
  threads=[t for t in threading.enumerate() if t.ident not in prethreads and '_launch_internal' in t.name]
  if not threads:raise RuntimeError('GUI_RENDER_THREAD_MISSING')
  def camera():
   with viewer.lock():
    viewer.cam.lookat[:]=[.55,0,.23];viewer.cam.distance=.85;viewer.cam.azimuth=135;viewer.cam.elevation=-20
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
  event('CAMERA_INITIAL_CONFIG',camera)
  event('FIRST_VIEWER_SYNC',viewer.sync)
  event('ONE_SECOND_GUI_WAIT',lambda:time.sleep(1))
  info=event('WINDOW_QUERY',window_info)
  event('WINDOW_ACTIVATION',lambda:raise_own_window(info['window_id']))
  event('SECOND_VIEWER_SYNC',viewer.sync)
  event('TWO_SECOND_GUI_WAIT',lambda:time.sleep(2))
  # The same relevant operations made by historical Visible.capture.
  event('CAPTURE_PRE_SYNC',viewer.sync)
  event('CAPTURE_CAMERA_CLOSEUP',lambda:configure_camera(viewer,'closeup'))
  event('CAPTURE_WINDOW_ACTIVATION',lambda:raise_own_window(window_info()['window_id']))
  event('CAPTURE_POST_SYNC',viewer.sync)
  event('CAPTURE_WAIT',lambda:time.sleep(.35))
  shot=out/'shared_I3_preflight.png'
  event('CAPTURE_SCREENSHOT',lambda:capture_desktop_window(shot,window_info()))
 except BaseException as e:errors.append({'boundary':'UNHANDLED','error':repr(e),'traceback':traceback.format_exc()})
 finally:
  if viewer is not None:
   try:receipt=event('VIEWER_CLOSE',lambda:close_and_join(viewer,threads))
   except BaseException as e:errors.append({'boundary':'VIEWER_CLOSE','error':repr(e)})
  mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=original
 earliest=next((x['boundary'] for x in records if x.get('against_registered')),None)
 result={'status':'COMPLETED' if not errors else 'ERROR','physics_steps':0,'policy_calls':0,'mujoco_version':mujoco.__version__,
  'source':'I3 registered initial restored on separate, disposable NativeGripperEnv; shared viewer architecture only',
  'registered_state':registered,'boundaries':records,'earliest_mismatch_in_this_forensic_trial':earliest,
  'historical_cause':'UNKNOWN: these observations do not establish which operation changed historical V1 I3 data',
  'gui_close':receipt,'errors':errors,'screenshot':str(shot) if shot and shot.exists() else None,
  'screenshot_sha256':hashlib.sha256(shot.read_bytes()).hexdigest() if shot and shot.exists() else None,
  'screenshot_qa':'PENDING_PRIVACY_AND_VISUAL_INSPECTION'}
 with (out/'shared_viewer_forensics.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'earliest_mismatch':earliest,'errors':errors,'output':str(out)}))
 return int(bool(errors))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(investigate(a.output))
