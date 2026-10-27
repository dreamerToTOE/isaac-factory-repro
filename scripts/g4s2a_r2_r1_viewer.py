#!/usr/bin/env python3
"""Interactive saved-state R2-R1 inspection; ZERO integration and ZERO policy.

Space play/pause; Right/Left single-step; Home/End first/last; L jumps to
three steps before first bilateral contact loss; S cycles slow playback speed.
V/Z switch overview/closeup, mouse adjusts camera, close the native window to
exit. Optional screenshots are original desktop pixels, not synthetic images.
Saved qpos/qvel and post-forward forces do not recover the old solver impulse.
"""
from __future__ import annotations
import argparse,json,sys,threading,time
from collections import deque
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s2a_r2_r1_model import make_model,model_identity
from scripts.g4s2a_r1_viewer import (require_gui_environment,PlaybackController,
    apply_archived_state,contact_forces,configure_camera,window_info,
    capture_desktop_window,close_and_join,sha,dump)
from scripts.capture_g4s2a_r1_comparison import raise_own_window

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--matrix',type=Path,default=ROOT/'evidence/g4_s/s2a_r2_r1/matrix_v1')
 p.add_argument('--run',default='I3_FULL_C0');p.add_argument('--capture',type=Path)
 p.add_argument('--steps',default='around_loss,last');p.add_argument('--overview',action='store_true');a=p.parse_args()
 require_gui_environment();run=a.matrix/'trajectories'/a.run
 identity=json.loads((run/'identity.json').read_text());env=make_model(identity['physical_selection'])
 if model_identity(env)!=identity['models']['plant']:raise ValueError('PLAYBACK_MODEL_IDENTITY_CHANGED')
 trace=run/'physics.jsonl'
 with trace.open() as source:
  rows=[item for line in source if (item:=json.loads(line))['event']=='physics_result']
 if not rows:raise ValueError('NO_RECORDED_PHYSICS_RESULTS')
 states=[identity['initial_snapshot']]+[row['state'] for row in rows]
 first_loss=next((row['step'] for row in rows if min(contact_forces(row['state']).values())<=1e-6),None)
 controller=PlaybackController(len(rows));commands=deque();viewer=None;threads=[];captures=[]
 if a.capture:a.capture=a.capture.resolve();a.capture.mkdir(parents=True,exist_ok=False)
 import mujoco.viewer,glfw
 try:
  before={t.ident for t in threading.enumerate()}
  viewer=mujoco.viewer.launch_passive(env.model,env.data,key_callback=lambda key:commands.append(key),show_left_ui=False,show_right_ui=False)
  threads=[t for t in threading.enumerate() if t.ident not in before and '_launch_internal' in t.name]
  if not threads:raise RuntimeError('GUI_RENDER_THREAD_MISSING')
  configure_camera(viewer,'overview' if a.overview else 'closeup')
  if not a.overview:
   with viewer.lock():viewer.cam.azimuth=45
  def frame(index):
   state=states[index]
   with viewer.lock():
    env.mj.mjv_defaultOption(viewer.opt);apply_archived_state(env,state)
    scene=viewer.user_scn;scene.ngeom=0
    for contact in state.get('contacts',[]):
     if scene.ngeom>=scene.maxgeom:break
     env.mj.mjv_initGeom(scene.geoms[scene.ngeom],env.mj.mjtGeom.mjGEOM_SPHERE,
      np.array([.0007]*3),np.array(contact['position_world_m']),np.eye(3).reshape(-1),np.array([0.,1.,1.,.85],dtype=np.float32))
     scene.ngeom+=1
   forces=contact_forces(state)
   viewer.set_texts((None,env.mj.mjtGridPos.mjGRID_TOPLEFT,
    f'SAVED_STATE_PLAYBACK | ZERO INTEGRATION | ZERO POLICY\n{a.run} | {identity["physical_selection"]}/{identity["controller_selection"]} | step={index} | t={index/120:.6f}s\n'
    f'Recorded post-forward L/R pad force={forces["left"]:.5g}/{forces["right"]:.5g} N\nCyan: saved positions; solver impulses NOT reconstructed\n'
    'Space pause/play | Left/Right single step | L before loss | S speed\nHome/End first/last | V/Z camera | Close window to exit',''))
   viewer.sync()
  frame(0);time.sleep(1.);info=window_info();raise_own_window(info['window_id']);frame(0);time.sleep(1.)
  if a.capture:
   selected=set()
   for value in a.steps.split(','):
    if value=='last':selected.add(len(rows))
    elif value=='around_loss':
     if first_loss is not None:selected.update(range(max(0,first_loss-3),min(len(rows),first_loss+1)+1))
    else:selected.add(int(value))
   for step in sorted(selected):
    if not 0<=step<len(states):raise ValueError('REQUESTED_STEP_OUTSIDE_SAVED_TRAJECTORY')
    frame(step);raise_own_window(window_info()['window_id']);frame(step);time.sleep(.5)
    path=a.capture/f'{a.run}_step_{step:03d}.png';capture_desktop_window(path,window_info())
    captures.append({'file':str(path.relative_to(ROOT)),'experiment_id':a.run,'initial_state_id':identity['initial_id'],
     'physical_model':identity['physical_selection'],'arm_controller_M':identity['controller_selection'],
     'model_xml_sha256':identity['models']['plant']['xml_sha256'],'physics_step':step,'episode_simulation_time_s':step/120,
     'source_trace_sha256':sha(trace),'sha256':sha(path),'capture_method':'Native Ubuntu X11 pixels of own raised MuJoCo window',
     'mode':'SAVED_STATE_PLAYBACK','new_integration_steps':0,'policy_calls':0,'window':window_info(),
     'visual_flags':{name:int(viewer.opt.flags[int(flag)]) for name,flag in env.mj.mjtVisFlag.__members__.items() if name!='mjNVISFLAG'},
     'note':'Post-forward saved state and force; actual previous solver impulse is not reconstructed.'})
   dump(a.capture/'manifest.json',captures)
  else:
   while viewer.is_running():
    while commands:
     key=commands.popleft()
     if key==glfw.KEY_SPACE:controller.toggle()
     elif key==glfw.KEY_RIGHT:controller.step()
     elif key==glfw.KEY_LEFT:controller.jump(max(0,controller.index-1))
     elif key==glfw.KEY_HOME:controller.jump(0)
     elif key==glfw.KEY_END:controller.jump(len(rows))
     elif key==glfw.KEY_L and first_loss is not None:controller.jump(max(0,first_loss-3))
     elif key==glfw.KEY_S:controller.cycle_speed()
     elif key in (glfw.KEY_V,glfw.KEY_Z):configure_camera(viewer,'overview' if key==glfw.KEY_V else 'closeup')
    frame(controller.tick(time.monotonic()));time.sleep(.01)
 finally:
  if viewer is not None:
   receipt=close_and_join(viewer,threads)
   if a.capture:dump(a.capture/'close_receipt.json',receipt)

if __name__=='__main__':main()
