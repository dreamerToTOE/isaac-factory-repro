#!/usr/bin/env python3
"""Interactive R2 SAVED-state inspection until user closes native GUI.

Zero integration; zero actor calls. Space pause/play, Right step, Home initial,
End last, S slow speed, V overview, Z grasp closeup; native mouse camera.
"""
from __future__ import annotations
import argparse
from collections import deque
import gzip,json,sys,threading,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s2a_r2_model import make_model,model_identity,VARIANTS
from scripts.g4s2a_r1_viewer import require_gui_environment,PlaybackController,apply_archived_state,contact_forces,configure_camera,window_info,capture_desktop_window,close_and_join,sha,dump
from scripts.capture_g4s2a_r1_comparison import raise_own_window

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--matrix',type=Path,default=ROOT/'evidence/g4_s/s2a_r2/matrix_v2')
 p.add_argument('--run',default='I1_D');p.add_argument('--capture',type=Path);p.add_argument('--steps',default='0,4,16,last');p.add_argument('--overview',action='store_true');a=p.parse_args()
 require_gui_environment();run=a.matrix/'trajectories'/a.run;identity=json.loads((run/'identity.json').read_text())
 env=make_model(bool(identity['plant_selection']))
 if model_identity(env)!=identity['models']['plant']:raise ValueError('PLAYBACK_MODEL_IDENTITY_CHANGED')
 rows=[json.loads(x) for x in gzip.open(run/'physics.jsonl.gz','rt') if json.loads(x)['event']=='physics_result']
 states=[identity['initial_snapshot']]+[r['state'] for r in rows]
 controller=PlaybackController(len(rows));commands=deque();viewer=None;threads=[];captures=[]
 if a.capture:a.capture=a.capture.resolve();a.capture.mkdir(parents=True,exist_ok=False)
 import mujoco.viewer,glfw
 try:
  before={t.ident for t in threading.enumerate()}
  viewer=mujoco.viewer.launch_passive(env.model,env.data,key_callback=lambda key:commands.append(key),show_left_ui=False,show_right_ui=False)
  threads=[t for t in threading.enumerate() if t.ident not in before and '_launch_internal' in t.name]
  configure_camera(viewer,'overview' if a.overview else 'closeup')
  # Native contact marker default scale can obscure fingers. Use the R1 saved
  # contact-position overlay (.7mm cyan markers), never synthetic forces.
  with viewer.lock():
   viewer.opt.flags[env.mj.mjtVisFlag.mjVIS_CONTACTPOINT]=False
   viewer.opt.flags[env.mj.mjtVisFlag.mjVIS_CONTACTFORCE]=False
   if not a.overview:viewer.cam.azimuth=45
  def frame(index):
   state=states[index]
   with viewer.lock():
    env.mj.mjv_defaultOption(viewer.opt)
    apply_archived_state(env,state);scene=viewer.user_scn;scene.ngeom=0
    for contact in state['contacts']:
     if scene.ngeom>=scene.maxgeom:break
     env.mj.mjv_initGeom(scene.geoms[scene.ngeom],env.mj.mjtGeom.mjGEOM_SPHERE,
      np.array([.0007]*3),np.array(contact['position_world_m']),np.eye(3).reshape(-1),np.array([0.,1.,1.,.85],dtype=np.float32))
     scene.ngeom+=1
   forces=contact_forces(state)
   viewer.set_texts((None,env.mj.mjtGridPos.mjGRID_TOPLEFT,
    f'R2 SAVED STATE PLAYBACK; ZERO INTEGRATION / ZERO ACTOR\n{a.run}: P{identity["plant_selection"]}+C{identity["controller_selection"]}; step={index}; t={index/120:.6f}s\n'
    f'Recorded pad force L/R={forces["left"]:.5g}/{forces["right"]:.5g} N\nCyan: recorded contact positions; normals/impulses NOT reconstructed\n'
    'Space play/pause | Right step | Home initial | End last | S speed\nV overview | Z closeup | Mouse camera; close window to exit',''))
   viewer.sync()
  frame(0);time.sleep(1.);info=window_info();raise_own_window(info['window_id']);viewer.sync();time.sleep(1.)
  if a.capture:
   selected=sorted(set(len(rows) if x=='last' else int(x) for x in a.steps.split(',')))
   for step in selected:
    if not 0<=step<len(states):continue
    frame(step);raise_own_window(window_info()['window_id']);viewer.sync();time.sleep(.5)
    path=a.capture/f'{a.run}_step_{step:03d}.png';capture_desktop_window(path,window_info())
    captures.append({'file':str(path.relative_to(ROOT)),'experiment_id':a.run,'initial_state_id':identity['initial_id'],
     'model_configuration':f'P{identity["plant_selection"]}+C{identity["controller_selection"]}',
     'model_xml_sha256':identity['models']['plant']['xml_sha256'],'step':step,'simulation_time_s':step/120,
     'sha256':sha(path),'capture_method':'Actual Ubuntu X11 screen rectangle, native raised MuJoCo window',
     'mode':'SAVED_DIAGNOSTIC_STATE_PLAYBACK','source_trace_sha256':sha(run/'physics.jsonl.gz'),
     'new_integration_steps':0,'policy_calls':0,'window':window_info(),
     'visual_flags':{name:int(viewer.opt.flags[int(flag)]) for name,flag in env.mj.mjtVisFlag.__members__.items() if name!='mjNVISFLAG'},
     'visual_geometry_groups':viewer.opt.geomgroup.tolist()})
   dump(a.capture/'manifest.json',captures)
  else:
   while viewer.is_running():
    while commands:
     key=commands.popleft()
     if key==glfw.KEY_SPACE:controller.toggle()
     elif key==glfw.KEY_RIGHT:controller.step()
     elif key==glfw.KEY_HOME:controller.jump(0)
     elif key==glfw.KEY_END:controller.jump(len(rows))
     elif key==glfw.KEY_S:controller.cycle_speed()
     elif key in (glfw.KEY_V,glfw.KEY_Z):configure_camera(viewer,'overview' if key==glfw.KEY_V else 'closeup')
    frame(controller.tick(time.monotonic()));time.sleep(.01)
 finally:
  if viewer is not None:
   close=close_and_join(viewer,threads)
   if a.capture:dump(a.capture/'close_receipt.json',close)

if __name__=='__main__':main()
