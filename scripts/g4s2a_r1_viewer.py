#!/usr/bin/env python3
"""Visible MuJoCo 3.5.0 archived-state viewer: no actor, no physics integration.

Space play/pause; Right single-step; Home reset; S speed cycle;
1 initial, 2 first action (same physical state), 3..8 physics1..6;
F first finger limit, A first arm limit, C hand/support collision; R toggles
NEWLY RECOMPUTED contact graphics. Cyan markers are ARCHIVED contact positions.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import contextmanager
import ctypes
from datetime import datetime,timezone
import gzip,hashlib,json,os
from pathlib import Path
import re,subprocess,sys,threading,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s1b_env import AdaptedPegInsertEnv

ARCHIVE=ROOT/'evidence/g4_s/s2a/run_20261026T140338'
XML_SHA='c60c33ee11eb4806294aee259f97d987308b23e83da62f2219cbc9950fb195c9'
EVENTS={'initial_grasp':0,'first_action':0,'physics_1':1,'physics_2':2,'first_slip':3,'grasp_loss':4,
        'physics_5':5,'physics_6':6,'physics_7':7,'physics_8':8,'first_finger_limit':61,'first_arm_limit':76,'hand_support_collision':685}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def load_archive(archive=ARCHIVE):
    archive=Path(archive)
    collection=json.loads((archive/'collection.json').read_text())
    for record in collection['artifacts']:
        local=archive/Path(record['raw_path']).relative_to(Path(collection['raw_directory'])) if Path(record['raw_path']).is_relative_to(Path(collection['raw_directory'])) else archive/Path(record['raw_path']).name
        if record['lossless_gzip']:local=Path(str(local)+'.gz')
        if sha(local)!=record['bundle_sha256']:raise ValueError('ARCHIVE_DIGEST_MISMATCH:'+str(local))
    rows=[json.loads(x) for x in gzip.open(archive/'seed_1001/physics.jsonl.gz','rt')]
    admission=json.loads((archive/'seed_1001/reset_admission.json').read_text())
    action=json.loads(gzip.open(archive/'seed_1001/policy.jsonl.gz','rt').readline())
    if len(rows)!=1192 or [r['physics_step'] for r in rows]!=list(range(1,1193)):raise ValueError('ARCHIVE_STEP_COUNT')
    return admission['final'],rows,action


def _finite_vector(value,size,label):
    a=np.asarray(value,dtype=float)
    if a.shape!=(size,) or not np.isfinite(a).all():raise ValueError('INVALID_ARCHIVED_'+label)
    return a.copy()


def apply_archived_state(env,state):
    """Validate completely before mutation; forward only, no integration.

    Loading qpos/qvel cannot restore missing solver warm starts or historical
    contact impulses. Recorded contacts remain separate cyan overlay markers.
    """
    q=_finite_vector(state['qpos'],env.model.nq,'QPOS')
    v=_finite_vector(state['qvel'],env.model.nv,'QVEL')
    timestamp=float(state['time_s'])
    if not np.isfinite(timestamp):raise ValueError('INVALID_ARCHIVED_TIME')
    torque=_finite_vector(state['software_limited_arm_torque_nm'],7,'ARM_TORQUE')
    fingers=_finite_vector(state.get('finger_command',{}).get('commanded_force_n',[0.,0.]),2,'FINGER_FORCE')
    hole=_finite_vector(state['hole_root_position'],3,'HOLE_POSITION')
    hole_quat=_finite_vector(state.get('s2a',{}).get('hole_quaternion_wxyz',[1.,0.,0.,0.]),4,'HOLE_QUATERNION')
    if abs(np.linalg.norm(hole_quat)-1)>1e-6:raise ValueError('INVALID_ARCHIVED_UNIT_HOLE_QUATERNION')
    peg_quat=q[env.peg_qpos_adr+3:env.peg_qpos_adr+7]
    if abs(np.linalg.norm(peg_quat)-1)>1e-6:raise ValueError('INVALID_ARCHIVED_UNIT_PEG_QUATERNION')
    env.data.qpos[:]=q;env.data.qvel[:]=v;env.data.time=timestamp
    env.data.ctrl[env.arm_actuator_ids]=torque
    env.data.ctrl[env.finger_actuator_ids]=fingers
    env.model.body_pos[env.hole_body_id]=hole;env.model.body_quat[env.hole_body_id]=hole_quat
    env.mj.mj_forward(env.model,env.data)


def window_info(model_name='g4_s1b_adapted_panda_peg_insert'):
    """Inspect only a viewable window whose title matches this model.

    The complete desktop tree is deliberately never returned or written here.
    """
    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
    matches=re.findall(r'(0x[0-9a-fA-F]+)\s+"(MuJoCo[^"\n]*)"',tree)
    scoped=[(wid,title) for wid,title in matches if model_name in title]
    for wid,title in reversed(scoped):
        info=subprocess.check_output(['xwininfo','-id',wid],text=True)
        if 'Map State: IsViewable' in info:
            def val(key):return int(re.search(re.escape(key)+r':\s*(-?\d+)',info).group(1))
            x,y,w,h=[val(k) for k in ('Absolute upper-left X','Absolute upper-left Y','Width','Height')]
            return {'window_id':wid,'title':title,'model_name':model_name,'map_state':'IsViewable','x':x,'y':y,'width':w,'height':h,'selected_window_xwininfo':info}
    raise RuntimeError('GUI_BLOCKED: no viewable MuJoCo window matching model '+model_name)


def close_and_join(viewer,render_threads,timeout_s=15.):
    """Wait for the passive viewer's own rendering thread before Python exit."""
    started=time.monotonic();viewer.close()
    names=[]
    for thread in render_threads:
        names.append(thread.name)
        thread.join(max(0.,timeout_s-(time.monotonic()-started)))
    remaining=[thread.name for thread in render_threads if thread.is_alive()]
    if remaining:raise RuntimeError('GUI_CLOSE_RENDER_THREAD_TIMEOUT:'+str(remaining))
    return {'viewer_close_requested':True,'render_threads_joined':names,'remaining_render_threads':remaining,'close_wall_s':time.monotonic()-started}


def configure_camera(viewer,preset='closeup'):
    """Explicit preset; normal playback never overwrites user camera rotation."""
    with viewer.lock():
        if preset=='overview':
            viewer.cam.lookat[:]=[.42,0.,.43];viewer.cam.distance=1.7;viewer.cam.azimuth=135;viewer.cam.elevation=-22
        elif preset=='closeup':
            viewer.cam.lookat[:]=[.605,-.015,.105];viewer.cam.distance=.32;viewer.cam.azimuth=135;viewer.cam.elevation=-14
        else:raise ValueError('UNKNOWN_CAMERA_PRESET')


class PlaybackController:
    """Pure recorded-frame navigation; this class has no simulator dependency."""
    def __init__(self,nsteps=1192):
        if not isinstance(nsteps,int) or nsteps<1:raise ValueError('INVALID_PLAYBACK_LENGTH')
        self.nsteps=nsteps;self.index=0;self.playing=False;self.speed=.1;self.last=None
    def jump(self,index):
        if not isinstance(index,int) or not 0<=index<=self.nsteps:raise ValueError('INVALID_PLAYBACK_INDEX')
        self.index=index;self.playing=False;self.last=None
    def step(self):self.jump(min(self.index+1,self.nsteps))
    def toggle(self):self.playing=not self.playing;self.last=None
    def cycle_speed(self):self.speed={.1:.25,.25:1.,1.:.1}[self.speed];self.last=None
    def tick(self,now):
        if not np.isfinite(now):raise ValueError('INVALID_PLAYBACK_CLOCK')
        if self.last is None:self.last=now;return self.index
        if now<self.last:raise ValueError('PLAYBACK_CLOCK_MOVED_BACKWARDS')
        if self.playing and now-self.last>=1/(120*self.speed):
            count=int((now-self.last)*120*self.speed)
            self.index=min(self.nsteps,self.index+count);self.last+=count/(120*self.speed)
            if self.index==self.nsteps:self.playing=False
        return self.index


def capture_desktop_window(path,info):
    # Actual X11 screen pixels; neither offscreen rendering nor image synthesis.
    helper="from PIL import ImageGrab; import sys; ImageGrab.grab(bbox=tuple(map(int,sys.argv[2:])),xdisplay=sys.argv[1]).save(sys.stdout.buffer,format='PNG')"
    box=[info['x'],info['y'],info['x']+info['width'],info['y']+info['height']]
    with Path(path).open('xb') as f:subprocess.run(['/usr/bin/python3','-c',helper,os.environ['DISPLAY'],*map(str,box)],stdout=f,check=True)


def contact_forces(state):
    force={'left':0.,'right':0.}
    for c in state['contacts']:
        if 'peg_collision' not in (c['geom1'],c['geom2']):continue
        for side in force:
            if any(g in {side+'_finger_geom_'+str(k) for k in range(3,8)} for g in (c['geom1'],c['geom2'])):
                force[side]+=max(0.,c['normal_force_n'])
    return force


def show(viewer,env,state,*,label,step,mode='ARCHIVED STATE PLAYBACK; NO INTEGRATION',extra=''):
    with viewer.lock():
        apply_archived_state(env,state)
        scene=viewer.user_scn;scene.ngeom=0
        for c in state.get('contacts',[]):
            if scene.ngeom>=scene.maxgeom:break
            env.mj.mjv_initGeom(scene.geoms[scene.ngeom],env.mj.mjtGeom.mjGEOM_SPHERE,
                               np.array([.0007]*3),np.array(c['position_world_m']),np.eye(3).reshape(-1),np.array([0.,1.,1.,.85],dtype=np.float32))
            scene.ngeom+=1
    forces=contact_forces(state)
    text=(f'{mode}\n{label}: physics {step}; episode t={step/120:.6f}s\n'
          f'ARCHIVED pad forces L/R: {forces["left"]:.5g}/{forces["right"]:.5g} N\n'
          'Cyan: RECORDED contact positions (normals UNKNOWN)\n'
          'R: toggle RECOMPUTED contacts/forces; these are not historical impulses\n'
          'Space play/pause | Right step | Home reset | S slow/normal\n'
          '1 initial | 2 action | 3..9,0 step1..8 | F finger | A arm | C collision\n'
          'V overview | Z closeup | Mouse rotates camera\n'+extra)
    viewer.set_texts((None,env.mj.mjtGridPos.mjGRID_TOPLEFT,text,''))
    viewer.sync()


def require_gui_environment():
    if threading.current_thread() is not threading.main_thread():raise RuntimeError('GUI_MAIN_THREAD_REQUIRED')
    if not os.environ.get('DISPLAY'):raise RuntimeError('GUI_BLOCKED: DISPLAY missing')
    if os.environ.get('MUJOCO_GL','').lower() in ('egl','osmesa'):raise RuntimeError('GUI_BLOCKED: offscreen backend is forbidden')
    subprocess.run(['xdpyinfo'],check=True,stdout=subprocess.DEVNULL)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ARCHIVE)
    parser.add_argument('--capture-events',type=Path)
    parser.add_argument('--auto-close',type=float,default=0.)
    args=parser.parse_args();require_gui_environment()
    if not np.isfinite(args.auto_close) or args.auto_close<0:raise ValueError('INVALID_AUTO_CLOSE')
    import mujoco.viewer,glfw
    initial,rows,action=load_archive(args.archive)
    env=AdaptedPegInsertEnv()
    if env.mj.__version__!='3.5.0' or hashlib.sha256(env.xml.encode()).hexdigest()!=XML_SHA:raise ValueError('FROZEN_MODEL_MISMATCH')
    commands=deque();playback=PlaybackController(len(rows));started=time.monotonic()
    events_by_key={glfw.KEY_1:0,glfw.KEY_2:0,**{getattr(glfw,'KEY_'+str(i)):i-2 for i in range(3,10)},glfw.KEY_0:8,glfw.KEY_F:61,glfw.KEY_A:76,glfw.KEY_C:685}
    def callback(key):commands.append(key)
    output=args.capture_events
    if output:output.mkdir(parents=True,exist_ok=False)
    viewer=None;render_threads=[];close_receipt=None;failure=None
    try:
        before_threads={t.ident for t in threading.enumerate()}
        viewer=mujoco.viewer.launch_passive(env.model,env.data,key_callback=callback,show_left_ui=False,show_right_ui=False)
        render_threads=[t for t in threading.enumerate() if t.ident not in before_threads and '_launch_internal' in t.name]
        if not render_threads:raise RuntimeError('GUI_RENDER_THREAD_NOT_IDENTIFIED')
        viewer.opt.flags[env.mj.mjtVisFlag.mjVIS_CONTACTPOINT]=False
        viewer.opt.flags[env.mj.mjtVisFlag.mjVIS_CONTACTFORCE]=False
        configure_camera(viewer,'closeup')
        show(viewer,env,initial,label='initial physical grasp',step=0)
        time.sleep(1.)
        info=window_info()
        print(json.dumps({'gui_status':'GUI_OPEN_VISIBLE','window':info,'new_physics_steps':0,'policy_inferences':0}),flush=True)
        if output:
            dump(output/'gui_visibility.json',{'status':'GUI_OPEN_VISIBLE','window':info,'DISPLAY':os.environ['DISPLAY'],'WAYLAND_DISPLAY':os.environ.get('WAYLAND_DISPLAY'),'session_type':os.environ.get('XDG_SESSION_TYPE'),'mujoco_version':env.mj.__version__,'model_xml_sha256':XML_SHA,'launch_thread':'main Python thread; MuJoCo Linux passive viewer owns rendering thread','new_physics_steps':0,'actor_calls':0})
            captures=[]
            capture_events={'initial_overview':0,**EVENTS}
            for label,step in capture_events.items():
                if not viewer.is_running():raise RuntimeError('GUI_CLOSED_BEFORE_EVENT_CAPTURE')
                if label=='initial_overview' or step>=61:configure_camera(viewer,'overview')
                elif label=='initial_grasp':configure_camera(viewer,'closeup')
                state=initial if step==0 else rows[step-1]['state']
                show(viewer,env,state,label=label,step=step,extra='First clipped PPO action (archived only): '+str(action['clipped']) if label=='first_action' else '')
                time.sleep(.6);viewer.sync();time.sleep(.15)
                path=output/(label+'.png');capture_desktop_window(path,window_info())
                captures.append({'file':str(path),'sha256':sha(path),'source_experiment':'S2A seed1001','mode':'ARCHIVED_STATE_PLAYBACK','physics_step':step,'episode_simulation_time_s':step/120,'recorded_absolute_time_s':state['time_s'],'model_xml_sha256':XML_SHA,'new_physics_steps':0,'recorded_contact_forces_n':contact_forces(state),'solver_internal_state':'UNKNOWN_NOT_RESTORED'})
            dump(output/'capture_manifest.json',captures)
            configure_camera(viewer,'closeup');show(viewer,env,initial,label='initial physical grasp - interactive',step=0)
        while viewer.is_running():
            if args.auto_close and time.monotonic()-started>args.auto_close:break
            while commands:
                key=commands.popleft()
                if key==glfw.KEY_SPACE:playback.toggle()
                elif key==glfw.KEY_RIGHT:playback.step()
                elif key==glfw.KEY_HOME:playback.jump(0)
                elif key==glfw.KEY_S:playback.cycle_speed()
                elif key in events_by_key:playback.jump(events_by_key[key])
                elif key==glfw.KEY_V:configure_camera(viewer,'overview')
                elif key==glfw.KEY_Z:configure_camera(viewer,'closeup')
                elif key==glfw.KEY_R:
                    with viewer.lock():
                        flag=not viewer.opt.flags[env.mj.mjtVisFlag.mjVIS_CONTACTPOINT]
                        viewer.opt.flags[env.mj.mjtVisFlag.mjVIS_CONTACTPOINT]=flag
                        viewer.opt.flags[env.mj.mjtVisFlag.mjVIS_CONTACTFORCE]=flag
            index=playback.tick(time.monotonic())
            show(viewer,env,initial if index==0 else rows[index-1]['state'],label='interactive',step=index,extra=f'Playback speed {playback.speed}x; playing={playback.playing}')
            time.sleep(.015)
    except Exception as exc:
        failure={'type':type(exc).__name__,'error':str(exc),'policy_inferences':0,'new_physics_steps':0}
    finally:
        if viewer is not None:
            try:close_receipt=close_and_join(viewer,render_threads)
            except Exception as exc:
                failure=failure or {'type':type(exc).__name__,'error':str(exc),'policy_inferences':0,'new_physics_steps':0}
    if failure:
        if output:dump(output/'viewer_failure.json',failure)
        raise RuntimeError(failure['error'])
    if output:dump(output/'viewer_exit.json',{'exit_code':0,'policy_inferences':0,'new_physics_steps':0,'wall_s':time.monotonic()-started,'graceful_close':close_receipt})
    print(json.dumps({'viewer_exit':0,'graceful_close':close_receipt,'new_physics_steps':0,'actor_calls':0}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
