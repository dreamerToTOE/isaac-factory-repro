#!/usr/bin/env python3
"""R1 archived-state GUI viewer: no integration and no policy execution.

Space pauses/plays; arrows single-step; Home/End select first/last; S cycles
slow/normal speed; L jumps to the first recorded breach; V/Z select cameras.
Cyan markers and forces are archived post-forward observations, not impulses.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
from scripts.g4s2a_gripper_v1_viewer import apply_state, first_failure
from scripts.g4s2a_r1_viewer import (require_gui_environment,contact_forces,configure_camera,
    window_info,capture_desktop_window,close_and_join,dump)
from scripts.capture_g4s2a_r1_comparison import raise_own_window

RUNS=('PREP_I2_RATE_LIMITED','PREP_I3_RATE_LIMITED','DIAG_I2_NATIVE','DIAG_I3_NATIVE')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_run(matrix, run):
    if run not in RUNS:raise ValueError('UNREGISTERED_RUN_NAME')
    directory=Path(matrix).resolve()/'runs'/run/'output'
    identity=json.loads((directory/'identity.json').read_text())
    if identity['run']!=run:raise ValueError('RUN_IDENTITY_MISMATCH')
    result_file=directory/'result.json'
    result=json.loads(result_file.read_text()) if result_file.exists() else None
    trace=directory/'physics.jsonl'
    rows=([v for line in trace.read_text().splitlines() if (v:=json.loads(line)).get('event')=='physics_result'] if trace.exists() else [])
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('INCONSISTENT_RECORDED_TRACE')
    if result is not None:
        if result['completed_steps']!=len(rows):raise ValueError('RESULT_TRACE_COUNT_MISMATCH')
        if result.get('trace_sha256') is not None and (not trace.exists() or sha(trace)!=result['trace_sha256']):
            raise ValueError('RESULT_TRACE_HASH_MISMATCH')
    gate_file=directory/'preparation_gate.json'
    gate=json.loads(gate_file.read_text()) if gate_file.exists() else None
    failure=first_failure(rows,run.startswith('PREP_'),gate)
    if not rows and result and (result.get('runtime_failure') or result.get('early_stop_reasons')):failure=0
    states=([rows[0]['pre_state']]+[r['state'] for r in rows]) if rows else [identity['initial']]
    return {'directory':directory,'identity':identity,'trace':trace if trace.exists() else None,
            'rows':rows,'states':states,'first_failure':failure,'result':result}

class Playback:
    """Pure saved-frame state machine, including a zero-integration trace."""
    def __init__(self,nsteps):
        if not isinstance(nsteps,int) or nsteps<0:raise ValueError('INVALID_PLAYBACK_LENGTH')
        self.nsteps=nsteps;self.index=0;self.playing=False;self.speed=.1;self.last=None
    def jump(self,index):
        if not isinstance(index,int) or not 0<=index<=self.nsteps:raise ValueError('INVALID_PLAYBACK_INDEX')
        self.index=index;self.playing=False;self.last=None
    def key(self,key,first_failure=None):
        if key=='SPACE':self.playing=not self.playing;self.last=None
        elif key=='RIGHT':self.jump(min(self.nsteps,self.index+1))
        elif key=='LEFT':self.jump(max(0,self.index-1))
        elif key=='HOME':self.jump(0)
        elif key=='END':self.jump(self.nsteps)
        elif key=='L' and first_failure is not None:self.jump(first_failure)
        elif key=='S':self.speed={.1:.25,.25:1.,1.:.1}[self.speed];self.last=None
        elif key=='V':return 'overview'
        elif key=='Z':return 'closeup'
        return None
    def tick(self,now):
        if not np.isfinite(now):raise ValueError('INVALID_PLAYBACK_CLOCK')
        if self.last is None:self.last=now;return self.index
        if now<self.last:raise ValueError('PLAYBACK_CLOCK_MOVED_BACKWARDS')
        if self.playing and now-self.last>=1/(120*self.speed):
            count=int((now-self.last)*120*self.speed)
            self.index=min(self.nsteps,self.index+count);self.last+=count/(120*self.speed)
            if self.index==self.nsteps:self.playing=False
        return self.index

def selection(text,count,failure):
    selected=set()
    for entry in text.split(','):
        if entry=='last':selected.add(count)
        elif entry=='failure':
            if failure is not None:selected.add(failure)
        else:selected.add(int(entry))
    if any(not 0<=i<=count for i in selected):raise ValueError('REQUESTED_STEP_OUTSIDE_TRACE')
    return sorted(selected)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True);p.add_argument('--run',choices=RUNS,required=True)
    p.add_argument('--capture',type=Path);p.add_argument('--steps',default='last');p.add_argument('--overview',action='store_true')
    a=p.parse_args();run=read_run(a.matrix,a.run);identity=run['identity'];rows=run['rows'];states=run['states'];failure=run['first_failure']
    require_gui_environment();env=NativeGripperEnv()
    if hashlib.sha256(env.xml.encode()).hexdigest()!=identity['model']['xml_sha256']:raise ValueError('PLAYBACK_MODEL_IDENTITY_CHANGED')
    controller=Playback(len(rows));commands=deque();viewer=None;threads=[];captures=[]
    if a.capture:a.capture=a.capture.resolve();a.capture.mkdir(parents=True,exist_ok=False)
    import mujoco.viewer,glfw
    try:
        before={t.ident for t in threading.enumerate()}
        viewer=mujoco.viewer.launch_passive(env.model,env.data,key_callback=lambda key:commands.append(key),show_left_ui=False,show_right_ui=False)
        threads=[t for t in threading.enumerate() if t.ident not in before and '_launch_internal' in t.name]
        if not threads:raise RuntimeError('GUI_RENDER_THREAD_MISSING')
        configure_camera(viewer,'overview' if a.overview else 'closeup')
        def frame(index):
            state=states[index]
            with viewer.lock():
                env.mj.mjv_defaultOption(viewer.opt);apply_state(env,state)
                scene=viewer.user_scn;scene.ngeom=0
                for c in state.get('contacts',[]):
                    if scene.ngeom>=scene.maxgeom:break
                    env.mj.mjv_initGeom(scene.geoms[scene.ngeom],env.mj.mjtGeom.mjGEOM_SPHERE,
                      np.array([.0007]*3),np.asarray(c['position_world_m']),np.eye(3).reshape(-1),np.array([0.,1.,1.,.85],dtype=np.float32));scene.ngeom+=1
            force=contact_forces(state)
            viewer.set_texts((None,env.mj.mjtGridPos.mjGRID_TOPLEFT,
               f'SAVED STATE PLAYBACK | ZERO INTEGRATION | ZERO POLICY\n{a.run} | saved step {index}/{len(rows)}\n'
               f'Archived POST-FORWARD pad force L/R {force["left"]:.5g}/{force["right"]:.5g} N\n'
               'Space pause/play | Left/Right step | L failure | S speed\nHome/End | V/Z camera | cyan saved contacts; no impulses reconstructed',''))
            viewer.sync()
        frame(0);time.sleep(1);raise_own_window(window_info()['window_id']);frame(0);time.sleep(1)
        if a.capture:
            for step in selection(a.steps,len(rows),failure):
                frame(step);raise_own_window(window_info()['window_id']);frame(step);time.sleep(.5)
                path=a.capture/f'{a.run}_step_{step:03d}.png';capture_desktop_window(path,window_info())
                captures.append({'file':str(path),'run':a.run,'saved_step':step,'model_xml_sha256':identity['model']['xml_sha256'],
                  'source_trace_sha256':sha(run['trace']) if run['trace'] else None,'sha256':sha(path),'mode':'SAVED_STATE_PLAYBACK',
                  'new_integration_steps':0,'policy_calls':0,'window':window_info(),
                  'note':'Archived state and post-forward contact force; no solver impulse reconstructed.'})
            dump(a.capture/'manifest.json',captures)
        else:
            keynames={glfw.KEY_SPACE:'SPACE',glfw.KEY_RIGHT:'RIGHT',glfw.KEY_LEFT:'LEFT',glfw.KEY_HOME:'HOME',
                glfw.KEY_END:'END',glfw.KEY_L:'L',glfw.KEY_S:'S',glfw.KEY_V:'V',glfw.KEY_Z:'Z'}
            while viewer.is_running():
                while commands:
                    preset=controller.key(keynames.get(commands.popleft(),''),failure)
                    if preset:configure_camera(viewer,preset)
                frame(controller.tick(time.monotonic()));time.sleep(.01)
    finally:
        if viewer is not None:
            receipt=close_and_join(viewer,threads)
            if a.capture:dump(a.capture/'close_receipt.json',receipt)
if __name__=='__main__':main()
