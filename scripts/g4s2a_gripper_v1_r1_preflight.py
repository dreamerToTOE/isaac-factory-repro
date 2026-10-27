#!/usr/bin/env python3
"""No-integration, visible-desktop state boundary investigation.

Receipts describe observed boundaries only.  They do not establish the cause of
historical V1 I3 corruption.  GUI screenshot images require independent privacy
QA before publication. The physics-step functions are made unavailable here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import mujoco
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
from scripts.g4s2a_gripper_v1_diagnostics import initial, read, R2
from scripts.g4s2a_gripper_v1_r1_state import (snapshot, compare, guard, IsolatedViewer,
    StateIntegrityError, copy_display_state)
from scripts.g4s2a_r1_viewer import configure_camera, window_info
from scripts.capture_g4s2a_r1_comparison import raise_own_window


def investigate(out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    old_step = mujoco.mj_step
    old_step1, old_step2 = mujoco.mj_step1, mujoco.mj_step2
    def forbidden(*args, **kwargs): raise RuntimeError('PREFLIGHT_INTEGRATION_FORBIDDEN')
    mujoco.mj_step = mujoco.mj_step1 = mujoco.mj_step2 = forbidden
    records, experiments, negative, errors = [], {}, [], []
    def event(name, env, operation, expected=None, permitted=False):
        before = snapshot(mujoco, env.model, env.data)
        try: operation()
        except BaseException as exc:
            errors.append({'boundary': name, 'error': repr(exc), 'traceback': traceback.format_exc()})
            raise
        after = snapshot(mujoco, env.model, env.data)
        changes = compare(before, after)
        expected_changes = [] if expected is None else compare(expected, after)
        records.append({'boundary': name, 'before': before, 'after': after,
                        'changes': changes, 'against_registered': expected_changes,
                        'changes_authorized': permitted})
        if changes and not permitted: errors.append({'boundary': name, 'unexpected_changes': changes})
        return after
    try:
        previous_env = None; previous_final = None
        for ident in ('I2', 'I3'):
            env = NativeGripperEnv()
            construction = snapshot(mujoco, env.model, env.data)
            records.append({'boundary': ident + '/ENVIRONMENT_CONSTRUCTION', 'before': None,
                            'after': construction, 'changes_authorized': True})
            # V1 reset operation calls are recorded at their actual boundaries.
            reset0, forward0 = mujoco.mj_resetData, mujoco.mj_forward
            def traced_reset(m, d): return event(ident+'/INITIAL_MJ_RESETDATA', env, lambda: reset0(m,d), permitted=True)
            def traced_forward(m, d): return event(ident+'/INITIAL_MJ_FORWARD', env, lambda: forward0(m,d))
            mujoco.mj_resetData, mujoco.mj_forward = traced_reset, traced_forward
            try: initial(env, ident)
            finally: mujoco.mj_resetData, mujoco.mj_forward = reset0, forward0
            registered = snapshot(mujoco, env.model, env.data)
            source = read(R2/'initials'/ident/'admission.json')['initial']
            match = np.array_equal(env.data.qpos, np.asarray(source['qpos'])) and np.array_equal(env.data.qvel, np.asarray(source['qvel'])) and env.data.time == 0
            records.append({'boundary': ident+'/REGISTERED_INITIAL_STATE', 'before': None, 'after': registered,
                            'registered_source_qpos_qvel_time_match': bool(match), 'changes_authorized': True})
            if not match: errors.append({'boundary': ident+'/REGISTERED_INITIAL_STATE','error':'REGISTERED_INITIAL_STATE_MISMATCH'})
            if previous_env is not None:
                previous_now = snapshot(mujoco, previous_env.model, previous_env.data)
                diff = compare(previous_final, previous_now)
                records.append({'boundary': 'SWITCH_I2_TO_I3', 'before': previous_final, 'after': previous_now,
                                'changes': diff, 'changes_authorized': False})
                if diff: errors.append({'boundary':'SWITCH_I2_TO_I3','unexpected_changes':diff})
            # Restore the same registered full state to expose reset/setState/forward boundaries.
            event(ident+'/MJ_SETSTATE_SAME_REGISTERED', env,
                  lambda: mujoco.mj_setState(env.model, env.data, np.asarray(registered['integration_state']), mujoco.mjtState.mjSTATE_INTEGRATION), registered)
            event(ident+'/MJ_FORWARD_REGISTERED', env, lambda: mujoco.mj_forward(env.model, env.data), registered)
            # Negative controls mutate copies only; no mj_step is reachable.
            for field, index in [('qpos',0),('qpos',7),('qpos',9),('qpos',12),('qvel',0),('time',None),('qfrc_applied',0),('qacc_warmstart',0)]:
                copy = mujoco.MjData(env.model)
                mujoco.mj_setState(env.model, copy, np.asarray(registered['integration_state']), mujoco.mjtState.mjSTATE_INTEGRATION)
                if index is None: copy.time += 1e-6
                else: getattr(copy,field)[index] += 1e-6
                try:
                    guard(registered, mujoco, env.model, copy, 'negative_control')
                    rejected = False; mismatch = []
                except StateIntegrityError as exc:
                    rejected = True; mismatch = exc.mismatches
                actual = snapshot(mujoco, env.model, copy)
                negative.append({'id':ident,'field':field,'index':index,'expected':registered,'actual':actual,'rejected':rejected,'mismatches':mismatch,'mj_step_calls':0})
                if not rejected: errors.append({'boundary':ident+'/NEGATIVE_CONTROL','field':field,'index':index})
            iso = IsolatedViewer(env, ident+'_PREFLIGHT_NO_INTEGRATION')
            if iso.model is env.model or iso.data is env.data: errors.append({'boundary':ident+'/MODEL_DATA_ISOLATION'})
            # Each operation below is checked against authoritative data; no shared-data viewer is used.
            event(ident+'/ISOLATED_VIEWER_LAUNCH_INITIAL_SYNC_CAMERA_ACTIVATION', env, iso.__enter__, registered)
            try:
                event(ident+'/VIEWER_SYNC_COPY_FORWARD', env, lambda: iso.sync(0), registered)
                def camera():
                    with iso.viewer.lock(): configure_camera(iso.viewer, 'closeup')
                # configure_camera itself acquires the viewer lock, so invoke directly.
                event(ident+'/CAMERA_CONFIGURE', env, lambda: configure_camera(iso.viewer,'closeup'), registered)
                event(ident+'/WINDOW_ACTIVATION', env, lambda: raise_own_window(window_info()['window_id']), registered)
                image = out/(ident+'_preflight.png')
                event(ident+'/SCREENSHOT_CAPTURE', env, lambda: iso.capture(image,0), registered)
                # Viewer mutation must stay one-way; restore display from physics before closing.
                def viewer_mutation():
                    with iso.viewer.lock(): iso.data.qpos[0] += .001
                event(ident+'/VIEWER_ONLY_QPOS_MUTATION', env, viewer_mutation, registered)
                event(ident+'/VIEWER_RESTORE_ONE_WAY', env, lambda: iso.sync(0), registered)
                restored = np.array_equal(iso.data.qpos, env.data.qpos)
                if not restored: errors.append({'boundary':ident+'/VIEWER_RESTORE_ONE_WAY','error':'VIEWER_DID_NOT_MATCH_PHYSICS'})
                experiments[ident] = {'gui_window':iso.info,'gui_sync_count':iso.sync_count,
                    'separate_model':iso.model is not env.model,'separate_data':iso.data is not env.data,
                    'model_fingerprint':registered['model_fingerprint'],'viewer_model_fingerprint':snapshot(mujoco,iso.model,iso.data)['model_fingerprint'],
                    'screenshot':str(image),'screenshot_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                    'registered_qpos_qvel_time_match':bool(match),'viewer_copy_restored':bool(restored)}
            finally:
                event(ident+'/VIEWER_CLOSE', env, lambda: iso.__exit__(None,None,None), registered)
            previous_env, previous_final = env, snapshot(mujoco, env.model, env.data)
        status = 'PASS' if not errors else 'FAIL'
    except BaseException as exc:
        status = 'FAIL'; errors.append({'boundary':'UNHANDLED','error':repr(exc),'traceback':traceback.format_exc()})
    finally: mujoco.mj_step, mujoco.mj_step1, mujoco.mj_step2 = old_step, old_step1, old_step2
    result = {'status':status,'passed':status == 'PASS','physics_steps':0,'policy_calls':0,'mujoco_version':mujoco.__version__,
              'experiments':experiments,'boundaries':records,'negative_controls':negative,'errors':errors,
              'historical_cause':'UNKNOWN; isolated preflight does not establish the historical corruption boundary',
              'screenshot_qa':'PENDING_PRIVACY_AND_VISUAL_INSPECTION'}
    path = out/'state_preflight.json'
    with path.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'status':status,'errors':errors,'output':str(path)}))
    return 0 if status == 'PASS' else 1

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raise SystemExit(investigate(a.output))
