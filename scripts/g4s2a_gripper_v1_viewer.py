#!/usr/bin/env python3
"""Native-gripper saved-state viewer. No physics integration and no PPO calls.

Space: pause/play. Left/Right: one recorded frame. Home/End: first/last.
S: cycle slow playback speed. L: jump to first detected failure. V/Z: camera
presets; mouse adjusts the native viewer camera. No new motion is integrated.
Saved-state forces shown are archived post-forward values, not impulses.
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
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
from scripts.g4s2a_r1_viewer import (require_gui_environment, PlaybackController,
    contact_forces, configure_camera, window_info, capture_desktop_window,
    close_and_join, dump)
from scripts.capture_g4s2a_r1_comparison import raise_own_window


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_failure(rows, preparation=False, final_gate=None):
    """Return first observable breach, not normal no-contact before closure."""
    for row in rows:
        state = row['state']; metrics = state.get('s2a', {})
        pad = row.get('pad_penetration_m', 0.)
        force = row.get('pad_force_n', {'left': 0., 'right': 0.})
        if row.get('stop_reasons') or pad > 50e-6 or metrics.get('peg_hole_penetration_m', 0.) > 10e-6:
            return int(row['step'])
        if not preparation and (min(force.values()) <= 1e-6 or
             metrics.get('peg_hand_slip_m', 0.) >= .001 or
             metrics.get('peg_hand_slip_rad', 0.) > .05 or
             max(abs(np.asarray(state['qvel'])[:7])) > 3. or
             metrics.get('arm_joint_limit_violation_rad', 0.) > 1e-6 or
             metrics.get('finger_joint_limit_violation_m', 0.) > 1e-6):
            return int(row['step'])
    if preparation and final_gate and (final_gate.get('reset_admission') != 'PASS' or
                                       final_gate.get('physical_quality') != 'PASS'):
        return int(rows[-1]['step']) if rows else None
    return None


def read_run(matrix, run):
    if run not in ('PREP_I2', 'PREP_I3', 'DIAG_I2', 'DIAG_I3'):
        raise ValueError('UNREGISTERED_RUN_NAME')
    directory = Path(matrix).resolve() / 'runs' / run / 'output'
    identity = json.loads((directory / 'identity.json').read_text())
    if identity['run'] != run:
        raise ValueError('RUN_IDENTITY_MISMATCH')
    trace = directory / 'physics.jsonl'
    rows = [row for line in trace.read_text().splitlines()
            if (row := json.loads(line)).get('event') == 'physics_result']
    if not rows or [row['step'] for row in rows] != list(range(1, len(rows)+1)):
        raise ValueError('EMPTY_OR_INCONSISTENT_RECORDED_TRACE')
    gate_path = directory / 'preparation_gate.json'
    gate = json.loads(gate_path.read_text()) if gate_path.is_file() else None
    return directory, identity, trace, rows, first_failure(rows, run.startswith('PREP_'), gate)


def apply_state(env, state):
    """Set and forward an independent playback data object; never step it."""
    q, v = np.asarray(state['qpos'], float), np.asarray(state['qvel'], float)
    arm = np.asarray(state['software_limited_arm_torque_nm'], float)
    hole = np.asarray(state['hole_root_position'], float)
    quat = np.asarray(state.get('s2a', {}).get('hole_quaternion_wxyz', [1, 0, 0, 0]), float)
    target = float(state.get('gripper_actuator_control_m', 0.))
    stamp = float(state['time_s'])
    if (q.shape != (env.model.nq,) or v.shape != (env.model.nv,) or arm.shape != (7,)
        or hole.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[q,v,arm,hole,quat,target,stamp]).all()
        or abs(np.linalg.norm(quat)-1.) > 1e-6 or not 0. <= target <= .08
        or abs(np.linalg.norm(q[env.peg_qpos_adr+3:env.peg_qpos_adr+7])-1.) > 1e-6):
        raise ValueError('INVALID_SAVED_STATE')
    env.data.qpos[:] = q; env.data.qvel[:] = v; env.data.time = stamp
    env.data.ctrl[env.arm_actuator_ids] = arm
    env.data.ctrl[env.finger_actuator_ids[0]] = target
    env.model.body_pos[env.hole_body_id] = hole; env.model.body_quat[env.hole_body_id] = quat
    env.mj.mj_forward(env.model, env.data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--run', choices=('PREP_I2','PREP_I3','DIAG_I2','DIAG_I3'), required=True)
    p.add_argument('--capture', type=Path, help='new, never-overwritten screenshot directory')
    p.add_argument('--steps', default='last', help='comma-separated saved indices; supports failure,last')
    p.add_argument('--overview', action='store_true')
    a = p.parse_args(); directory, identity, trace, rows, failure = read_run(a.matrix, a.run)
    require_gui_environment(); env = NativeGripperEnv()
    if hashlib.sha256(env.xml.encode()).hexdigest() != identity['model']['xml_sha256']:
        raise ValueError('PLAYBACK_MODEL_IDENTITY_CHANGED')
    # The executed pre-state is evidence; an admitted identity may differ if
    # GUI interaction reset the active model before integration.
    states = [rows[0]['pre_state']] + [r['state'] for r in rows]
    invalid_initial = not np.array_equal(np.asarray(identity['initial']['qpos']), np.asarray(rows[0]['pre_state']['qpos']))
    controller = PlaybackController(len(rows)); commands = deque(); viewer = None; threads = []; captures = []
    if a.capture:
        a.capture = a.capture.resolve(); a.capture.mkdir(parents=True, exist_ok=False)
    import mujoco.viewer, glfw
    try:
        before = {t.ident for t in threading.enumerate()}
        viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=lambda key: commands.append(key),
                                              show_left_ui=False, show_right_ui=False)
        threads = [t for t in threading.enumerate() if t.ident not in before and '_launch_internal' in t.name]
        if not threads: raise RuntimeError('GUI_RENDER_THREAD_MISSING')
        configure_camera(viewer, 'overview' if a.overview else 'closeup')
        def frame(index):
            state = states[index]
            with viewer.lock():
                env.mj.mjv_defaultOption(viewer.opt); apply_state(env, state)
                scene = viewer.user_scn; scene.ngeom = 0
                for contact in state.get('contacts', []):
                    if scene.ngeom >= scene.maxgeom: break
                    env.mj.mjv_initGeom(scene.geoms[scene.ngeom], env.mj.mjtGeom.mjGEOM_SPHERE,
                        np.array([.0007]*3), np.asarray(contact['position_world_m']), np.eye(3).reshape(-1),
                        np.array([0., 1., 1., .85], dtype=np.float32))
                    scene.ngeom += 1
            forces = contact_forces(state)
            viewer.set_texts((None,env.mj.mjtGridPos.mjGRID_TOPLEFT,
                f'SAVED STATE PLAYBACK | ZERO INTEGRATION | ZERO POLICY\n{a.run} | saved step {index}/{len(rows)} | '+('INVALID INITIAL STATE' if invalid_initial else 'INITIAL VERIFIED')+'\n'
                f'Archived post-forward pad force L/R {forces["left"]:.5g}/{forces["right"]:.5g} N\n'
                'Space pause/play | Left/Right step | L first failure | S speed\n'
                'Home/End | V/Z camera | cyan saved contacts; no impulses reconstructed',''))
            viewer.sync()
        frame(0); time.sleep(1.); raise_own_window(window_info()['window_id']); frame(0); time.sleep(1.)
        if a.capture:
            selected = set()
            for entry in a.steps.split(','):
                if entry == 'last': selected.add(len(rows))
                elif entry == 'failure':
                    if failure is not None: selected.add(failure)
                else: selected.add(int(entry))
            for step in sorted(selected):
                if not 0 <= step < len(states): raise ValueError('REQUESTED_STEP_OUTSIDE_TRACE')
                frame(step); raise_own_window(window_info()['window_id']); frame(step); time.sleep(.5)
                path = a.capture / f'{a.run}_step_{step:03d}.png'
                capture_desktop_window(path, window_info())
                captures.append({'file': str(path), 'run': a.run, 'saved_step': step,
                    'model_xml_sha256': identity['model']['xml_sha256'], 'source_trace_sha256': sha(trace),
                    'sha256': sha(path), 'mode': 'SAVED_STATE_PLAYBACK', 'new_integration_steps': 0,
                    'policy_calls': 0, 'window': window_info(),
                    'note': 'Recorded state and post-forward forces only; earlier solver impulse not reconstructed.'})
            dump(a.capture/'manifest.json', captures)
        else:
            while viewer.is_running():
                while commands:
                    key = commands.popleft()
                    if key == glfw.KEY_SPACE: controller.toggle()
                    elif key == glfw.KEY_RIGHT: controller.step()
                    elif key == glfw.KEY_LEFT: controller.jump(max(0, controller.index-1))
                    elif key == glfw.KEY_HOME: controller.jump(0)
                    elif key == glfw.KEY_END: controller.jump(len(rows))
                    elif key == glfw.KEY_L and failure is not None: controller.jump(failure)
                    elif key == glfw.KEY_S: controller.cycle_speed()
                    elif key in (glfw.KEY_V, glfw.KEY_Z): configure_camera(viewer, 'overview' if key == glfw.KEY_V else 'closeup')
                frame(controller.tick(time.monotonic())); time.sleep(.01)
    finally:
        if viewer is not None:
            receipt = close_and_join(viewer, threads)
            if a.capture: dump(a.capture/'close_receipt.json', receipt)


if __name__ == '__main__': main()
