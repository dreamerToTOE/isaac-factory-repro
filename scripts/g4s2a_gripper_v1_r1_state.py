"""R1 state integrity and isolated rendering, with no integration or policy.

All comparisons of authoritative state are bitwise, including warmstart and the
complete MuJoCo integration-state vector. Derived quantities (contacts, xpos,
qacc and solver outputs) are deliberately not equated across forward stages.
"""
from __future__ import annotations
import hashlib
import json
import threading
import time
import numpy as np

_FIELDS = (
    ('time', 'mjSTATE_TIME'), ('qpos', 'mjSTATE_QPOS'), ('qvel', 'mjSTATE_QVEL'),
    ('act', 'mjSTATE_ACT'), ('history', 'mjSTATE_HISTORY'),
    ('qacc_warmstart', 'mjSTATE_WARMSTART'), ('ctrl', 'mjSTATE_CTRL'),
    ('qfrc_applied', 'mjSTATE_QFRC_APPLIED'), ('xfrc_applied', 'mjSTATE_XFRC_APPLIED'),
    ('eq_active', 'mjSTATE_EQ_ACTIVE'), ('mocap_pos', 'mjSTATE_MOCAP_POS'),
    ('mocap_quat', 'mjSTATE_MOCAP_QUAT'), ('userdata', 'mjSTATE_USERDATA'),
    ('plugin', 'mjSTATE_PLUGIN'),
)
_MODEL_PREFIXES = ('body_', 'jnt_', 'dof_', 'geom_', 'site_', 'actuator_',
                   'tendon_', 'wrap_', 'eq_', 'pair_', 'exclude_', 'mesh_',
                   'hfield_', 'flex_', 'key_')
_OPT_FIELDS = ('timestep', 'gravity', 'wind', 'magnetic', 'density', 'viscosity',
               'impratio', 'o_margin', 'o_solref', 'o_solimp', 'integrator',
               'cone', 'jacobian', 'solver', 'iterations', 'tolerance',
               'ls_iterations', 'ls_tolerance', 'noslip_iterations',
               'noslip_tolerance', 'ccd_iterations', 'ccd_tolerance',
               'disableflags', 'enableflags')


def model_fingerprint(model):
    """Digest compiled physical arrays and numerical solver options.

    Includes geometry vertices, transmissions, masses and inertias. This is a
    fingerprint of these selected physical parameters, not a claim that every
    visualization-only or opaque plugin byte is identical.
    """
    h = hashlib.sha256()
    def add(name, value):
        arr = np.asarray(value)
        if arr.dtype.kind not in 'biuf':
            return
        arr = np.ascontiguousarray(arr)
        h.update(name.encode() + b'\0' + arr.dtype.str.encode() + b'\0')
        h.update(json.dumps(arr.shape).encode() + b'\0' + arr.tobytes())
    for name in sorted(dir(model)):
        if name.startswith(_MODEL_PREFIXES):
            value = getattr(model, name)
            if isinstance(value, np.ndarray):
                add(name, value)
    for name in _OPT_FIELDS:
        if hasattr(model.opt, name):
            add('opt.' + name, getattr(model.opt, name))
    for name in ('nq', 'nv', 'na', 'nu', 'nbody', 'ngeom', 'njnt', 'neq',
                 'ntendon', 'nmocap', 'nuserdata', 'nplugin'):
        if hasattr(model, name):
            add(name, getattr(model, name))
    return h.hexdigest()


def snapshot(mj, model, data):
    """Capture every MuJoCo 3.5 integration-state component and its union."""
    def read(spec):
        v = np.zeros(mj.mj_stateSize(model, spec), dtype=np.float64)
        mj.mj_getState(model, data, v, spec)
        return v.tolist()
    state = {key: read(getattr(mj.mjtState, flag)) for key, flag in _FIELDS}
    state['integration_state'] = read(mj.mjtState.mjSTATE_INTEGRATION)
    state['model_fingerprint'] = model_fingerprint(model)
    return state


def compare(expected, actual):
    """Return actionable first differences per field, with exact float bits."""
    differences = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            differences.append({'field': key, 'reason': 'missing_field'})
        elif key == 'model_fingerprint':
            if expected[key] != actual[key]:
                differences.append({'field': key, 'expected': expected[key], 'actual': actual[key]})
        else:
            e, a = np.asarray(expected[key], dtype=np.float64), np.asarray(actual[key], dtype=np.float64)
            if e.shape != a.shape:
                differences.append({'field': key, 'reason': 'shape', 'expected': list(e.shape), 'actual': list(a.shape)})
                continue
            bad = np.flatnonzero(e.view(np.uint64) != a.view(np.uint64))
            if len(bad):
                index = int(bad[0]); differences.append({'field': key, 'reason': 'bitwise',
                    'first_flat_index': index, 'mismatch_count': int(len(bad)),
                    'expected': float(e.flat[index]), 'actual': float(a.flat[index])})
            if not np.isfinite(a).all():
                differences.append({'field': key, 'reason': 'nonfinite'})
    return differences


class StateIntegrityError(RuntimeError):
    def __init__(self, boundary, expected, actual, mismatches):
        super().__init__('STATE_INTEGRITY_FAILURE:' + boundary + ':' + ','.join(x['field'] for x in mismatches))
        self.boundary = boundary
        self.expected = expected
        self.actual = actual
        self.mismatches = mismatches


def guard(expected, mj, model, data, boundary):
    actual = snapshot(mj, model, data)
    differences = compare(expected, actual)
    if differences:
        raise StateIntegrityError(boundary, expected, actual, differences)
    return actual


def authorized_ctrl(mj, model, expected, ctrl):
    """Derive next expected state after *only* the requested control update.

    The caller writes exactly these controls to the live data, then guards it
    immediately at mj_step. A temporary data object prevents any live write.
    """
    if expected['model_fingerprint'] != model_fingerprint(model):
        raise ValueError('MODEL_CHANGED_BEFORE_CONTROL_AUTHORIZATION')
    scratch = mj.MjData(model)
    spec = mj.mjtState.mjSTATE_INTEGRATION
    mj.mj_setState(model, scratch, np.asarray(expected['integration_state']), spec)
    ctrl = np.asarray(ctrl, dtype=np.float64)
    if ctrl.shape != scratch.ctrl.shape or not np.isfinite(ctrl).all():
        raise ValueError('INVALID_AUTHORIZED_CONTROL')
    scratch.ctrl[:] = ctrl
    return snapshot(mj, model, scratch)


def copy_display_state(mj, physics_model, physics_data, viewer_model, viewer_data):
    """Copy physics -> viewer only. Never set or forward authoritative data."""
    if physics_model is viewer_model or physics_data is viewer_data:
        raise ValueError('PHYSICS_AND_VIEWER_MUST_BE_DISTINCT')
    expected = snapshot(mj, physics_model, physics_data)
    if expected['model_fingerprint'] != model_fingerprint(viewer_model):
        raise ValueError('PHYSICAL_MODEL_FINGERPRINT_MISMATCH')
    mj.mj_setState(viewer_model, viewer_data, np.asarray(expected['integration_state']), mj.mjtState.mjSTATE_INTEGRATION)
    mj.mj_forward(viewer_model, viewer_data)
    guard(expected, mj, physics_model, physics_data, 'viewer_copy_and_forward')
    return expected


class IsolatedViewer:
    """Actual passive viewer on an independent model/data; no physics calls."""
    def __init__(self, env, label):
        self.env, self.label, self.viewer = env, label, None
        self.sync_count = 0
        self.threads = []
        self.receipt = None
        self.model = env.mj.MjModel.from_xml_string(env.xml)
        # The registered initial reset changes only the world hole root position.
        hole = env.mj.mj_name2id(self.model, env.mj.mjtObj.mjOBJ_BODY, 'hole')
        if hole < 0:
            raise ValueError('MISSING_HOLE_BODY')
        self.model.body_pos[hole] = env.model.body_pos[env.hole_body_id]
        self.data = env.mj.MjData(self.model)
        if model_fingerprint(env.model) != model_fingerprint(self.model):
            raise ValueError('INDEPENDENT_COMPILED_MODEL_MISMATCH')

    def __enter__(self):
        from scripts.g4s2a_r1_viewer import require_gui_environment, window_info, close_and_join
        from scripts.capture_g4s2a_r1_comparison import raise_own_window
        import mujoco.viewer
        require_gui_environment()
        before = snapshot(self.env.mj, self.env.model, self.env.data)
        copy_display_state(self.env.mj, self.env.model, self.env.data, self.model, self.data)
        threads_before = {t.ident for t in threading.enumerate()}
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.threads = [t for t in threading.enumerate() if t.ident not in threads_before and '_launch_internal' in t.name]
        try:
            if not self.threads:
                raise RuntimeError('GUI_RENDER_THREAD_MISSING')
            guard(before, self.env.mj, self.env.model, self.env.data, 'viewer_launch')
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [.55, 0, .23]
                self.viewer.cam.distance = .85
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -20
                self.viewer.opt.flags[self.env.mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            guard(before, self.env.mj, self.env.model, self.env.data, 'camera_setup')
            self.sync(0)
            time.sleep(1)
            self.info = window_info()
            raise_own_window(self.info['window_id'])
            guard(before, self.env.mj, self.env.model, self.env.data, 'window_activation')
            self.sync(0)
        except BaseException:
            self.receipt = close_and_join(self.viewer, self.threads)
            raise
        return self

    def sync(self, step, state=None, extra=''):
        if not self.viewer.is_running():
            raise RuntimeError('GUI_CLOSED_BEFORE_NEXT_INTEGRATION')
        with self.viewer.lock():
            before = copy_display_state(self.env.mj, self.env.model, self.env.data, self.model, self.data)
        self.viewer.set_texts((None, self.env.mj.mjtGridPos.mjGRID_TOPLEFT,
            f'LIVE PHYSICS STATE DISPLAY | {self.label}\nstep={step} | no PPO | viewer is separate data\n' + extra, ''))
        self.viewer.sync()
        self.sync_count += 1
        guard(before, self.env.mj, self.env.model, self.env.data, 'viewer_sync')
        return before

    def capture(self, path, step, state=None, identity=None):
        from scripts.g4s2a_r1_viewer import configure_camera, window_info, capture_desktop_window
        from scripts.capture_g4s2a_r1_comparison import raise_own_window
        before = snapshot(self.env.mj, self.env.model, self.env.data)
        self.sync(step, state)
        configure_camera(self.viewer, 'closeup')
        guard(before, self.env.mj, self.env.model, self.env.data, 'capture_camera')
        raise_own_window(window_info()['window_id'])
        guard(before, self.env.mj, self.env.model, self.env.data, 'capture_activation')
        self.sync(step, state)
        time.sleep(.35)
        capture_desktop_window(path, window_info())
        guard(before, self.env.mj, self.env.model, self.env.data, 'screenshot_capture')

    def __exit__(self, *args):
        from scripts.g4s2a_r1_viewer import close_and_join
        if self.viewer is not None:
            before = snapshot(self.env.mj, self.env.model, self.env.data)
            self.receipt = close_and_join(self.viewer, self.threads)
            guard(before, self.env.mj, self.env.model, self.env.data, 'viewer_close')
