"""Independent pre-integration MuJoCo native-gripper audit.

Only compiles and evaluates forward dynamics in separate MjData objects.  No
mj_step, PPO model, physical experiment, or mutation of historical artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
from scripts.g4s2a_r2_model import make_model


def _check(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _same(a, b, label):
    _check(np.array_equal(np.asarray(a), np.asarray(b)), 'FROZEN_PROPERTY_CHANGED:' + label)


def audit():
    candidate = NativeGripperEnv()
    reference = make_model(True)
    m, old = candidate.model, reference.model
    _check(mj.__version__ == '3.5.0', 'MUJOCO_VERSION')
    _check((m.nq, m.nv, m.nu, m.ntendon) == (old.nq, old.nv, 8, 1), 'MODEL_DIMENSIONS')
    _check(m.neq == 0, 'UNEXPECTED_EQUALITY_OR_WELD')
    _check(int(m.jnt_type[m.joint('peg_free').id]) == int(mj.mjtJoint.mjJNT_FREE), 'PEG_NOT_FREE')
    _check(m.body('peg').id != m.body('hand').id, 'PEG_NOT_INDEPENDENT')
    # Compare compiled bodies, collision surfaces, mass and P_FULL inertia to
    # the baseline, not merely the source XML's descriptive labels.
    props = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
             'body_parentid', 'body_gravcomp', 'jnt_type', 'jnt_range',
             'jnt_limited', 'jnt_bodyid', 'jnt_pos', 'jnt_axis',
             'dof_armature', 'dof_damping', 'dof_frictionloss',
             'geom_type', 'geom_bodyid', 'geom_dataid', 'geom_contype',
             'geom_conaffinity', 'geom_condim', 'geom_size', 'geom_pos',
             'geom_quat', 'geom_friction', 'geom_margin', 'geom_gap',
             'geom_solref', 'geom_solimp')
    for name in props:
        _same(getattr(m, name), getattr(old, name), name)
    for body in range(1, m.nbody):
        if m.body_mass[body] > 0:
            moments = m.body_inertia[body]
            _check(np.all(moments > 0) and 2 * max(moments) <= sum(moments) + 1e-12,
                   'INVALID_BODY_INERTIA:' + str(body))
    peg = m.geom('peg_collision').id
    def can_collide(a, b):
        return bool((m.geom_contype[a] & m.geom_conaffinity[b]) or
                    (m.geom_contype[b] & m.geom_conaffinity[a]))
    pads = [m.geom(f'{side}_finger_geom_{i}').id
            for side in ('left', 'right') for i in range(3, 8)]
    _check(all(can_collide(peg, pad) for pad in pads), 'PEG_PAD_COLLISIONS_DISABLED')
    _check(can_collide(peg, m.geom('hole_wall_000').id), 'PEG_HOLE_COLLISION_DISABLED')
    for name in ('timestep', 'integrator', 'cone', 'solver', 'iterations',
                 'tolerance', 'gravity'):
        _same(getattr(m.opt, name), getattr(old.opt, name), 'option.' + name)
    for i in range(1, 8):
        n = f'arm_torque_{i}'
        a, b = m.actuator(n).id, old.actuator(n).id
        for attr in ('actuator_trntype', 'actuator_trnid', 'actuator_gear',
                     'actuator_ctrlrange', 'actuator_ctrllimited', 'actuator_forcerange',
                     'actuator_forcelimited', 'actuator_gaintype', 'actuator_gainprm',
                     'actuator_biastype', 'actuator_biasprm'):
            _same(getattr(m, attr)[a], getattr(old, attr)[b], n + '.' + attr)
    joints = [m.joint('finger_joint1').id, m.joint('finger_joint2').id]
    _same(m.jnt_range[joints], [[0., .04], [0., .04]], 'finger_joint_limits')
    _check(all(m.jnt_limited[joints]), 'FINGER_JOINTS_NOT_LIMITED')
    dofs = m.jnt_dofadr[joints].astype(int)
    qadr = m.jnt_qposadr[joints].astype(int)
    actuator = m.actuator('gripper_v1_position').id
    tendon = m.tendon('gripper_v1_sum').id
    _check(int(m.actuator_trntype[actuator]) == int(mj.mjtTrn.mjTRN_TENDON), 'ACTUATOR_NOT_TENDON')
    _check(int(m.actuator_trnid[actuator, 0]) == tendon, 'WRONG_ACTUATOR_TENDON')
    _check(bool(m.actuator_ctrllimited[actuator]) and bool(m.actuator_forcelimited[actuator]), 'UNLIMITED_ACTUATOR')
    _same(m.actuator_ctrlrange[actuator], [0, .08], 'control_range')
    _same(m.actuator_forcerange[actuator], [-40, 40], 'force_range')
    _same(m.actuator_gainprm[actuator, 0], 3750., 'gain_kp')
    _same(m.actuator_biasprm[actuator, :3], [0, -3750, -86.5], 'position_velocity_feedback')
    _same(m.actuator_gear[actuator], [1, 0, 0, 0, 0, 0], 'actuator_gear')
    # Force tests use separate data, with all arm controls at zero. The result
    # is static numerical forward evaluation; no dynamics step is performed.
    samples = []
    cases = [([.005, .005], [0., 0.], .010),
             ([.003, .006], [0., 0.], .010),
             ([.003, .006], [.001, .002], .010),
             ([.040, .040], [0., 0.], .000),
             ([.000, .000], [0., 0.], .080)]
    moment = None
    for q, v, target in cases:
        d = mj.MjData(m)
        d.qpos[qadr] = q
        d.qvel[dofs] = v
        d.ctrl[actuator] = target
        mj.mj_forward(m, d)
        start, count = int(d.moment_rowadr[actuator]), int(d.moment_rownnz[actuator])
        full_moment = np.zeros(m.nv)
        full_moment[d.moment_colind[start:start+count]] = d.actuator_moment[start:start+count]
        expected_moment = np.zeros(m.nv); expected_moment[dofs] = 1.
        _check(np.allclose(full_moment, expected_moment, atol=1e-12, rtol=0), 'WRONG_MOMENT_ARMS')
        moment = full_moment.tolist()
        _check(np.isclose(d.ten_length[tendon], sum(q), atol=1e-12, rtol=0), 'WRONG_TENDON_LENGTH')
        _check(np.isclose(d.ten_velocity[tendon], sum(v), atol=1e-12, rtol=0), 'WRONG_TENDON_VELOCITY')
        expected = float(np.clip(3750. * (target - sum(q)) - 86.5 * sum(v), -40., 40.))
        actual = float(d.actuator_force[actuator])
        _check(np.isclose(actual, expected, atol=1e-10, rtol=0), 'BAD_FORCE_OR_CLIPPING')
        _check(np.allclose(d.qfrc_actuator[dofs], expected * full_moment[dofs], atol=1e-10, rtol=0), 'BAD_JOINT_FORCE_TRANSMISSION')
        _check(d.time == 0., 'UNEXPECTED_PHYSICS_STEP')
        samples.append({'finger_position_m': q, 'finger_velocity_m_s': v,
                        'target_tendon_m': target, 'tendon_length_m': float(d.ten_length[tendon]),
                        'tendon_velocity_m_s': float(d.ten_velocity[tendon]),
                        'expected_actuator_force_n': expected, 'actuator_force_n': actual,
                        'joint_generalized_actuator_force_n': d.qfrc_actuator[dofs].tolist()})
    # Independently differentiate tendon length w.r.t each finger qpos.
    eps = 1e-7; derivatives = []
    for i in range(2):
        values = []
        for sign in (-1., 1.):
            d = mj.MjData(m); d.qpos[qadr] = [.01, .01]; d.qpos[qadr[i]] += sign * eps
            mj.mj_forward(m, d); values.append(float(d.ten_length[tendon]))
        derivatives.append((values[1] - values[0]) / (2 * eps))
    _check(np.allclose(derivatives, [1., 1.], atol=1e-9, rtol=0), 'FINITE_DIFFERENCE_FAILED')
    # Draws and the pre-closure snapshots are preserved for identity checking;
    # none of the historical post-closure states is used as a new preparation.
    initials = {}
    for label, seed in [('I2', 2101), ('I3', 2102)]:
        root = Path('evidence/g4_s/s2a_r2/matrix_v2/initials') / label
        draw_path, admission_path, ik_path = root / 'reset_draw.json', root / 'admission.json', root / 'ik.json'
        draw = json.loads(draw_path.read_text()); admission = json.loads(admission_path.read_text()); ik = json.loads(ik_path.read_text())
        _check(draw['seed'] == seed and draw == admission['draw'], 'DRAW_IDENTITY_MISMATCH:' + label)
        _check(ik == admission['ik'] and ik['accepted'], 'IK_IDENTITY_MISMATCH:' + label)
        _check('qpos' in admission['initial'], 'NO_PRE_CLOSURE_STATE:' + label)
        initials[label] = {'seed': seed, 'reset_draw_sha256': hashlib.sha256(draw_path.read_bytes()).hexdigest(),
                           'admission_sha256': hashlib.sha256(admission_path.read_bytes()).hexdigest(),
                           'ik_sha256': hashlib.sha256(ik_path.read_bytes()).hexdigest(),
                           'preclosure_qpos': admission['initial']['qpos'],
                           'historical_initial_state_is_post_closure': True}
    return {'status': 'PASS', 'mujoco_version': mj.__version__, 'physics_steps': 0,
            'candidate_xml_sha256': hashlib.sha256(candidate.xml.encode()).hexdigest(),
            'reference_p_full_xml_sha256': hashlib.sha256(reference.xml.encode()).hexdigest(),
            'frozen_compiled_properties': list(props), 'equality_count': int(m.neq),
            'peg_joint_type': 'free', 'actuator_moment_full': moment,
            'finite_difference_dlength_dq': derivatives, 'force_tests': samples,
            'initials': initials, 'scope': 'static forward computations only, not a dynamic-grasp validation'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = audit()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(path)}))


if __name__ == '__main__': main()
