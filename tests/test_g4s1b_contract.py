"""No-policy CPU tests, with independent unmodified pinned-source AST oracles.

Original installed Isaac functions are read locally after hash checks, selected
into an in-memory namespace, and executed only on CPU Torch tensors. No Isaac
module, simulator, actor, checkpoint, network or USD asset is loaded.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import g4s1b_contract as c

REPO = Path(__file__).resolve().parents[1]
IDENTITY = json.loads((REPO/'evidence/g4_i/i0/checkpoint_audit.json').read_text())
FACTORY = IDENTITY['source_files']
ROTATIONS = Path('/home/ubuntu2004/isaacsim-4.5.0/exts/isaacsim.core.utils/isaacsim/core/utils/torch/rotations.py')
TORCH_MATHS = ROTATIONS.with_name('maths.py')
ISAAC_MATH = Path(FACTORY['factory_control.py']['path']).parents[4]/'isaaclab/isaaclab/utils/math.py'
PINS = {
    str(ROTATIONS): '40e0b7c318fdf77c171dd09bc20aa3185be076fe94a312dcfd11a2a02c8122b0',
    str(TORCH_MATHS): '25eb6af449529061369ccb627714557ee133dde2d03d3e2bd1eef85c49d7a21c',
    str(ISAAC_MATH): '902973439cc588a5c98a3fa1b2ec3656a2ca7cfaba865f8ef5544014f4f81205',
}


def checked_tree(path, sha):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, f'frozen source changed: {path}'
    return ast.parse(data)


def selected_functions(path, names, namespace, sha):
    tree = checked_tree(path, sha)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []  # Only disable JIT decoration, not function bodies.
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), str(path), 'exec'), namespace)


def source_oracle():
    ns = {'torch': torch, 'np': np, 'math': math}
    selected_functions(TORCH_MATHS, ['normalize', 'copysign'], ns, PINS[str(TORCH_MATHS)])
    selected_functions(ROTATIONS, ['quat_mul', 'quat_conjugate', 'quat_unit', 'quat_from_angle_axis',
                                 'get_euler_xyz', 'quat_from_euler_xyz'], ns, PINS[str(ROTATIONS)])
    selected_functions(ISAAC_MATH, ['axis_angle_from_quat'], ns, PINS[str(ISAAC_MATH)])
    ns['torch_utils'] = SimpleNamespace(**{n: ns[n] for n in ('quat_mul', 'quat_conjugate', 'quat_from_angle_axis', 'get_euler_xyz', 'quat_from_euler_xyz')})
    src = FACTORY['factory_control.py']
    selected_functions(src['path'], ['compute_dof_torque', 'get_pose_error', '_apply_task_space_gains'], ns, src['sha256'])
    src = FACTORY['factory_env.py']
    tree = checked_tree(src['path'], src['sha256'])
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FactoryEnv')
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_apply_action', '_pre_physics_step', '_get_observations')]
    assert len(nodes) == 3
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), src['path'], 'exec'), ns)
    # Exact source reset-action statements, selected by AST positions: exclude
    # preceding simulation/grasp and subsequent gain setup, keep their bodies.
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'randomize_initial_state')
    start = next(i for i, n in enumerate(reset.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'self.actions' and 'zeros_like' in ast.unparse(n.value))
    end = next(i for i, n in enumerate(reset.body[start:], start) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == 'self.actions[:, 5]')
    selected = ast.FunctionDef(name='reset_action_cache_oracle', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')],
                             kwonlyargs=[], kw_defaults=[], defaults=[]), body=reset.body[start:end+1], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module([selected], type_ignores=[])), src['path'], 'exec'), ns)
    intermediate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_intermediate_values')
    start = next(i for i, n in enumerate(intermediate.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'self.ee_linvel_fd')
    end = next(i for i, n in enumerate(intermediate.body[start:], start) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == 'self.prev_fingertip_quat')
    selected = ast.FunctionDef(name='physics_fd_oracle', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self'), ast.arg(arg='dt')],
                             kwonlyargs=[], kw_defaults=[], defaults=[]), body=intermediate.body[start:end+1], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module([selected], type_ignores=[])), src['path'], 'exec'), ns)
    return ns


def tensor(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32).unsqueeze(0)


def action_proxy(p, q, cache, frame):
    return SimpleNamespace(fingertip_midpoint_pos=tensor(p), fingertip_midpoint_quat=tensor(q),
        actions=tensor(cache), prev_actions=tensor(cache), pos_threshold=tensor([.02]*3),
        rot_threshold=tensor([.097]*3), cfg_task=SimpleNamespace(unidirectional_rot=False),
        cfg=SimpleNamespace(ctrl=SimpleNamespace(pos_action_bounds=[.05]*3, ema_factor=.2),
                            obs_order=['fingertip_pos_rel_fixed', 'fingertip_quat', 'ee_linvel', 'ee_angvel'], state_order=[]),
        fixed_pos_action_frame=tensor(frame), fixed_pos_obs_frame=tensor(frame), init_fixed_pos_obs_noise=tensor([0.]*3),
        identity_quat=tensor([1., 0., 0., 0.]), num_envs=1, device='cpu',
        last_update_timestamp=0., _robot=SimpleNamespace(_data=SimpleNamespace(_sim_timestamp=0.)),
        generate_ctrl_signals=lambda: None)


def oracle_measurements():
    """Return quantitative parity evidence from fixed synthetic (no actor) inputs."""
    ns = source_oracle()
    rng = np.random.default_rng(20261026)
    target_errors, torque_errors, wrench_errors, pose_errors, null_residuals = [], [], [], [], []
    for i in range(24):
        p = rng.uniform(-.04, .04, 3)
        q = c.quat_from_euler(2.5+rng.uniform(-.3, .3), rng.uniform(-.3, .3), rng.uniform(-math.pi, math.pi))
        if i % 2:
            q = -q
        cache = rng.uniform(-1.4, 1.4, 6)
        frame = np.array([.005, -.007, .003])
        proxy = action_proxy(p, q, cache, frame)
        ns['_apply_action'](proxy)
        got = c.cartesian_targets(p, q, cache, frame)
        target_errors.append(float(max(np.max(np.abs(got['position_target']-proxy.ctrl_target_fingertip_midpoint_pos.numpy()[0])),
                                       np.max(np.abs(got['quaternion_target']-proxy.ctrl_target_fingertip_midpoint_quat.numpy()[0])))))
        j = np.c_[np.eye(6), rng.uniform(-.2, .2, 6)] + rng.uniform(-.02, .02, (6, 7))
        a = rng.uniform(-.2, .2, (7, 7)); m = a.T@a+np.diag(np.linspace(.5, 2., 7))
        arm = c.DEFAULT_Q+rng.uniform(-.3, .3, 7)
        velocity = rng.uniform(-.1, .1, 7)
        lin, ang = rng.uniform(-.02, .02, 3), rng.uniform(-.05, .05, 3)
        # Source controller gets source-generated targets, so compare the
        # controller separately from float32 target-construction roundoff.
        tp = proxy.ctrl_target_fingertip_midpoint_pos.numpy()[0].astype(float)
        tq = proxy.ctrl_target_fingertip_midpoint_quat.numpy()[0].astype(float)
        result = c.torque_command(arm, velocity, p, q, lin, ang, j, m, tp, tq)
        cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1), ctrl=SimpleNamespace(default_dof_pos_tensor=c.DEFAULT_Q.tolist(), kp_null=10., kd_null=6.3246))
        official, wrench = ns['compute_dof_torque'](cfg, tensor(np.r_[arm, 0., 0.]), tensor(np.r_[velocity, 0., 0.]),
            tensor(p), tensor(q), tensor(lin), tensor(ang), tensor(j), tensor(m), tensor(tp), tensor(tq), tensor(c.KP), tensor(c.KD), 'cpu')
        torque_errors.append(float(np.max(np.abs(result['software_limited']-official.numpy()[0, :7]))))
        wrench_errors.append(float(np.max(np.abs(result['wrench']-wrench.numpy()[0]))))
        pe, ae = ns['get_pose_error'](tensor(p), tensor(q), tensor(tp), tensor(tq), 'geometric', 'axis_angle')
        pose_errors.append(float(np.max(np.abs(result['pose_error']-np.r_[pe.numpy()[0], ae.numpy()[0]]))))
        null_residuals.append(float(np.linalg.norm(j@np.linalg.solve(m, result['null_torque']))))
    return {'fixtures': 24, 'seed': 20261026,
            'target_max_abs_error': max(target_errors), 'software_torque_max_abs_error_Nm': max(torque_errors),
            'wrench_max_abs_error': max(wrench_errors), 'pose_error_max_abs_error': max(pose_errors),
            'dynamic_nullspace_max_residual': max(null_residuals), 'oracle_precision': 'unmodified source Torch float32 CPU',
            'port_precision': 'NumPy float64; final raw observations/RMS float32',
            'source_sha256': {**PINS, **{v['path']: v['sha256'] for k, v in FACTORY.items() if k in ('factory_control.py', 'factory_env.py')}}}


def test_source_target_and_torque_oracle_numeric_bounds():
    result = oracle_measurements()
    assert result['target_max_abs_error'] < 1e-6
    assert result['software_torque_max_abs_error_Nm'] < 3e-5
    assert result['wrench_max_abs_error'] < 3e-5
    assert result['pose_error_max_abs_error'] < 1e-6
    assert result['dynamic_nullspace_max_residual'] < 1e-12


def test_reset_cache_against_actual_source_including_unclipped_translation_and_yaw():
    ns = source_oracle()
    for yaw in (-2.9, -1.2, 0., 1.5, 2.9):
        p = np.array([.11, -.063, .082]); q = c.quat_from_euler(3.14159, .07, yaw)
        proxy = action_proxy(p, q, np.ones(6), np.zeros(3))
        proxy.fixed_pos_obs_frame = tensor([0., 0., .025])
        proxy.init_fixed_pos_obs_noise = tensor([.001, -.002, .003])
        ns['reset_action_cache_oracle'](proxy)
        contract = c.PhysicsContract()
        obs = contract.reset(p, q, np.zeros(3), [1., 0., 0., 0.], [.001, -.002, .003])
        np.testing.assert_allclose(contract.action_cache, proxy.actions.numpy()[0], atol=4e-7, rtol=1e-6)
        assert contract.action_cache[0] > 1
        np.testing.assert_array_equal(obs[7:13], np.zeros(6))
        before = contract.action_cache.copy()
        proxy.reset_buf = torch.zeros(1, dtype=torch.bool)
        ns['_pre_physics_step'](proxy, tensor(np.clip([2., -2., .3, .1, .2, .3], -1, 1)))
        after = contract.policy_update([2., -2., .3, .1, .2, .3])
        np.testing.assert_allclose(after, .8*before+.2*np.array([1, -1, .3, .1, .2, .3]))
        np.testing.assert_allclose(after, proxy.actions.numpy()[0], atol=4e-7, rtol=1e-6)


def test_every_raw_observation_field_against_actual_source_and_raw_quaternion_sign():
    ns = source_oracle()
    con = c.PhysicsContract()
    q = -c.quat_from_euler(3.14159, -.1, -.7)
    con.reset([.3, .2, .4], q, [.27, .21, .35], [1., 0., 0., 0.], [.001, .002, -.001])
    con.policy_update(np.arange(6)/7)
    con.update_physics([.3001, .1999, .4002], q, c.PHYSICS_DT)
    proxy = action_proxy(con.position, con.quaternion, con.action_cache, con.fixed_tip)
    proxy.init_fixed_pos_obs_noise = tensor(con.observation_noise)
    proxy.ee_linvel_fd, proxy.ee_angvel_fd = tensor(con.linear_fd), tensor(con.angular_fd)
    for attr, value in [('fingertip_midpoint_linvel', [0.]*3), ('fingertip_midpoint_angvel', [0.]*3),
                        ('joint_pos', [0.]*9), ('held_pos', [0.]*3), ('held_quat', [1., 0., 0., 0.]),
                        ('fixed_pos', [0.]*3), ('fixed_quat', [1., 0., 0., 0.]), ('task_prop_gains', c.KP)]:
        setattr(proxy, attr, tensor(value))
    expected = ns['_get_observations'](proxy)['policy'].numpy()[0]
    np.testing.assert_array_equal(con.observation(), expected)
    assert con.observation()[4] < 0  # No positive-w or other sign canonicalization.


def test_fd_world_difference_source_sign_zero_and_physics_timestamp_cache():
    con = c.PhysicsContract(); con.reset(np.zeros(3), [1., 0., 0., 0.], np.zeros(3), [1., 0., 0., 0.])
    v, w = con.update_physics([1/120, 0., 0.], c.quat_from_euler(0., 0., .01), c.PHYSICS_DT)
    np.testing.assert_allclose(v, [1., 0., 0.]); np.testing.assert_allclose(w, [0., 0., 1.2], atol=1e-12)
    same = con.update_physics(con.position, con.quaternion, c.PHYSICS_DT)
    np.testing.assert_array_equal(same, (v, w))
    with pytest.raises(ValueError): con.update_physics(con.position, con.quaternion, 3*c.PHYSICS_DT)
    with pytest.raises(ValueError): con.update_physics(np.ones(3), con.quaternion, c.PHYSICS_DT)
    con.reset(np.zeros(3), [1., 0., 0., 0.], np.zeros(3), [1., 0., 0., 0.])
    _, w = con.update_physics(np.zeros(3), [0., 1., 0., 0.], c.PHYSICS_DT)
    np.testing.assert_array_equal(w, np.zeros(3))


def test_fd_against_actual_source_statements_including_raw_sign_flip():
    ns = source_oracle(); rng = np.random.default_rng(312)
    for sign in (1., -1.):
        old_p = np.array([.3, -.2, .5]); new_p = old_p + rng.uniform(-.001, .001, 3)
        old_q = c.quat_from_euler(3.14159, .1, .3)
        new_q = sign*c.quat_mul(c.quat_from_euler(.002, -.001, .003), old_q)
        con = c.PhysicsContract(); con.reset(old_p, old_q, np.zeros(3), [1., 0., 0., 0.])
        v, w = con.update_physics(new_p, new_q, c.PHYSICS_DT)
        proxy = SimpleNamespace(fingertip_midpoint_pos=tensor(new_p), prev_fingertip_pos=tensor(old_p),
                                fingertip_midpoint_quat=tensor(new_q), prev_fingertip_quat=tensor(old_q))
        ns['physics_fd_oracle'](proxy, c.PHYSICS_DT)
        # Differencing source float32 world positions amplifies conversion error
        # by 120; document this tolerance instead of claiming bitwise parity.
        np.testing.assert_allclose(v, proxy.ee_linvel_fd.numpy()[0], atol=5e-6, rtol=1e-5)
        np.testing.assert_allclose(w, proxy.ee_angvel_fd.numpy()[0], atol=5e-5, rtol=1e-4)


def test_target_uses_current_pose_each_substep_and_keeps_world_left_rotation():
    p = np.zeros(3); q = c.quat_from_euler(2.7, .2, -.7); a = np.array([.5, 0., 0., .4, -.2, .3])
    first = c.cartesian_targets(p, q, a, np.zeros(3))
    second = c.cartesian_targets([.001, 0., 0.], q, a, np.zeros(3))
    assert first['position_target'][0] == .01 and second['position_target'][0] == .011
    assert c.source_euler(first['quaternion_target'])[0] == pytest.approx(3.14159)
    assert math.sin(c.source_euler(first['quaternion_target'])[1]) == pytest.approx(0., abs=1e-14)
    r = .097*a[3:]; theta = np.linalg.norm(r); dq = np.r_[np.cos(theta/2), r/theta*np.sin(theta/2)]
    np.testing.assert_allclose(first['quaternion_pre_constraint'], c.quat_mul(dq, q), atol=1e-15)
    assert not np.allclose(first['quaternion_pre_constraint'], c.quat_mul(q, dq))


def simple_torque(**updates):
    args = dict(q7=c.DEFAULT_Q.copy(), qdot7=np.zeros(7), position=np.zeros(3), quaternion=[1., 0., 0., 0.],
                linear_fd=np.zeros(3), angular_fd=np.zeros(3), jacobian=np.c_[np.eye(6), np.zeros(6)],
                mass_matrix=2*np.eye(7), target_position=np.zeros(3), target_quaternion=[1., 0., 0., 0.])
    return c.torque_command(**{**args, **updates})


def test_pd_sign_nullspace_wrap_and_software_clipping_have_known_independent_numbers():
    q = c.DEFAULT_Q.copy(); q[6] -= 1
    out = simple_torque(q7=q, target_position=[.01, 0., 0.])
    np.testing.assert_allclose(out['requested'], [1., 0., 0., 0., 0., 0., 20.], atol=1e-14)
    out = simple_torque(linear_fd=[1., 0., 0.], target_position=[.01, 0., 0.])
    assert out['requested'][0] == pytest.approx(-19.)
    out = simple_torque(target_position=[2., -2., 0.])
    np.testing.assert_array_equal(out['requested'][:2], [200., -200.])
    np.testing.assert_array_equal(out['software_limited'][:2], [100., -100.])
    wrapped = c.DEFAULT_Q.copy(); wrapped[6] += 2*math.pi-.2
    assert simple_torque(q7=wrapped)['null_torque'][6] == pytest.approx(4.)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_inputs_rejected(bad):
    with pytest.raises(ValueError): c.cartesian_targets([bad, 0., 0.], [1., 0., 0., 0.], np.zeros(6), np.zeros(3))
    with pytest.raises(ValueError): simple_torque(qdot7=[bad]*7)


@pytest.mark.parametrize('mass,jac', [(np.zeros((7, 7)), np.c_[np.eye(6), np.zeros(6)]),
                                    (-np.eye(7), np.c_[np.eye(6), np.zeros(6)]),
                                    (np.eye(7), np.zeros((6, 7))),
                                    (np.eye(7), np.ones((6, 7)))])
def test_singular_nonphysical_states_are_explicitly_rejected_without_regularization(mass, jac):
    with pytest.raises(ValueError, match='inversion rejected'):
        simple_torque(mass_matrix=mass, jacobian=jac)


def test_frozen_rms_interface_matches_archived_primary_statistics_without_checkpoint_load():
    rms = IDENTITY['checkpoints']['g2_epoch200_primary']['rms']
    mean, var = np.array(rms['running_mean']), np.array(rms['running_var'])
    obs = np.linspace(-2, 2, 19, dtype=np.float32)
    untouched = (mean.copy(), var.copy())
    got = c.normalize_observation(obs, mean, var)
    expected = np.clip((obs-mean.astype(np.float32))/np.sqrt(var.astype(np.float32)+np.float32(1e-5)), -5., 5.)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(mean, untouched[0]); np.testing.assert_array_equal(var, untouched[1])
    assert got.dtype == np.float32 and got.shape == (19,)
    with pytest.raises(ValueError): c.normalize_observation(obs, mean, -np.ones(19))


def test_invalid_quaternion_and_uninitialized_contract_rejected():
    with pytest.raises(ValueError): c.cartesian_targets(np.zeros(3), np.zeros(4), np.zeros(6), np.zeros(3))
    with pytest.raises(ValueError): c.PhysicsContract().policy_update(np.zeros(6))


class IndependentClockFixture:
    """Explicit fake kinematics for scheduling assertions, not a physics oracle."""
    arm_qpos_ids = np.arange(7)
    arm_dof_ids = np.arange(7)
    hole_body_id, peg_body_id = 0, 1

    def reset(self, seed=0, **kwargs):
        self.rng = np.random.default_rng(seed)
        self.data = SimpleNamespace(time=0., qpos=c.DEFAULT_Q.copy(), qvel=np.zeros(7),
                                    xpos=np.array([[0., 0., -.025], [0., 0., -.03]]),
                                    xquat=np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]))
        self.calls = []
        return self.snapshot()

    def tcp_state(self):
        return np.array([len(self.calls)*.0001, 0., 0.]), c.quat_from_euler(3.14159, 0., 0.)

    def controller_jacobian(self):
        return np.c_[np.eye(6), np.zeros(6)]

    def arm_mass_matrix(self):
        return np.eye(7)

    def physics_step(self, torque, finger_target=0.):
        self.calls.append({'torque': torque.copy(), 'finger_target': finger_target})
        self.data.time += c.PHYSICS_DT
        return self.snapshot()

    def snapshot(self):
        return {'time_s': self.data.time, 'peg_z': float(self.data.xpos[self.peg_body_id, 2])}

    def insertion_diagnostics(self):
        # Source geometric success can be true below the hole floor. Keep this
        # independent physical outcome separate, instead of rewriting success.
        return {'physical_insertion_candidate': False, 'known_below_floor': True}


def test_reference_bridge_ema_once_eight_rebuilt_targets_and_final_fd_before_observation():
    env = IndependentClockFixture(); runner = c.ReferenceRunner(env)
    reset = runner.reset(observation_noise=np.zeros(3))
    assert reset['lstm_reset_pending'] is True
    out = runner.step_reference_action([1., 0., 0., 0., 0., 0.])
    assert len(env.calls) == 8 and runner.contract.policy_updates == 1
    assert out['episode_steps'] == 1 and out['episode_simulation_time_s'] == pytest.approx(1/15)
    assert out['ema_cache'][0] == pytest.approx(.2)
    np.testing.assert_allclose([x['target_position_m'][0] for x in out['substeps']], .004+np.arange(8)*.0001, atol=1e-15)
    assert out['observation'][7] == pytest.approx(.012)
    assert out['official_geometric_success'] is True
    assert out['physical_insertion_diagnostic']['physical_insertion_candidate'] is False
    assert out['state_is_pre_reset'] is True and out['reset_required'] is False
    assert all(x['finger_target'] == 0. for x in env.calls)


def test_reference_reset_seeded_noise_and_pre_reset_timeout_accounting():
    env = IndependentClockFixture(); runner = c.ReferenceRunner(env)
    a = runner.reset(seed=731)['episode_noise_m']
    b = runner.reset(seed=731)['episode_noise_m']
    d = runner.reset(seed=732)['episode_noise_m']
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, d)
    assert runner.contract.policy_updates == 0
    runner.reset(observation_noise=np.zeros(3))
    runner.episode_steps = 148
    env.data.time = 148*8*c.PHYSICS_DT
    runner.contract.timestamp = env.data.time
    out = runner.step_reference_action(np.zeros(6))
    assert out['episode_steps'] == 149
    assert out['episode_simulation_time_s'] == pytest.approx(149*8/120)
    assert out['terminated'] is True and out['truncated'] is True and out['reset_required'] is True
    assert out['official_geometric_success'] is True and env.data.xpos[env.peg_body_id, 2] == -.03
    with pytest.raises(ValueError): runner.step_reference_action(np.zeros(6))
    runner.reset()
    assert runner.ever_success is False and runner.first_success_step is None


def test_reference_controller_fault_rejects_before_physics_and_requires_explicit_reset():
    env = IndependentClockFixture(); runner = c.ReferenceRunner(env)
    runner.reset(observation_noise=np.zeros(3))
    env.controller_jacobian = lambda: np.zeros((6, 7))
    with pytest.raises(ValueError, match='inversion rejected'):
        runner.step_reference_action(np.ones(6))
    assert len(env.calls) == 0 and env.data.time == 0
    assert runner.reset_required is True and runner.last_fault['type'] == 'ValueError'
    with pytest.raises(ValueError, match='reset is required'):
        runner.step_reference_action(np.zeros(6))


@pytest.mark.parametrize('position,expected', [([0., 0., .000999], True), ([0., 0., .001], False),
                                           ([.0025, 0., 0.], False), ([.002499, 0., 0.], True),
                                           ([0., 0., -1.], True), ([.0025, .0025, -1.], False)])
def test_original_geometric_success_against_actual_source_and_independent_boundary_values(position, expected):
    src = FACTORY['factory_env.py']; tree = checked_tree(src['path'], src['sha256'])
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FactoryEnv')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_curr_successes')
    ns = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module([node], type_ignores=[])), src['path'], 'exec'), ns)
    proxy = SimpleNamespace(num_envs=1, device='cpu', held_base_pos=tensor(position), target_held_base_pos=tensor(np.zeros(3)),
                            cfg_task=SimpleNamespace(name='peg_insert', fixed_asset_cfg=SimpleNamespace(height=.025)))
    assert bool(ns['_get_curr_successes'](proxy, .04, False)[0]) is expected
    assert c.original_success(position, np.zeros(3)) is expected
