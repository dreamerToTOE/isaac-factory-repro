"""Standalone frozen PegInsert inference; optional official Player parity oracle.

Run through run_sim.sh. This entry never constructs a training Runner or optimizer.
All robot commands use the original Factory environment/controller unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import MethodType, SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
_ACTIVE_WRAPPER = None
sys.path.insert(0, str(REPO))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def close_with_exit_status(app, code):
    """Notify Kit before fast shutdown, preserving a pending runtime failure."""
    try:
        if app is not None:
            app.app.post_quit(code)
            app.close()
    except SystemExit as exc:
        if code == 0 and exc.code not in (None, 0): code = 1
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        raise SystemExit(code)


def run(args, app):
    global _ACTIVE_WRAPPER
    import numpy as np
    import torch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import isaacsim.core.utils.torch as tu
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
    from isaaclab_tasks.direct.factory import factory_control as fc
    from scripts.g4i1_policy import FrozenPolicy, CommandBoundary
    from scripts.g4i0_interface import validate_policy_output
    from scripts.g4i1_parity import compare_stages, REQUIRED_CASES
    from scripts.success_metrics import PreResetSuccessCapture, SuccessTracker
    from scripts.evaluation_runtime import reset_player_rnn_states

    def clone(x):
        return x.detach().cpu().numpy().copy() if torch.is_tensor(x) else np.asarray(x).copy()
    def plain(x):
        return clone(x).tolist()
    def rng():
        return {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(args.device).clone()}
    def same_rng(a, b):
        return all(torch.equal(a[k], b[k]) for k in a)
    def stamp(env):
        return {"robot_scene_timestamp_s": float(env._robot._data._sim_timestamp),
                "simulation_context_time_s": float(env.sim.current_time),
                "policy_physics_step_counter": int(env._sim_step_counter),
                "wall_monotonic_s": time.monotonic()}

    i2 = getattr(args, 'i2', False)
    count = args.num_envs if i2 else 1
    seed = args.seed if i2 else 1001
    policy = FrozenPolicy.from_checkpoint(device=args.device)
    state = policy.initial_state(count)
    oracle = None
    if args.validate:
        # Same Player class and get_action method invoked by evaluate_checkpoint.py,
        # with env_info supplied directly; no Runner or second simulator is needed.
        import gym as old_gym
        from rl_games.algos_torch.players import PpoPlayerContinuous
        audit = json.loads((REPO / 'evidence/g4_i/i0/checkpoint_audit.json').read_text())
        params = copy.deepcopy(audit['checkpoints']['g2_epoch200_primary']['agent_configuration']['params'])
        params['config'].update(device=args.device, device_name=args.device, num_actors=count,
            env_info={'observation_space': old_gym.spaces.Box(-np.inf, np.inf, (19,), np.float32),
                      'action_space': old_gym.spaces.Box(-1., 1., (6,), np.float32), 'agents': 1, 'value_size': 1})
        params['config']['player'] = {'deterministic': True}
        oracle = PpoPlayerContinuous(params)
        oracle.model.load_state_dict(policy.model.state_dict(), strict=True)
        oracle.model.eval()
        oracle.get_batch_size(torch.zeros(count, 19, device=args.device), 1)
        oracle.init_rnn()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = parse_env_cfg('Isaac-Factory-PegInsert-Direct-v0', device=args.device, num_envs=count)
    cfg.seed = seed
    from isaaclab.utils.io import dump_yaml
    dump_yaml(str(args.output/'env.yaml'), cfg)
    raw = gym.make('Isaac-Factory-PegInsert-Direct-v0', cfg=cfg)
    env = raw.unwrapped
    if abs(env.physics_dt-1/120)>1e-12 or env.cfg.decimation!=8 or env.num_envs!=count:
        raise RuntimeError('frozen physics/control/environment-count mismatch')
    wrapper = RlGamesVecEnvWrapper(raw, args.device, float('inf'), 1.0)
    _ACTIVE_WRAPPER = wrapper
    env_before_reset = stamp(env)
    # Match the evaluator's reset seed after model/player construction.
    torch.manual_seed(seed)
    obs = wrapper.reset()
    initial_stamp = stamp(env)
    current = obs['obs']
    telemetry = None
    if i2:
        from scripts.g4i2_runtime import Telemetry
        telemetry = Telemetry(args, env, policy, state, current)
    pending_reference_cache = None
    tap_state = {'in_apply': False, 'step': 0, 'episode': 0, 'physics_calls': 0, 'aux_controller_calls': 0}
    pairs_ref, pairs_actual = {}, {}
    def add(stage, reference, actual):
        if i2 and not args.validate: return
        pairs_ref.setdefault(stage, []).append(clone(reference))
        pairs_actual.setdefault(stage, []).append(clone(actual))
    def targets(cache, position, quat):
        target_pos = env.fixed_pos_action_frame + torch.clip(position + cache[:, :3] * env.pos_threshold - env.fixed_pos_action_frame,
                       -env.cfg.ctrl.pos_action_bounds[0], env.cfg.ctrl.pos_action_bounds[1])
        rot = cache[:, 3:] * env.rot_threshold
        angle = torch.linalg.vector_norm(rot, dim=-1)
        axis = rot / angle[:, None]
        delta = tu.quat_from_angle_axis(angle, axis)
        delta = torch.where((angle > 1e-6)[:, None], delta, torch.tensor([1., 0., 0., 0.], device=args.device)[None])
        q = tu.quat_mul(delta, quat)
        xyz = torch.stack(tu.get_euler_xyz(q), dim=1)
        xyz[:, 0] = 3.14159; xyz[:, 1] = 0.
        return target_pos, tu.quat_from_euler_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    controller_fields = {
        'dof_pos': 'joint_pos', 'dof_vel': 'joint_vel',
        'fingertip_midpoint_pos': 'fingertip_midpoint_pos', 'fingertip_midpoint_quat': 'fingertip_midpoint_quat',
        'fingertip_midpoint_linvel': 'ee_linvel_fd', 'fingertip_midpoint_angvel': 'ee_angvel_fd',
        'jacobian': 'fingertip_midpoint_jacobian', 'arm_mass_matrix': 'arm_mass_matrix',
        'task_prop_gains': 'task_prop_gains', 'task_deriv_gains': 'task_deriv_gains'}
    trace = (args.output / 'trace.jsonl').open('x')
    def record(row):
        trace.write(json.dumps(row, allow_nan=False) + '\n'); trace.flush()
    original_generate, original_apply, original_pre = env.generate_ctrl_signals, env._apply_action, env._pre_physics_step
    def pre(instance, actions):
        if telemetry:
            with telemetry.probe.section('ema', cuda=True): original_pre(actions)
        else: original_pre(actions)
        tap_state["actual_ema"] = instance.actions.clone()
        add("ema", pending_reference_cache, instance.actions)
    def generate(instance):
        if not tap_state['in_apply']:
            tap_state['aux_controller_calls'] += 1
            return original_generate()
        if telemetry: telemetry.target_end(instance)
        if i2 and not args.validate:
            original_generate()
            telemetry.controller(instance)
            tap_state['physics_calls'] += 1
            return
        # Preserve the exact PhysX view strides for the duplicated controller call.
        # Packing a sliced9x9->7x7 mass matrix can select different CUDA inverse kernels.
        inputs = {key: getattr(instance, attr) for key, attr in controller_fields.items()}
        ref_pos, ref_quat = targets(pending_reference_cache, inputs['fingertip_midpoint_pos'], inputs['fingertip_midpoint_quat'])
        actual_pos, actual_quat = instance.ctrl_target_fingertip_midpoint_pos.clone(), instance.ctrl_target_fingertip_midpoint_quat.clone()
        inputs['ctrl_target_fingertip_midpoint_pos'] = actual_pos
        inputs['ctrl_target_fingertip_midpoint_quat'] = actual_quat
        reference_inputs = {**inputs, 'ctrl_target_fingertip_midpoint_pos': ref_pos, 'ctrl_target_fingertip_midpoint_quat': ref_quat}
        if any(not torch.isfinite(value).all() for value in inputs.values()):
            raise RuntimeError("nonfinite controller input; no new command submitted")
        reference_torque, reference_wrench = fc.compute_dof_torque(cfg=instance.cfg, device=instance.device, **reference_inputs)
        if not torch.isfinite(reference_torque).all() or not torch.isfinite(reference_wrench).all():
            raise RuntimeError("nonfinite controller result; no new command submitted")
        original_generate()
        if telemetry: telemetry.controller(instance)
        add('target_position', ref_pos, actual_pos); add('target_quaternion', ref_quat, actual_quat)
        for key in inputs: add('controller__' + key, reference_inputs[key], inputs[key])
        add('controller__joint_effort', reference_torque, instance.joint_torque)
        add('controller__applied_wrench', reference_wrench, instance.applied_wrench)
        add('controller__gripper_target', torch.zeros(1,device=instance.device), torch.as_tensor([instance.ctrl_target_gripper_dof_pos],device=instance.device))
        add('controller__finger_position_target', torch.zeros((count,2),device=instance.device), instance.ctrl_target_joint_pos[:,7:9])
        add('controller__joint_position_target', instance.ctrl_target_joint_pos, instance.ctrl_target_joint_pos)
        tap_state['physics_calls'] += 1
        if i2: return
        record({'phase': 'controller', 'step': tap_state['step'], 'episode': tap_state['episode'], **stamp(instance),
                'target_position': plain(actual_pos), 'target_quaternion': plain(actual_quat),
                'inputs': {key: plain(value) for key, value in inputs.items()},
                'input_strides':{key:list(value.stride()) for key,value in inputs.items()},
                'joint_position_command':plain(instance.ctrl_target_joint_pos), 'gripper_target':float(instance.ctrl_target_gripper_dof_pos),
                'joint_effort': plain(instance.joint_torque)})
    def apply(instance):
        tap_state['in_apply'] = True
        if telemetry: telemetry.target_begin()
        try: return original_apply()
        finally: tap_state['in_apply'] = False
    env._pre_physics_step = MethodType(pre, env)
    env.generate_ctrl_signals = MethodType(generate, env)
    env._apply_action = MethodType(apply, env)
    boundary = CommandBoundary(policy, telemetry.submit(wrapper.step) if telemetry else wrapper.step, batch_size=count)
    def verify_reset_cache(observation, mask=None):
        expected = torch.zeros_like(env.actions)
        bounds = torch.tensor(env.cfg.ctrl.pos_action_bounds, device=env.device)
        expected[:, :3] = (env.fingertip_midpoint_pos-env.fixed_pos_action_frame) @ torch.diag(1.0/bounds)
        unrot = torch.tensor([[-np.pi,0.,0.]], device=env.device)
        q = tu.quat_from_euler_xyz(unrot[:,0],unrot[:,1],unrot[:,2]).repeat(count, 1)
        yaw = tu.get_euler_xyz(tu.quat_mul(q,env.fingertip_midpoint_quat))[-1]
        yaw = torch.where(yaw>torch.pi/2,yaw-2*torch.pi,yaw)
        yaw = torch.where(yaw < -torch.pi,yaw+2*torch.pi,yaw)
        expected[:,5] = (yaw+np.deg2rad(180.))/np.deg2rad(270.)*2.-1.
        torch.testing.assert_close(env.actions if mask is None else env.actions[mask], expected if mask is None else expected[mask],atol=1e-5,rtol=1e-5)
        boundary.initialize_cache(observation,env.actions)
        return {'expected_cache':plain(expected),'actual_cache':plain(env.actions),
                'max_absolute_error':float((expected-env.actions).abs().max())}
    initial_cache_check = verify_reset_cache(current)
    initial = {'phase': 'initial_reset', **initial_stamp, 'observation': plain(current), 'cache': plain(env.actions),
               'h': plain(state[0]), 'c': plain(state[1]), 'before_reset': env_before_reset,
               'joint_pos':plain(env.joint_pos), 'joint_vel':plain(env.joint_vel),
               'fingertip_pos':plain(env.fingertip_midpoint_pos), 'fingertip_quat':plain(env.fingertip_midpoint_quat),
               'held_pos':plain(env.held_pos), 'held_quat':plain(env.held_quat),
               'fixed_pos':plain(env.fixed_pos), 'fixed_quat':plain(env.fixed_quat),
               'fixed_observation_frame':plain(env.fixed_pos_obs_frame), 'fixed_action_frame':plain(env.fixed_pos_action_frame),
               'episode_observation_noise':plain(env.init_fixed_pos_obs_noise), 'pose_cache_check':initial_cache_check}
    record(initial)
    # Edge coverage uses original _apply_action on a detached tensor-only proxy:
    # no physics step, task state modification or simulator command submission.
    fixtures = []
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
    for case, mu, position in ([] if i2 and not args.validate else [
        ('normal', [.2, -.1, .3, .1, .2, -.1], [0., 0., 0.]),
        ('action_clipping', [2., -2., 3., -3., 4., -4.], [0., 0., 0.]),
        ('near_zero_rotation', [.2, .1, -.1, 0., 0., 0.], [0., 0., 0.]),
        ('position_saturation', [1., 1., 1., .3, .1, .1], [.2, -.2, .2])]):
        action = torch.tensor([mu], device=args.device); cache = .2 * action.clamp(-1, 1)
        pos = env.fixed_pos_action_frame + torch.tensor([position], device=args.device)
        proxy = SimpleNamespace(actions=cache.clone(), pos_threshold=env.pos_threshold, rot_threshold=env.rot_threshold,
            fingertip_midpoint_pos=pos, fingertip_midpoint_quat=env.fingertip_midpoint_quat.clone(),
            fixed_pos_action_frame=env.fixed_pos_action_frame, cfg=env.cfg, cfg_task=env.cfg_task,
            device=env.device, num_envs=1, last_update_timestamp=0.,
            _robot=SimpleNamespace(_data=SimpleNamespace(_sim_timestamp=0.)), generate_ctrl_signals=lambda: None)
        FactoryEnv._apply_action(proxy)
        expected_pos, expected_quat = targets(cache, pos, proxy.fingertip_midpoint_quat)
        result = compare_stages({'target_position': clone(expected_pos), 'target_quaternion': clone(expected_quat)},
                 {'target_position': clone(proxy.ctrl_target_fingertip_midpoint_pos), 'target_quaternion': clone(proxy.ctrl_target_fingertip_midpoint_quat)},
                 device_kind='gpu', require_all_stages=False, cases=[case])
        fixtures.append({'case': case, 'result': result, 'raw_mu': plain(action), 'clipped': plain(action.clamp(-1, 1)),
                         'cache': plain(cache), 'target_position': plain(proxy.ctrl_target_fingertip_midpoint_pos),
                         'target_quaternion': plain(proxy.ctrl_target_fingertip_midpoint_quat), 'simulator_commands_submitted': 0})
    write_json(args.output / 'edge_fixtures.json', fixtures)
    if any(not x['result']['passed'] for x in fixtures):
        raise RuntimeError('original Factory edge-fixture parity failed; episode not started')
    tracker = SuccessTracker(count)
    resets = []
    rng_checks = []
    started = time.monotonic()
    steps = args.max_steps if i2 else (2 if args.mode == 'preflight' else 150)
    try:
        with PreResetSuccessCapture(env, convert=plain, snapshot=lambda e,r: {
                'held_pos': plain(e.held_pos), 'held_quat': plain(e.held_quat),
                'fingertip_pos': plain(e.fingertip_midpoint_pos), 'fingertip_quat':plain(e.fingertip_midpoint_quat),
                'held_base_pos':plain(e.held_base_pos), 'target_held_base_pos':plain(e.target_held_base_pos),
                'joint_pos':plain(e.joint_pos), 'joint_vel':plain(e.joint_vel), 'cache': plain(e.actions), **stamp(e)}) as capture:
            for index in range(steps):
                tap_state['step'] = index + 1
                if telemetry: telemetry.begin(index + 1)
                before_cache = env.actions.clone()
                if not torch.equal(current[:, 13:19], before_cache):
                    raise ValueError('actor observation EMA cache does not match original environment cache')
                reference = {}
                before_rng = rng()
                if oracle is not None:
                    hook = oracle.model.register_forward_hook(lambda model, inp, out: reference.update(out))
                    with torch.random.fork_rng(devices=[torch.device(args.device).index or 0]), torch.inference_mode():
                        ref_norm = oracle.model.norm_obs(current.clone())
                        ref_action = oracle.get_action(current.clone(), is_deterministic=True)
                        expected_rng = rng()
                    hook.remove()
                    assert same_rng(before_rng, rng()), 'oracle isolation changed live RNG'
                with torch.inference_mode():
                    step = policy.infer(current, state, profiler=telemetry.probe if telemetry else None)
                state = step.state
                if oracle is not None:
                    rng_equal = same_rng(expected_rng, rng())
                    rng_checks.append(rng_equal)
                    if not rng_equal: raise RuntimeError('discarded sample RNG consumption mismatch')
                    for name, ref, actual in [('observation', current, step.observation), ('normalized', ref_norm, step.normalized),
                            ('mu', reference['mus'], step.mu), ('h', reference['rnn_states'][0], state[0]),
                            ('c', reference['rnn_states'][1], state[1]), ('clipped', ref_action, step.clipped_action)]: add(name, ref, actual)
                    ref_ema = .2 * ref_action + .8 * before_cache
                else:
                    ref_ema = .2 * step.clipped_action + .8 * before_cache
                pending_reference_cache = ref_ema
                if not i2: record({'phase': 'policy', 'step': index + 1, 'episode': tap_state['episode'], **stamp(env),
                        'observation': plain(current), 'normalized': plain(step.normalized), 'mu': plain(step.mu),
                        'h': plain(state[0]), 'c': plain(state[1]), 'clipped': plain(step.clipped_action),
                        'cache_before': plain(before_cache), 'ema': plain(.2*step.clipped_action+.8*before_cache)})
                validate_policy_output(step.clipped_action, batch_size=count)
                if step.clipped_action.device != torch.device(args.device):
                    raise ValueError("wrong action device; no command submitted")
                if telemetry: telemetry.action(step, before_cache)
                with torch.inference_mode():
                    obs, reward, done, extras = boundary.submit_precomputed(step,current,before_cache)
                cap = capture.consume()
                if not i2: record({'phase':'pre_reset_success','step':index+1,'episode':tap_state['episode'],'capture':cap})
                finalized = tracker.update(cap['success'], plain(done), plain(extras['time_outs']), cap['step_numbers'])
                if telemetry: telemetry.episodes(finalized, cap, done, obs['obs'], state)
                if bool(done.any()):
                    before_h, before_c = state[0].clone(), state[1].clone()
                    state = boundary.reset_state(state, done.bool())
                    if oracle is not None: reset_player_rnn_states(oracle, done.bool())
                    if state[0][:, done.bool(), :].count_nonzero() or state[1][:, done.bool(), :].count_nonzero(): raise RuntimeError('recurrent reset failed')
                    reset = {'phase': 'timeout_reset', 'step': index+1, 'pre_reset': cap,
                             'post_reset_observation': plain(obs['obs']), 'post_reset_cache': plain(env.actions),
                             'h_before': plain(before_h), 'c_before': plain(before_c), 'h_after': plain(state[0]), 'c_after': plain(state[1]), **stamp(env)}
                    reset['pose_cache_check'] = verify_reset_cache(obs['obs'], done.bool())
                    resets.append(reset)
                    if not i2: record(reset)
                    tap_state['episode'] += 1
                current = obs['obs']
                if telemetry: telemetry.end(current, state)
                if i2 and len(tracker.completed) >= count * args.episodes_per_env and (not args.validate or index+1>=150): break
                if (index+1)%25 == 0: print('G4I1_PROGRESS', index+1, len(tracker.completed), flush=True)
        if not i2 and args.mode == 'episode' and (len(tracker.completed) != 1 or tracker.completed[0].steps != 149 or len(resets) != 1):
            raise RuntimeError('expected exactly one complete149-step episode plus first next-episode action')
        if i2 and len(tracker.completed) != count * args.episodes_per_env:
            raise RuntimeError('I2 requested episode denominator not reached')
        reference_arrays = {key: np.stack(value) for key,value in pairs_ref.items()}
        actual_arrays = {key: np.stack(value) for key,value in pairs_actual.items()}
        cases = [x['case'] for x in fixtures] + (['timeout', 'post_reset_first_action'] if resets else [])
        parity = compare_stages(reference_arrays, actual_arrays, device_kind='gpu', cases=cases, require_cases=REQUIRED_CASES if args.mode=='episode' else REQUIRED_CASES[:4]) if oracle is not None else None
        np.savez_compressed(args.output / 'paired_trace.npz', **{'reference__'+k:v for k,v in reference_arrays.items()}, **{'actual__'+k:v for k,v in actual_arrays.items()})
        if parity is not None:
            write_json(args.output / 'parity.json', parity)
            if not parity['passed']: raise RuntimeError('intermediate parity tolerance violation')
        if not policy.rms_unchanged(): raise RuntimeError('frozen normalization buffers changed')
        final_stamp = stamp(env)
        completed = [e.to_dict() for e in tracker.completed]
        result = {'mode': args.mode, 'checkpoint_identity': policy.identity, 'seed':seed, 'torch_seed':seed,
            'num_envs':count, 'fp32':True, 'deterministic':True, 'training_runner_instantiated':False, 'optimizer_restored':False,
            'policy_steps': index+1, 'physics_substeps':tap_state['physics_calls'], 'physics_dt':env.physics_dt, 'step_dt':env.step_dt,
            'episode_loop_wall_s':time.monotonic()-started, 'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device), 'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device), 'training_runner_module_loaded':'rl_games.torch_runner' in sys.modules, 'episodes':completed, 'denominator':len(completed),
            'ever_success':sum(e['ever_success'] for e in completed), 'terminal_success':sum(e['terminal_success'] for e in completed),
            'first_success_seconds':[None if e['first_success_step'] is None else e['first_success_step']*env.step_dt for e in completed],
            'initial_timestamp':initial_stamp, 'final_timestamp':final_stamp,
            'auxiliary_physics_s_during_loop':final_stamp['robot_scene_timestamp_s']-initial_stamp['robot_scene_timestamp_s']-(index+1)*env.step_dt,
            'auxiliary_controller_calls':tap_state['aux_controller_calls'], 'reset_count':len(resets),
            'post_reset_first_action_recorded':bool(resets and index+1>149), 'rng_policy':'replicate discarded Normal sample; oracle isolated via fork_rng',
            'rng_comparisons':len(rng_checks), 'rng_all_equal':all(rng_checks), 'rms_unchanged':True,
            'parity':parity, 'edge_cases':[x['case'] for x in fixtures],
            'status':'COMPLETED', 'scope':'I1 only; one completed episode does not reproduce288episode population evaluation'}
        if telemetry:
            result['i2'] = telemetry.finish(tracker, result)
            result['scope'] = 'G4-I2 inference profiling and fixed evaluation; no Sim2Sim or training'
        write_json(args.output / 'result.json', result)
        print('G4I1_RESULT='+json.dumps({k:result[k] for k in ['status','mode','policy_steps','denominator','ever_success','terminal_success','rng_all_equal']}),flush=True)
    finally:
        trace.close()
        if telemetry: telemetry.close()
        env.generate_ctrl_signals, env._apply_action, env._pre_physics_step = original_generate, original_apply, original_pre
        wrapper.close()
        _ACTIVE_WRAPPER = None


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['preflight','episode','exit-probe'], default='episode')
    parser.add_argument('--validate', action='store_true', help='Compare to the unchanged official Player oracle; no training Runner.')
    parser.add_argument('--i2', action='store_true')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--episodes-per-env', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--profiling', choices=['off', 'events'], default='events')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--diagnostics', choices=['full','minimal'], default='full')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.headless or args.device != 'cuda:0': parser.error('I1 registered run is headless cuda:0, one env, seed1001')
    if args.i2 and args.validate and args.num_envs != 1: parser.error('I2 parity oracle uses the single-env acceptance fixture')
    if args.i2 and (args.num_envs not in (1,32) or args.seed not in (1001,1002,1003) or args.episodes_per_env < 1): parser.error('I2 preregistered allocation mismatch')
    args.output = args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'invocation.json', {'argv':sys.argv,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'mode':args.mode,'validate':args.validate,'seed':args.seed if args.i2 else 1001,'num_envs':args.num_envs if args.i2 else 1, 'i2':args.i2, 'profiling':args.profiling,'dtype':'float32','status':'INITIALIZING'})
    app = None; code = 0
    try:
        app = AppLauncher(args).app
        if args.mode == 'exit-probe':
            raise RuntimeError('Intentional I1 exit-status probe; no environment or robot command created')
        run(args, app)
    except BaseException as exc:
        write_json(args.output/'runtime_error.json', {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        traceback.print_exc(file=sys.stderr); sys.stderr.flush()
        code = 1
        # Setup can fail before the episode-loop finally; release the original
        # environment before Kit shutdown, retaining the original failure code.
        if _ACTIVE_WRAPPER is not None:
            try: _ACTIVE_WRAPPER.close()
            except BaseException: traceback.print_exc(file=sys.stderr)
    finally:
        write_json(args.output/'process_outcome.json', {'requested_exit_code':code,'runtime_result_exists':(args.output/'result.json').exists(), 'runtime_error_exists':(args.output/'runtime_error.json').exists()})
        close_with_exit_status(app,code)


if __name__ == '__main__': main()
