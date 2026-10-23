"""Evaluate an RL-Games Factory checkpoint with official pre-reset success masks.

This is an evaluation-only harness.  It does not edit Isaac Lab or alter any
task condition.  The temporary method hook records the value returned by the
official FactoryEnv._get_curr_successes call while DirectRLEnv.step is still in
its reward phase, before the automatic reset.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate an RL-Games Factory checkpoint.")
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10, help="Completed episodes per environment.")
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stochastic", action="store_true", help="Sample policy actions; default is deterministic mu.")
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]]
if not args_cli.checkpoint.is_file():
    parser.error("--checkpoint must name an existing local file")
if args_cli.num_envs < 1 or args_cli.episodes < 1 or args_cli.max_steps < 1:
    parser.error("num_envs, episodes and max_steps must be positive")
if args_cli.output is not None and args_cli.output.exists():
    parser.error("--output already exists; use a new path to preserve previous evidence")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _tensor_list(value):
    return value.detach().to(device="cpu").tolist()


def main():
    if not args_cli.checkpoint.is_file():
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.num_envs < 1 or args_cli.episodes < 1 or args_cli.max_steps < 1:
        raise ValueError("num_envs, episodes and max_steps must be positive")

    import gymnasium as gym
    import numpy as np
    import torch
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # Import the pure accounting code after AppLauncher so this script works
    # both as a module from the checkout root and through run_sim.sh.
    try:
        from success_metrics import PreResetSuccessCapture, SuccessTracker
        from evaluation_runtime import reset_player_rnn_states, tensor_tree_is_finite
    except ImportError:
        from scripts.success_metrics import PreResetSuccessCapture, SuccessTracker
        from scripts.evaluation_runtime import reset_player_rnn_states, tensor_tree_is_finite

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["device"] = args_cli.device
    agent_cfg["params"]["config"]["device_name"] = args_cli.device
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(args_cli.checkpoint.resolve())
    agent_cfg["params"]["config"].setdefault("player", {})["deterministic"] = not args_cli.stochastic

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    base_env = raw_env.unwrapped
    predicate_source = inspect.getsource(base_env._get_curr_successes)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(raw_env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(args_cli.checkpoint.resolve()))
    agent.reset()
    obs = env.reset()
    _ = agent.get_batch_size(agent.obs_to_torch(obs), 1)
    if agent.is_rnn:
        agent.init_rnn()

    tracker = SuccessTracker(args_cli.num_envs)
    steps = 0
    started = time.monotonic()
    terminal_checks = []
    with PreResetSuccessCapture(base_env, convert=_tensor_list) as capture:
        while len(tracker.completed) < args_cli.episodes * args_cli.num_envs and steps < args_cli.max_steps:
            with torch.inference_mode():
                model_obs = agent.obs_to_torch(obs)
                actions = agent.get_action(model_obs, is_deterministic=not args_cli.stochastic)
                if not torch.isfinite(actions).all():
                    raise RuntimeError(f"nonfinite policy action at policy step {steps + 1}")
                obs, rew, dones, extras = env.step(actions)
            cap = capture.consume()
            if not tensor_tree_is_finite(rew) or not tensor_tree_is_finite(obs):
                raise RuntimeError(f"nonfinite observation/reward at policy step {steps + 1}")
            done_list = [bool(x) for x in _tensor_list(dones)]
            timeout_tensor = extras.get("time_outs", torch.zeros_like(dones))
            timeout_list = [bool(x) for x in _tensor_list(timeout_tensor)]
            tracker.update(cap["success"], done_list, timeout_list, cap["step_numbers"])
            if any(done_list):
                if not all(done_list):
                    raise RuntimeError("Factory synchronous-reset assumption changed; evaluation cannot count per-env episodes")
                # This check matches the official logging denominator. Factory
                # resets synchronously in the unmodified timeout-only baseline.
                observed = float(extras["successes"].item())
                captured = sum(bool(x) for x in cap["success"]) / args_cli.num_envs
                if not math.isclose(observed, captured, abs_tol=1e-7):
                    raise RuntimeError("captured terminal success disagrees with official extras['successes']")
                reset_steps = _tensor_list(base_env.episode_length_buf)
                if any(int(value) != 0 for value in reset_steps):
                    raise RuntimeError("returned counters are not post-reset at timeout")
                terminal_checks.append({"policy_step": steps + 1, "official_rate": observed,
                                        "captured_rate": captured, "pre_reset_episode_steps": cap["step_numbers"],
                                        "returned_episode_steps_after_reset": reset_steps})
            if agent.is_rnn and agent.states is not None and any(done_list):
                reset_player_rnn_states(agent, done_list)
            steps += 1
        capture_count = capture.captures_consumed

    completed = tracker.completed[: args_cli.episodes * args_cli.num_envs]
    if len(completed) < args_cli.episodes * args_cli.num_envs:
        raise RuntimeError(
            f"evaluation stopped after {steps} steps with {len(completed)} completed episodes; "
            f"requested {args_cli.episodes * args_cli.num_envs}"
        )

    aggregate = SuccessTracker(args_cli.num_envs)
    aggregate.completed = completed
    result = {
        "task": args_cli.task,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args_cli.checkpoint.read_bytes()).hexdigest(),
        "num_envs": args_cli.num_envs,
        "episodes_per_env_requested": args_cli.episodes,
        "episodes_completed": len(completed),
        "max_steps": args_cli.max_steps,
        "seed": args_cli.seed,
        "deterministic": not args_cli.stochastic,
        "official_success_capture": "pre_reset_factory_get_curr_successes",
        "official_predicate_source_sha256": hashlib.sha256(predicate_source.encode()).hexdigest(),
        "official_success_threshold": float(base_env.cfg_task.success_threshold),
        "official_rotation_check": base_env.cfg_task.name == "nut_thread",
        "capture_calls": capture_count,
        "auto_reset_order_verified": bool(terminal_checks),
        "terminal_consistency_checks": terminal_checks,
        "runtime_packages": {name: {"version": module.__version__, "path": module.__file__}
                             for name, module in {"torch": torch, "numpy": np, "gymnasium": gym}.items()},
        "step_dt_seconds": float(base_env.step_dt),
        "policy_observation_shape": list(obs["obs"].shape) if isinstance(obs, dict) and "obs" in obs else None,
        "episodes": [dict(episode.to_dict(), seed=args_cli.seed,
                          first_success_time_seconds=(None if episode.first_success_step is None
                                                      else episode.first_success_step * base_env.step_dt))
                     for episode in completed],
        "aggregate": aggregate.aggregate(args_cli.episodes * args_cli.num_envs),
        "steps_executed": steps,
        "elapsed_seconds": time.monotonic() - started,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args_cli.output is None:
        print("FACTORY_EVALUATION_RESULT=" + text, flush=True)
    else:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(text)
        print("FACTORY_EVALUATION_OUTPUT=" + str(args_cli.output.resolve()), flush=True)

    env.close()


try:
    main()
finally:
    simulation_app.close()
