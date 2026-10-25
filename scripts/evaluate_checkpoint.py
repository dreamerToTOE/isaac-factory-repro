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
parser.add_argument("--progress-output", type=Path, default=None,
                    help="Atomic partial result; default: OUTPUT.partial.json when --output is given.")
parser.add_argument("--progress-interval", type=int, default=10, help="Write partial evidence every N policy steps.")
parser.add_argument("--snapshot-limit", type=int, default=0,
                    help="Optional total cap of successful pre-reset geometry records; zero disables.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]]
if not args_cli.checkpoint.is_file():
    parser.error("--checkpoint must name an existing local file")
if args_cli.num_envs < 1 or args_cli.episodes < 1 or args_cli.max_steps < 1:
    parser.error("num_envs, episodes and max_steps must be positive")
if args_cli.output is not None and args_cli.output.exists():
    parser.error("--output already exists; use a new path to preserve previous evidence")
if args_cli.progress_interval < 1 or args_cli.snapshot_limit < 0 or args_cli.snapshot_limit == 1:
    parser.error("progress-interval must be positive; snapshot-limit must be zero or at least two")
progress_path = args_cli.progress_output
if progress_path is None and args_cli.output is not None:
    progress_path = args_cli.output.with_name(args_cli.output.name + ".partial.json")
if progress_path is not None and (progress_path.exists() or progress_path == args_cli.output):
    parser.error("progress output must be a new path distinct from --output")
try:
    from evaluation_reporting import add_intervals, atomic_write_json
except ImportError:
    from scripts.evaluation_reporting import add_intervals, atomic_write_json
if progress_path is not None:
    atomic_write_json(progress_path, {"status": "initializing", "seed": args_cli.seed,
                                    "checkpoint": str(args_cli.checkpoint.resolve()),
                                    "episodes_completed": 0,
                                    "episodes_requested": args_cli.episodes * args_cli.num_envs,
                                    "episodes": [], "aggregate": None})

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _tensor_list(value):
    return value.detach().to(device="cpu").tolist()


def main():
    if not args_cli.checkpoint.is_file():
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.num_envs < 1 or args_cli.episodes < 1 or args_cli.max_steps < 1:
        raise ValueError("num_envs, episodes and max_steps must be positive")
    checkpoint_digest = hashlib.sha256()
    with args_cli.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint_digest.update(block)
    checkpoint_sha256 = checkpoint_digest.hexdigest()

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

    # Shared provenance makes completed episodes in partial snapshots usable by
    # the same strict analysis checks as the final output. This is logging only.
    initialized_metadata = {
        "checkpoint_sha256": checkpoint_sha256,
        "deterministic": not args_cli.stochastic,
        "inference_settings": {
            "action_selection": "sample" if args_cli.stochastic else "policy_mean",
            "model_training": bool(agent.model.training),
            "player_clip_actions": bool(agent.clip_actions),
            "normalization": "checkpoint buffers in eval mode",
            "rnn_reset": "zero hidden/cell states at completed episode boundaries",
            "torch_inference_mode": True,
        },
        "official_success_capture": "pre_reset_factory_get_curr_successes",
        "official_predicate_source_sha256": hashlib.sha256(predicate_source.encode()).hexdigest(),
        "official_success_threshold": float(base_env.cfg_task.success_threshold),
        "official_rotation_check": base_env.cfg_task.name == "nut_thread",
        "step_dt_seconds": float(base_env.step_dt),
    }

    tracker = SuccessTracker(args_cli.num_envs)
    steps = 0
    started = time.monotonic()
    terminal_checks = []
    geometric_snapshots = []
    seen_success = [False] * args_cli.num_envs
    snapshot_episode_ids = [0] * args_cli.num_envs
    snapshot_counts = {"first_success": 0, "terminal_success": 0}
    snapshot_limits = {"first_success": (args_cli.snapshot_limit + 1) // 2,
                       "terminal_success": args_cli.snapshot_limit // 2}

    def snapshot_geometry(instance, record):
        """Read original tensors before reset; never add a success criterion."""
        snapshots = []
        done = _tensor_list(instance.reset_buf)
        for env_id, success in enumerate(record["success"]):
            triggers = []
            if success and not seen_success[env_id]:
                triggers.append("first_success")
            if success and done[env_id]:
                triggers.append("terminal_success")
            for trigger in triggers:
                if snapshot_counts[trigger] >= snapshot_limits[trigger]:
                    continue
                held = instance.held_base_pos[env_id]
                target = instance.target_held_base_pos[env_id]
                xy_dist = float(torch.linalg.vector_norm(target[:2] - held[:2]).item())
                z_disp = float((held[2] - target[2]).item())
                fixed = instance.cfg_task.fixed_asset_cfg
                height = float(fixed.thread_pitch if instance.cfg_task.name == "nut_thread" else fixed.height)
                height_threshold = height * float(instance.cfg_task.success_threshold)
                yaw = float(instance.curr_yaw[env_id].item()) if record["check_rot"] else None
                yaw_threshold = float(instance.cfg_task.ee_success_yaw) if record["check_rot"] else None
                geometry_success = xy_dist < 0.0025 and z_disp < height_threshold
                if record["check_rot"]:
                    geometry_success = geometry_success and yaw < yaw_threshold
                snapshots.append({"trigger": trigger, "seed": args_cli.seed, "env_id": env_id,
                                  "episode_id": snapshot_episode_ids[env_id],
                                  "episode_step": record["step_numbers"][env_id],
                                  "held_base_pos": _tensor_list(held),
                                  "target_held_base_pos": _tensor_list(target),
                                  "xy_dist_m": xy_dist, "z_disp_m": z_disp,
                                  "official_xy_threshold_m": 0.0025,
                                  "official_height_threshold_m": height_threshold,
                                  "official_success_threshold": float(instance.cfg_task.success_threshold),
                                  "official_check_rot": record["check_rot"],
                                  "curr_yaw_rad": yaw, "official_yaw_threshold_rad": yaw_threshold,
                                  "official_predicate_result": bool(success),
                                  "geometry_recheck_result": bool(geometry_success)})
                if not geometry_success:
                    raise RuntimeError("recorded successful geometry disagrees with official predicate")
                snapshot_counts[trigger] += 1
            seen_success[env_id] = seen_success[env_id] or bool(success)
            if done[env_id]:
                seen_success[env_id] = False
                snapshot_episode_ids[env_id] += 1
        return snapshots

    def write_progress(status, capture_count):
        if progress_path is None:
            return
        atomic_write_json(progress_path, {
            **initialized_metadata,
            "status": status, "task": args_cli.task, "seed": args_cli.seed,
            "checkpoint": str(args_cli.checkpoint.resolve()), "num_envs": args_cli.num_envs,
            "episodes_requested": args_cli.episodes * args_cli.num_envs,
            "episodes_completed": len(tracker.completed),
            "episodes": [dict(item.to_dict(), seed=args_cli.seed,
                              first_success_time_seconds=(None if item.first_success_step is None
                                                          else item.first_success_step * base_env.step_dt))
                         for item in tracker.completed],
            "aggregate": add_intervals(tracker.aggregate(args_cli.episodes * args_cli.num_envs)),
            "steps_executed": steps, "capture_calls": capture_count,
            "elapsed_seconds": time.monotonic() - started,
            "geometric_snapshots": geometric_snapshots,
            "auto_reset_order_verified": bool(terminal_checks),
            "terminal_consistency_checks": terminal_checks,
            "partial_denominator_rule": "completed environment episodes only; unfinished episodes excluded",
        })

    write_progress("running", 0)
    with PreResetSuccessCapture(base_env, convert=_tensor_list,
                               snapshot=snapshot_geometry if args_cli.snapshot_limit else None) as capture:
        while len(tracker.completed) < args_cli.episodes * args_cli.num_envs and steps < args_cli.max_steps:
            with torch.inference_mode():
                model_obs = agent.obs_to_torch(obs)
                actions = agent.get_action(model_obs, is_deterministic=not args_cli.stochastic)
                if not torch.isfinite(actions).all():
                    raise RuntimeError(f"nonfinite policy action at policy step {steps + 1}")
                obs, rew, dones, extras = env.step(actions)
            cap = capture.consume()
            geometric_snapshots.extend(cap.get("geometric_snapshots", []))
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
            if any(done_list) or steps % args_cli.progress_interval == 0:
                write_progress("running", capture.captures_consumed)
        capture_count = capture.captures_consumed

    completed = tracker.completed[: args_cli.episodes * args_cli.num_envs]
    if len(completed) < args_cli.episodes * args_cli.num_envs:
        write_progress("incomplete_max_steps", capture_count)
        raise RuntimeError(
            f"evaluation stopped after {steps} steps with {len(completed)} completed episodes; "
            f"requested {args_cli.episodes * args_cli.num_envs}"
        )

    aggregate = SuccessTracker(args_cli.num_envs)
    aggregate.completed = completed
    result = {
        **initialized_metadata,
        "status": "complete",
        "task": args_cli.task,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "num_envs": args_cli.num_envs,
        "episodes_per_env_requested": args_cli.episodes,
        "episodes_completed": len(completed),
        "max_steps": args_cli.max_steps,
        "seed": args_cli.seed,
        "capture_calls": capture_count,
        "auto_reset_order_verified": bool(terminal_checks),
        "terminal_consistency_checks": terminal_checks,
        "geometric_snapshot_limit": args_cli.snapshot_limit,
        "geometric_snapshots": geometric_snapshots,
        "runtime_packages": {name: {"version": module.__version__, "path": module.__file__}
                             for name, module in {"torch": torch, "numpy": np, "gymnasium": gym}.items()},
        "policy_observation_shape": list(obs["obs"].shape) if isinstance(obs, dict) and "obs" in obs else None,
        "episodes": [dict(episode.to_dict(), seed=args_cli.seed,
                          first_success_time_seconds=(None if episode.first_success_step is None
                                                      else episode.first_success_step * base_env.step_dt))
                     for episode in completed],
        "aggregate": add_intervals(aggregate.aggregate(args_cli.episodes * args_cli.num_envs)),
        "steps_executed": steps,
        "elapsed_seconds": time.monotonic() - started,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args_cli.output is None:
        print("FACTORY_EVALUATION_RESULT=" + text, flush=True)
    else:
        atomic_write_json(args_cli.output, result)
        print("FACTORY_EVALUATION_OUTPUT=" + str(args_cli.output.resolve()), flush=True)
    if progress_path is not None:
        atomic_write_json(progress_path, result)

    env.close()


try:
    main()
except (Exception, KeyboardInterrupt) as exc:
    if progress_path is not None and progress_path.exists():
        partial = json.loads(progress_path.read_text())
        if partial.get("status") != "incomplete_max_steps":
            partial["status"] = "incomplete_interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete_error"
        partial["error_type"] = type(exc).__name__
        partial["error"] = str(exc)
        atomic_write_json(progress_path, partial)
    raise
finally:
    simulation_app.close()
