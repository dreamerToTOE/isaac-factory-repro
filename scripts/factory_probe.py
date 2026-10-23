"""Register all Factory IDs and run a bounded CUDA environment reset/step probe."""
import argparse
import json
import time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 1:
    parser.error("--steps and --num_envs must be positive")
launcher = AppLauncher(args)
app = launcher.app
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    import numpy as np
    print("RUNTIME_PACKAGES=" + json.dumps({name: {"version": mod.__version__, "path": mod.__file__}
          for name, mod in {"torch": torch, "numpy": np, "gymnasium": gym}.items()}), flush=True)
    from isaaclab_tasks.utils import parse_env_cfg

    ids = ["Isaac-Factory-PegInsert-Direct-v0", "Isaac-Factory-GearMesh-Direct-v0", "Isaac-Factory-NutThread-Direct-v0"]
    registered = {name: name in gym.registry for name in ids}
    print("FACTORY_REGISTRATION=" + json.dumps(registered), flush=True)
    assert all(registered.values())
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 0
    env = gym.make(args.task, cfg=cfg)
    obs, _ = env.reset()
    assert all(torch.isfinite(x).all() for x in obs.values())
    assert all(x.is_cuda for x in obs.values())
    start = time.perf_counter()
    reward_min, reward_max, resets = float("inf"), float("-inf"), 0
    reset_steps = []
    max_joint_speed, max_held_speed = 0., 0.
    for index in range(args.steps):
        actions = torch.zeros((args.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device)
        # Bounded alternating input exercises control without a trained policy.
        actions[:, 0] = .05 if index % 2 else -.05
        obs, rew, terminated, truncated, _ = env.step(actions)
        assert all(torch.isfinite(x).all() for x in obs.values()), f"Nonfinite observation at {index}"
        assert torch.isfinite(rew).all(), f"Nonfinite reward at {index}"
        assert torch.isfinite(env.unwrapped._robot.data.joint_pos).all(), f"Nonfinite joints at {index}"
        for name, asset in env.unwrapped.scene.articulations.items():
            assert torch.isfinite(asset.data.root_state_w).all(), f"Nonfinite {name} state at {index}"
            assert torch.isfinite(asset.data.joint_vel).all(), f"Nonfinite {name} joint speed at {index}"
        max_joint_speed = max(max_joint_speed, env.unwrapped._robot.data.joint_vel.abs().max().item())
        max_held_speed = max(max_held_speed, env.unwrapped._held_asset.data.root_lin_vel_w.norm(dim=-1).max().item())
        reward_min, reward_max = min(reward_min, rew.min().item()), max(reward_max, rew.max().item())
        resets += int((terminated | truncated).sum())
        if bool((terminated | truncated).any()):
            reset_steps.append(index + 1)
    elapsed = time.perf_counter() - start
    print("FACTORY_PROBE_RESULT=" + json.dumps({"passed": True, "task": args.task, "num_envs": args.num_envs,
          "steps": args.steps, "observation_shapes": {key: list(val.shape) for key, val in obs.items()},
          "device": str(env.unwrapped.device), "reward_min": reward_min, "reward_max": reward_max,
          "max_abs_joint_speed_rad_s": max_joint_speed, "max_held_linear_speed_m_s": max_held_speed,
          "asset_url": env.unwrapped.cfg_task.held_asset_cfg.usd_path,
          "episode_resets": resets, "reset_steps": reset_steps, "step_seconds": elapsed, "environment_steps_per_second": args.steps * args.num_envs / elapsed}), flush=True)
    env.close()
finally:
    app.close()
