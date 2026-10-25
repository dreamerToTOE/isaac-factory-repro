"""Supplemental rendered playback, separate from frozen quantitative evaluation.

Read-only success taps capture geometry and images in the official reward phase,
before DirectRLEnv automatically resets a terminal environment. Rendering calls
advance no physics; counters and asset/robot tensors are checked around capture.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--task", default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--max-steps", type=int, default=650)
parser.add_argument("--width", type=int, default=640, help="Presentation render width")
parser.add_argument("--height", type=int, default=480, help="Presentation render height")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
if not args.checkpoint.is_file() or args.output_dir.exists():
    parser.error("checkpoint must exist and output-dir must be new")
if args.episodes < 1 or args.max_steps < 1 or not args.enable_cameras:
    parser.error("positive episode/step limits and --enable_cameras are required")
args.output_dir.mkdir(parents=True)
from evaluation_reporting import atomic_write_json
result = {"status": "initializing", "seed": args.seed, "num_envs": 1,
          "checkpoint": str(args.checkpoint.resolve()), "episodes": [], "frames": [],
          "scope": "Supplemental rendered rollout; excluded from primary G2 denominator. This is not an exact replay of a primary evaluation episode."}
output = args.output_dir / "playback.json"
atomic_write_json(output, result)
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from success_metrics import PreResetSuccessCapture, SuccessTracker
    from evaluation_runtime import reset_player_rnn_states, tensor_tree_is_finite

    def values(x):
        return x.detach().cpu().tolist()

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    # Camera is presentation-only; task, action and physics settings are official.
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.eye = (0.85, 0.25, 0.25)
    env_cfg.viewer.lookat = (0.6, 0.0, 0.08)
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    config = agent_cfg["params"]["config"]
    config["num_actors"] = 1
    config["device"] = config["device_name"] = args.device
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(args.checkpoint.resolve())
    config.setdefault("player", {})["deterministic"] = True
    raw = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    base = raw.unwrapped
    if base.cfg_task.name != "peg_insert":
        raise ValueError("This supplemental geometry renderer is PegInsert-only")
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(raw, args.device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(args.checkpoint.resolve()))
    agent.reset()
    obs = env.reset()
    agent.get_batch_size(agent.obs_to_torch(obs), 1)
    if agent.is_rnn:
        agent.init_rnn()

    result.update({"status": "running", "task": args.task,
                   "checkpoint_sha256": hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest()
                   if hasattr(hashlib, "file_digest") else hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   "deterministic": True, "action_selection": "policy_mean",
                   "model_training": bool(agent.model.training),
                   "observation_normalization": "checkpoint buffers in eval mode",
                   "step_dt_seconds": float(base.step_dt), "physics_dt_seconds": float(base.physics_dt),
                   "episodes_requested": args.episodes, "max_steps": args.max_steps,
                   "official_predicate_source_sha256": hashlib.sha256(inspect.getsource(base._get_curr_successes).encode()).hexdigest(),
                   "capture_phase": "official _get_curr_successes during reward computation, before automatic reset",
                   "camera_resolution": list(env_cfg.viewer.resolution),
                   "rendering_mode": args.rendering_mode or "balanced",
                   "render_physics_steps": 0, "geometry_trace": [], "terminal_checks": []})
    tracker = SuccessTracker(1)
    episode_id = 0
    seen_success = False
    camera = None
    started = time.monotonic()

    def physics_state(instance):
        return {"held_root": instance._held_asset.data.root_state_w.clone(),
                "fixed_root": instance._fixed_asset.data.root_state_w.clone(),
                "joint_pos": instance._robot.data.joint_pos.clone(),
                "joint_vel": instance._robot.data.joint_vel.clone(),
                "episode_steps": instance.episode_length_buf.clone(),
                "held_base": instance.held_base_pos.clone(),
                "target_base": instance.target_held_base_pos.clone()}

    def capture_frame(instance, geometry, triggers):
        nonlocal camera
        before = physics_state(instance)
        clock_before = float(instance.sim.current_time)
        steps_before = int(instance._sim_step_counter)
        if geometry["episode_step"] == 1:
            target = np.asarray(values(instance.target_held_base_pos[0])) + np.asarray(values(instance.scene.env_origins[0]))
            lookat = target + np.array([0.0, 0.0, 0.030])
            eye = target + np.array([0.22, 0.22, 0.16])
            camera = {"eye_world_m": eye.tolist(), "lookat_world_m": lookat.tolist()}
            instance.sim.set_camera_view(eye=eye, target=lookat, camera_prim_path=instance.cfg.viewer.cam_prim_path)
        # First call initializes the RGB annotator; render-only updates fill it.
        instance.render()
        for _ in range(4):
            instance.sim.render()
        rgb = instance.render()
        if rgb is None or rgb.shape != (args.height, args.width, 3) or not np.any(rgb):
            raise RuntimeError("Renderer did not produce a nonempty RGB frame of the requested size")
        after = physics_state(instance)
        unchanged = all(torch.equal(before[key], after[key]) for key in before)
        unchanged = unchanged and float(instance.sim.current_time) == clock_before and int(instance._sim_step_counter) == steps_before
        if not unchanged:
            raise RuntimeError("render-only capture changed physics state or counters")
        filename = f"episode_{episode_id:02d}_step_{geometry['episode_step']:03d}_{'_'.join(triggers)}.png"
        image_path = args.output_dir / filename
        Image.fromarray(rgb.copy()).save(image_path)
        record = {**geometry, "triggers": triggers, "file": filename,
                  "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                  "camera": camera, "physics_state_unchanged_by_render": unchanged,
                  "sim_time_seconds": clock_before, "sim_step_counter": steps_before,
                  "capture_precedes_auto_reset": True}
        result["frames"].append(record)
        atomic_write_json(output, result)
        print("PLAYBACK_FRAME=" + json.dumps(record), flush=True)

    def snapshot(instance, official):
        nonlocal seen_success
        held = instance.held_base_pos[0]
        target = instance.target_held_base_pos[0]
        xy = float(torch.linalg.vector_norm(target[:2] - held[:2]).item())
        z = float((held[2] - target[2]).item())
        height = float(instance.cfg_task.fixed_asset_cfg.height * instance.cfg_task.success_threshold)
        geometry_success = xy < 0.0025 and z < height
        success = bool(official["success"][0])
        terminal = bool(instance.reset_buf[0].item())
        step = int(official["step_numbers"][0])
        geometry = {"episode_id": episode_id, "env_id": 0, "seed": args.seed, "episode_step": step,
                    "held_base_pos_m": values(held), "target_held_base_pos_m": values(target),
                    "held_quaternion_wxyz": values(instance.held_quat[0]),
                    "fixed_quaternion_wxyz": values(instance.fixed_quat[0]),
                    "xy_dist_m": xy, "z_disp_m": z,
                    "official_xy_threshold_m": 0.0025, "official_height_threshold_m": height,
                    "official_success": success, "independent_geometry_success": geometry_success,
                    "terminal_pre_reset": terminal}
        if success != geometry_success:
            raise RuntimeError("Independent geometric predicate disagrees with official mask")
        result["geometry_trace"].append(geometry)
        triggers = []
        if step == 1:
            triggers.append("start")
        if success and not seen_success:
            triggers.append("first_success")
        if terminal:
            triggers.append("terminal_success" if success else "terminal_failure")
        if triggers:
            capture_frame(instance, geometry, triggers)
        seen_success = seen_success or success
        return []

    atomic_write_json(output, result)
    with PreResetSuccessCapture(base, convert=values, snapshot=snapshot) as capture:
        for steps in range(args.max_steps):
            with torch.inference_mode():
                actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("nonfinite action")
                obs, rewards, dones, extras = env.step(actions)
            if not tensor_tree_is_finite(obs) or not tensor_tree_is_finite(rewards):
                raise RuntimeError("nonfinite observation/reward")
            record = capture.consume()
            done = [bool(x) for x in values(dones)]
            timeout = [bool(x) for x in values(extras.get("time_outs", torch.zeros_like(dones)))]
            tracker.update(record["success"], done, timeout, record["step_numbers"])
            if done[0]:
                terminal_rate = float(extras["successes"].item())
                post_reset = values(base.episode_length_buf)
                if terminal_rate != float(record["success"][0]) or post_reset != [0]:
                    raise RuntimeError("official terminal/post-reset consistency check failed")
                result["terminal_checks"].append({"episode_id": episode_id,
                    "official_terminal_rate": terminal_rate, "pre_reset_step": record["step_numbers"][0],
                    "returned_post_reset_steps": post_reset})
                if agent.is_rnn and agent.states is not None:
                    reset_player_rnn_states(agent, done)
                episode_id += 1
                seen_success = False
            result["episodes"] = [episode.to_dict() for episode in tracker.completed]
            result["steps_executed"] = steps + 1
            if done[0] or (steps + 1) % 25 == 0:
                atomic_write_json(output, result)
            if len(tracker.completed) >= args.episodes:
                break
    result["status"] = "complete" if len(tracker.completed) == args.episodes else "incomplete_max_steps"
    result["aggregate"] = tracker.aggregate(args.episodes)
    result["elapsed_seconds"] = time.monotonic() - started
    result["all_geometry_rechecks_match"] = all(x["official_success"] == x["independent_geometry_success"] for x in result["geometry_trace"])
    atomic_write_json(output, result)
    env.close()
    if result["status"] != "complete":
        raise RuntimeError("Playback did not complete all requested episodes")
    print("PLAYBACK_COMPLETE=" + str(output), flush=True)


try:
    main()
except (Exception, KeyboardInterrupt) as exc:
    result["status"] = "incomplete_error"
    result["error"] = f"{type(exc).__name__}: {exc}"
    atomic_write_json(output, result)
    raise
finally:
    app.close()
