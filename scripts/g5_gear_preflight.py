"""G5 GearMesh bounded environment preflight; no policy training.

Read-only pre-reset capture reuses the project's official-predicate tap. JSONL is
flushed after every decision so a supervised interruption preserves evidence.
"""
import argparse
import json
import math
import time
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--steps', type=int, default=300)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if not 1 <= a.steps <= 500 or a.device != 'cuda:0':
    p.error('registered budget 1..500 and cuda:0 required')
app = AppLauncher(a).app
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from success_metrics import PreResetSuccessCapture, SuccessTracker
    task = 'Isaac-Factory-GearMesh-Direct-v0'
    assert task in gym.registry
    cfg = parse_env_cfg(task, device=a.device, num_envs=1)
    cfg.seed = 0
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    assert u.max_episode_length == 300 and cfg.decimation == 8
    assert u.sim.is_gpu_dynamics_enabled(), 'GPU dynamics disabled'
    obs, _ = env.reset()
    assert all(t.is_cuda and torch.isfinite(t).all() for t in obs.values())
    assert obs['policy'].shape == (1, 19) and obs['critic'].shape == (1, 43)
    tensor = u._held_asset.root_physx_view.get_transforms()
    assert tensor.is_cuda and torch.isfinite(tensor).all()
    meta = {'task':task, 'seed':0, 'registered':True, 'gpu_dynamics':u.sim.is_gpu_dynamics_enabled(),
            'physx_view_device':str(tensor.device), 'sim_device':str(u.device), 'horizon':u.max_episode_length,
            'dt':u.physics_dt, 'decimation':cfg.decimation, 'observation_shapes':{k:list(v.shape) for k,v in obs.items()},
            'torch_version':torch.__version__, 'torch_path':torch.__file__, 'isaaclab_tasks_path':isaaclab_tasks.__file__}
    print('G5_RUNTIME_META='+json.dumps(meta), flush=True)
    def conv(x): return x.detach().cpu().tolist()
    def snapshot(e, record):
        return {'held_base':conv(e.held_base_pos), 'target_base':conv(e.target_held_base_pos),
                'joint_pos':conv(e._robot.data.joint_pos), 'joint_vel':conv(e._robot.data.joint_vel),
                'held_root':conv(e._held_asset.data.root_state_w), 'fixed_root':conv(e._fixed_asset.data.root_state_w),
                'small_root':conv(e._small_gear_asset.data.root_state_w), 'large_root':conv(e._large_gear_asset.data.root_state_w),
                'keypoint_dist':conv(e.keypoint_dist)}
    tracker = SuccessTracker(1)
    start = time.monotonic()
    with open(a.output, 'x', buffering=1) as f, PreResetSuccessCapture(u, convert=conv, snapshot=snapshot) as tap:
        for i in range(a.steps):
            action = torch.zeros((1,6), device=u.device)
            action[:,0] = .05 if i % 2 else -.05
            obs, rew, term, trunc, _ = env.step(action)
            capture = tap.consume()
            snap = capture['geometric_snapshots']
            for name, val in snap.items():
                assert torch.isfinite(torch.tensor(val)).all(), f'nonfinite {name} at decision {i+1}'
            assert all(torch.isfinite(v).all() for v in obs.values()), f'nonfinite observation {i+1}'
            assert torch.isfinite(rew).all(), f'nonfinite reward {i+1}'
            for name, asset in u.scene.articulations.items():
                assert torch.isfinite(asset.data.root_state_w).all() and torch.isfinite(asset.data.joint_vel).all(), f'nonfinite {name}'
            done = term | trunc
            tracker.update(capture['success'], conv(done), conv(trunc), capture['step_numbers'])
            row = {'decision':i+1, 'episode_step':capture['step_numbers'][0], 'official_success':capture['success'][0],
                   'terminated':conv(term)[0], 'truncated':conv(trunc)[0], 'reward':conv(rew)[0],
                   'post_reset_episode_counter':int(u.episode_length_buf[0]), 'physics_counter':u._sim_step_counter,
                   **{k:v[0] for k,v in snap.items()}}
            f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
    print('G5_RUNTIME_RESULT='+json.dumps({'decisions':a.steps,'step_loop_seconds':time.monotonic()-start,
          'physics_decision_steps':u._sim_step_counter, 'episodes':[x.to_dict() for x in tracker.completed],
          'summary':tracker.aggregate(), 'metadata':meta}, allow_nan=False), flush=True)
    env.close()
finally:
    app.close()
