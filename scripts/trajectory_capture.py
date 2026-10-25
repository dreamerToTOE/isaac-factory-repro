"""Read-only, vectorized pre-reset Factory trajectory capture for G3B.

This module deliberately imports neither Isaac Sim nor Torch at import time.
Quaternions use Isaac Lab's scalar-first (w, x, y, z) convention. Saved positions
are world positions; official XY/Z errors use the task's held/target base frames.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MethodType


SCHEMA_VERSION = "g3b_factory_trajectory_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_trajectory_npz(path, arrays, metadata):
    """Write without pickle or overwrite; metadata travels inside the NPZ.

    This function also supports small CPU serialization controls without a
    simulator. All arrays must share the first two axes (policy time, env).
    """
    import numpy as np

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    converted = {name: np.asarray(value) for name, value in arrays.items()}
    if not converted:
        raise ValueError("trajectory has no arrays")
    first_axes = next(iter(converted.values())).shape[:2]
    if len(first_axes) != 2 or any(size < 1 for size in first_axes):
        raise ValueError("trajectory arrays require nonempty [time, env, ...] axes")
    for name, array in converted.items():
        if array.ndim < 2 or array.shape[:2] != first_axes:
            raise ValueError(f"trajectory axes disagree: {name}")
        if array.dtype.hasobject:
            raise ValueError(f"object arrays are forbidden: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"nonfinite trajectory data: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **converted,
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        stream.flush()
        os.fsync(stream.fileno())
    # No replace(): preserve prior diagnostic evidence if another run races us.
    os.link(temporary, path)
    temporary.unlink()
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                       for name, array in converted.items()}}


class VectorTrajectoryRecorder:
    """Clone existing GPU tensors, then transfer to CPU once capture ends.

    The official pre-physics method and success/engagement predicate retain
    their arguments, return objects, call order and tensor contents.
    """

    def __init__(self, env, max_steps, seed):
        import torch

        self.env = env
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.records = []
        self.policy_actions = None
        self.episode_ids = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self.env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
        self.original_pre_physics = env._pre_physics_step
        recorder = self

        def observed_pre_physics(instance, action):
            # Input is the exact action passed by RlGamesVecEnvWrapper, before
            # the task's official exponential moving average.
            recorder.policy_actions = action.detach().clone()
            return recorder.original_pre_physics(action)

        self.observed_pre_physics = MethodType(observed_pre_physics, env)

    def install_action_tap(self):
        self.env._pre_physics_step = self.observed_pre_physics

    def restore_action_tap(self):
        self.env._pre_physics_step = self.original_pre_physics

    def record_success(self, instance, success_tensor):
        import torch

        if len(self.records) >= self.max_steps:
            raise RuntimeError("diagnostic capture exceeded its fixed policy-step bound")
        if self.policy_actions is None:
            raise RuntimeError("success sample has no associated policy action")
        origins = instance.scene.env_origins
        held_base = instance.held_base_pos
        target_base = instance.target_held_base_pos
        data = {
            "seed": torch.full_like(self.env_ids, self.seed),
            "env_id": self.env_ids,
            "episode_id": self.episode_ids,
            "env_origins": origins,
            "policy_step": instance.episode_length_buf,
            "global_policy_step": torch.full_like(self.env_ids, len(self.records) + 1),
            "peg_pos_w": instance.held_pos + origins,
            "peg_quat_w": instance.held_quat,
            "hole_pos_w": instance.fixed_pos + origins,
            "hole_quat_w": instance.fixed_quat,
            "held_base_pos_w": held_base + origins,
            "held_base_quat_w": instance.held_base_quat,
            "target_held_base_pos_w": target_base + origins,
            "target_held_base_quat_w": instance.target_held_base_quat,
            "ee_pos_w": instance.fingertip_midpoint_pos + origins,
            "ee_quat_w": instance.fingertip_midpoint_quat,
            "ee_lin_vel_w": instance.fingertip_midpoint_linvel,
            "ee_ang_vel_w": instance.fingertip_midpoint_angvel,
            "policy_actions": self.policy_actions,
            "actions": instance.actions,
            "controller_target_ee_pos_w": instance.ctrl_target_fingertip_midpoint_pos + origins,
            "controller_target_ee_quat_w": instance.ctrl_target_fingertip_midpoint_quat,
            "xy_error_m": torch.linalg.vector_norm(target_base[:, :2] - held_base[:, :2], dim=-1),
            "axial_displacement_m": held_base[:, 2] - target_base[:, 2],
            "official_success": success_tensor,
            "done": instance.reset_buf,
            "timeout": instance.reset_time_outs,
        }
        self.records.append({name: value.detach().clone() for name, value in data.items()})
        # This bookkeeping only reads the pre-reset done mask; it never resets
        # an environment, a task tensor or recurrent policy state.
        self.episode_ids = self.episode_ids + instance.reset_buf.to(dtype=torch.int64)
        return []

    def record_engagement(self, result):
        if not self.records or "engaged" in self.records[-1]:
            raise RuntimeError("engagement predicate call order/count changed")
        self.records[-1]["engaged"] = result.detach().clone()

    def arrays(self):
        import torch

        if not self.records or any("engaged" not in item for item in self.records):
            raise RuntimeError("incomplete success/engagement trajectory capture")
        # Stacking and host transfer occur after all stepping has ended.
        return {name: torch.stack([item[name] for item in self.records]).cpu().numpy()
                for name in self.records[0]}

