"""CPU audit helpers for the frozen PegInsert interface, not an inference runner.

No simulator, policy forward, optimizer, or controller is instantiated here.
The contract deliberately rejects broadcasting, implicit dtype conversion of
runtime inputs, non-finite data, and incompatible checkpoint metadata.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch

OBSERVATION_ORDER = (
    "fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel", "prev_actions",
)
OBSERVATION_DIM, CRITIC_DIM, ACTION_DIM = 19, 43, 6
RMS_EPSILON, RMS_CLIP = 1e-5, 5.0


def _vector(value, size, name):
    if torch.is_tensor(value):
        valid_dtype = value.dtype == torch.float32
        finite = torch.isfinite(value).all().item()
    else:
        value = np.asarray(value)
        valid_dtype = value.dtype == np.float32
        finite = np.isfinite(value).all()
    if value.ndim not in (1, 2) or value.shape[-1] != size:
        raise ValueError(f"{name} shape must be [{size}] or [batch,{size}]")
    if value.ndim == 2 and value.shape[0] == 0:
        raise ValueError(f"{name} batch must be positive")
    if not valid_dtype:
        raise ValueError(f"{name} dtype must be float32")
    if not finite:
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def _same_layout(a, b):
    if a.shape != b.shape or torch.is_tensor(a) != torch.is_tensor(b):
        raise ValueError("shape/backend mismatch; broadcasting is forbidden")
    if torch.is_tensor(a) and (a.dtype != b.dtype or a.device != b.device):
        raise ValueError("dtype/device mismatch")


def validate_observation(observation):
    return _vector(observation, OBSERVATION_DIM, "observation")


def validate_policy_output(mu, *, batch_size=None):
    mu = _vector(mu, ACTION_DIM, "policy mean")
    if batch_size is not None and (mu.ndim != 2 or mu.shape[0] != batch_size):
        raise ValueError("policy output batch mismatch")
    return mu


def validate_rms(mean, var, count):
    """Checkpoint RMS is float64; calculation uses its float32 projection."""
    for name, value in (("running_mean", mean), ("running_var", var)):
        a = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if a.shape != (19,) or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError(f"{name} must be finite float64 [19]")
    a = count.detach().cpu().numpy() if torch.is_tensor(count) else np.asarray(count)
    if a.shape != () or a.dtype != np.float64 or not np.isfinite(a) or a <= 0:
        raise ValueError("RMS count must be a finite positive float64 scalar")
    a = var.detach().cpu().numpy() if torch.is_tensor(var) else np.asarray(var)
    if np.any(a < 0):
        raise ValueError("RMS variance must be non-negative")


def normalize_observation(observation, running_mean, running_var, *, epsilon=RMS_EPSILON, clip=RMS_CLIP):
    obs = validate_observation(observation)
    # count is separately verified on loading; normalization never updates it.
    validate_rms(running_mean, running_var, np.array(1.0, dtype=np.float64))
    if epsilon != RMS_EPSILON or clip != RMS_CLIP:
        raise ValueError("epsilon/clip differ from the frozen RL-Games contract")
    if torch.is_tensor(obs):
        for value in (running_mean, running_var):
            if torch.is_tensor(value) and value.device != obs.device:
                raise ValueError("RMS device mismatch")
        mean = torch.as_tensor(running_mean, device=obs.device).float()
        var = torch.as_tensor(running_var, device=obs.device).float()
        out = torch.clamp((obs - mean) / torch.sqrt(var + epsilon), -clip, clip)
    else:
        mean, var = np.asarray(running_mean, dtype=np.float32), np.asarray(running_var, dtype=np.float32)
        out = np.clip((obs - mean) / np.sqrt(var + np.float32(epsilon)), -clip, clip)
    return validate_observation(out)


@dataclass(frozen=True)
class RNNStateSpec:
    layers: int = 2
    hidden_size: int = 1024

    def initialize(self, batch_size, *, dtype=torch.float32, device="cpu"):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if dtype != torch.float32:
            raise ValueError("recurrent dtype must be float32")
        shape = (self.layers, batch_size, self.hidden_size)
        return tuple(torch.zeros(shape, dtype=dtype, device=device) for _ in range(2))

    def validate(self, state, batch_size, *, device=None):
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise ValueError("state must be (hidden, cell)")
        for value in state:
            if not torch.is_tensor(value) or tuple(value.shape) != (self.layers, batch_size, self.hidden_size):
                raise ValueError("recurrent shape mismatch")
            if value.dtype != torch.float32 or value.device != state[0].device:
                raise ValueError("recurrent dtype/device mismatch")
            if device is not None and value.device != torch.device(device):
                raise ValueError("recurrent device mismatch")
            if not torch.isfinite(value).all():
                raise ValueError("recurrent state must be finite")
        return state

    def reset(self, state, done):
        if not isinstance(state, (tuple, list)) or len(state) != 2 or not torch.is_tensor(state[0]) or state[0].ndim != 3:
            raise ValueError("state must be (hidden, cell)")
        self.validate(state, state[0].shape[1])
        mask = torch.as_tensor(done, device=state[0].device)
        if mask.dtype != torch.bool or tuple(mask.shape) != (state[0].shape[1],):
            raise ValueError("done must be a bool vector with one entry per actor")
        out = tuple(value.detach().clone() for value in state)
        for value in out:
            value[:, mask, :] = 0
        return out


def clip_policy_action(mu):
    mu = validate_policy_output(mu)
    return torch.clamp(mu, -1.0, 1.0) if torch.is_tensor(mu) else np.clip(mu, -1.0, 1.0)


def smooth_action(action, previous):
    action, previous = validate_policy_output(action), validate_policy_output(previous)
    _same_layout(action, previous)
    # Do not clip the cache here: reset initializes it from pose, not from mu.
    return validate_policy_output(0.2 * action + 0.8 * previous)


def action_targets(smoothed_action, current_position, fixed_action_frame):
    """CPU check of translation/axis-angle quantities before quaternion projection.

    Exact upright quaternion conversion and physical controller parity remain
    preregistered simulator tests. The EMA cache must NOT be clipped again.
    """
    action = validate_policy_output(smoothed_action)
    pos = _vector(current_position, 3, "position")
    frame = _vector(fixed_action_frame, 3, "fixed frame")
    _same_layout(pos, frame)
    _same_layout(action[..., :3], pos)
    target = pos + action[..., :3] * 0.02
    delta = target - frame
    delta = torch.clamp(delta, -0.05, 0.05) if torch.is_tensor(action) else np.clip(delta, -0.05, 0.05)
    return {"position_target_m": frame + delta, "axis_angle_delta_rad": action[..., 3:] * 0.097}


def verify_checkpoint_identity(path, expected_sha256):
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("invalid SHA-256 identity")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("checkpoint SHA-256 identity mismatch; do not deserialize")
    return digest.hexdigest()


def validate_checkpoint_compatibility(checkpoint, expected_shapes, *, epoch, frame):
    if not isinstance(checkpoint, Mapping) or checkpoint.get("epoch") != epoch or checkpoint.get("frame") != frame:
        raise ValueError("checkpoint epoch/frame mismatch")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or set(state) != set(expected_shapes):
        raise ValueError("checkpoint model keys mismatch")
    for name, shape in expected_shapes.items():
        value = state[name]
        if not torch.is_tensor(value) or list(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor incompatible: {name}")
        expected_dtype = torch.float64 if name.startswith(("running_mean_std.", "value_mean_std.")) else torch.float32
        if value.dtype != expected_dtype:
            raise ValueError(f"checkpoint tensor dtype incompatible: {name}")
    validate_rms(*(state["running_mean_std." + k] for k in ("running_mean", "running_var", "count")))
    return state
