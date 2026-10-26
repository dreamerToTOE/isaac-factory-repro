"""Frozen FP32 Factory actor, independent of the PPO Runner and optimizer.

Only reviewed checkpoint hashes reach trusted pickle deserialization. The
official RL-Games architecture is reused, while its training loop is absent.
The caller owns the original Factory EMA, target generation and controller.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import torch

from scripts.g4i0_interface import (
    RNNStateSpec, validate_checkpoint_compatibility, verify_checkpoint_identity,
    validate_rms,
)

PRIMARY_SHA256 = "f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f"
ROLES = ("g2_epoch200_primary", "g3c_r4_epoch400_secondary", "g3c_r5_epoch400_additional")
STATE_SPEC = RNNStateSpec()


def _tensor(value, shape, device, name):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape must be {tuple(shape)}")
    # Device must be checked before any operation that would read a meta tensor.
    if value.device != torch.device(device):
        raise ValueError(f"{name} device mismatch")
    if value.dtype != torch.float32 or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous float32")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def _batch(obs):
    if not torch.is_tensor(obs) or obs.ndim != 2 or obs.shape[1] != 19 or obs.shape[0] < 1:
        raise ValueError("observation shape must be [positive_batch,19]")
    return obs.shape[0]


@dataclass(frozen=True)
class PolicyStep:
    observation: torch.Tensor
    normalized: torch.Tensor
    mu: torch.Tensor
    logstd: torch.Tensor
    state: tuple[torch.Tensor, torch.Tensor]
    clipped_action: torch.Tensor
    consumed_discarded_sample: bool

    @property
    def h(self):
        return self.state[0]

    @property
    def c(self):
        return self.state[1]


class FrozenPolicy:
    """Functional actor forward; recurrent state is explicit to the caller."""

    def __init__(self, model, metadata, *, device="cpu", params=None):
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.identity = copy.deepcopy(metadata)
        self.metadata = self.identity
        self.params = copy.deepcopy(params)
        self.states = None
        rms = self.model.running_mean_std
        validate_rms(rms.running_mean, rms.running_var, rms.count)
        if rms.epsilon != 1e-5 or rms.norm_only:
            raise ValueError("normalizer differs from frozen epsilon/norm_only")
        self._rms = {key: value.detach().clone() for key, value in rms.state_dict().items()}

    @classmethod
    def from_checkpoint(cls, checkpoint=None, *, workspace=None, repo=None,
                        role="g2_epoch200_primary", device="cpu"):
        repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
        workspace = Path(workspace) if workspace is not None else repo.parents[1]
        if role not in ROLES:
            raise ValueError("checkpoint role is not allowlisted")
        identity = json.loads((repo / "evidence/g4_i/i0/checkpoint_and_source_identity.json").read_text())
        audit = json.loads((repo / "evidence/g4_i/i0/checkpoint_audit.json").read_text())
        record, values = identity["checkpoints"][role], audit["checkpoints"][role]
        if role == ROLES[0] and record["sha256"] != PRIMARY_SHA256:
            raise ValueError("primary checkpoint allowlist has changed")
        if record["sha256"] != values["sha256"]:
            raise ValueError("I0 checkpoint identity records disagree")
        path = Path(checkpoint) if checkpoint is not None else workspace / record["workspace_relative_path"]
        digest = verify_checkpoint_identity(path, record["sha256"])
        # Explicit trusted load applies only after the known file hash passes.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if set(saved) != set(record["checkpoint_keys"]) or saved.get("env_state") is not None:
            raise ValueError("checkpoint metadata keys/env_state mismatch")
        state = validate_checkpoint_compatibility(saved, record["model_tensor_shapes"],
                                                  epoch=record["epoch"], frame=record["frame"])
        for name, expected in values["rms"].items():
            actual = state["running_mean_std." + name]
            if not torch.equal(actual, torch.tensor(expected, dtype=torch.float64)):
                raise ValueError(f"frozen RMS mismatch: {name}")
        params = copy.deepcopy(values["agent_configuration"]["params"])
        if params["model"]["name"] != record["architecture"]["model"] or params["network"] != record["architecture"]["actor_critic"]:
            raise ValueError("frozen architecture metadata mismatch")
        from rl_games.algos_torch.model_builder import ModelBuilder
        # Building initializes weights. Keep that initialization out of the
        # inference/reset RNG stream; callers register seed 1001 after loading.
        with torch.random.fork_rng(devices=[]):
            model = ModelBuilder().load(params).build({
                "actions_num": 6, "input_shape": (19,), "num_seqs": 1,
                "value_size": 1, "normalize_value": True, "normalize_input": True,
            })
        model.load_state_dict(state, strict=True)
        metadata = {"role": role, "path": str(path.resolve()), "sha256": digest,
                    "epoch": int(saved["epoch"]), "frame": int(saved["frame"]),
                    "strict_state_dict": True, "frozen_rms_exact": True,
                    "fp32": True, "optimizer_restored": False,
                    "rng_policy": "discard the same Normal.sample as official deterministic Player"}
        return cls(model, metadata, device=device, params=params)

    def initial_state(self, batch_size=1):
        return STATE_SPEC.initialize(batch_size, device=self.device)

    def rms_unchanged(self):
        return all(torch.equal(value, self._rms[key])
                   for key, value in self.model.running_mean_std.state_dict().items())

    def reset_state(self, state, done):
        STATE_SPEC.validate(state, state[0].shape[1], device=self.device)
        out = STATE_SPEC.reset(state, done)
        mask = torch.as_tensor(done, device=self.device)
        for before, after in zip(state, out):
            if torch.count_nonzero(after[:, mask, :]).item() or not torch.equal(before[:, ~mask, :], after[:, ~mask, :]):
                raise RuntimeError("recurrent reset verification failed")
        return out

    def reset(self, done):
        if self.states is None:
            raise ValueError("recurrent states have not been initialized")
        self.states = self.reset_state(self.states, done)
        return self.states

    def infer(self, obs, state, *, consume_rng=True):
        batch_size = _batch(obs)
        _tensor(obs, (batch_size, 19), self.device, "observation")
        STATE_SPEC.validate(state, batch_size, device=self.device)
        if self.model.training or self.model.running_mean_std.training or not self.rms_unchanged():
            raise RuntimeError("model/RMS must remain frozen in evaluation mode")
        rms = self.model.running_mean_std
        with torch.inference_mode():
            normalized = torch.clamp((obs - rms.running_mean.float()) /
                                     torch.sqrt(rms.running_var.float() + 1e-5), -5.0, 5.0)
            _tensor(normalized, (batch_size, 19), self.device, "normalized observation")
            # The inner network receives an already-normalized observation;
            # calling the outer model here would apply normalization twice.
            mu, logstd, _value, out_state = self.model.a2c_network({
                "is_train": False, "prev_actions": None, "obs": normalized,
                "rnn_states": state,
            })
            _tensor(mu, (batch_size, 6), self.device, "policy mean")
            _tensor(logstd, (batch_size, 6), self.device, "policy logstd")
            STATE_SPEC.validate(out_state, batch_size, device=self.device)
            sigma = torch.exp(logstd)
            _tensor(sigma, (batch_size, 6), self.device, "policy sigma")
            if (sigma <= 0).any().item():
                raise ValueError("policy sigma must remain positive")
            if consume_rng:
                # ModelA2CContinuousLogStd.forward samples even when Player
                # ultimately selects mu. Preserve exactly that RNG effect.
                torch.distributions.Normal(mu, sigma, validate_args=False).sample()
            clipped = torch.clamp(mu, -1.0, 1.0)
            if not self.rms_unchanged():
                raise RuntimeError("RMS changed during inference")
            return PolicyStep(obs.detach().clone(), normalized.detach().clone(),
                              mu.detach().clone(), logstd.detach().clone(),
                              tuple(value.detach().clone() for value in out_state),
                              clipped.detach().clone(), bool(consume_rng))

    def step(self, obs, *, consume_rng=True):
        if self.states is None:
            self.states = self.initial_state(_batch(obs))
        out = self.infer(obs, self.states, consume_rng=consume_rng)
        self.states = out.state
        return {"observation": out.observation, "normalized": out.normalized,
                "mu": out.mu, "logstd": out.logstd, "h": out.h, "c": out.c,
                "clipped": out.clipped_action}


class CommandBoundary:
    """Validate a complete policy step before invoking a robot command callback.

    This is a software admission gate and makes no hardware safety claim. The
    callback keeps the original Factory action/EMA/controller path intact.
    """

    def __init__(self, policy, submit: Callable, *, batch_size=1, admit=None):
        self.policy, self.submit, self.batch_size, self.admit = policy, submit, batch_size, admit
        self.cache_initialized = False
        self.healthy = True

    def initialize_cache(self, observation, cache):
        _tensor(observation, (self.batch_size, 19), self.policy.device, "observation")
        _tensor(cache, (self.batch_size, 6), self.policy.device, "EMA cache")
        if not torch.equal(observation[:, 13:19], cache):
            raise ValueError("EMA cache must equal the original pose-derived observation cache")
        self.cache_initialized = True

    def reset_state(self, state, done):
        self.healthy = False
        self.cache_initialized = False
        STATE_SPEC.validate(state, self.batch_size, device=self.policy.device)
        done = torch.as_tensor(done, device=self.policy.device)
        if done.dtype != torch.bool or done.shape != (self.batch_size,):
            raise ValueError("done must have one bool per actor")
        result = self.policy.reset_state(state, done)
        STATE_SPEC.validate(result, self.batch_size, device=self.policy.device)
        for before, after in zip(state, result):
            if torch.count_nonzero(after[:, done, :]).item() or not torch.equal(before[:, ~done, :], after[:, ~done, :]):
                raise RuntimeError("recurrent reset failed; commands remain blocked")
        self.healthy = True
        return result

    def step(self, obs, state, cache):
        self._validate_input(obs, cache)
        STATE_SPEC.validate(state, self.batch_size, device=self.policy.device)
        result = self.policy.infer(obs, state)
        self.submit_precomputed(result, obs, cache)
        return result

    def _validate_input(self, obs, cache):
        if not self.healthy or not self.cache_initialized:
            raise RuntimeError("reset/cache initialization incomplete; command blocked")
        _tensor(obs, (self.batch_size, 19), self.policy.device, "observation")
        _tensor(cache, (self.batch_size, 6), self.policy.device, "EMA cache")
        if not torch.equal(obs[:, 13:19], cache):
            raise ValueError("EMA cache and observation disagree")

    def submit_precomputed(self, result, obs, cache):
        """Submit a previously traced forward without another RNG-consuming call."""
        self._validate_input(obs, cache)
        _tensor(result.observation, (self.batch_size, 19), self.policy.device, "traced observation")
        _tensor(result.normalized, (self.batch_size, 19), self.policy.device, "normalized observation")
        if not torch.equal(result.observation, obs):
            raise ValueError("traced observation differs from current controller observation")
        _tensor(result.clipped_action, (self.batch_size, 6), self.policy.device, "controller action")
        _tensor(result.mu, (self.batch_size, 6), self.policy.device, "policy mean")
        _tensor(result.logstd, (self.batch_size, 6), self.policy.device, "policy logstd")
        STATE_SPEC.validate(result.state, self.batch_size, device=self.policy.device)
        if (result.clipped_action.abs() > 1).any().item():
            raise ValueError("controller action exceeds original clipping bounds")
        if self.admit is not None and self.admit(result.clipped_action) is not True:
            raise RuntimeError("controller command rejected before submission")
        return self.submit(result.clipped_action)
