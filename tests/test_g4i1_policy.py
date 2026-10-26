"""Focused CPU failure injection at the standalone command admission boundary."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from scripts.g4i1_policy import CommandBoundary, FrozenPolicy


class MockRMS(torch.nn.Module):
    epsilon = 1e-5
    norm_only = False

    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.arange(19, dtype=torch.float64) / 10)
        self.register_buffer("running_var", torch.arange(1, 20, dtype=torch.float64) / 10)
        self.register_buffer("count", torch.tensor(123.0, dtype=torch.float64))


class MockActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        obs = inputs["obs"]
        h, c = inputs["rnn_states"]
        return obs[:, :6] * 4, torch.zeros_like(obs[:, :6]), obs[:, :1], (h + .1, c + .2)


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.running_mean_std = MockRMS()
        self.a2c_network = MockActor()


@pytest.fixture
def policy():
    return FrozenPolicy(MockModel(), {"fixture": True})


@pytest.fixture
def gate(policy):
    callback = Mock()
    boundary = CommandBoundary(policy, callback)
    obs = torch.linspace(-1, 1, 19).reshape(1, 19)
    cache = obs[:, 13:].contiguous()
    boundary.initialize_cache(obs, cache)
    return boundary, obs, policy.initial_state(), cache, callback


def assert_blocked(gate, obs=None, state=None, cache=None):
    boundary, original_obs, original_state, original_cache, callback = gate
    with pytest.raises((ValueError, RuntimeError)):
        boundary.step(original_obs if obs is None else obs,
                      original_state if state is None else state,
                      original_cache if cache is None else cache)
    callback.assert_not_called()


def test_frozen_normalization_once_and_state_extension(policy):
    obs = torch.linspace(-1, 1, 19).reshape(1, 19)
    state = policy.initial_state()
    before = {key: value.clone() for key, value in policy.model.running_mean_std.state_dict().items()}
    actual = policy.infer(obs, state)
    rms = policy.model.running_mean_std
    expected = ((obs - rms.running_mean.float()) / (rms.running_var.float() + 1e-5).sqrt()).clamp(-5, 5)
    torch.testing.assert_close(actual.normalized, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.mu, expected[:, :6] * 4, rtol=0, atol=0)
    assert actual.clipped_action.abs().max() <= 1
    assert torch.count_nonzero(state[0]) == 0
    assert torch.all(actual.h == .1) and torch.all(actual.c == .2)
    assert all(torch.equal(before[key], value) for key, value in rms.state_dict().items())
    assert not actual.mu.requires_grad


def test_discarded_sample_matches_official_rng_consumption(policy):
    obs = torch.linspace(-1, 1, 19).reshape(1, 19)
    torch.manual_seed(1001)
    initial = torch.get_rng_state()
    out = policy.infer(obs, policy.initial_state())
    after_actual = torch.get_rng_state()
    torch.set_rng_state(initial)
    torch.distributions.Normal(out.mu, out.logstd.exp(), validate_args=False).sample()
    assert torch.equal(after_actual, torch.get_rng_state())
    torch.set_rng_state(initial)
    policy.infer(obs, policy.initial_state(), consume_rng=False)
    assert torch.equal(initial, torch.get_rng_state())


def test_boundary_submits_only_valid_original_clipped_action(gate):
    boundary, obs, state, cache, callback = gate
    result = boundary.step(obs, state, cache)
    callback.assert_called_once()
    assert torch.equal(callback.call_args.args[0], result.clipped_action)


def test_precomputed_bad_action_rejects_before_controller_command(gate):
    boundary, obs, state, cache, callback = gate
    result = boundary.policy.infer(obs, state)
    bad = replace(result, clipped_action=torch.full((1, 6), float("nan")))
    with pytest.raises(ValueError, match="NaN/Inf"):
        boundary.submit_precomputed(bad, obs, cache)
    callback.assert_not_called()


def test_precomputed_valid_action_submits_without_second_forward_or_sample(gate, monkeypatch):
    boundary, obs, state, cache, callback = gate
    result = boundary.policy.infer(obs, state)
    forward = Mock(side_effect=RuntimeError("must not forward twice"))
    monkeypatch.setattr(boundary.policy, "infer", forward)
    before = torch.get_rng_state()
    boundary.submit_precomputed(result, obs, cache)
    assert torch.equal(before, torch.get_rng_state())
    callback.assert_called_once()
    forward.assert_not_called()


@pytest.mark.parametrize("shape", [(19,), (1, 18), (1, 20), (0, 19), (1, 1, 19)])
def test_invalid_observation_shape_rejects_without_command(gate, shape):
    assert_blocked(gate, obs=torch.zeros(shape))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_observation_rejects_without_command(gate, bad):
    obs = gate[1].clone()
    obs[0, 0] = bad
    assert_blocked(gate, obs=obs)


def test_wrong_device_rejects_without_reading_meta_or_command(gate):
    assert_blocked(gate, obs=torch.empty((1, 19), device="meta"))


def test_wrong_dtype_rejects_without_command(gate):
    assert_blocked(gate, obs=gate[1].double())


@pytest.mark.parametrize("shape", [(1, 1, 1024), (2, 2, 1024), (2, 1, 1023)])
def test_incorrect_lstm_dimensions_reject_without_command(gate, shape):
    assert_blocked(gate, state=(torch.zeros(shape), torch.zeros(shape)))


def test_state_wrong_device_rejects_without_command(gate):
    assert_blocked(gate, state=(torch.empty((2, 1, 1024), device="meta"),) * 2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_state_nonfinite_rejects_without_command(gate, bad):
    state = tuple(t.clone() for t in gate[2])
    state[0][0, 0, 0] = bad
    assert_blocked(gate, state=state)


@pytest.mark.parametrize("shape", [(6,), (1, 5), (2, 6)])
def test_invalid_action_shape_rejects_before_submission(gate, monkeypatch, shape):
    boundary, obs, state, cache, _ = gate
    result = boundary.policy.infer(obs, state)
    monkeypatch.setattr(boundary.policy, "infer", lambda *a, **k: replace(result, clipped_action=torch.zeros(shape)))
    assert_blocked(gate)


@pytest.mark.parametrize("field", ["mu", "logstd", "clipped_action"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_outputs_reject_before_submission(gate, monkeypatch, field, bad):
    boundary, obs, state, cache, _ = gate
    result = boundary.policy.infer(obs, state)
    broken = getattr(result, field).clone()
    broken[0, 0] = bad
    monkeypatch.setattr(boundary.policy, "infer", lambda *a, **k: replace(result, **{field: broken}))
    assert_blocked(gate)


def test_actual_actor_nan_rejected_before_normal_sampling_and_command(gate, monkeypatch):
    policy = gate[0].policy
    original = policy.model.a2c_network.forward
    def broken(inputs):
        mu, logstd, value, states = original(inputs)
        mu[0, 0] = float("nan")
        return mu, logstd, value, states
    monkeypatch.setattr(policy.model.a2c_network, "forward", broken)
    assert_blocked(gate)


def test_logstd_overflow_rejects_without_command(gate, monkeypatch):
    policy = gate[0].policy
    original = policy.model.a2c_network.forward
    def broken(inputs):
        mu, logstd, value, states = original(inputs)
        return mu, logstd + 1000, value, states
    monkeypatch.setattr(policy.model.a2c_network, "forward", broken)
    assert_blocked(gate)


def test_pose_derived_cache_not_replaced_by_zero(gate):
    boundary, obs, state, cache, callback = gate
    boundary.cache_initialized = False
    with pytest.raises(ValueError, match="pose-derived"):
        boundary.initialize_cache(obs, torch.zeros_like(cache))
    assert_blocked(gate)


def test_cache_may_exceed_one_but_must_match_observation(gate):
    boundary, obs, state, cache, callback = gate
    obs[:, 13:] = 2.5
    cache = torch.full((1, 6), 2.5)
    boundary.initialize_cache(obs, cache)
    boundary.step(obs, state, cache)
    callback.assert_called_once()


def test_stale_cache_rejects_without_command(gate):
    assert_blocked(gate, cache=torch.zeros((1, 6)))


def test_successful_reset_zeros_only_done_actor(policy):
    state = (torch.ones((2, 2, 1024)), torch.full((2, 2, 1024), 2.0))
    result = policy.reset_state(state, torch.tensor([True, False]))
    for before, after in zip(state, result):
        assert torch.count_nonzero(after[:, 0]) == 0
        assert torch.equal(before[:, 1], after[:, 1])
        assert torch.count_nonzero(before[:, 0]) > 0


def test_failed_recurrent_reset_latches_boundary_and_blocks_command(gate, monkeypatch):
    boundary, obs, state, cache, callback = gate
    nonzero = tuple(torch.ones_like(t) for t in state)
    monkeypatch.setattr(boundary.policy, "reset_state", lambda state, done: state)
    with pytest.raises(RuntimeError, match="reset failed"):
        boundary.reset_state(nonzero, torch.tensor([True]))
    assert not boundary.healthy
    assert_blocked(gate)


def test_reset_requires_fresh_pose_cache_before_next_command(gate):
    boundary, obs, state, cache, callback = gate
    cleared = boundary.reset_state(state, torch.tensor([True]))
    assert_blocked(gate, state=cleared)
    boundary.initialize_cache(obs, cache)
    boundary.step(obs, cleared, cache)
    callback.assert_called_once()


def test_controller_admission_rejection_submits_no_command(gate):
    gate[0].admit = lambda action: False
    assert_blocked(gate)


def test_mutated_rms_rejects_before_command(gate):
    gate[0].policy.model.running_mean_std.running_mean[0] += .1
    assert_blocked(gate)


def test_training_mode_rejects_before_command(gate):
    gate[0].policy.model.train()
    assert_blocked(gate)


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "nan", "count", "epsilon"])
def test_incompatible_or_missing_rms_prevents_policy_and_commands(change):
    model, callback = MockModel(), Mock()
    if change == "missing":
        del model.running_mean_std.running_mean
    elif change == "shape":
        model.running_mean_std.running_mean = torch.zeros(18, dtype=torch.float64)
    elif change == "dtype":
        model.running_mean_std.running_mean = torch.zeros(19)
    elif change == "nan":
        model.running_mean_std.running_var[0] = float("nan")
    elif change == "count":
        model.running_mean_std.count.fill_(0)
    elif change == "epsilon":
        model.running_mean_std.epsilon = 1e-3
    with pytest.raises((AttributeError, ValueError)):
        policy = FrozenPolicy(model, {})
        CommandBoundary(policy, callback).step(torch.zeros((1, 19)), policy.initial_state(), torch.zeros((1, 6)))
    callback.assert_not_called()


def test_wrong_checkpoint_sha_never_deserializes_or_submits(tmp_path, monkeypatch):
    path = tmp_path / "not_a_checkpoint.pth"
    path.write_bytes(b"untrusted-input-never-deserialize")
    load, callback = Mock(), Mock()
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="SHA-256"):
        policy = FrozenPolicy.from_checkpoint(path)
        CommandBoundary(policy, callback).step(torch.zeros((1, 19)), policy.initial_state(), torch.zeros((1, 6)))
    load.assert_not_called()
    callback.assert_not_called()


def test_unregistered_checkpoint_role_is_rejected_without_loading(monkeypatch):
    load = Mock()
    monkeypatch.setattr(torch, "load", load)
    with pytest.raises(ValueError, match="allowlisted"):
        FrozenPolicy.from_checkpoint(role="best_new_model")
    load.assert_not_called()
