"""CPU contract and failure-injection tests; no policy forward or simulator."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.g4i0_interface import (
    OBSERVATION_ORDER, RNNStateSpec, action_targets, clip_policy_action,
    normalize_observation, smooth_action, validate_checkpoint_compatibility,
    validate_observation, validate_policy_output, validate_rms, verify_checkpoint_identity,
)


def rms():
    return np.zeros(19, np.float64), np.ones(19, np.float64), np.array(1., np.float64)


def test_order_and_spec_slices_reconstruct_distinct_features():
    spec = json.loads((Path(__file__).parents[1] / "evidence/g4_i/i0/interface_spec.json").read_text())
    assert tuple(v["name"] for v in spec["input"]["ordering"]) == OBSERVATION_ORDER
    parts = [np.full(n, i, np.float32) for i, n in enumerate([3, 4, 3, 3, 6])]
    obs = validate_observation(np.concatenate(parts))
    for part, entry in zip(parts, spec["input"]["ordering"]):
        start, end = map(int, entry["index"].split(":"))
        np.testing.assert_array_equal(obs[start:end], part)


@pytest.mark.parametrize("value", [np.zeros(18, np.float32), np.zeros((1, 1, 19), np.float32), np.zeros((0, 19), np.float32)])
def test_bad_observation_shape(value):
    with pytest.raises(ValueError, match="shape|batch"):
        validate_observation(value)


@pytest.mark.parametrize("value", [np.zeros(19, np.int32), torch.zeros(19, dtype=torch.float64)])
def test_bad_runtime_dtype(value):
    with pytest.raises(ValueError, match="dtype"):
        validate_observation(value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rejected_at_inputs_outputs_and_controller(bad):
    obs, action, pos = torch.zeros(19), torch.zeros(6), torch.zeros(3)
    obs[0] = action[0] = pos[0] = bad
    for call in (lambda: validate_observation(obs), lambda: validate_policy_output(action),
                 lambda: action_targets(torch.zeros(6), pos, torch.zeros(3))):
        with pytest.raises(ValueError, match="NaN/Inf"):
            call()


def test_normalization_casts_float64_metadata_and_clips_without_mutation():
    mean, var, count = rms()
    mean[:] = 0.123456789
    var[:] = 0.00001
    obs = torch.linspace(-1., 1., 19).repeat(2, 1)
    got = normalize_observation(obs, mean, var)
    expected = ((obs - torch.from_numpy(mean).float()) / torch.sqrt(torch.from_numpy(var).float() + 1e-5)).clamp(-5, 5)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert got.min() == -5 and got.max() == 5 and count == 1
    np.testing.assert_allclose(normalize_observation(obs.numpy(), mean, var), expected.numpy(), atol=1e-6)
    assert mean.dtype == var.dtype == np.float64


@pytest.mark.parametrize("kind", ["length", "rank", "negative", "nan", "dtype", "count_zero", "count_nan", "count_rank"])
def test_invalid_rms(kind):
    mean, var, count = rms()
    if kind == "length": mean = mean[:-1]
    if kind == "rank": mean = mean[None, :]
    if kind == "negative": var[0] = -1
    if kind == "nan": var[0] = np.nan
    if kind == "dtype": mean = mean.astype(np.float32)
    if kind == "count_zero": count[...] = 0
    if kind == "count_nan": count[...] = np.nan
    if kind == "count_rank": count = count[None]
    with pytest.raises(ValueError):
        validate_rms(mean, var, count)


@pytest.mark.parametrize("kwargs", [{"epsilon": 0}, {"epsilon": 1e-8}, {"clip": None}, {"clip": 10}])
def test_normalization_cannot_silently_change_frozen_settings(kwargs):
    with pytest.raises(ValueError, match="frozen"):
        normalize_observation(np.zeros(19, np.float32), *rms()[:2], **kwargs)


def test_lstm_reset_preserves_unfinished_columns_and_input():
    spec = RNNStateSpec()
    h, c = spec.initialize(3)
    assert h.shape == c.shape == (2, 3, 1024) and h.dtype == torch.float32
    assert not h.any() and not c.any()
    h.fill_(2); c.fill_(-3)
    out_h, out_c = spec.reset((h, c), [False, True, False])
    assert not out_h[:, 1].any() and not out_c[:, 1].any()
    torch.testing.assert_close(out_h[:, [0, 2]], h[:, [0, 2]])
    torch.testing.assert_close(out_c[:, [0, 2]], c[:, [0, 2]])
    assert h[:, 1].sum() == 4096  # reset did not mutate the supplied state


@pytest.mark.parametrize("kind", ["layers", "hidden", "batch", "dtype", "nan", "mask_length", "mask_dtype", "state_arity"])
def test_lstm_contract_violations(kind):
    spec = RNNStateSpec(); h, c = spec.initialize(2); mask = [False, True]
    if kind == "layers": h = h[:1]
    if kind == "hidden": c = c[..., :-1]
    if kind == "batch": h = h[:, :1]
    if kind == "dtype": c = c.double()
    if kind == "nan": c[0, 0, 0] = float("nan")
    if kind == "mask_length": mask = [True]
    if kind == "mask_dtype": mask = [0, 1]
    with pytest.raises(ValueError):
        spec.reset((h,) if kind == "state_arity" else (h, c), mask)


def test_mock_policy_boundary_and_deterministic_replay():
    # A transparent mock checks orchestration, not trained-policy correctness.
    def mock_policy(obs, state):
        obs = validate_observation(obs)
        RNNStateSpec().validate(state, obs.shape[0], device=obs.device)
        return validate_policy_output(obs[:, :6].clone(), batch_size=obs.shape[0])
    obs = torch.arange(38, dtype=torch.float32).reshape(2, 19)
    state = RNNStateSpec().initialize(2)
    torch.testing.assert_close(mock_policy(obs, state), mock_policy(obs, state), atol=0, rtol=0)
    with pytest.raises(ValueError, match="batch"):
        validate_policy_output(torch.zeros(1, 6), batch_size=2)
    with pytest.raises(ValueError, match="shape"):
        validate_policy_output(torch.zeros(2, 7))


def test_clip_ema_scale_and_saturation_are_distinct_stages():
    raw = torch.tensor([2., -2., .5, 1., -1., 0.])
    clipped = clip_policy_action(raw)
    smoothed = smooth_action(clipped, torch.zeros(6))
    torch.testing.assert_close(smoothed, torch.tensor([.2, -.2, .1, .2, -.2, 0.]))
    target = action_targets(smoothed, torch.zeros(3), torch.zeros(3))
    torch.testing.assert_close(target["position_target_m"], torch.tensor([.004, -.004, .002]))
    torch.testing.assert_close(target["axis_angle_delta_rad"], torch.tensor([.0194, -.0194, 0.]))
    out = action_targets(torch.ones(6), torch.ones(3), torch.zeros(3))
    torch.testing.assert_close(out["position_target_m"], torch.full((3,), .05))


def test_reset_cache_is_not_clipped_after_ema():
    cache = torch.full((6,), 2.)
    smoothed = smooth_action(torch.zeros(6), cache)
    out = action_targets(smoothed, torch.zeros(3), torch.zeros(3))
    torch.testing.assert_close(out["position_target_m"], torch.full((3,), .032))
    torch.testing.assert_close(out["axis_angle_delta_rad"], torch.full((3,), .1552))


def test_no_accidental_batch_broadcast_or_mixed_backend():
    with pytest.raises(ValueError, match="mismatch"):
        smooth_action(torch.zeros(2, 6), torch.zeros(6))
    with pytest.raises(ValueError, match="mismatch"):
        action_targets(torch.zeros(2, 6), torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match="mismatch"):
        smooth_action(torch.zeros(6), np.zeros(6, np.float32))


def test_simulation_clock_and_substep_target_regeneration():
    spec = json.loads((Path(__file__).parents[1] / "evidence/g4_i/i0/interface_spec.json").read_text())
    timing = spec["timing"]
    step_dt = timing["physics_dt_seconds"] * timing["decimation"]
    assert step_dt == pytest.approx(1 / 15)
    assert math.ceil(10 / step_dt) == 150
    assert timing["actual_timeout_policy_steps"] == 149
    assert timing["actual_timeout_sim_seconds"] == pytest.approx(149 * step_dt)
    # One policy cache, eight feedback samples: no physics or wall-time claim.
    cache = torch.ones(6); frame = torch.zeros(3)
    positions = [torch.full((3,), i * .001) for i in range(timing["decimation"])]
    targets = [action_targets(cache, pos, frame)["position_target_m"] for pos in positions]
    torch.testing.assert_close(targets[0], torch.full((3,), .02))
    torch.testing.assert_close(targets[-1], torch.full((3,), .027))
    assert torch.equal(cache, torch.ones(6))  # cache held, target not held


@pytest.mark.parametrize("kind", ["obs_nan", "obs_shape", "state_batch", "output_inf", "output_shape"])
def test_invalid_boundary_submits_no_mock_command(kind):
    commands = []
    obs = torch.zeros(1, 19); state = RNNStateSpec().initialize(1)
    output = torch.zeros(1, 6)
    if kind == "obs_nan": obs[0, 0] = float("nan")
    if kind == "obs_shape": obs = torch.zeros(1, 43)
    if kind == "state_batch": state = RNNStateSpec().initialize(2)
    if kind == "output_inf": output[0, 0] = float("inf")
    if kind == "output_shape": output = torch.zeros(1, 7)
    # Audit mock only: an invalid boundary must raise before controller submit.
    with pytest.raises(ValueError):
        validate_observation(obs)
        RNNStateSpec().validate(state, obs.shape[0], device=obs.device)
        validate_policy_output(output, batch_size=obs.shape[0])
        commands.append(clip_policy_action(output))
    assert commands == []


def test_checkpoint_hash_rejected_before_any_deserialization(tmp_path):
    path = tmp_path / "checkpoint.pth"; path.write_bytes(b"trusted fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_checkpoint_identity(path, digest) == digest
    path.write_bytes(b"altered fixture")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_checkpoint_identity(path, digest)
    with pytest.raises(ValueError, match="invalid SHA"):
        verify_checkpoint_identity(path, "unknown")
    with pytest.raises(FileNotFoundError):
        verify_checkpoint_identity(tmp_path / "missing.pth", digest)


def fixture_checkpoint():
    state = {"running_mean_std." + key: torch.as_tensor(v) for key, v in zip(("running_mean", "running_var", "count"), rms())}
    state["a2c_network.mu.weight"] = torch.zeros(6, 2)
    shapes = {key: list(v.shape) for key, v in state.items()}
    return {"model": state, "epoch": 200, "frame": 1638400}, shapes


@pytest.mark.parametrize("kind", ["valid", "missing_model", "missing_key", "extra_key", "shape", "dtype", "epoch", "frame", "nonfinite", "bad_rms"])
def test_checkpoint_compatibility_failures(kind):
    ckpt, shapes = fixture_checkpoint()
    if kind == "missing_model": del ckpt["model"]
    elif kind == "missing_key": del ckpt["model"]["a2c_network.mu.weight"]
    elif kind == "extra_key": ckpt["model"]["extra"] = torch.zeros(1)
    elif kind == "shape": ckpt["model"]["a2c_network.mu.weight"] = torch.zeros(7, 2)
    elif kind == "dtype": ckpt["model"]["a2c_network.mu.weight"] = torch.zeros(6, 2).double()
    elif kind == "epoch": ckpt["epoch"] = 400
    elif kind == "frame": ckpt["frame"] = 0
    elif kind == "nonfinite": ckpt["model"]["a2c_network.mu.weight"][0, 0] = float("inf")
    elif kind == "bad_rms": ckpt["model"]["running_mean_std.running_var"][0] = -1
    if kind == "valid":
        assert validate_checkpoint_compatibility(ckpt, shapes, epoch=200, frame=1638400) is ckpt["model"]
    else:
        with pytest.raises(ValueError):
            validate_checkpoint_compatibility(ckpt, shapes, epoch=200, frame=1638400)
