"""CPU bridge parity on synthetic observations only; no simulator or rollout.

The integration test executes five actor forwards: three in a fresh subprocess
and two through the previously validated direct FrozenPolicy interface. The
checkpoint is immutable and no physical environment is constructed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from scripts.g4i1_policy import FrozenPolicy
from scripts.g4s2a_inference import BridgeError, InferenceClient, finite_vector
from scripts.g4s2a_smoke import CHECKPOINT, CHECKPOINT_SHA, ROOT


@pytest.mark.parametrize('value', [
    [0.]*18,
    [0.]*20,
    [[0.]*19],
    [True]+[0.]*18,
    [None]+[0.]*18,
    ['0']+[0.]*18,
    [float('nan')]+[0.]*18,
    [float('inf')]+[0.]*18,
    [1e50]+[0.]*18,
])
def test_bridge_observation_rejection_requires_no_actor(value):
    with pytest.raises(ValueError):
        finite_vector(value, 19, 'observation', fp32=True)


def test_fp32_transport_bytes_match_independent_numpy_rounding():
    values = np.linspace(-.1234567890123, .9876543210123, 19, dtype=np.float64)
    got = finite_vector(values, 19, 'observation', fp32=True)
    np.testing.assert_array_equal(np.array(got, dtype=np.float32), values.astype(np.float32))


def assert_receipt_matches_direct(receipt, direct):
    for key, tensor in [('observation', direct.observation), ('normalized', direct.normalized),
                        ('mu', direct.mu), ('logstd', direct.logstd),
                        ('clipped', direct.clipped_action)]:
        np.testing.assert_array_equal(np.asarray(receipt[key], dtype=np.float32), tensor[0].numpy())
    np.testing.assert_array_equal(np.asarray(receipt['h'], dtype=np.float32), direct.h.numpy())
    np.testing.assert_array_equal(np.asarray(receipt['c'], dtype=np.float32), direct.c.numpy())
    assert receipt['rms_unchanged'] is True
    assert receipt['consumed_discarded_sample'] is True


def test_real_frozen_bridge_matches_direct_once_normalized_recurrent_sequence_and_reset(tmp_path, monkeypatch):
    if not CHECKPOINT.is_file():
        pytest.skip('local allowlisted checkpoint unavailable; synthetic bridge checks still run')
    # Keep direct and child CPU arithmetic configuration explicit and identical.
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy = FrozenPolicy.from_checkpoint(CHECKPOINT, repo=ROOT, role='g2_epoch200_primary', device='cpu')
        rms_before = {key: value.clone() for key, value in policy.model.running_mean_std.state_dict().items()}
        archived = json.loads((ROOT/'evidence/g4_i/i0/checkpoint_audit.json').read_text())
        rms = archived['checkpoints']['g2_epoch200_primary']['rms']
        mean = torch.tensor(rms['running_mean'], dtype=torch.float32)
        variance = torch.tensor(rms['running_var'], dtype=torch.float32)
        first_obs = torch.linspace(-.6, .7, 19, dtype=torch.float32).reshape(1, 19)
        second_obs = torch.linspace(.9, -.8, 19, dtype=torch.float32).reshape(1, 19)
        first_state = policy.initial_state(1)
        assert all(torch.count_nonzero(value).item() == 0 for value in first_state)
        torch.manual_seed(1001)
        direct_first = policy.infer(first_obs, first_state, consume_rng=True)
        direct_second = policy.infer(second_obs, direct_first.state, consume_rng=True)
        expected_normalized = ((first_obs-mean)/(variance+1e-5).sqrt()).clamp(-5., 5.)
        torch.testing.assert_close(direct_first.normalized, expected_normalized, rtol=0., atol=0.)
        assert all(torch.equal(value, rms_before[key])
                   for key, value in policy.model.running_mean_std.state_dict().items())
        assert not policy.model.training and policy.rms_unchanged()

        with InferenceClient(sys.executable, CHECKPOINT, repo=ROOT,
                             stderr_path=tmp_path/'actor.stderr.log', timeout_s=60.) as bridge:
            assert bridge.ready['info']['checkpoint_sha256'] == CHECKPOINT_SHA
            assert bridge.ready['info']['infer_count'] == 0
            with pytest.raises(BridgeError, match='reset is required'):
                bridge.infer(first_obs[0])
            reset = bridge.reset(1001)
            assert np.count_nonzero(reset['h']) == 0 and np.count_nonzero(reset['c']) == 0
            first = bridge.infer(first_obs[0])
            assert_receipt_matches_direct(first, direct_first)
            second = bridge.infer(second_obs[0])
            assert_receipt_matches_direct(second, direct_second)
            assert first['infer_count'] == 1 and second['infer_count'] == 2
            assert np.count_nonzero(first['h']) > 0 and np.count_nonzero(first['c']) > 0
            # A rejected local input cannot advance the subprocess h/c or RNG.
            with pytest.raises(ValueError):
                bridge.infer([float('nan')]+[0.]*18)
            assert bridge.info()['infer_count'] == 2
            repeated_reset = bridge.reset(1001)
            assert repeated_reset['reset_index'] == 2
            assert np.count_nonzero(repeated_reset['h']) == 0 and np.count_nonzero(repeated_reset['c']) == 0
            repeated_first = bridge.infer(first_obs[0])
            assert_receipt_matches_direct(repeated_first, direct_first)
            assert repeated_first['infer_index'] == 1 and repeated_first['infer_count'] == 3
            info = bridge.info()
            assert info['rms_unchanged'] and info['eval_mode'] and info['all_parameters_frozen']
            assert info['infer_count'] == 3 and info['reset_count'] == 2
        assert bridge.returncode == 0
        assert bridge.close()['child_returncode'] == 0
    finally:
        torch.set_num_threads(old_threads)
