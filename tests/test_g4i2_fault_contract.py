"""CPU synthetic fault controls composed from the reviewed software gates.

No GPU, simulator, wall-clock sleep or robot is used. Existing G4-I1 fixtures
supply the actor and command boundary; this file does not reimplement a policy.
Set G4I2_FAULT_EVIDENCE to a NEW JSON path to preserve machine-readable results.
Such evidence proves these mocked software tests, not runtime fault injection or
real hardware safety. No hold/last-command hardware behavior is implemented.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from scripts.g4i1_policy import FrozenPolicy
from scripts.g4i2_performance import DeadlineGuard
from test_g4i1_policy import gate, policy  # noqa: F401; reuse original pytest fixtures


FAULT_CASES = (
    'delayed_inference_reject', 'nan_observation', 'nonfinite_policy_output',
    'incorrect_action_dimensions', 'stale_observation', 'incorrect_lstm_state',
    'controller_command_rejection', 'checkpoint_sha_mismatch',
)


class SyntheticMonotonicClock:
    """Manually advanced fake time; never a latency measurement."""

    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def advance_ms(self, milliseconds):
        self.ns += int(milliseconds * 1_000_000)


@pytest.fixture(scope='session')
def fault_evidence():
    rows = []
    yield rows
    output = os.environ.get('G4I2_FAULT_EVIDENCE')
    if output:
        payload = {
            'schema': 'g4i2.cpu_synthetic_fault_controls.v1',
            'scope': 'CPU synthetic composed software gates; no GPU/simulator/hardware fault execution',
            'clock': 'manually advanced synthetic monotonic nanoseconds; not measured latency',
            'hold_implemented': False,
            'hardware_safety_certification': False,
            'expected_case_count': len(FAULT_CASES) + 3,
            'recorded_case_count': len(rows),
            'status': 'COMPLETE' if len(rows) == len(FAULT_CASES) + 3 else 'INCOMPLETE',
            'cases': rows,
        }
        path = Path(output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Evidence is append-once by unique path; never overwrite prior output.
        with path.open('x') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')


def record(rows, case, callback, *, reason, **details):
    assert all(row['case'] != case for row in rows)
    rows.append({'case': case, 'passed': True, 'submit_count': callback.call_count,
                 'reason': reason, 'hold_implemented': False,
                 'scope': 'CPU synthetic software contract', **details})


def with_fake_inference_delay(boundary, monkeypatch, clock, delay_ms):
    original = boundary.policy.infer
    def infer(*args, **kwargs):
        out = original(*args, **kwargs)
        clock.advance_ms(delay_ms)
        return out
    monkeypatch.setattr(boundary.policy, 'infer', infer)


@pytest.mark.parametrize('case', FAULT_CASES)
def test_fault_contract_never_submits_rejected_command(case, gate, monkeypatch,
                                                      tmp_path, fault_evidence):
    boundary, obs, state, cache, callback = gate
    clock = SyntheticMonotonicClock()
    guard = None
    extra = {}
    if case == 'delayed_inference_reject':
        guard = DeadlineGuard(deadline_ms=10, max_observation_age_ms=100, mode='reject')
        with_fake_inference_delay(boundary, monkeypatch, clock, 11)
        boundary.admit = lambda action: guard.check(
            observation_ns=0, inference_started_ns=0, now_ns=clock()).allowed
        operation = lambda: boundary.step(obs, state, cache)
        message = 'controller command rejected'
    elif case == 'nan_observation':
        obs[0, 0] = float('nan')
        operation = lambda: boundary.step(obs, state, cache)
        message = 'NaN/Inf'
    elif case == 'nonfinite_policy_output':
        valid = boundary.policy.infer(obs, state)
        bad = valid.clipped_action.clone()
        bad[0, 0] = float('nan')
        monkeypatch.setattr(boundary.policy, 'infer', lambda *a, **k: replace(valid, clipped_action=bad))
        operation = lambda: boundary.step(obs, state, cache)
        message = 'NaN/Inf'
    elif case == 'incorrect_action_dimensions':
        valid = boundary.policy.infer(obs, state)
        monkeypatch.setattr(boundary.policy, 'infer', lambda *a, **k: replace(valid, clipped_action=torch.zeros(1, 5)))
        operation = lambda: boundary.step(obs, state, cache)
        message = 'shape'
    elif case == 'stale_observation':
        clock.advance_ms(31)
        guard = DeadlineGuard(deadline_ms=10, max_observation_age_ms=20, mode='log_only')
        boundary.admit = lambda action: guard.check(
            observation_ns=0, inference_started_ns=clock(), now_ns=clock()).allowed
        operation = lambda: boundary.step(obs, state, cache)
        message = 'controller command rejected'
    elif case == 'incorrect_lstm_state':
        wrong_state = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
        operation = lambda: boundary.step(obs, wrong_state, cache)
        message = 'recurrent shape mismatch'
    elif case == 'controller_command_rejection':
        boundary.admit = lambda action: False
        operation = lambda: boundary.step(obs, state, cache)
        message = 'controller command rejected'
    else:
        path = tmp_path / 'synthetic_invalid_checkpoint.pth'
        path.write_bytes(b'synthetic hash mismatch; never deserialize')
        load = Mock()
        monkeypatch.setattr(torch, 'load', load)
        operation = lambda: FrozenPolicy.from_checkpoint(path)
        message = 'SHA-256'
    with pytest.raises((ValueError, RuntimeError), match=message) as error:
        operation()
    callback.assert_not_called()
    if guard is not None:
        assert len(guard.decisions) == 1
        decision = guard.decisions[0]
        assert not decision['allowed']
        assert decision['reason'] == ('INFERENCE_DEADLINE_MISSED' if case == 'delayed_inference_reject' else 'STALE_OBSERVATION')
        extra['deadline_decision'] = decision
    if case == 'checkpoint_sha_mismatch':
        load.assert_not_called()
        extra['checkpoint_deserialization_count'] = load.call_count
    record(fault_evidence, case, callback, reason=str(error.value), **extra)


def test_delayed_fresh_inference_log_only_submits_original_action(gate, monkeypatch,
                                                                fault_evidence):
    boundary, obs, state, cache, callback = gate
    clock = SyntheticMonotonicClock()
    guard = DeadlineGuard(deadline_ms=10, max_observation_age_ms=100, mode='log_only')
    with_fake_inference_delay(boundary, monkeypatch, clock, 11)
    boundary.admit = lambda action: guard.check(
        observation_ns=0, inference_started_ns=0, now_ns=clock()).allowed
    result = boundary.step(obs, state, cache)
    callback.assert_called_once()
    assert torch.equal(callback.call_args.args[0], result.clipped_action)
    decision = guard.decisions[0]
    assert decision['deadline_missed'] and decision['allowed']
    assert decision['action'] == 'log_only' and not decision['stop_requested']
    record(fault_evidence, 'delayed_inference_log_only', callback,
           reason=decision['reason'], deadline_decision=decision)


def test_delayed_inference_stop_latch_blocks_later_fresh_command(gate, monkeypatch,
                                                               fault_evidence):
    boundary, obs, state, cache, callback = gate
    clock = SyntheticMonotonicClock()
    guard = DeadlineGuard(deadline_ms=10, max_observation_age_ms=100, mode='stop')
    with_fake_inference_delay(boundary, monkeypatch, clock, 11)
    boundary.admit = lambda action: guard.check(
        observation_ns=0, inference_started_ns=0, now_ns=clock()).allowed
    with pytest.raises(RuntimeError, match='controller command rejected'):
        boundary.step(obs, state, cache)
    callback.assert_not_called()
    assert guard.stopped
    boundary.admit = lambda action: guard.check(
        observation_ns=clock(), inference_started_ns=clock(), now_ns=clock()).allowed
    with pytest.raises(RuntimeError, match='controller command rejected'):
        boundary.step(obs, state, cache)
    callback.assert_not_called()
    assert guard.decisions[-1]['reason'] == 'STOP_LATCHED'
    assert all(row['stop_requested'] for row in guard.decisions)
    record(fault_evidence, 'deadline_stop_latch', callback,
           reason='STOP_LATCHED', deadline_decisions=guard.decisions,
           actual_simulator_stop_performed=False)


def test_hold_not_implemented_or_claimed(gate, fault_evidence):
    callback = gate[-1]
    with pytest.raises(ValueError, match='unknown deadline policy'):
        DeadlineGuard(mode='hold')
    callback.assert_not_called()
    record(fault_evidence, 'hold_not_implemented', callback,
           reason='HOLD_NOT_IMPLEMENTED; rejection is no new submission, not physical hold')
