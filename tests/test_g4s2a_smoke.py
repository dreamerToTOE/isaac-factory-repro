"""Synthetic CPU admission and bridge tests; never execute a policy or physics.

The fake clock below has prescribed kinematics and does not claim to represent
contact dynamics. It detects ordering, action-transfer and reset-accounting bugs
independently of the MuJoCo rollout and its recorded results.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import g4s1b_contract as contract
from scripts import g4s2a_smoke as smoke


def valid_admission():
    return {
        'bilateral_last10': True,
        'stability_m': .0002,
        'angular_stability_rad': .002,
        'q_limit_violation': 0.,
        'finite': True,
        'independent_peg': True,
        'starts_above_hole': True,
        'interference_bodies': [],
    }


def policy_receipt():
    # Distinct raw and normalized values make accidental normalization visible.
    mean = np.array([1.4, -2., .25, -.3, .1, 0.], dtype=np.float32)
    return {
        'observation': np.linspace(-.7, .9, 19, dtype=np.float32).tolist(),
        'normalized': np.linspace(-4., 3., 19, dtype=np.float32).tolist(),
        'mu': mean.tolist(),
        'clipped': np.clip(mean, -1., 1.).tolist(),
        'h': np.zeros((2, 1, 1024), dtype=np.float32).tolist(),
        'c': np.zeros((2, 1, 1024), dtype=np.float32).tolist(),
        'rms_unchanged': True,
    }


def test_identity_gate_compares_actual_bytes_and_rejects_both_tamper_types(tmp_path, monkeypatch):
    checkpoint = tmp_path/'checkpoint.pth'
    checkpoint.write_bytes(b'independently specified checkpoint fixture')
    xml = '<mujoco model="frozen-fixture"/>\n'
    monkeypatch.setattr(smoke, 'CHECKPOINT_SHA', hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(smoke, 'XML_SHA', hashlib.sha256(xml.encode()).hexdigest())
    smoke.verify_identity(checkpoint, xml)
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises((ValueError, RuntimeError)):
        smoke.verify_identity(checkpoint, xml)
    checkpoint.write_bytes(b'independently specified checkpoint fixture')
    with pytest.raises((ValueError, RuntimeError)):
        smoke.verify_identity(checkpoint, xml+' ')


def test_reset_admission_positive_control_and_fixed_stability_boundaries():
    assert smoke.admission_errors(valid_admission()) == []
    exact = valid_admission()
    exact.update(stability_m=.001, angular_stability_rad=.05)
    assert smoke.admission_errors(exact) == []


@pytest.mark.parametrize('key,value', [
    ('bilateral_last10', False),
    ('stability_m', .001001),
    ('stability_m', float('nan')),
    ('angular_stability_rad', .050001),
    ('angular_stability_rad', float('inf')),
    ('q_limit_violation', .0001),
    ('finite', False),
    ('independent_peg', False),
    ('starts_above_hole', False),
    ('interference_bodies', ['hand']),
    ('interference_bodies', ['hole']),
])
def test_reset_admission_negative_controls_cannot_be_promoted(key, value):
    metrics = valid_admission()
    metrics[key] = value
    assert smoke.admission_errors(metrics), f'{key} was not rejected'


def test_geometric_success_never_substitutes_for_physical_candidate():
    diagnostic = {'physical_insertion_candidate': False, 'min_contact_distance_m': 0.,
                  'official_geometric_success': True}
    assert smoke.contact_quality(diagnostic, 0.) is False


def test_contact_quality_preserves_endpoint_and_whole_episode_ten_micrometer_threshold():
    diagnostic = {'physical_insertion_candidate': True, 'min_contact_distance_m': -1e-5}
    assert smoke.contact_quality(diagnostic, 1e-5) is True
    assert smoke.contact_quality(diagnostic, 1.00001e-5) is False
    diagnostic['min_contact_distance_m'] = -1.00001e-5
    assert smoke.contact_quality(diagnostic, 0.) is False
    diagnostic['min_contact_distance_m'] = float('nan')
    assert smoke.contact_quality(diagnostic, 0.) is False


def test_policy_receipt_is_validation_only_and_does_not_normalize_again():
    receipt = policy_receipt()
    before = deepcopy(receipt)
    smoke.validate_policy_result(receipt)
    assert receipt == before
    assert receipt['observation'] != receipt['normalized']


@pytest.mark.parametrize('field', ['observation', 'normalized', 'mu', 'clipped', 'h', 'c'])
def test_nonfinite_receipt_is_rejected_before_action_transfer(field):
    receipt = policy_receipt()
    value = np.asarray(receipt[field], dtype=np.float32)
    value.flat[0] = np.nan
    receipt[field] = value.tolist()
    with pytest.raises((ValueError, RuntimeError)):
        smoke.validate_policy_result(receipt)


@pytest.mark.parametrize('field', ['observation', 'normalized', 'mu', 'clipped', 'h', 'c'])
def test_wrong_shape_receipt_is_rejected_before_action_transfer(field):
    receipt = policy_receipt()
    receipt[field] = np.asarray(receipt[field]).reshape(-1)[:-1].tolist()
    with pytest.raises((ValueError, RuntimeError)):
        smoke.validate_policy_result(receipt)


def test_changed_rms_or_nondeterministic_clipping_is_rejected():
    receipt = policy_receipt()
    receipt['rms_unchanged'] = False
    with pytest.raises((ValueError, RuntimeError)):
        smoke.validate_policy_result(receipt)
    receipt = policy_receipt()
    receipt['clipped'][2] += .1
    with pytest.raises((ValueError, RuntimeError)):
        smoke.validate_policy_result(receipt)


class PrescribedClock:
    """No simulator: each fake substep advances a prescribed TCP by 0.1 mm."""
    arm_qpos_ids = np.arange(7)
    arm_dof_ids = np.arange(7)
    hole_body_id, peg_body_id = 0, 1

    def __init__(self):
        self.reset_calls = 0
        self.reset(seed=1001)
        # Represent a physically prepared state, which a second reset destroys.
        self.data.time = .25

    def reset(self, seed=0, **kwargs):
        self.reset_calls += 1
        self.rng = np.random.default_rng(seed)
        self.calls = []
        self.data = SimpleNamespace(time=0., qpos=contract.DEFAULT_Q.copy(), qvel=np.zeros(7),
            xpos=np.array([[.6, 0., .05], [.6, 0., .0876]]),
            xquat=np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]))
        return self.snapshot()

    def tcp_state(self):
        return np.array([.6+len(self.calls)*.0001, 0., .122]), contract.quat_from_euler(np.pi, 0., 0.)

    def controller_jacobian(self):
        return np.c_[np.eye(6), np.zeros(6)]

    def arm_mass_matrix(self):
        return np.eye(7)

    def physics_step(self, torque, finger_target=0.):
        self.calls.append({'torque': np.array(torque), 'finger_target': finger_target})
        self.data.time += contract.PHYSICS_DT
        return self.snapshot()

    def snapshot(self):
        return {'time_s': self.data.time, 'peg_root_m': self.data.xpos[self.peg_body_id].tolist()}

    def insertion_diagnostics(self):
        return {'physical_insertion_candidate': False, 'min_contact_distance_m': 0.}


def prepared_runner():
    env = PrescribedClock()
    view = smoke.PreparedResetView(env, admitted=True)
    runner = contract.ReferenceRunner(view)
    reset = runner.reset(seed=1001, observation_noise=np.zeros(3))
    return env, view, runner, reset


def test_prepared_reset_consumes_admission_once_without_teleport_or_clock_reset():
    env, view, runner, reset = prepared_runner()
    assert env.reset_calls == 1
    assert env.data.time == .25
    assert reset['state']['time_s'] == .25
    assert reset['lstm_reset_pending'] is True
    assert reset['ema_cache'][2] == pytest.approx((.122-.075)/.05)
    np.testing.assert_array_equal(reset['observation'][7:13], np.zeros(6))
    with pytest.raises((ValueError, RuntimeError)):
        runner.reset(seed=1001)
    assert env.reset_calls == 1 and env.data.time == .25


def test_unadmitted_reset_or_physical_reset_override_never_reaches_environment():
    env = PrescribedClock()
    with pytest.raises((ValueError, RuntimeError)):
        smoke.PreparedResetView(env, admitted=False).reset(seed=1001)
    assert env.reset_calls == 1
    with pytest.raises((ValueError, RuntimeError)):
        smoke.PreparedResetView(env, admitted=True).reset(seed=1001, arm_qpos=np.zeros(7))
    assert env.reset_calls == 1


def test_first_action_ema_once_then_all_eight_targets_rebuilt_from_current_tcp():
    env, _, runner, reset = prepared_runner()
    before = reset['ema_cache'].copy()
    raw = np.array([2., -2., .3, .4, -.5, .6], dtype=np.float32)
    expected = .8*before+.2*np.clip(raw, -1., 1.)
    result = runner.step_reference_action(raw)
    np.testing.assert_allclose(result['ema_cache'], expected, rtol=0., atol=1e-7)
    assert runner.contract.policy_updates == 1 and len(env.calls) == 8
    assert all(call['finger_target'] == 0. for call in env.calls)
    targets = np.array([sub['target_position_m'] for sub in result['substeps']])
    # Source recomputes targets from the moving TCP, even with a fixed EMA.
    np.testing.assert_allclose(np.diff(targets[:, 0]), .0001, rtol=0., atol=1e-14)
    np.testing.assert_allclose(result['observation'][13:19], expected, rtol=0., atol=1e-7)
    assert result['observation'][7] == pytest.approx(.012)
    assert result['state_is_pre_reset'] is True


def test_149_decision_fake_clock_preserves_terminal_state_and_blocks_extra_step():
    env, _, runner, _ = prepared_runner()
    initial_root = env.data.xpos[env.peg_body_id].copy()
    for index in range(149):
        result = runner.step_reference_action(np.zeros(6))
        assert result['episode_steps'] == index+1
        assert result['state_is_pre_reset'] is True
        assert result['reset_required'] is (index == 148)
    assert len(env.calls) == 149*8 and runner.contract.policy_updates == 149
    assert result['episode_simulation_time_s'] == pytest.approx(149*8/120.)
    np.testing.assert_array_equal(env.data.xpos[env.peg_body_id], initial_root)
    assert env.reset_calls == 1
    with pytest.raises(ValueError, match='reset is required'):
        runner.step_reference_action(np.zeros(6))
    assert len(env.calls) == 149*8


def test_controller_fault_does_not_transfer_action_or_allow_favorable_retry():
    env, _, runner, _ = prepared_runner()
    env.controller_jacobian = lambda: np.zeros((6, 7))
    with pytest.raises(ValueError, match='inversion rejected'):
        runner.step_reference_action(np.ones(6))
    assert env.calls == [] and env.data.time == .25
    assert runner.reset_required is True and runner.last_fault['type'] == 'ValueError'
    with pytest.raises((ValueError, RuntimeError)):
        runner.reset(seed=1002)
    assert env.reset_calls == 1


def test_evidence_digest_gate_detects_tamper_and_missing_artifacts(tmp_path):
    artifact = tmp_path/'physics.jsonl'
    artifact.write_text('{"step": 1}\n')
    manifest = {str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()}
    smoke.verify_artifacts(manifest)
    artifact.write_text('{"step": 2}\n')
    with pytest.raises((ValueError, RuntimeError)):
        smoke.verify_artifacts(manifest)
    artifact.unlink()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        smoke.verify_artifacts(manifest)
