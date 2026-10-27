"""Reference-generator tests, intentionally without any MuJoCo integration."""
import pytest
from scripts.g4s2a_gripper_v1_r1_schedule import (
    advance, PreparationState, INITIAL_OPENING_M, MAX_TARGET_DELTA_M,
    MAX_TARGET_RATE_M_S, MAX_PREPARATION_STEPS, POSITIVE_CONTACT_N,
    ACQUISITION_BILATERAL_STEPS, HOLD_BILATERAL_STEPS,
)


def test_bounded_ramp_and_original_tendon_coordinate():
    state = PreparationState()
    for i in range(100):
        old = state.target_m
        state, record = advance(state, 0, 0)
        assert old - state.target_m == pytest.approx(MAX_TARGET_DELTA_M, abs=1e-18)
        assert record['tendon_target_m'] == 2 * state.target_m
        assert abs(record['target_rate_m_s']) <= MAX_TARGET_RATE_M_S + 1e-15
        assert state.target_m == pytest.approx(INITIAL_OPENING_M - (i + 1) * MAX_TARGET_DELTA_M)


def test_first_unilateral_contact_freezes_without_resumption():
    state, _ = advance(PreparationState(), 0, 0)
    target = state.target_m
    state, record = advance(state, POSITIVE_CONTACT_N * 2, 0)
    assert state.phase == 'CONTACT_ACQUISITION'
    assert state.target_m == target and record['target_delta_m'] == 0
    for _ in range(20):
        state, _ = advance(state, 0, 0)
        assert state.target_m == target


def test_hold_is_additional_and_requires_continuous_bilateral_contact():
    state = PreparationState()
    for _ in range(ACQUISITION_BILATERAL_STEPS):
        state, _ = advance(state, 0.1, 0.2)
    assert state.phase == 'GRASP_HOLD'  # initial bilateral detection counts as first
    for _ in range(HOLD_BILATERAL_STEPS - 1):
        state, _ = advance(state, 0.1, 0.2)
        assert state.phase == 'GRASP_HOLD'
    state, record = advance(state, 0.1, 0.2)
    assert state.phase == 'COMPLETE' and record['complete']
    with pytest.raises(ValueError):
        advance(state, 0.1, 0.2)


def test_contact_loss_resets_hold_without_resuming_closure():
    state = PreparationState(phase='GRASP_HOLD', target_m=.003, hold_streak=5)
    state, _ = advance(state, 0.1, 0)
    assert state.phase == 'CONTACT_ACQUISITION' and state.hold_streak == 0
    for _ in range(2):
        state, _ = advance(state, 0, 0)
        assert state.target_m == .003


def test_threshold_is_strict_and_horizon_is_not_extendable():
    state, _ = advance(PreparationState(), POSITIVE_CONTACT_N, POSITIVE_CONTACT_N)
    assert state.phase == 'FREE_SPACE_APPROACH'
    for _ in range(MAX_PREPARATION_STEPS - 1):
        state, _ = advance(state, 0, 0)
    assert state.commands == MAX_PREPARATION_STEPS
    assert state.target_m > 0
    with pytest.raises(ValueError):
        advance(state, 0, 0)


@pytest.mark.parametrize('left,right', [(float('nan'), 0), (0, float('inf')), (-1, 0)])
def test_bad_force_is_rejected(left, right):
    with pytest.raises(ValueError):
        advance(PreparationState(), left, right)
