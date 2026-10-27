"""Preregistered, actor-free reference generation for one recovery protocol.

The bound constrains the *reference* only, not the physical finger displacement,
contact force, or penetration. Observations supplied to ``advance`` must come
from the current initial state or the previous completed physical step; never
from a later state. No MuJoCo or mutable model objects are used here.
"""
from dataclasses import dataclass
import math

DT_S = 1.0 / 120.0
MAX_TARGET_DELTA_M = 10e-6
MAX_TARGET_RATE_M_S = MAX_TARGET_DELTA_M / DT_S
POSITIVE_CONTACT_N = 1e-6  # identical to the existing bilateral preparation gate
ACQUISITION_BILATERAL_STEPS = 10
HOLD_BILATERAL_STEPS = 10
MAX_PREPARATION_STEPS = 240
INITIAL_OPENING_M = 0.00499125


@dataclass(frozen=True)
class PreparationState:
    target_m: float = INITIAL_OPENING_M
    phase: str = 'FREE_SPACE_APPROACH'
    bilateral_streak: int = 0
    hold_streak: int = 0
    commands: int = 0


def advance(state: PreparationState, left_force_n: float, right_force_n: float):
    """Return (next state, audit record) before a step; never performs physics.

    First positive pad contact freezes the target permanently. Ten consecutive
    bilateral observations establish acquisition; then ten *additional*
    consecutive bilateral observations establish hold. Contact loss resets the
    corresponding counter without resuming closure. A run must stop at 240
    commands whether or not grasp has been acquired.
    """
    if state.phase not in ('FREE_SPACE_APPROACH', 'CONTACT_ACQUISITION', 'GRASP_HOLD', 'COMPLETE'):
        raise ValueError('unknown phase')
    if not math.isfinite(state.target_m) or not 0 <= state.target_m <= INITIAL_OPENING_M:
        raise ValueError('invalid target')
    if not all(math.isfinite(v) and v >= 0 for v in (left_force_n, right_force_n)):
        raise ValueError('contact normal force must be finite and nonnegative')
    if state.commands >= MAX_PREPARATION_STEPS or state.phase == 'COMPLETE':
        raise ValueError('preparation already finished')
    left = left_force_n > POSITIVE_CONTACT_N
    right = right_force_n > POSITIVE_CONTACT_N
    bilateral = left and right
    phase, target, acq, hold = state.phase, state.target_m, state.bilateral_streak, state.hold_streak
    if phase == 'FREE_SPACE_APPROACH':
        if left or right:
            phase = 'CONTACT_ACQUISITION'
            acq = int(bilateral)
        else:
            target = max(0.0, target - MAX_TARGET_DELTA_M)
    elif phase == 'CONTACT_ACQUISITION':
        acq = acq + 1 if bilateral else 0
        if acq >= ACQUISITION_BILATERAL_STEPS:
            phase = 'GRASP_HOLD'
            hold = 0
    elif phase == 'GRASP_HOLD':
        if bilateral:
            hold += 1
            if hold >= HOLD_BILATERAL_STEPS:
                phase = 'COMPLETE'
        else:
            phase, acq, hold = 'CONTACT_ACQUISITION', 0, 0
    next_state = PreparationState(target, phase, acq, hold, state.commands + 1)
    return next_state, {
        'phase_before': state.phase, 'phase_after': phase,
        'observed_left_force_n': float(left_force_n),
        'observed_right_force_n': float(right_force_n),
        'positive_contact_threshold_n': POSITIVE_CONTACT_N,
        'bilateral_observed': bilateral, 'target_m': target,
        'target_delta_m': target - state.target_m,
        'target_rate_m_s': (target - state.target_m) / DT_S,
        'tendon_target_m': 2.0 * target,
        'bilateral_streak': acq, 'hold_streak': hold,
        'command_number': next_state.commands,
        'complete': phase == 'COMPLETE',
    }
