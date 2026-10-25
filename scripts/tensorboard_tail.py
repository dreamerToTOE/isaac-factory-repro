"""[ENGINEERING] Detect a missing or non-finite TensorBoard tail."""
from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Mapping


def _is_finite(value: object) -> bool:
    """Return whether a scalar value can be represented as a finite float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def validate_tail(
    points: Iterable[Mapping[str, object]],
    expected_epoch: int,
    expected_tags: Iterable[str] | None = None,
) -> dict:
    """Validate scalar tail records without synthesizing missing values.

    Each point must expose ``tag``, ``step``, ``epoch`` and finite ``value``.
    ``expected_tags`` should be supplied when the caller knows the complete
    scalar set; otherwise the observed tags are used for backwards-compatible
    inspection of a single stream.  A tail is complete only when every
    required tag has an event at the expected epoch and no non-finite value is
    present.
    """
    rows = list(points)
    observed_tags = {str(r.get("tag")) for r in rows}
    required_tags = observed_tags | {str(tag) for tag in (expected_tags or ())}
    nonfinite = [r for r in rows if not _is_finite(r.get("value"))]
    missing = [
        tag
        for tag in sorted(required_tags)
        if not any(str(r.get("tag")) == tag and r.get("epoch") == expected_epoch for r in rows)
    ]
    return {
        "status": "complete" if rows and not nonfinite and not missing else "incomplete",
        "expected_epoch": expected_epoch,
        "point_count": len(rows),
        "tags": sorted(required_tags),
        "missing_tags_at_expected_epoch": missing,
        "nonfinite_count": len(nonfinite),
    }


# These are present once per optimization epoch in pinned RL-Games 1.6.1.
# Central-value loss uses the post-increment frame; actor/performance tags use
# the pre-increment frame. This difference is part of the official logger.
RL_GAMES_EPOCH_TAGS = (
    "info/epochs",
    "losses/a_loss",
    "losses/c_loss",
    "losses/entropy",
    "losses/bounds_loss",
    "performance/step_inference_rl_update_fps",
    "performance/step_inference_time",
    "performance/rl_update_time",
)
RL_GAMES_CENTRAL_TAGS = ("losses/cval_loss", "info/cval_lr")
RL_GAMES_REWARD_TAGS = ("rewards/iter", "shaped_rewards/iter", "episode_lengths/iter")


def validate_rlgames_scalars(
    scalars: Mapping[str, Iterable[Mapping[str, object]]],
    expected_epochs: int,
    transitions_per_epoch: int,
    *,
    reward_first_epoch: int = 2,
) -> dict:
    """Check the exact counts/steps needed by the new short Factory smoke.

    This is intentionally stricter than :func:`validate_tail`: required PPO,
    central-value and reward tags are fixed independently of the observed
    events. For the 149-step Factory episode and 128-step horizon, the short
    smoke first emits episodic reward at epoch 2; callers must state a
    different ``reward_first_epoch`` if testing a different episode schedule.
    No missing scalar is synthesized.
    """
    if expected_epochs < 1 or transitions_per_epoch < 1:
        raise ValueError("expected epochs and transitions per epoch must be positive")
    if reward_first_epoch < 1:
        raise ValueError("reward_first_epoch must be positive")
    records = {str(tag): list(values) for tag, values in scalars.items()}
    issues = []
    required = (*RL_GAMES_EPOCH_TAGS, *RL_GAMES_CENTRAL_TAGS)
    if expected_epochs >= reward_first_epoch:
        required += RL_GAMES_REWARD_TAGS
    missing = [tag for tag in required if tag not in records]
    issues += [f"missing required tag: {tag}" for tag in missing]
    bad = [
        {"tag": tag, "index": index}
        for tag, values in records.items()
        for index, row in enumerate(values)
        if not _is_finite(row.get("value"))
    ]
    if bad:
        issues.append(f"non-finite scalar count: {len(bad)}")
    summaries = {}
    for tag in required:
        values = records.get(tag, [])
        if tag in RL_GAMES_CENTRAL_TAGS:
            expected_steps = [epoch * transitions_per_epoch for epoch in range(1, expected_epochs + 1)]
        elif tag in RL_GAMES_REWARD_TAGS:
            expected_steps = list(range(reward_first_epoch, expected_epochs + 1))
        else:
            expected_steps = [epoch * transitions_per_epoch for epoch in range(expected_epochs)]
        actual_steps = [row.get("step") for row in values]
        if actual_steps != expected_steps:
            issues.append(f"count/step mismatch: {tag}")
        summaries[tag] = {
            "count": len(values),
            "expected_count": len(expected_steps),
            "steps": actual_steps,
            "expected_steps": expected_steps,
            "last_value": values[-1].get("value") if values and _is_finite(values[-1].get("value")) else None,
        }
    epochs = [row.get("value") for row in records.get("info/epochs", [])]
    if epochs != list(range(1, expected_epochs + 1)):
        issues.append("info/epochs values do not match every completed epoch")
    return {
        "status": "complete" if not issues else "incomplete",
        "expected_epochs": expected_epochs,
        "transitions_per_epoch": transitions_per_epoch,
        "reward_first_epoch": reward_first_epoch,
        "required_tags": list(required),
        "missing_tags": missing,
        "nonfinite_count": len(bad),
        "scalar_tag_count": len(records),
        "summaries": summaries,
        "issues": issues,
    }


def inspect_rlgames_events(directory, expected_epochs: int, transitions_per_epoch: int, *, reward_first_epoch: int = 2) -> dict:
    """Read TensorBoard events and run the strict short-smoke validation."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    scalars = {
        tag: [{"step": event.step, "value": event.value, "wall_time": event.wall_time}
              for event in events.Scalars(tag)]
        for tag in events.Tags()["scalars"]
    }
    result = validate_rlgames_scalars(scalars, expected_epochs, transitions_per_epoch, reward_first_epoch=reward_first_epoch)
    result["directory"] = str(directory)
    return result


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--transitions-per-epoch", type=int, required=True)
    parser.add_argument("--reward-first-epoch", type=int, default=2)
    args = parser.parse_args(argv)
    result = inspect_rlgames_events(args.directory, args.expected_epochs, args.transitions_per_epoch,
                                   reward_first_epoch=args.reward_first_epoch)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
