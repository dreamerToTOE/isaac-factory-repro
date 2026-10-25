"""Exact count/step checks for the pinned 2-epoch Factory smoke logger."""
from scripts.tensorboard_tail import (
    RL_GAMES_CENTRAL_TAGS, RL_GAMES_EPOCH_TAGS, RL_GAMES_REWARD_TAGS,
    validate_rlgames_scalars,
)


def complete_scalars():
    rows = {tag: [{"step": 0, "value": 1.0}, {"step": 8192, "value": 1.0}]
            for tag in RL_GAMES_EPOCH_TAGS}
    rows["info/epochs"] = [{"step": 0, "value": 1.0}, {"step": 8192, "value": 2.0}]
    rows.update({tag: [{"step": 8192, "value": 1.0}, {"step": 16384, "value": 1.0}]
                 for tag in RL_GAMES_CENTRAL_TAGS})
    rows.update({tag: [{"step": 2, "value": 1.0}] for tag in RL_GAMES_REWARD_TAGS})
    return rows


def test_two_epoch_smoke_requires_exact_counts_and_official_offsets():
    result = validate_rlgames_scalars(complete_scalars(), 2, 8192)
    assert result["status"] == "complete"
    assert result["summaries"]["losses/a_loss"]["expected_steps"] == [0, 8192]
    assert result["summaries"]["losses/cval_loss"]["expected_steps"] == [8192, 16384]
    assert result["summaries"]["rewards/iter"]["expected_count"] == 1


def test_two_epoch_smoke_rejects_entire_missing_required_tag():
    rows = complete_scalars()
    del rows["losses/a_loss"]
    result = validate_rlgames_scalars(rows, 2, 8192)
    assert result["status"] == "incomplete"
    assert "losses/a_loss" in result["missing_tags"]


def test_two_epoch_smoke_rejects_tail_missing_from_required_tag():
    rows = complete_scalars()
    rows["losses/a_loss"].pop()
    result = validate_rlgames_scalars(rows, 2, 8192)
    assert result["status"] == "incomplete"
    assert "count/step mismatch: losses/a_loss" in result["issues"]


def test_two_epoch_smoke_rejects_duplicate_epoch():
    rows = complete_scalars()
    rows["info/epochs"].append({"step": 8192, "value": 2.0})
    assert validate_rlgames_scalars(rows, 2, 8192)["status"] == "incomplete"


def test_two_epoch_smoke_rejects_nonfinite_extra_tag():
    rows = complete_scalars()
    rows["unexpected/tag"] = [{"step": 2, "value": float("-inf")}]
    result = validate_rlgames_scalars(rows, 2, 8192)
    assert result["status"] == "incomplete" and result["nonfinite_count"] == 1


def test_two_epoch_smoke_rejects_forged_epoch_values_even_with_matching_count():
    rows = complete_scalars()
    rows["info/epochs"][1]["value"] = 1.0
    assert validate_rlgames_scalars(rows, 2, 8192)["status"] == "incomplete"
