import csv
import json

import pytest

from scripts.analyze_g3cr2 import (budget_summary, matched_comparison,
                                   numerical_summary, read_supervised_run,
                                   scalar_transition_points)


def test_interruption_counts_completed_epochs_not_registered_budget():
    epochs = [{"epoch": n, "environment_transitions": 4096} for n in range(1, 11)]
    result = budget_summary(epochs)
    assert result["completed_transitions"] == 40960
    assert result["planned_transitions"] == 1638400
    assert result["completed_budget_fraction"] == .025
    assert result["registered_transition_budget_completed"] is False
    assert budget_summary([])["completed_transitions"] == 0
    with pytest.raises(ValueError, match="registered vector batch"):
        budget_summary([{"epoch": 1, "environment_transitions": 8192}])
    assert not budget_summary([{"epoch": 2, "environment_transitions": 4096}])["audit_contiguous_from_epoch_1"]


def test_missing_evaluation_not_zero_rate_or_historical_pooling():
    empty = {"episodes_completed": 0, "ever_success_rate": None, "terminal_success_rate": None}
    historical = {"episodes_completed": 96, "ever_success_rate": .75, "terminal_success_rate": .75}
    new = {"checkpoints": [{"checkpoint_epoch": 100, "aggregate": dict(empty)}]}
    old = {"checkpoints": [{"checkpoint_epoch": 50, "aggregate": dict(historical)}]}
    result = matched_comparison(new, old)
    first = result["rows"][0]
    assert first["environment_transitions"] == 409600
    assert first["g3cr2"]["aggregate"] == empty
    assert first["g2_historical"]["aggregate"]["episodes_completed"] == 96
    assert first["ever_success_rate_delta"] is None
    assert first["between_protocol_delta_assessed"] is False
    assert result["rows"][-1]["g3cr2"] is None
    new["checkpoints"][0]["aggregate"] = {"episodes_completed": 32, "ever_success_rate": .5, "terminal_success_rate": .5}
    first = matched_comparison(new, old)["rows"][0]
    assert first["ever_success_rate_delta"] == -.25
    assert first["g3cr2"]["aggregate"]["episodes_completed"] == 32


def test_scalar_mapping_respects_actor_logging_offset():
    scalars = {"tags": {
        "info/epochs": [{"step": 0, "value": 1, "finite": True}],
        "losses/a_loss": [{"step": 0, "value": .4, "finite": True}],
        "losses/cval_loss": [{"step": 4096, "value": .3, "finite": True}],
        "rewards/iter": [{"step": 2, "value": 10, "finite": True}],
    }}
    assert scalar_transition_points(scalars, "losses/a_loss", 4096) == [(4096, .4)]
    assert scalar_transition_points(scalars, "losses/cval_loss", 4096) == [(4096, .3)]
    assert scalar_transition_points(scalars, "rewards/iter", 4096) == [(8192, 10)]
    scalars["tags"]["losses/a_loss"][0]["finite"] = False
    assert scalar_transition_points(scalars, "losses/a_loss", 4096) == []


def test_guard_gap_blocks_coverage_despite_exit_zero_and_memory_margin(tmp_path):
    (tmp_path / "guard").mkdir()
    (tmp_path / "supervisor").mkdir()
    def record(path, value):
        (tmp_path / path).write_text(json.dumps(value))
    record("guard/status.json", {"continuous_monitoring": True, "policy": {"gap_limit_seconds": .5}})
    record("supervisor/child_identity.json", {"monotonic_seconds": 1.05})
    record("supervisor/child_exit.json", {"monotonic_seconds": 1.7, "raw_wait_returncode": 0})
    with (tmp_path / "guard/samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["elapsed_seconds", "monotonic_seconds", "used_MiB", "free_MiB"])
        writer.writerows([[0, 1, 6000, 1800], [.1, 1.1, 6000, 1800], [.7, 1.7, 6000, 1800], [.8, 1.8, 6000, 1800]])
    issues = {"errors": [], "warnings": []}
    result, _ = read_supervised_run(tmp_path, issues)
    assert result["child_raw_wait_returncode"] == 0
    assert result["samples_bracket_child"] is True
    assert result["min_free_MiB"] == 1800
    assert result["sampled_coverage_complete"] is False
    assert len(result["observed_gaps"]) == 1
    assert not issues["errors"]


def test_observed_amp_skip_never_erased_by_finite_loss_summary():
    audit = {"optimizer_steps": {"actor": 319, "central_value": 320},
             "actor_attempts": 320, "amp_skips": [{"epoch": 1, "attempt": 1}]}
    result = numerical_summary(audit, [{"epoch_loss_arrays_finite": True}])
    assert result["all_observed_epoch_loss_arrays_finite"] is True
    assert result["amp_skipped_actor_updates"] == 1
    assert result["actual_optimizer_steps"]["actor"] == 319
    assert numerical_summary({}, [])["all_observed_epoch_loss_arrays_finite"] is None
