"""Scientific accounting checks for partial/duplicate/missing G2 evidence."""
import json

import pytest

from scripts.analyze_g2 import (read_audit, read_evaluations, read_monitor_segments,
                                summarize_episodes, training_summary, validate_episodes)


def episode(env, ever=False, terminal=False, first=None):
    return {"env_id": env, "episode_id": 0, "steps": 149,
            "ever_success": ever, "terminal_success": terminal,
            "first_success_step": first, "terminal_kind": "timeout"}


def result(episodes, status="complete", seed=1001):
    return {"checkpoint_epoch": 50, "checkpoint": "checkpoint_ep_50.pth",
            "checkpoint_sha256": "test-identity", "seed": seed,
            "status": status, "step_dt_seconds": 1 / 15,
            "episodes_completed": len(episodes), "episodes": episodes,
            "deterministic": True, "official_success_capture": "pre_reset_factory_get_curr_successes"}


def write(path, data):
    path.write_text(json.dumps(data))


def test_transient_and_terminal_success_and_conditional_time():
    data = result([episode(0, True, False, 30), episode(1, True, True, 90), episode(2)])
    metrics = summarize_episodes(validate_episodes(data))
    assert metrics["episodes_completed"] == 3
    assert metrics["ever_success_rate"] == 2 / 3
    assert metrics["terminal_success_rate"] == 1 / 3
    assert metrics["transient_only_count"] == 1
    assert metrics["first_success_step_conditional"]["n"] == 2
    assert metrics["first_success_step_conditional"]["mean"] == 60
    assert metrics["first_success_time_seconds_conditional"]["mean"] == 4


def test_no_success_is_zero_rate_but_undefined_conditional_time():
    metrics = summarize_episodes(validate_episodes(result([episode(0)])))
    assert metrics["ever_success_rate"] == 0
    assert metrics["terminal_success_rate"] == 0
    assert metrics["ever_success_wilson95"][1] > 0
    assert metrics["first_success_step_conditional"]["n"] == 0
    assert metrics["first_success_step_conditional"]["mean"] is None


def test_missing_evaluation_does_not_become_zero_success(tmp_path):
    issues = {"errors": [], "warnings": []}
    evaluation = read_evaluations(tmp_path, [50, 200], [1001, 1002, 1003], 32, 3, issues)
    final = evaluation["checkpoints"][-1]
    assert final["episodes_requested"] == 288
    assert final["aggregate"]["episodes_completed"] == 0
    assert final["aggregate"]["ever_success_rate"] is None
    assert final["aggregate"]["ever_success_wilson95"] is None
    assert not final["complete_preregistered_seeds"]


def test_partial_result_uses_actual_denominator(tmp_path):
    write(tmp_path / "metrics.partial.json", result([episode(0, True, False, 25)], status="incomplete_error"))
    issues = {"errors": [], "warnings": []}
    evaluation = read_evaluations(tmp_path, [50, 200], [1001, 1002, 1003], 32, 3, issues)
    row = evaluation["checkpoints"][0]
    assert row["episodes_requested"] == 96
    assert row["aggregate"]["episodes_completed"] == 1
    assert row["aggregate"]["ever_success_rate"] == 1
    assert not row["complete_preregistered_seeds"]
    assert not issues["errors"]


def test_complete_and_atomic_snapshot_are_not_double_counted(tmp_path):
    full = result([episode(0, True, True, 20), episode(1)])
    write(tmp_path / "metrics.json", full)
    write(tmp_path / "metrics.json.partial.json", full)
    issues = {"errors": [], "warnings": []}
    row = read_evaluations(tmp_path, [50, 200], [1001], 32, 3, issues)["checkpoints"][0]
    assert row["aggregate"]["episodes_completed"] == 2
    assert row["aggregate"]["ever_success_rate"] == .5
    assert not issues["errors"]


def test_conflicting_duplicate_episode_is_reported(tmp_path):
    write(tmp_path / "metrics.json", result([episode(0, True, False, 20)]))
    write(tmp_path / "metrics.partial.json", result([episode(0)], status="running"))
    issues = {"errors": [], "warnings": []}
    read_evaluations(tmp_path, [50, 200], [1001], 32, 3, issues)
    assert any("Conflicting duplicate" in error for error in issues["errors"])


def test_extra_seed_is_not_silently_pooled_into_preregistered_protocol(tmp_path):
    write(tmp_path / "registered.json", result([episode(0)], seed=1001))
    write(tmp_path / "extra.json", result([episode(0, True, True, 20)], seed=999))
    issues = {"errors": [], "warnings": []}
    row = read_evaluations(tmp_path, [50, 200], [1001], 32, 3, issues)["checkpoints"][0]
    assert row["aggregate"]["episodes_completed"] == 1
    assert row["aggregate"]["ever_success_rate"] == 0
    assert len(row["per_seed"]) == 2
    assert issues["warnings"]


@pytest.mark.parametrize("change", [
    {"episodes_completed": 32},
    {"aggregate": {"ever_success_count": 1}},
    {"episodes": [episode(0, False, True)]},
    {"episodes": [episode(0, True, False, None)]},
    {"episodes": [episode(0), episode(0)]},
])
def test_inconsistent_evaluation_is_rejected(change):
    data = result([episode(0)])
    data.update(change)
    with pytest.raises(ValueError):
        validate_episodes(data)


def test_initializing_partial_with_null_aggregate_is_valid_empty_evidence():
    data = result([], status="initializing")
    data["aggregate"] = None
    assert validate_episodes(data) == []


def test_jsonl_truncation_and_duplicate_epochs_are_visible(tmp_path):
    path = tmp_path / "epochs.jsonl"
    path.write_text('{"epoch":1,"environment_transitions":8192}\n'
                    '{"epoch":1,"environment_transitions":8192}\n'
                    '{"epoch":2,')
    issues = {"errors": [], "warnings": []}
    audit, epochs = read_audit(path, issues)
    assert len(epochs) == 1
    assert epochs[0]["environment_transitions"] == 8192
    assert issues["errors"] and issues["warnings"]


def monitor_segment(tmp_path, name, rows, result_data=None):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "gpu.csv").write_text("elapsed_seconds,total_used_MiB,free_MiB\n" +
                                       "".join(f"{elapsed},{used},{free}\n" for elapsed, used, free in rows))
    if result_data is not None:
        write(directory / "result.json", result_data)
    return {"name": name, "directory": name,
            "elapsed_origin_utc": "2026-10-23T12:00:00+00:00",
            "elapsed_origin_source": "test fixture", "elapsed_origin_uncertainty_seconds": .01}


def test_disconnected_monitors_preserve_gap_and_unknown_training_exit(tmp_path):
    first = monitor_segment(tmp_path, "original", [(0, 6000, 1800), (.1, 6100, 1700)])
    second = monitor_segment(tmp_path, "reattached", [(0, 6200, 1600), (.1, 6150, 1650)],
                             {"returncode": 0, "duration_seconds": .1})
    second["elapsed_origin_utc"] = "2026-10-23T12:00:10+00:00"
    path = tmp_path / "monitor.json"
    write(path, {"wall_evidence": {"start_utc": first["elapsed_origin_utc"],
                                  "end_utc": "2026-10-23T12:00:10.200000+00:00",
                                  "end_uncertainty_seconds": .1, "returncode": None},
                 "segments": [first, second]})
    issues = {"errors": [], "warnings": []}
    summary, rows = read_monitor_segments(path, issues)
    assert [r["elapsed_seconds"] for r in rows] == pytest.approx([0, .1, 10, 10.1])
    assert rows[2]["segment_elapsed_seconds"] == 0
    assert rows[0]["coverage_piece"] != rows[2]["coverage_piece"]
    assert summary["peak_total_gpu_used_MiB"] == 6200
    assert summary["min_free_MiB"] == 1600
    assert summary["wall_duration_seconds"] == pytest.approx(10.2)
    assert summary["sampled_span_coverage_seconds"] == pytest.approx(.2)
    assert [g["duration_seconds"] for g in summary["coverage_gaps"]] == pytest.approx([9.9, .1], abs=1e-6)
    assert not summary["training_child_exit_code_available"]
    assert summary["training_child_returncode"] is None  # Replacement guard exited zero, child status stays unknown.
    assert not issues["errors"]
    assert any("unavailable" in warning for warning in issues["warnings"])
    scalars = {"tags": {}, "all_logged_scalars_finite": None}
    epoch = {"epoch": 1, "environment_transitions": 8192, "official_total_seconds": 5}
    train = training_summary(scalars, {}, [epoch], summary, 64, 128, issues)
    assert train["official_ppo_aggregate_fps"] == 8192 / 5
    assert train["end_to_end_transitions_per_wall_second"] == pytest.approx(8192 / 10.2)
    assert train["launched_process_wall_uncertainty_seconds"] == .1


def test_monitor_internal_gap_and_no_disappearance_do_not_invent_wall_time(tmp_path):
    segment = monitor_segment(tmp_path, "one", [(0, 6000, 1800), (.1, 6100, 1700),
                                               (3, 6000, 1800), (3.1, 6000, 1800)])
    path = tmp_path / "monitor.json"
    write(path, {"wall_evidence": {"start_utc": segment["elapsed_origin_utc"]}, "segments": [segment]})
    issues = {"errors": [], "warnings": []}
    summary, rows = read_monitor_segments(path, issues)
    assert summary["wall_duration_seconds"] is None
    assert summary["sampled_span_coverage_fraction"] is None
    assert len(summary["coverage_gaps"]) == 1
    assert summary["coverage_gaps"][0]["duration_seconds"] == pytest.approx(2.9)
    assert rows[1]["coverage_piece"] != rows[2]["coverage_piece"]


def test_monitor_overlapping_segments_do_not_double_count_coverage(tmp_path):
    first = monitor_segment(tmp_path, "first", [(0, 6000, 1800), (.1, 6100, 1700)])
    second = monitor_segment(tmp_path, "second", [(0, 6000, 1800), (.1, 6100, 1700)])
    path = tmp_path / "monitor.json"
    write(path, {"wall_evidence": {"start_utc": first["elapsed_origin_utc"],
                                  "end_utc": "2026-10-23T12:00:00.100000+00:00"},
                 "segments": [first, second]})
    summary, _ = read_monitor_segments(path, {"errors": [], "warnings": []})
    assert summary["sample_count"] == 4
    assert summary["sampled_span_coverage_seconds"] == pytest.approx(.1)
    assert summary["sampled_span_coverage_fraction"] == pytest.approx(1)


@pytest.mark.parametrize("wall", [
    {"start_utc": "2026-10-23T12:00:00"},
    {"start_utc": "2026-10-23T12:00:00Z", "end_utc": "2026-10-23T11:00:00Z"},
    {"start_utc": "2026-10-23T12:00:00Z", "end_uncertainty_seconds": -1},
    {"start_utc": "2026-10-23T12:00:00Z", "returncode": True},
])
def test_monitor_rejects_ambiguous_or_invalid_wall_evidence(tmp_path, wall):
    path = tmp_path / "monitor.json"
    write(path, {"wall_evidence": wall, "segments": []})
    with pytest.raises(ValueError):
        read_monitor_segments(path, {"errors": [], "warnings": []})
