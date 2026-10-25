import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation_reporting import add_intervals, atomic_write_json, wilson_interval
from scripts.success_metrics import SuccessTracker


def test_partial_result_uses_completed_denominator_and_omits_unfinished_episode(tmp_path):
    tracker = SuccessTracker(2)
    tracker.update([True, False], [True, False], [True, False], [149, 20])
    result = {"status": "running", "aggregate": add_intervals(tracker.aggregate(96)),
              "episodes": [item.to_dict() for item in tracker.completed]}
    path = tmp_path / "eval.partial.json"
    atomic_write_json(path, result)
    actual = json.loads(path.read_text())
    assert actual["aggregate"]["episodes_requested"] == 96
    assert actual["aggregate"]["episodes_completed"] == 1
    assert actual["aggregate"]["terminal_success_rate"] == 1
    assert len(actual["episodes"]) == 1
    assert not (tmp_path / "eval.partial.json.tmp").exists()


def test_atomic_update_replaces_valid_json_and_rejects_nonfinite(tmp_path):
    path = tmp_path / "eval.partial.json"
    atomic_write_json(path, {"episodes_completed": 32})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"episodes_completed": 64, "rate": math.nan})
    assert json.loads(path.read_text()) == {"episodes_completed": 32}
    atomic_write_json(path, {"episodes_completed": 64})
    assert json.loads(path.read_text()) == {"episodes_completed": 64}


def test_wilson_zero_and_full_success_are_not_zero_width():
    assert wilson_interval(0, 0) is None
    zero = wilson_interval(0, 288)
    full = wilson_interval(288, 288)
    assert zero[0] == 0 and 0.01 < zero[1] < 0.02
    assert 0.98 < full[0] < 1 and full[1] == 1
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_wilson_g2_gate_reliability_example():
    # Gate thresholds are project review heuristics, independent of task geometry.
    lower, upper = wilson_interval(260, 288)
    assert 260 / 288 >= .9
    assert lower >= .85
    assert upper < 1


def test_actual_evaluator_partial_writer_retains_analyzable_provenance(tmp_path):
    """Run the production writer without importing the simulator entry point.

    Extracting its metadata assignment and nested writer lets this regression
    test the actual emitted schema, rather than supplying an idealized partial
    fixture that can accidentally hide missing fields in the real evaluator.
    """
    from scripts.analyze_g2 import read_evaluations

    source_path = Path(__file__).parents[1] / "scripts/evaluate_checkpoint.py"
    source = ast.parse(source_path.read_text())
    main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    metadata = next(node for node in main.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "initialized_metadata"
                            for target in node.targets))
    writer = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "write_progress")
    tracker = SuccessTracker(2)
    tracker.update([True, False], [False, False], [False, False], [30, 30])
    tracker.update([False, True], [True, True], [True, True], [149, 149])
    checkpoint_hash = hashlib.sha256(b"test checkpoint identity").hexdigest()
    path = tmp_path / "metrics.json.partial.json"
    namespace = {
        "checkpoint_sha256": checkpoint_hash,
        "args_cli": SimpleNamespace(checkpoint=tmp_path / "last_Factory_ep_200_rew__1_.pth",
                                     stochastic=False, task="Isaac-Factory-PegInsert-Direct-v0",
                                     num_envs=2, episodes=3, seed=1001),
        "agent": SimpleNamespace(model=SimpleNamespace(training=False), clip_actions=True),
        "base_env": SimpleNamespace(step_dt=1 / 15,
                                    cfg_task=SimpleNamespace(name="peg_insert", success_threshold=.04)),
        "hashlib": hashlib, "predicate_source": "official predicate test identity",
        "progress_path": path, "atomic_write_json": atomic_write_json,
        "add_intervals": add_intervals, "tracker": tracker,
        "steps": 149, "started": 0, "time": SimpleNamespace(monotonic=lambda: 12),
        "geometric_snapshots": [], "terminal_checks": [{"policy_step": 149}],
    }
    exec(compile(ast.Module(body=[metadata, writer], type_ignores=[]), str(source_path), "exec"), namespace)
    namespace["write_progress"]("incomplete_max_steps", 149)
    payload = json.loads(path.read_text())
    assert payload["checkpoint_sha256"] == checkpoint_hash
    assert payload["step_dt_seconds"] == 1 / 15
    assert payload["inference_settings"]["action_selection"] == "policy_mean"
    assert payload["inference_settings"]["model_training"] is False
    assert payload["auto_reset_order_verified"] is True
    issues = {"errors": [], "warnings": []}
    row = read_evaluations(tmp_path, [200], [1001], 2, 3, issues)["checkpoints"][0]
    assert not issues["errors"]
    assert row["aggregate"]["episodes_completed"] == 2
    assert row["aggregate"]["ever_success_rate"] == 1
    assert row["aggregate"]["terminal_success_rate"] == .5
    assert row["episodes_requested"] == 6
    assert not row["complete_preregistered_seeds"]
