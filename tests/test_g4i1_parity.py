"""Fixed acceptance thresholds, intermediate coverage, and malformed trace tests."""
import json

import numpy as np
import pytest

from scripts.g4i1_parity import REQUIRED_CASES, compare_stages, main, read_paired_npz


def trace():
    data = {"observation": np.zeros((3, 1, 19), np.float32),
            "normalized": np.zeros((3, 1, 19), np.float32),
            "mu": np.array([[[3., -2., 0., 0., 0., .4]]], np.float32),
            "h": np.zeros((3, 2, 1, 1024), np.float32),
            "c": np.zeros((3, 2, 1, 1024), np.float32),
            "ema": np.zeros((3, 1, 6), np.float32),
            "target_position": np.zeros((3, 1, 3), np.float32),
            "target_quaternion": np.tile([1., 0., 0., 0.], (3, 1, 1)),
            "controller__joint_pos": np.zeros((3, 1, 7), np.float32),
            "controller__jacobian": np.zeros((3, 1, 6, 7), np.float32)}
    data["clipped"] = data["mu"].clip(-1, 1)
    return data


def partial(reference, actual, **kwargs):
    return compare_stages(reference, actual, require_all_stages=False, **kwargs)


def test_full_trace_counts_all_elements_and_registered_cases():
    data = trace()
    result = compare_stages(data, data, cases=REQUIRED_CASES, require_cases=REQUIRED_CASES)
    assert result["passed"] and result["coverage"]["passed"]
    assert result["comparison_count"] == sum(v.size for v in data.values())
    assert result["violation_count"] == result["max_absolute_error"] == 0
    assert result["stages"]["h"]["comparison_count"] == 3 * 2 * 1024
    assert result["stages"]["target_quaternion"]["orientation_comparison_count"] == 3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("device,atol", [("cpu", 1e-6), ("gpu", 1e-5)])
def test_frozen_absolute_threshold_accepts_below_and_rejects_above(device, atol):
    ref = {"mu": np.zeros((1, 6), np.float64)}
    got = {"mu": np.full((1, 6), np.nextafter(atol, 0.))}
    assert partial(ref, got, device_kind=device)["passed"]
    got["mu"][0, 0] = np.nextafter(atol, np.inf)
    result = partial(ref, got, device_kind=device)
    assert not result["passed"]
    assert result["violation_count"] == 1
    assert result["tolerances"]["atol"] == atol


def test_relative_term_is_reference_based_not_actual_based():
    ref = {"mu": np.ones((1, 6), np.float64)}
    got = {"mu": ref["mu"] + (2e-6 + 1e-12)}
    result = partial(ref, got)
    assert not result["passed"] and result["violation_count"] == 6
    assert result["max_relative_error"] == pytest.approx(2e-6 + 1e-12)


def test_raw_mean_mismatch_cannot_hide_behind_identical_clipped_actions():
    ref = {"mu": np.full((1, 6), 2.), "clipped": np.ones((1, 6))}
    got = {"mu": np.full((1, 6), 3.), "clipped": np.ones((1, 6))}
    result = partial(ref, got)
    assert not result["passed"]
    assert not result["stages"]["mu"]["passed"]
    assert result["stages"]["clipped"]["passed"]


def test_quaternion_sign_change_has_zero_orientation_and_component_error():
    ref = {"target_quaternion": np.array([[.5, .5, .5, .5]])}
    got = {"target_quaternion": -ref["target_quaternion"]}
    result = partial(ref, got)
    assert result["passed"]
    assert result["max_absolute_error"] == 0
    assert result["stages"]["target_quaternion"]["max_orientation_error_rad"] == 0


def test_small_quaternion_angles_remain_resolvable_and_threshold_is_frozen():
    ref = {"target_quaternion": np.array([[1., 0., 0., 0.]])}
    angle = 1.00001e-4
    got = {"target_quaternion": np.array([[np.cos(angle / 2), 0., 0., np.sin(angle / 2)]])}
    result = partial(ref, got, device_kind="gpu")
    stage = result["stages"]["target_quaternion"]
    assert stage["max_orientation_error_rad"] == pytest.approx(angle, abs=1e-14)
    assert stage["orientation_violation_count"] == 1
    assert not stage["orientation_passed"] and not result["passed"]


def test_missing_mandatory_stage_and_controller_arguments_fail():
    data = trace()
    del data["normalized"]
    result = compare_stages(data, data)
    assert not result["passed"] and result["coverage"]["missing_stages"] == ["normalized"]
    data = {key: value for key, value in trace().items() if not key.startswith("controller__")}
    result = compare_stages(data, data)
    assert not result["passed"] and result["coverage"]["missing_stages"] == ["controller_inputs"]


def test_unmatched_controller_key_and_missing_case_fail():
    ref, got = trace(), trace()
    del got["controller__jacobian"]
    result = compare_stages(ref, got, cases=["normal"], require_cases=REQUIRED_CASES)
    assert not result["passed"]
    assert result["coverage"]["missing_in_actual"] == ["controller__jacobian"]
    assert "timeout" in result["coverage"]["missing_cases"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_arrays_fail_and_output_is_valid_json(bad):
    ref = {"mu": np.zeros((1, 6))}
    got = {"mu": np.array([[bad, 0, 0, 0, 0, 0.]])}
    result = partial(ref, got)
    assert not result["passed"] and result["violation_count"] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("stage,shape", [("mu", (1, 5)), ("h", (1, 1, 1024)),
                                         ("observation", (1, 18))])
def test_matching_but_invalid_stage_dimensions_are_rejected(stage, shape):
    data = {stage: np.zeros(shape)}
    assert not partial(data, data)["passed"]


def test_broadcastable_shape_difference_and_empty_trace_are_rejected():
    assert not partial({"mu": np.zeros(6)}, {"mu": np.zeros((1, 6))})["passed"]
    assert not partial({"mu": np.empty((0, 6))}, {"mu": np.empty((0, 6))})["passed"]
    assert not partial({}, {})["passed"]


def test_zero_quaternion_and_non_numeric_arrays_are_rejected():
    q = {"target_quaternion": np.zeros((1, 4))}
    assert not partial(q, q)["passed"]
    strings = {"mu": np.array(["1"] * 6)}
    assert not partial(strings, strings)["passed"]


def test_nested_controller_mapping_reports_each_argument_and_missing_value():
    data = trace()
    del data["controller__joint_pos"], data["controller__jacobian"]
    data["controller_inputs"] = {"joint_pos": np.zeros((1, 7))}
    result = compare_stages(data, data)
    assert result["passed"] and "controller__joint_pos" in result["stages"]
    got = dict(data, controller_inputs={})
    assert not compare_stages(data, got)["passed"]


def test_unknown_device_cannot_select_unregistered_thresholds():
    with pytest.raises(ValueError, match="frozen"):
        partial({}, {}, device_kind="amp")


def test_npz_cli_roundtrip_exit_and_failure_reports(tmp_path):
    input_path, output = tmp_path / "pairs.npz", tmp_path / "report.json"
    ref = trace()
    np.savez(input_path, **{"reference__" + key: value for key, value in ref.items()},
             **{"actual__" + key: value for key, value in ref.items()})
    assert set(read_paired_npz(input_path)[0]) == set(ref)
    assert main(["--input", str(input_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["passed"]
    failed_output = tmp_path / "failed-report.json"
    assert main(["--input", str(input_path), "--output", str(failed_output),
                 "--require-cases", "timeout"]) == 1
    assert not json.loads(failed_output.read_text())["passed"]


def test_cli_refuses_overwrite_without_changing_old_evidence(tmp_path, capsys):
    input_path, output = tmp_path / "pairs.npz", tmp_path / "existing-report.json"
    np.savez(input_path, reference__mu=np.zeros((1, 6)), actual__mu=np.zeros((1, 6)))
    original = b'{"historical_evidence":true}\n'
    output.write_bytes(original)
    assert main(["--input", str(input_path), "--output", str(output),
                 "--allow-partial-stages"]) == 1
    assert output.read_bytes() == original
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_object_npz_is_not_deserialized(tmp_path):
    input_path, output = tmp_path / "object.npz", tmp_path / "report.json"
    np.savez(input_path, reference__mu=np.array([{"unsafe": True}], dtype=object))
    assert main(["--input", str(input_path), "--output", str(output)]) == 1
    assert "Object arrays cannot be loaded" in json.loads(output.read_text())["error"]
