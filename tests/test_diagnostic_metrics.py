"""Hand-computed controls for G3B diagnostic summaries (no simulator imports)."""
import copy
import json
import math

import numpy as np
import pytest

from scripts.diagnostic_metrics import (
    classify_failure,
    episode_summary,
    official_geometry,
    quaternion_axis_angle_deg,
    quaternion_distance_deg,
    quaternion_z_axis,
    scalar_stats,
    summarize_boolean_timing,
)


def trajectory(success=(False, False, False), xy=(.01, .005, .003), z=(.03, .01, .002)):
    n = len(success)
    return {
        "seed": np.full(n, 1001), "env_id": np.full(n, 11), "episode_id": np.zeros(n, dtype=int),
        "policy_step": np.arange(1, n + 1), "done": np.arange(n) == n - 1,
        "official_success": np.asarray(success), "xy_error_m": np.asarray(xy),
        "axial_displacement_m": np.asarray(z), "engaged": np.ones(n, dtype=bool),
        "peg_quat_w": np.tile([1., 0., 0., 0.], (n, 1)),
        "hole_quat_w": np.tile([1., 0., 0., 0.], (n, 1)),
    }


def test_wxyz_local_z_basis_vectors_and_quaternion_sign():
    root_half = math.sqrt(.5)
    assert quaternion_z_axis([1, 0, 0, 0]) == pytest.approx([0, 0, 1])
    assert quaternion_z_axis([root_half, root_half, 0, 0]) == pytest.approx([0, -1, 0], abs=1e-14)
    assert quaternion_z_axis([root_half, 0, root_half, 0]) == pytest.approx([1, 0, 0], abs=1e-14)
    assert quaternion_z_axis([0, 1, 0, 0]) == pytest.approx([0, 0, -1])
    assert quaternion_z_axis([-root_half, -root_half, 0, 0]) == pytest.approx([0, -1, 0], abs=1e-14)


def test_axis_tilt_ignores_yaw_but_relative_orientation_records_it():
    q = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
    assert quaternion_axis_angle_deg(q, [1, 0, 0, 0]) == pytest.approx(0)
    assert quaternion_distance_deg(q, [1, 0, 0, 0]) == pytest.approx(90)
    assert quaternion_distance_deg(q, [-x for x in q]) == pytest.approx(0, abs=2e-6)
    assert quaternion_axis_angle_deg([0, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(180)


def test_axis_and_relative_angles_invariant_to_common_world_rotation():
    # Initially peg Rx90 and hole identity. After common Rz90, products are
    # peg (1/2,1/2,1/2,1/2), hole (sqrt(1/2),0,0,sqrt(1/2)).
    common_hole = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
    assert quaternion_axis_angle_deg([.5, .5, .5, .5], common_hole) == pytest.approx(90)
    assert quaternion_distance_deg([.5, .5, .5, .5], common_hole) == pytest.approx(90)


def test_invalid_quaternion_is_missing_instead_of_identity():
    assert np.isnan(quaternion_axis_angle_deg([0, 0, 0, 0], [1, 0, 0, 0]))
    assert np.isnan(quaternion_distance_deg([float("nan"), 0, 0, 0], [1, 0, 0, 0]))
    with pytest.raises(ValueError, match="dimension 4"):
        quaternion_z_axis([1, 0, 0])


def test_official_world_xy_and_z_are_translation_invariant():
    xy, z = official_geometry([[10.003, 20.004, 30.006]], [[10., 20., 30.]])
    assert xy.tolist() == pytest.approx([.005])
    assert z.tolist() == pytest.approx([.006])
    xy2, z2 = official_geometry([[.003, .004, .006]], [[0, 0, 0]])
    assert xy == pytest.approx(xy2)
    assert z == pytest.approx(z2)


def test_world_z_and_hole_local_z_remain_distinct_and_nominal_depth_labelled():
    data = trajectory(success=(False,), xy=(.005,), z=(.006,))
    data["held_base_pos_w"] = [[10.003, 20.004, 30.006]]
    data["target_held_base_pos_w"] = [[10., 20., 30.]]
    # Hole local +Z points along world +X.
    data["hole_quat_w"] = [[math.sqrt(.5), 0, math.sqrt(.5), 0]]
    result = episode_summary(data, policy_dt_s=.1, hole_height_m=.025, plausible_z_range_m=(0, .025))
    assert result["geometry"]["axial_displacement_m"]["final"] == pytest.approx(.006)
    assert result["secondary"]["hole_axis_projected_base_gap_m"]["final"] == pytest.approx(.003)
    assert result["secondary"]["hole_axis_transverse_error_m"]["final"] == pytest.approx(math.sqrt(.004**2 + .006**2))
    assert result["secondary"]["nominal_insertion_depth_proxy_m"]["final"] == pytest.approx(.022)
    assert result["secondary"]["terminal_in_plausible_z_range"] is True
    assert result["secondary"]["terminal_secondary_quality"] is False
    assert result["geometry"]["supplied_vs_pose_max_abs_xy_error_m"] < 1e-14


def test_success_timing_and_disjoint_sustained_runs_are_hand_counted():
    result = summarize_boolean_timing([False, True, True, False, True], [1, 2, 3, 4, 5], .2)
    assert result["observed_success_samples"] == 3
    assert result["sampled_success_duration_s"] == pytest.approx(.6)
    assert result["longest_observed_success_run_steps"] == 2
    assert result["longest_observed_success_run_s"] == pytest.approx(.4)
    assert result["observed_success_runs"] == 2
    assert result["first_success_step"] == 2
    assert result["first_success_time_s"] == pytest.approx(.4)
    assert result["last_success_step"] == 5
    assert result["terminal_success"] is True
    assert result["first_success_is_exact"] is True


def test_first_success_missing_prefix_is_not_claimed_exact():
    result = summarize_boolean_timing([None, True, False], [1, 2, 3], .1)
    assert result["ever_success"] is True
    assert result["first_success_step"] == 2
    assert result["first_success_is_exact"] is False
    assert result["sampled_success_duration_s"] is None
    assert result["observed_success_duration_lower_bound_s"] == pytest.approx(.1)
    assert result["terminal_success"] is False
    unknown = summarize_boolean_timing([None, False, None], [1, 2, 3], .1)
    assert unknown["ever_success"] is None
    assert unknown["terminal_success"] is None


@pytest.mark.parametrize("steps", [[1, 2, 1], [0, 1, 2], [1, 3, 4], [2, 3, 4]])
def test_success_timing_rejects_cross_reset_or_missing_steps(steps):
    with pytest.raises(ValueError, match="reset leakage"):
        summarize_boolean_timing([False] * 3, steps, .1)


def test_episode_summary_rejects_ids_changing_and_done_before_final():
    data = trajectory()
    data["episode_id"] = [0, 0, 1]
    with pytest.raises(ValueError, match="episode_id changes"):
        episode_summary(data, policy_dt_s=.1)
    data = trajectory()
    data["done"] = [True, False, True]
    with pytest.raises(ValueError, match="done only"):
        episode_summary(data, policy_dt_s=.1)
    data["done"] = [False, False, False]
    with pytest.raises(ValueError, match="complete episode"):
        episode_summary(data, policy_dt_s=.1)


def test_official_strict_threshold_boundaries_and_negative_z_control():
    data = trajectory(xy=(.003, .0025, .0025), z=(.002, .001, .001))
    result = episode_summary(data, policy_dt_s=.1)
    assert [x["category"] for x in result["failure_categories"]] == ["A", "B"]
    assert result["geometry"]["official_predicate_disagreement_samples"] == 0
    # Geometric success permits arbitrary negative Z. A secondary proxy must
    # never rewrite that official success, even if a new plausible range fails.
    success = trajectory(success=(True,), xy=(.00249,), z=(-.02,))
    success["held_base_pos_w"] = [[.00249, 0, -.02]]
    success["target_held_base_pos_w"] = [[0, 0, 0]]
    successful = episode_summary(success, policy_dt_s=.1, plausible_z_range_m=(0, .025))
    assert successful["timing"]["terminal_success"] is True
    assert successful["failure_categories"] == []
    assert successful["secondary"]["terminal_in_plausible_z_range"] is False
    assert successful["secondary"]["terminal_secondary_quality"] is False


def test_orientation_secondary_boundary_is_explicit_and_no_yaw_failure():
    result = episode_summary(trajectory(), policy_dt_s=.1)
    result["orientation"]["axis_angle_deg"]["final"] = 5.
    assert "C" not in [x["category"] for x in classify_failure(result)]
    result["orientation"]["axis_angle_deg"]["final"] = 5.01
    assert "C" in [x["category"] for x in classify_failure(result)]
    yaw = trajectory()
    yaw["peg_quat_w"] = np.tile([0, 0, 0, 1], (3, 1))
    assert "C" not in [x["category"] for x in episode_summary(yaw, policy_dt_s=.1)["failure_categories"]]


def test_temporary_success_is_category_e_even_if_final_xy_and_z_fail():
    data = trajectory(success=(False, True, False), xy=(.004, .002, .003), z=(.03, .0005, .002))
    result = episode_summary(data, policy_dt_s=.1)
    assert [x["category"] for x in result["failure_categories"]] == ["A", "B", "E"]
    assert result["timing"]["first_success_step"] == 2
    assert result["timing"]["sampled_success_duration_s"] == pytest.approx(.1)


def test_stationary_ee_active_actions_and_forces_cannot_establish_obstruction():
    data = trajectory()
    data["ee_pos_w"] = [[0, 0, 0]] * 3
    data["actions"] = [[1] * 6] * 3
    # Even an extra force array does not independently prove obstruction.
    data["contact_force_w"] = [[100, 0, 0]] * 3
    result = episode_summary(data, policy_dt_s=.1)
    assert "D" not in [x["category"] for x in result["failure_categories"]]
    assert result["contact"]["classification_D_supported"] is False
    assert result["motion"]["ee_sampled_path_length_m"] == 0
    assert result["actions"]["official_ema_actions"]["fraction_abs_ge_0p99"] == 1


def test_missing_metrics_stay_null_and_unknown_boolean_is_not_failure():
    result = episode_summary({"policy_step": [1, 2]}, policy_dt_s=.1)
    assert result["timing"]["ever_success"] is None
    assert result["timing"]["terminal_success"] is None
    assert result["geometry"]["xy_error_m"]["final"] is None
    assert result["orientation"]["axis_angle_deg"]["final"] is None
    assert result["actions"]["official_ema_actions"] is None
    assert [x["category"] for x in result["failure_categories"]] == ["F"]
    json.dumps(result, allow_nan=False)


def test_missing_final_value_does_not_silently_substitute_previous_value():
    stats = scalar_stats([.001, .002, float("nan")])
    assert stats["final"] is None
    assert stats["min"] == .001
    assert stats["max"] == .002
    assert stats["valid_samples"] == 2
    assert stats["missing_samples"] == 1


def test_motion_path_velocity_and_action_stats_hand_computed():
    data = trajectory()
    data["ee_pos_w"] = [[0, 0, 0], [3, 0, 0], [3, 4, 0]]
    data["ee_lin_vel_w"] = [[3, 4, 0]] * 3
    data["ee_ang_vel_w"] = [[0, 0, 2]] * 3
    data["actions"] = [[1, -1], [0, 0], [-1, 1]]
    result = episode_summary(data, policy_dt_s=.1)
    assert result["motion"]["ee_sampled_path_length_m"] == 7
    assert result["motion"]["ee_final_minus_initial_m"] == 5
    assert result["motion"]["ee_linear_speed_m_s"]["final"] == 5
    assert result["motion"]["ee_angular_speed_rad_s"]["mean"] == 2
    assert result["actions"]["official_ema_actions"]["rms_all_components"] == pytest.approx(math.sqrt(2/3))
    assert result["actions"]["official_ema_actions"]["fraction_abs_ge_0p99"] == pytest.approx(2/3)


def test_json_serialization_and_provenance_do_not_mutate_trajectory():
    data = trajectory()
    before = copy.deepcopy(data)
    result = episode_summary(data, policy_dt_s=1/15)
    serialized = json.dumps(result, allow_nan=False)
    loaded = json.loads(serialized)
    assert loaded["seed"] == 1001
    assert loaded["env_id"] == 11
    assert loaded["episode_id"] == 0
    assert loaded["samples"] == 3
    assert loaded["coordinate_conventions"]["quaternion_order"] == "wxyz"
    for key in data:
        np.testing.assert_equal(data[key], before[key])


def test_authoritative_success_is_preserved_and_geometry_mismatch_flagged():
    data = trajectory(success=(True,), xy=(.02,), z=(.02,))
    result = episode_summary(data, policy_dt_s=.1)
    assert result["timing"]["ever_success"] is True
    assert result["geometry"]["official_predicate_disagreement_samples"] == 1
    assert result["failure_categories"] == []


@pytest.mark.parametrize("change", [
    {"policy_step": [1., 2., 2.5]},
    {"policy_step": [1, 2, 1]},
    {"xy_error_m": [[0], [0], [0]]},
    {"official_success": [False, 2, True]},
    {"actions": [[1], [2]]},
])
def test_invalid_schema_is_rejected(change):
    data = trajectory()
    data.update(change)
    with pytest.raises(ValueError):
        episode_summary(data, policy_dt_s=.1)


def test_end_effector_quaternion_rotation_path_uses_shortest_so3_increments():
    data = trajectory()
    half = math.sqrt(.5)
    data["ee_quat_w"] = [[1, 0, 0, 0], [half, 0, 0, half], [0, 0, 0, 1]]
    result = episode_summary(data, policy_dt_s=.1)
    assert result["motion"]["ee_orientation_from_initial_deg"]["final"] == pytest.approx(180)
    assert result["motion"]["ee_sampled_rotation_path_deg"] == pytest.approx(180)
