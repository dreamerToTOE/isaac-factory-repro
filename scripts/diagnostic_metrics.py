"""CPU-only, read-only summaries of one pre-reset Factory PegInsert episode.

Quaternion components are Isaac Lab's (w, x, y, z), describing local-to-world
rotations.  The official predicate uses world XY and world Z of the held/target
base frames; it does not use hole-local displacement or an orientation gate.
All added thresholds are labelled secondary.  This module never steps an env.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

SCHEMA_VERSION = 1
COORDINATE_CONVENTIONS = {
    "quaternion_order": "wxyz",
    "quaternion_rotation": "local_to_world",
    "axis_error": "directed angle between peg/hole local +Z axes in world; [0,180] degrees",
    "relative_orientation": "shortest SO(3) geodesic angle; yaw included; q and -q equivalent",
    "official_xy": "norm((held_base_pos_w - target_held_base_pos_w)[:2]); world XY",
    "official_axial_displacement": "held_base_pos_w.z - target_held_base_pos_w.z; world Z; positive above target",
    "sampling": "one sample per policy step at official reward predicate, before automatic reset",
    "success_duration": "count of successful sampled policy steps times policy_dt_s; not continuous physics-time proof",
}


def _quat_unit(q):
    values = np.asarray(q, dtype=float)
    if values.shape[-1:] != (4,):
        raise ValueError("quaternions must have trailing dimension 4 (wxyz)")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    valid = np.isfinite(values).all(axis=-1, keepdims=True) & (norm > 0)
    return np.divide(values, norm, out=np.full_like(values, np.nan), where=valid)


def quaternion_z_axis(q):
    """Rotate local unit +Z by a wxyz quaternion, accepting a leading batch."""
    unit = _quat_unit(q)
    w, x, y, z = np.moveaxis(unit, -1, 0)
    return np.stack((2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)), axis=-1)


def quaternion_axis_angle_deg(peg_quat_w, hole_quat_w):
    """Directed +Z-axis angle; axial roll/yaw does not count as tilt."""
    peg_axis = quaternion_z_axis(peg_quat_w)
    hole_axis = quaternion_z_axis(hole_quat_w)
    return np.degrees(np.arccos(np.clip(np.sum(peg_axis * hole_axis, axis=-1), -1, 1)))


def quaternion_distance_deg(first_quat_w, second_quat_w):
    """SO(3) angle including yaw, invariant to quaternion sign."""
    first, second = _quat_unit(first_quat_w), _quat_unit(second_quat_w)
    dot = np.abs(np.sum(first * second, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))


def official_geometry(held_base_pos_w, target_held_base_pos_w):
    held, target = np.asarray(held_base_pos_w, dtype=float), np.asarray(target_held_base_pos_w, dtype=float)
    if held.shape != target.shape or held.shape[-1:] != (3,):
        raise ValueError("held and target base positions require equal [...,3] shapes")
    delta = held - target
    return np.linalg.norm(delta[..., :2], axis=-1), delta[..., 2]


def _number(value):
    result = float(value)
    return result if math.isfinite(result) else None


def scalar_stats(values):
    """Missing/nonfinite samples stay absent; final always means final row."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("scalar_stats requires one dimension")
    finite = values[np.isfinite(values)]
    return {
        "valid_samples": int(finite.size),
        "missing_samples": int(values.size - finite.size),
        "min": _number(np.min(finite)) if finite.size else None,
        "max": _number(np.max(finite)) if finite.size else None,
        "mean": _number(np.mean(finite)) if finite.size else None,
        "rms": _number(np.sqrt(np.mean(finite**2))) if finite.size else None,
        "initial": _number(values[0]) if values.size else None,
        "final": _number(values[-1]) if values.size else None,
    }


def _mask(values, n):
    if values is None:
        return np.full(n, np.nan)
    result = np.asarray(values, dtype=float)
    if result.shape != (n,):
        raise ValueError("boolean sample fields must have shape [T]")
    if np.any(np.isfinite(result) & (result != 0) & (result != 1)):
        raise ValueError("boolean sample fields accept only bool/0/1/missing")
    return result


def summarize_boolean_timing(values, policy_step, policy_dt_s):
    steps = np.asarray(policy_step, dtype=int)
    if not math.isfinite(policy_dt_s) or policy_dt_s <= 0:
        raise ValueError("policy_dt_s must be finite and positive")
    if steps.ndim != 1 or not np.array_equal(steps, np.arange(1, steps.size + 1)):
        raise ValueError("one episode requires contiguous policy_step 1..T; reset leakage rejected")
    samples = _mask(values, len(steps))
    known = np.isfinite(samples)
    successes = samples == 1
    count, run, longest, runs = int(successes.sum()), 0, 0, 0
    for success in successes:
        if success:
            if run == 0:
                runs += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    success_steps = steps[successes]
    all_known = bool(known.all())
    # A missing sample cannot prove the absence of an ever-success event.
    ever = True if count else (False if all_known else None)
    terminal = bool(samples[-1]) if len(samples) and known[-1] else None
    first = int(success_steps[0]) if count else None
    return {
        "observed_success_samples": count,
        "unknown_samples": int((~known).sum()),
        "ever_success": ever,
        "terminal_success": terminal,
        "first_success_step": first,
        "first_success_time_s": first * policy_dt_s if first is not None else None,
        "first_success_is_exact": bool(first is not None and known[:first].all()),
        "last_success_step": int(success_steps[-1]) if count else None,
        "sampled_success_duration_s": count * policy_dt_s if all_known else None,
        "observed_success_duration_lower_bound_s": count * policy_dt_s,
        "longest_observed_success_run_steps": longest,
        "longest_observed_success_run_s": longest * policy_dt_s,
        "observed_success_runs": runs,
        "timing_complete": all_known,
    }


def _vector_stats(values, n, width=None):
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != n or (width is not None and array.shape[1] != width):
        raise ValueError(f"vector field requires shape [T,{width or 'D'}]")
    row_valid = np.isfinite(array).all(axis=1)
    norm = np.linalg.norm(array, axis=1)
    norm[~row_valid] = np.nan
    valid = array[row_valid]
    return {
        "norm": scalar_stats(norm),
        "per_component": [scalar_stats(array[:, i]) for i in range(array.shape[1])],
        "rms_all_components": _number(np.sqrt(np.mean(valid**2))) if valid.size else None,
        "max_abs_component": _number(np.max(np.abs(valid))) if valid.size else None,
        "fraction_abs_ge_0p99": float(np.mean(np.abs(valid) >= .99)) if valid.size else None,
    }


def classify_failure(summary):
    """Attach descriptive observed failure conditions, never inferred causes.

    A/B use exactly the official terminal gates.  C is a newly specified tilt
    descriptor.  D cannot be established from pose stagnation or net force alone.
    A successful terminal sample receives no failure category.
    """
    timing = summary["timing"]
    categories = []
    if timing["terminal_success"] is True:
        return categories
    geometry, thresholds = summary["geometry"], summary["thresholds"]
    xy, z = geometry["xy_error_m"]["final"], geometry["axial_displacement_m"]["final"]
    if timing["terminal_success"] is False and xy is not None and xy >= thresholds["official_xy_threshold_m"]:
        categories.append({"category": "A", "description": "terminal world-XY alignment gate not satisfied",
                           "evidence": {"final_xy_error_m": xy, "strict_upper_threshold_m": thresholds["official_xy_threshold_m"]},
                           "confidence": "high for observed condition; causal mechanism unresolved"})
    if timing["terminal_success"] is False and z is not None and z >= thresholds["official_z_threshold_m"]:
        categories.append({"category": "B", "description": "terminal official one-sided axial gate not satisfied",
                           "evidence": {"final_axial_displacement_m": z, "strict_upper_threshold_m": thresholds["official_z_threshold_m"]},
                           "confidence": "high for observed condition; causal mechanism unresolved"})
    axis = summary["orientation"]["axis_angle_deg"]["final"]
    if timing["terminal_success"] is False and axis is not None and axis > thresholds["secondary_axis_threshold_deg"]:
        categories.append({"category": "C", "description": "terminal peg/hole +Z tilt exceeds secondary diagnostic threshold",
                           "evidence": {"final_axis_angle_deg": axis, "secondary_threshold_deg": thresholds["secondary_axis_threshold_deg"]},
                           "confidence": "high for measured tilt; association only, not official failure cause"})
    if timing["ever_success"] is True and timing["terminal_success"] is False:
        categories.append({"category": "E", "description": "official success observed before terminal loss",
                           "evidence": {"first_success_step": timing["first_success_step"], "terminal_success": False},
                           "confidence": "high for sampled predicate transition; intervening physics states unsampled"})
    if not categories:
        categories.append({"category": "F", "description": "other / insufficient evidence",
                           "evidence": {"terminal_success": timing["terminal_success"]},
                           "confidence": "insufficient evidence"})
    return categories


def episode_summary(
    trajectory: Mapping,
    *,
    policy_dt_s: float,
    official_xy_threshold_m: float = .0025,
    official_z_threshold_m: float = .001,
    secondary_axis_threshold_deg: float = 5.,
    plausible_z_range_m: tuple[float, float] | None = None,
    hole_height_m: float = .025,
):
    """Summarize one complete episode from the diagnostic NPZ time slices.

    Optional vector fields are [T,D], scalar fields [T]. Missing fields become
    null statistics. Passed nonfinite values become missing, not false success.
    The authoritative success mask is never replaced by a recomputed predicate.
    """
    if "policy_step" not in trajectory:
        raise ValueError("policy_step provenance is required")
    raw_steps = np.asarray(trajectory["policy_step"])
    if raw_steps.ndim != 1 or not np.isfinite(raw_steps).all() or not np.equal(raw_steps, raw_steps.astype(int)).all():
        raise ValueError("policy_step must be an integer vector")
    steps, n = raw_steps.astype(int), len(raw_steps)
    if not n:
        raise ValueError("episode must contain samples")
    policy_dt_s = float(policy_dt_s)
    official_xy_threshold_m = float(official_xy_threshold_m)
    official_z_threshold_m = float(official_z_threshold_m)
    secondary_axis_threshold_deg = float(secondary_axis_threshold_deg)
    hole_height_m = float(hole_height_m)
    thresholds = (official_xy_threshold_m, official_z_threshold_m, secondary_axis_threshold_deg, hole_height_m)
    if not all(math.isfinite(x) and x > 0 for x in thresholds):
        raise ValueError("thresholds must be finite and positive")
    for name in ("seed", "env_id", "episode_id"):
        if name in trajectory:
            value = np.asarray(trajectory[name])
            if value.ndim > 0 and (value.shape != (n,) or not np.all(value == value[0])):
                raise ValueError(f"{name} changes within one episode")
    if "done" in trajectory:
        done = _mask(trajectory["done"], n)
        if not np.isfinite(done).all() or np.any(done[:-1]) or done[-1] != 1:
            raise ValueError("complete episode requires done only at its final pre-reset sample")
    timing = summarize_boolean_timing(trajectory.get("official_success"), steps, policy_dt_s)

    def scalar(name):
        values = np.asarray(trajectory.get(name, np.full(n, np.nan)), dtype=float)
        if values.shape != (n,):
            raise ValueError(f"{name} requires shape [T]")
        return values

    xy, z = scalar("xy_error_m"), scalar("axial_displacement_m")
    xy_from_pose = z_from_pose = None
    if "held_base_pos_w" in trajectory and "target_held_base_pos_w" in trajectory:
        xy_from_pose, z_from_pose = official_geometry(trajectory["held_base_pos_w"], trajectory["target_held_base_pos_w"])
        if xy_from_pose.shape != (n,):
            raise ValueError("base poses require shape [T,3]")
        if "xy_error_m" not in trajectory:
            xy = xy_from_pose
        if "axial_displacement_m" not in trajectory:
            z = z_from_pose
    quat_pair_present = "peg_quat_w" in trajectory and "hole_quat_w" in trajectory
    if quat_pair_present:
        pq, hq = np.asarray(trajectory["peg_quat_w"]), np.asarray(trajectory["hole_quat_w"])
        if pq.shape != (n, 4) or hq.shape != (n, 4):
            raise ValueError("peg/hole quaternions require shape [T,4]")
        axis, relative = quaternion_axis_angle_deg(pq, hq), quaternion_distance_deg(pq, hq)
    else:
        axis, relative = np.full(n, np.nan), np.full(n, np.nan)
    hole_local_z_gap = np.full(n, np.nan)
    hole_local_xy_error = np.full(n, np.nan)
    if quat_pair_present and "held_base_pos_w" in trajectory and "target_held_base_pos_w" in trajectory:
        delta = np.asarray(trajectory["held_base_pos_w"], dtype=float) - np.asarray(trajectory["target_held_base_pos_w"], dtype=float)
        hole_axis = quaternion_z_axis(hq)
        hole_local_z_gap = np.sum(delta * hole_axis, axis=-1)
        hole_local_xy_error = np.sqrt(np.maximum(0., np.sum(delta**2, axis=-1) - hole_local_z_gap**2))
    nominal_depth = hole_height_m - hole_local_z_gap
    geom_valid = np.isfinite(xy) & np.isfinite(z)
    success = _mask(trajectory.get("official_success"), n)
    comparison = geom_valid & np.isfinite(success)
    recomputed = (xy < official_xy_threshold_m) & (z < official_z_threshold_m)
    mismatch = int(np.sum(recomputed[comparison] != success[comparison].astype(bool)))
    out = {
        "schema_version": SCHEMA_VERSION,
        "samples": n,
        "policy_dt_s": float(policy_dt_s),
        "sampled_episode_duration_s": n * policy_dt_s,
        "coordinate_conventions": COORDINATE_CONVENTIONS.copy(),
        "thresholds": {"official_xy_threshold_m": official_xy_threshold_m,
                       "official_z_threshold_m": official_z_threshold_m,
                       "secondary_axis_threshold_deg": secondary_axis_threshold_deg,
                       "secondary_axis_threshold_is_official": False},
        "timing": timing,
        "engagement": summarize_boolean_timing(trajectory.get("engaged"), steps, policy_dt_s),
        "geometry": {"xy_error_m": scalar_stats(xy), "axial_displacement_m": scalar_stats(z),
                     "official_geometry_comparison_samples": int(comparison.sum()),
                     "official_predicate_disagreement_samples": mismatch,
                     "supplied_vs_pose_max_abs_xy_error_m": _number(np.nanmax(np.abs(xy - xy_from_pose))) if xy_from_pose is not None and np.isfinite(xy - xy_from_pose).any() else None,
                     "supplied_vs_pose_max_abs_z_error_m": _number(np.nanmax(np.abs(z - z_from_pose))) if z_from_pose is not None and np.isfinite(z - z_from_pose).any() else None},
        "orientation": {"axis_angle_deg": scalar_stats(axis), "relative_rotation_angle_deg": scalar_stats(relative),
                        "axis_angle_has_official_gate": False},
        "actions": {"official_ema_actions": _vector_stats(trajectory.get("actions"), n),
                    "policy_pre_ema_actions": _vector_stats(trajectory.get("policy_actions"), n),
                    "interpretation": "controller input; neither measured force nor measured contact"},
        "motion": {"ee_linear_speed_m_s": None, "ee_angular_speed_rad_s": None,
                   "ee_sampled_path_length_m": None, "ee_final_minus_initial_m": None,
                   "ee_orientation_from_initial_deg": None, "ee_sampled_rotation_path_deg": None,
                   "ee_path_valid_segments": 0, "interpretation": "measured pose/velocity; small motion does not establish contact obstruction"},
        "contact": {"available": False, "classification_D_supported": False,
                    "limitation": "no independently established contact-obstruction evidence supplied; action/motion are not contact measurements"},
        "secondary": {"hole_axis_projected_base_gap_m": scalar_stats(hole_local_z_gap),
                      "hole_axis_transverse_error_m": scalar_stats(hole_local_xy_error),
                      "nominal_insertion_depth_proxy_m": scalar_stats(nominal_depth),
                      "nominal_depth_formula": "hole_height_m - dot(held_base-target_base, hole_local_+Z_world)",
                      "hole_height_m": hole_height_m,
                      "plausible_z_range_m": [float(x) for x in plausible_z_range_m] if plausible_z_range_m is not None else None,
                      "plausible_z_range_coordinate": "hole_axis_projected_base_gap_m",
                      "plausible_z_range_fraction": None, "terminal_in_plausible_z_range": None,
                      "terminal_secondary_quality": None,
                      "interpretation": "optional asset-dimension-based axial geometric proxy, inclusive bounds; not collision/interpenetration measurement"},
        "limitations": ["Classification describes observed terminal conditions and does not identify a causal mechanism.",
                        "No independent interpenetration measurement is available from these scalar geometric proxies."],
    }
    for name in ("seed", "env_id", "episode_id"):
        if name in trajectory:
            value = np.asarray(trajectory[name])
            out[name] = int(value if value.ndim == 0 else value[0])
    for field, output in (("ee_lin_vel_w", "ee_linear_speed_m_s"), ("ee_ang_vel_w", "ee_angular_speed_rad_s")):
        if field in trajectory:
            values = np.asarray(trajectory[field], dtype=float)
            if values.shape != (n, 3):
                raise ValueError(f"{field} requires shape [T,3]")
            speed = np.linalg.norm(values, axis=1)
            speed[~np.isfinite(values).all(axis=1)] = np.nan
            out["motion"][output] = scalar_stats(speed)
    if "ee_pos_w" in trajectory:
        positions = np.asarray(trajectory["ee_pos_w"], dtype=float)
        if positions.shape != (n, 3):
            raise ValueError("ee_pos_w requires shape [T,3]")
        valid = np.isfinite(positions).all(axis=1)
        segment_valid = valid[1:] & valid[:-1]
        out["motion"]["ee_path_valid_segments"] = int(segment_valid.sum())
        out["motion"]["ee_sampled_path_length_m"] = _number(np.linalg.norm(np.diff(positions, axis=0)[segment_valid], axis=1).sum()) if segment_valid.any() or n == 1 and valid[0] else None
        out["motion"]["ee_final_minus_initial_m"] = _number(np.linalg.norm(positions[-1] - positions[0])) if valid[0] and valid[-1] else None
    if "ee_quat_w" in trajectory:
        quaternions = np.asarray(trajectory["ee_quat_w"], dtype=float)
        if quaternions.shape != (n, 4):
            raise ValueError("ee_quat_w requires shape [T,4]")
        from_initial = quaternion_distance_deg(quaternions, np.broadcast_to(quaternions[0], quaternions.shape))
        out["motion"]["ee_orientation_from_initial_deg"] = scalar_stats(from_initial)
        increments = quaternion_distance_deg(quaternions[1:], quaternions[:-1])
        out["motion"]["ee_sampled_rotation_path_deg"] = _number(increments.sum()) if np.isfinite(increments).all() else None
    if plausible_z_range_m is not None:
        lower, upper = plausible_z_range_m
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError("plausible_z_range_m requires finite lower < upper")
        finite = np.isfinite(hole_local_z_gap)
        in_range = (hole_local_z_gap >= lower) & (hole_local_z_gap <= upper)
        out["secondary"]["plausible_z_range_fraction"] = float(in_range[finite].mean()) if finite.any() else None
        out["secondary"]["terminal_in_plausible_z_range"] = bool(in_range[-1]) if finite[-1] else None
        quality_known = finite[-1] and np.isfinite(axis[-1]) and np.isfinite(xy[-1]) and np.isfinite(success[-1])
        if quality_known:
            out["secondary"]["terminal_secondary_quality"] = bool(success[-1] and in_range[-1] and axis[-1] <= secondary_axis_threshold_deg and xy[-1] < official_xy_threshold_m)
    out["failure_categories"] = classify_failure(out)
    return out
