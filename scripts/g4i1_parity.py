"""Offline, fixed-tolerance G4-I1 parity analysis; never imports the simulator.

NPZ input uses ``reference__<stage>`` and ``actual__<stage>`` arrays. Controller
arguments are named ``controller__<argument>``. Arrays may have leading trace
dimensions; their shapes must match exactly, with no implicit broadcasting.
Relative errors use a reporting denominator floor of 1e-12; this floor does
not affect the preregistered absolute-plus-relative acceptance criterion.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


TOLERANCES = {"cpu": (1e-6, 1e-6), "gpu": (1e-5, 1e-5)}
QUATERNION_ANGLE_TOLERANCE_RAD = 1e-4
REQUIRED_STAGES = (
    "observation", "normalized", "mu", "h", "c", "clipped", "ema",
    "target_position", "target_quaternion", "controller_inputs",
)
REQUIRED_CASES = (
    "normal", "action_clipping", "near_zero_rotation", "position_saturation",
    "timeout", "post_reset_first_action",
)
_WIDTHS = {"observation": 19, "normalized": 19, "mu": 6, "clipped": 6,
           "ema": 6, "target_position": 3, "target_quaternion": 4}


def _failure(reason: str) -> dict:
    return {"passed": False, "comparison_count": 0, "sample_count": 0,
            "max_absolute_error": None, "max_relative_error": None,
            "violation_count": 0, "reason": reason}


def _compare_array(name: str, reference, actual, atol: float, rtol: float) -> dict:
    try:
        ref, got = np.asarray(reference), np.asarray(actual)
        if ref.dtype.kind not in "biuf" or got.dtype.kind not in "biuf":
            return _failure("non-numeric array")
        if ref.shape != got.shape:
            return _failure(f"shape mismatch: reference {ref.shape}, actual {got.shape}")
        if ref.size == 0:
            return _failure("empty array")
        if name in _WIDTHS and (ref.ndim == 0 or ref.shape[-1] != _WIDTHS[name]):
            return _failure(f"invalid final dimension for {name}: {ref.shape}")
        if name in ("h", "c") and (ref.ndim < 3 or ref.shape[-3] != 2 or ref.shape[-1] != 1024):
            return _failure(f"invalid recurrent state dimensions: {ref.shape}")
        ref, got = ref.astype(np.float64), got.astype(np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        return _failure(f"invalid array: {exc}")

    finite = np.isfinite(ref) & np.isfinite(got)
    if not finite.all():
        result = _failure("NaN/Inf in reference or actual")
        result.update(comparison_count=int(ref.size), violation_count=int((~finite).sum()),
                      nonfinite_count=int((~finite).sum()))
        return result

    orientation = None
    if name == "target_quaternion":
        ref_norm, got_norm = np.linalg.norm(ref, axis=-1), np.linalg.norm(got, axis=-1)
        if np.any(ref_norm == 0) or np.any(got_norm == 0):
            return _failure("zero-norm quaternion")
        # q and -q encode the same orientation; compare their aligned components.
        got = got * np.where(np.sum(ref * got, axis=-1, keepdims=True) < 0, -1., 1.)
        unit_ref, unit_got = ref / ref_norm[..., None], got / got_norm[..., None]
        # Stable at zero, unlike acos(dot), which loses small FP32-angle detail.
        angles = 4. * np.arctan2(np.linalg.norm(unit_ref - unit_got, axis=-1),
                                 np.linalg.norm(unit_ref + unit_got, axis=-1))
        angle_violations = int((angles > QUATERNION_ANGLE_TOLERANCE_RAD).sum())
        orientation = {"orientation_comparison_count": int(angles.size),
                       "max_orientation_error_rad": float(angles.max()),
                       "orientation_tolerance_rad": QUATERNION_ANGLE_TOLERANCE_RAD,
                       "orientation_violation_count": angle_violations,
                       "orientation_passed": angle_violations == 0}

    error = np.abs(got - ref)
    violations = int((error > atol + rtol * np.abs(ref)).sum())
    result = {
        "passed": violations == 0 and (orientation is None or orientation["orientation_passed"]),
        "comparison_count": int(error.size),
        "sample_count": int(np.prod(ref.shape[:-1])) if ref.ndim else 1,
        "shape": list(ref.shape),
        "max_absolute_error": float(error.max()),
        "max_relative_error": float((error / np.maximum(np.abs(ref), 1e-12)).max()),
        "violation_count": violations,
    }
    if orientation is not None:
        result.update(orientation)
    return result


def compare_stages(reference: Mapping, actual: Mapping, *, device_kind: str = "cpu",
                   cases: Sequence[str] = (), require_all_stages: bool = True,
                   require_cases: Sequence[str] = ()) -> dict:
    """Compare captured identical-input stage arrays without rerunning a policy.

    ``cases`` are evidence labels supplied by the caller, not inferred from the
    numerical data. A caller combining real traces and source-based edge-case
    fixtures must require their union separately via ``require_cases``. A partial
    comparison can pass only when explicitly selected for a stage-specific fixture.
    """
    if device_kind not in TOLERANCES:
        raise ValueError("device_kind must be cpu or gpu; tolerances are frozen")
    atol, rtol = TOLERANCES[device_kind]
    ref_keys, actual_keys = set(reference), set(actual)
    if any(not isinstance(k, str) for k in ref_keys | actual_keys):
        raise ValueError("stage names must be strings")
    common = sorted(ref_keys & actual_keys)
    stages = {name: _compare_array(name, reference[name], actual[name], atol, rtol)
              for name in common}
    covered = set(common)
    if any(name.startswith("controller__") for name in common):
        covered.add("controller_inputs")
    # A nested controller mapping is also accepted by programmatic callers.
    if "controller_inputs" in common and isinstance(reference["controller_inputs"], Mapping):
        ref_controller, got_controller = reference["controller_inputs"], actual["controller_inputs"]
        if isinstance(got_controller, Mapping):
            del stages["controller_inputs"]
            for name in sorted(set(ref_controller) | set(got_controller)):
                key = "controller__" + name
                stages[key] = (_compare_array(key, ref_controller[name], got_controller[name], atol, rtol)
                               if name in ref_controller and name in got_controller
                               else _failure("controller argument missing from one implementation"))
            if not ref_controller or not got_controller:
                stages["controller_inputs"] = _failure("empty controller input mapping")
        else:
            stages["controller_inputs"] = _failure("controller input mapping mismatch")
    missing = sorted(set(REQUIRED_STAGES) - covered) if require_all_stages else []
    recorded_cases = sorted(set(cases))
    missing_cases = sorted(set(require_cases) - set(recorded_cases))
    unmatched_reference = sorted(ref_keys - actual_keys)
    unmatched_actual = sorted(actual_keys - ref_keys)
    coverage_passed = bool(common) and not (missing or missing_cases or unmatched_reference or unmatched_actual)
    passed = bool(coverage_passed and all(stage["passed"] for stage in stages.values()))
    return {
        "schema_version": "g4_i1_parity_v1", "passed": passed,
        "status": "PASS" if passed else "FAIL", "device_kind": device_kind,
        "tolerances": {"atol": atol, "rtol": rtol,
                       "quaternion_angle_rad": QUATERNION_ANGLE_TOLERANCE_RAD},
        "relative_error_reporting_denominator_floor": 1e-12,
        "comparison_count": sum(v["comparison_count"] for v in stages.values()),
        "violation_count": sum(v["violation_count"] for v in stages.values()),
        "orientation_violation_count": sum(v.get("orientation_violation_count", 0) for v in stages.values()),
        "max_absolute_error": max((v["max_absolute_error"] for v in stages.values()
                                    if v["max_absolute_error"] is not None), default=None),
        "max_relative_error": max((v["max_relative_error"] for v in stages.values()
                                    if v["max_relative_error"] is not None), default=None),
        "coverage": {"passed": coverage_passed, "required_all_stages": require_all_stages,
                     "missing_stages": missing, "cases": recorded_cases,
                     "required_cases": sorted(set(require_cases)), "missing_cases": missing_cases,
                     "missing_in_actual": unmatched_reference, "missing_in_reference": unmatched_actual},
        "stages": stages,
    }


def read_paired_npz(path: str | Path) -> tuple[dict, dict]:
    """Read only numeric arrays; pickle/object deserialization is disabled."""
    reference, actual = {}, {}
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            if name.startswith("reference__"):
                reference[name[len("reference__"):]] = data[name]
            elif name.startswith("actual__"):
                actual[name[len("actual__"):]] = data[name]
    return reference, actual


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Paired reference__/actual__ NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-kind", choices=tuple(TOLERANCES), default="cpu")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--require-cases", nargs="*", default=[])
    parser.add_argument("--allow-partial-stages", action="store_true")
    args = parser.parse_args(argv)
    try:
        reference, actual = read_paired_npz(args.input)
        result = compare_stages(reference, actual, device_kind=args.device_kind,
                                cases=args.cases, require_cases=args.require_cases,
                                require_all_stages=not args.allow_partial_stages)
    except (OSError, ValueError, TypeError) as exc:
        result = {"schema_version": "g4_i1_parity_v1", "passed": False,
                  "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except FileExistsError:
        print(f"Refusing to overwrite existing evidence: {args.output}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
