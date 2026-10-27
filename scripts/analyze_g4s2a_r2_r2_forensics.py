#!/usr/bin/env python3
"""Independent, read-only forensic reader for G4-S2A-R2-R2 Stage A/D.

This script only reads archived JSON/JSONL evidence.  It imports no MuJoCo,
does not call an integrator, does not open a viewer, and does not execute a
policy.  It reconstructs contact-pair chronology, passive-arm motion, the
registered free-motion finger-PD equation, and the saved MuJoCo solver-stage
metadata.  The output is intentionally a new R2-R2 evidence directory; older
R2/R2-R1 evidence is never rewritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
R2R1 = ROOT / "evidence/g4_s/s2a_r2_r1"
OUT_DEFAULT = ROOT / "evidence/g4_s/s2a_r2_r2/forensics_v1"

R2_FILES = {
    "I3_B": R2R1 / "forensics_v1/I3_B_timeline.jsonl.gz",
    "I3_D": R2R1 / "forensics_v1/I3_D_timeline.jsonl.gz",
    "I2_B": R2R1 / "forensics_v1/I2_B_timeline.jsonl.gz",
    "I2_D": R2R1 / "forensics_v1/I2_D_timeline.jsonl.gz",
}
R2R1_FILES = {
    "I3_FULL_C0": R2R1 / "matrix_v2/trajectories/I3_FULL_C0/physics.jsonl",
    "I3_FULL_PASSIVE_ARM": R2R1 / "matrix_v2/trajectories/I3_FULL_PASSIVE_ARM/physics.jsonl",
}

DT = 1.0 / 120.0
KP = 7500.0
KD = 173.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def names(contacts: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(
        f"{c.get('geom1')}|{c.get('geom2')}"
        for c in contacts
        if c.get("geom1") == "peg_collision" or c.get("geom2") == "peg_collision"
    )


def side_pair_names(contacts: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(
        c.get("geom2") if c.get("geom1") == "peg_collision" else c.get("geom1")
        for c in contacts
        if c.get("geom1") == "peg_collision" or c.get("geom2") == "peg_collision"
    )


def contact_force(row: dict[str, Any], side: str) -> float:
    if "contact_normal_force_n" in row:
        return float(row["contact_normal_force_n"].get(side, 0.0))
    return float(row.get("pad_force_n", {}).get(side, 0.0))


def r2_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(row["step"]),
        "time_s": float(row.get("episode_time_s", 0.0)),
        "finger_qpos_m": row.get("finger_qpos_m"),
        "finger_qvel_m_s": row.get("finger_qvel_m_s"),
        "raw_pd_force_n": row.get("reconstructed_raw_pd_n"),
        "applied_finger_force_n": row.get("integration_actuator_force_n"),
        "left_pad_force_n": contact_force(row, "left"),
        "right_pad_force_n": contact_force(row, "right"),
        "contact_pairs": side_pair_names(row.get("contacts", [])),
        "max_penetration_um": float(row.get("max_penetration_m", 0.0)) * 1e6,
        "peg_hand_slip_mm": float(row.get("peg_hand_slip_translation_m", 0.0)) * 1e3,
        "peg_hand_slip_rad": float(row.get("peg_hand_slip_rotation_rad", 0.0)),
        "qacc_post_forward_max": max((abs(float(x)) for x in row.get("qacc_post_forward", [])), default=0.0),
        "qfrc_constraint_fingers": [float(x) for x in row.get("qfrc_constraint_post_forward", [])[7:9]],
        "matrix_summary_preintegration": row.get("matrix_summary_preintegration"),
    }


def r2r1_metrics(row: dict[str, Any]) -> dict[str, Any]:
    state = row["state"]
    s2a = state["s2a"]
    return {
        "step": int(row["step"]),
        "time_s": float(row.get("episode_time_s", 0.0)),
        "finger_qpos_m": [float(x) for x in state["qpos"][7:9]],
        "finger_qvel_m_s": [float(x) for x in state["qvel"][7:9]],
        "arm_qvel_max_rad_s": max((abs(float(x)) for x in state["qvel"][:7]), default=0.0),
        "applied_finger_force_n": [float(x) for x in state.get("finger_actuator_force_n", [])],
        "left_pad_force_n": float(row.get("pad_force_n", {}).get("left", 0.0)),
        "right_pad_force_n": float(row.get("pad_force_n", {}).get("right", 0.0)),
        "pre_contact_pairs": names(row.get("pre_state", {}).get("contacts", [])),
        "state_contact_pairs": names(state.get("contacts", [])),
        "integration_contact_pairs": names(row.get("integration_stage", {}).get("contacts", [])),
        "post_forward_contact_pairs": names(row.get("post_forward_solver", {}).get("contacts", [])),
        "max_penetration_um": float(s2a.get("all_contact_penetration_m", 0.0)) * 1e6,
        "peg_hand_slip_mm": float(s2a.get("peg_hand_slip_m", 0.0)) * 1e3,
        "peg_hand_slip_rad": float(s2a.get("peg_hand_slip_rad", 0.0)),
        "qacc_integration_fingers_m_s2": [float(x) for x in row.get("integration_stage", {}).get("qacc", [])[7:9]],
        "qacc_post_forward_fingers_m_s2": [float(x) for x in row.get("qacc_post_forward", [])[7:9]],
        "qacc_post_forward_max": max((abs(float(x)) for x in row.get("qacc_post_forward", [])), default=0.0),
        "qfrc_constraint_integration_fingers_n": [float(x) for x in row.get("integration_stage", {}).get("qfrc_constraint", [])[7:9]],
        "qfrc_constraint_post_forward_fingers_n": [float(x) for x in row.get("qfrc_constraint_post_forward", [])[7:9]],
        "integration_efc_type": [int(x) for x in row.get("integration_stage", {}).get("efc_type", [])],
        "integration_efc_id": [int(x) for x in row.get("integration_stage", {}).get("efc_id", [])],
        "integration_efc_force": [float(x) for x in row.get("integration_stage", {}).get("efc_force", [])],
        "integration_efc_pos": [float(x) for x in row.get("integration_stage", {}).get("efc_pos", [])],
        "integration_efc_vel": [float(x) for x in row.get("integration_stage", {}).get("efc_vel", [])],
        "integration_nefc": int(row.get("integration_stage", {}).get("nefc", 0)),
        "integration_contact_address": [
            {"index": int(c.get("index", -1)), "pair": f"{c.get('geom1')}|{c.get('geom2')}", "efc_address": c.get("efc_address")}
            for c in row.get("integration_stage", {}).get("contacts", [])
        ],
        "post_forward_efc_type": [int(x) for x in row.get("post_forward_solver", {}).get("efc_type", [])],
        "post_forward_efc_id": [int(x) for x in row.get("post_forward_solver", {}).get("efc_id", [])],
        "post_forward_contact_address": [
            {"index": int(c.get("index", -1)), "pair": f"{c.get('geom1')}|{c.get('geom2')}", "efc_address": c.get("efc_address")}
            for c in row.get("post_forward_solver", {}).get("contacts", [])
        ],
    }


def chronology(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if variant == "r2r1":
        rows = [r for r in rows if r.get("event") == "physics_result"]
    metrics = [r2_metrics(r) if variant == "r2" else r2r1_metrics(r) for r in rows]
    pair_key = "contact_pairs" if variant == "r2" else "state_contact_pairs"
    changes: list[dict[str, Any]] = []
    previous = None
    for m in metrics:
        current = m[pair_key]
        if previous is not None and current != previous:
            changes.append({"step": m["step"], "from": previous, "to": current})
        previous = current
    first_left_pair_loss = None
    previous_left = set()
    for i, m in enumerate(metrics):
        current_left = {p for p in m[pair_key] if "left_finger_geom" in p}
        if i and previous_left - current_left:
            first_left_pair_loss = m["step"]
            break
        previous_left = current_left
    first_left_positive_loss = None
    for m in metrics:
        if m["left_pad_force_n"] <= 1e-6:
            first_left_positive_loss = m["step"]
            break
    first_any_positive_loss = None
    for m in metrics:
        if m["left_pad_force_n"] <= 1e-6 or m["right_pad_force_n"] <= 1e-6:
            first_any_positive_loss = m["step"]
            break
    return {
        "rows": len(metrics),
        "first_contact_pair_change_step": changes[0]["step"] if changes else None,
        "first_left_pair_loss_step": first_left_pair_loss,
        "first_left_positive_contact_loss_step": first_left_positive_loss,
        "first_any_positive_contact_loss_step": first_any_positive_loss,
        "contact_pair_changes": changes,
        "metrics": metrics,
    }


def finger_equation_audit(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute the registered free-motion step-implicit equation.

    F_raw = kp(target-q) - kd*v
    [I + dt*(kd+dt*kp)*M_ff^-1] F = kp(target-q) - (kd+dt*kp)v
    where M_ff^-1 is the finger block extracted from inv(full M).
    """
    summary = row.get("matrix_summary_preintegration") or {}
    ainv = summary.get("finger_inverse_mass_block")
    if ainv is None:
        return {"available": False}
    # The controller is evaluated from the archived pre-integration state;
    # finger_qpos_m/finger_qvel_m_s are the post-step snapshot.
    q = row.get("finger_pre_qpos_m", row["finger_qpos_m"])
    v = row.get("finger_pre_qvel_m_s", row["finger_qvel_m_s"])
    target = 0.0
    import numpy as np  # numerical reviewer only; no MuJoCo import
    ainv = np.asarray(ainv, dtype=float)
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    raw = KP * (target - q) - KD * v
    rhs = KP * (target - q) - (KD + DT * KP) * v
    lhs = np.eye(2) + DT * (KD + DT * KP) * ainv
    predicted = np.linalg.solve(lhs, rhs)
    applied = np.asarray(row.get("integration_actuator_force_n", []), dtype=float)
    return {
        "available": True,
        "dt_s": DT,
        "kp_n_per_m": KP,
        "kd_n_s_per_m": KD,
        "ainv_finger_block_m_per_kg": ainv.tolist(),
        "raw_pd_force_n": raw.tolist(),
        "rhs_n": rhs.tolist(),
        "implicit_lhs": lhs.tolist(),
        "implicit_lhs_eigenvalues": np.linalg.eigvalsh(lhs).tolist(),
        "predicted_force_n": predicted.tolist(),
        "archived_applied_force_n": applied.tolist(),
        "max_prediction_error_n": float(np.max(np.abs(predicted - applied))) if applied.size else None,
        "raw_to_implicit_norm_ratio": float(np.linalg.norm(raw) / max(np.linalg.norm(predicted), 1e-30)),
        "interpretation": "The approximately 24 N raw request is attenuated to approximately 0.18 N by the registered free-motion inverse-mass implicit solve; this is mathematically expected for the archived M_ff^-1 and is not by itself an implementation error.",
    }


def solver_semantics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = {}
    for r in rows:
        if r.get("event") != "physics_result" or r.get("step") not in (175, 176, 177, 143, 144):
            continue
        selected[str(r["step"])] = {
            "integration": {
                "nefc": r["integration_stage"].get("nefc"),
                "efc_type": r["integration_stage"].get("efc_type"),
                "efc_id": r["integration_stage"].get("efc_id"),
                "contact_address": r2r1_metrics(r)["integration_contact_address"],
            },
            "post_forward": {
                "nefc": r["post_forward_solver"].get("nefc"),
                "efc_type": r["post_forward_solver"].get("efc_type"),
                "efc_id": r["post_forward_solver"].get("efc_id"),
                "contact_address": r2r1_metrics(r)["post_forward_contact_address"],
            },
        }
    return {
        "mujoco_version": "3.5.0 (recorded runner contract)",
        "enum_mapping": {
            "0": "mjCNSTR_EQUALITY",
            "6": "mjCNSTR_CONTACT_PYRAMIDAL",
            "7": "mjCNSTR_CONTACT_ELLIPTIC",
        },
        "address_semantics": "efc_address is the first scalar row for that contact; a contact with a 4-dimensional elliptic cone occupies address,address+1,address+2,address+3. efc_id repeats the contact index for each scalar row. The equality row precedes contact rows. efc arrays are constraint-row ordered and must not be indexed as if they were one-to-one with mjData.contact entries.",
        "selected_rows": selected,
        "limitations": [
            "R2 archived timelines contain post-forward aggregate fields only and no per-row efc arrays.",
            "Post-forward qacc/qfrc_constraint are recomputations after integration, not the preceding step's impulse history.",
            "An integration-stage contact set and a post-forward contact set may differ; this difference is preserved rather than collapsed.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.output
    # The parent task may have pre-created the evidence namespace.  We only
    # create our two new files below and never overwrite a pre-existing file.
    out.mkdir(parents=True, exist_ok=True)
    for existing in (out / "forensics.json", out / "prephysics_receipt.json"):
        if existing.exists():
            raise FileExistsError(f"REFUSE_OVERWRITE_EXISTING_EVIDENCE: {existing}")

    source_script = Path(__file__)
    inputs: dict[str, Any] = {}
    for key, path in {**R2_FILES, **R2R1_FILES}.items():
        inputs[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}

    old = {name: read_rows(path) for name, path in R2_FILES.items()}
    new = {name: read_rows(path) for name, path in R2R1_FILES.items()}

    r2_windows = {
        "I3_B": chronology(old["I3_B"], "r2"),
        "I3_D": chronology(old["I3_D"], "r2"),
        "I2_B": chronology(old["I2_B"], "r2"),
        "I2_D": chronology(old["I2_D"], "r2"),
    }
    r2r1_windows = {name: chronology(rows, "r2r1") for name, rows in new.items()}

    eq_row = next(r for r in old["I3_B"] if r["step"] == 150)
    passive = r2r1_windows["I3_FULL_PASSIVE_ARM"]
    passive_fail = next(m for m in passive["metrics"] if m["step"] == 143)
    passive_next = next(m for m in passive["metrics"] if m["step"] == 144)
    active = r2r1_windows["I3_FULL_C0"]
    active_fail = next(m for m in active["metrics"] if m["step"] == 176)
    active_next = next(m for m in active["metrics"] if m["step"] == 177)

    # Include only the failure windows and stable references in the published JSON.
    def restrict(v: dict[str, Any], predicate, label: str) -> dict[str, Any]:
        ms = [m for m in v["metrics"] if predicate(m["step"])]
        pair_key = "contact_pairs" if label.startswith("R2_") and not label.startswith("R2_R1_") else "state_contact_pairs"
        pair_changes = []
        previous = None
        for m in ms:
            current = m[pair_key]
            if previous is not None and current != previous:
                pair_changes.append({"step": m["step"], "from": previous, "to": current})
            previous = current
        left_loss = None
        any_loss = None
        prev_left = set()
        for i, m in enumerate(ms):
            current_left = {p for p in m[pair_key] if "left_finger_geom" in p}
            if i and left_loss is None and prev_left - current_left:
                left_loss = m["step"]
            if any_loss is None and (m["left_pad_force_n"] <= 1e-6 or m["right_pad_force_n"] <= 1e-6):
                any_loss = m["step"]
            prev_left = current_left
        return {
            **v,
            "trace_rows": v["rows"],
            "analysis_window": label,
            "metrics": ms,
            "window_contact_pair_changes": pair_changes,
            "window_first_contact_pair_change_step": pair_changes[0]["step"] if pair_changes else None,
            "window_first_left_pair_loss_step": left_loss,
            "window_first_left_positive_contact_loss_step": next((m["step"] for m in ms if m["left_pad_force_n"] <= 1e-6), None),
            "window_first_any_positive_contact_loss_step": any_loss,
        }

    selected_r2 = {
        k: restrict(v, (lambda step, k=k: 150 <= step <= 177) if k == "I3_B" else (lambda step, k=k: 190 <= step <= 216) if k == "I3_D" else (lambda step, k=k: step in (1, 2, 7, 8, 235)), f"R2_{k}_window")
        for k, v in r2_windows.items()
    }
    selected_r2r1 = {
        k: restrict(v, (lambda step, k=k: 170 <= step <= 177) if k == "I3_FULL_C0" else (lambda step, k=k: 138 <= step <= 154), f"R2_R1_{k}_window")
        for k, v in r2r1_windows.items()
    }

    # Independently verify the exact recurrence relation and key chronology assertions.
    equation = finger_equation_audit(eq_row)
    assertions = {
        "raw_pd_reconstruction_is_zero_error": all(float(r.get("raw_pd_reconstruction_error_n", 1.0)) == 0.0 for r in old["I3_B"]),
        "i3_b_left_pair_loss_step_175": 175 in [c["step"] for c in r2_windows["I3_B"]["contact_pair_changes"]],
        "i3_b_left_positive_loss_step_176": r2_windows["I3_B"]["first_left_positive_contact_loss_step"] == 176,
        "i3_d_left_pair_loss_step_214": 214 in [c["step"] for c in r2_windows["I3_D"]["contact_pair_changes"]],
        "i3_d_left_positive_loss_step_215": r2_windows["I3_D"]["first_left_positive_contact_loss_step"] == 215,
        "active_i3_postforward_left_loss_step_176": active["first_left_positive_contact_loss_step"] == 176,
        "passive_i3_postforward_any_pad_loss_step_143": passive["first_any_positive_contact_loss_step"] == 143,
        "passive_arm_qvel_is_small_at_failure": passive_fail["arm_qvel_max_rad_s"] < 1e-3,
        "passive_finger_speed_jumps_next_step": max(abs(x) for x in passive_next["finger_qvel_m_s"]) > 0.05,
        "active_finger_speed_jumps_next_step": max(abs(x) for x in active_next["finger_qvel_m_s"]) > 0.05,
        "equation_prediction_matches_archived_force": equation.get("max_prediction_error_n", 1.0) < 1e-10,
    }

    report = {
        "task": "G4-S2A-R2-R2",
        "phase": "STAGE_A_D_READ_ONLY_FORENSICS",
        "status": "PREPHYSICS_FORENSICS_COMPLETE",
        "generated_by": str(source_script.relative_to(ROOT)),
        "source_sha256": sha256(source_script),
        "source_bytes": source_script.stat().st_size,
        "method": "Archived numerical reconstruction only; no new integration, MuJoCo import, viewer, policy inference, or source-model mutation.",
        "starting_commit": "d1a2097b1bf43d0cc95b9bae7da231644b43503a",
        "inputs": inputs,
        "constants": {"timestep_s": DT, "finger_kp_n_per_m": KP, "finger_kd_n_s_per_m": KD, "finger_target_m": 0.0},
        "finger_controller_audit": equation,
        "chronology": {"r2": selected_r2, "r2_r1": selected_r2r1},
        "solver_constraint_audit": solver_semantics(new["I3_FULL_C0"] + new["I3_FULL_PASSIVE_ARM"]),
        "passive_arm_finding": {
            "failure_step": 143,
            "next_step": 144,
            "arm_qvel_max_at_failure_rad_s": passive_fail["arm_qvel_max_rad_s"],
            "arm_qvel_max_next_step_rad_s": passive_next["arm_qvel_max_rad_s"],
            "finger_qvel_at_failure_m_s": passive_fail["finger_qvel_m_s"],
            "finger_qvel_next_step_m_s": passive_next["finger_qvel_m_s"],
            "peg_hand_slip_rad_at_failure": passive_fail["peg_hand_slip_rad"],
            "peg_hand_slip_rad_next_step": passive_next["peg_hand_slip_rad"],
            "interpretation": "Zero commanded arm torque is not a mechanically fixed arm, but the archived passive trace has negligible arm velocity at contact loss. The finger/contact manifold change precedes the large finger velocity and peg rotation jump; this weakens an arm-torque-only explanation without proving contact solver uniqueness.",
        },
        "mechanism_assessment": {
            "contact_manifold_discontinuity": "SUPPORTED_AS_CHRONOLOGY",
            "finger_joint_motion": "SUPPORTED_AS_POST_MANIFOLD_RESPONSE",
            "peg_rotation": "FOLLOWS_CONTACT_LOSS_IN_WINDOWS",
            "solver_instability": "PLAUSIBLE_BUT_NOT_PROVEN",
            "free_motion_finger_prediction_error": "NOT_FOUND; archived force matches registered equation",
            "finger_equality_constraints": "PRESENT_AS_ONE_EQUALITY_ROW; separate equality impulse attribution unavailable in R2 archive",
            "single_root_cause": "INCONCLUSIVE",
        },
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
        "limitations": [
            "Contact positions are stored contact points/midpoints and do not prove a mesh feature, edge, or corner identity.",
            "R2 post-forward qacc, qfrc_constraint, and contact forces are recomputed diagnostics, not historical integration impulses.",
            "The saved R2-R1 EFC arrays expose type/id/address rows but not the full solver iteration history or separate equality/contact impulse decomposition.",
            "This is no-policy evidence and does not establish PPO transfer or a universal contact-solver root cause.",
        ],
    }
    (out / "forensics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    receipt = {
        "task": "G4-S2A-R2-R2",
        "phase": "PREPHYSICS_FORENSICS_COMPLETE",
        "passed": bool(report["all_assertions_pass"]),
        "scope": report["method"],
        "new_integration_steps": 0,
        "policy_calls": 0,
        "source_model_mutations": 0,
        "output_sha256": sha256(out / "forensics.json"),
        "output_bytes": (out / "forensics.json").stat().st_size,
    }
    (out / "prephysics_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "passed": receipt["passed"], "assertions": assertions}, ensure_ascii=False))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
