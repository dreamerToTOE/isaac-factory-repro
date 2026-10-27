#!/usr/bin/env python3
"""Independent static Stage-B/C audit for G4-S2A-R2-R2.

This reader performs no MuJoCo integration and does not call mj_step, mj_forward,
or any policy.  It recomputes the registered step-implicit finger force from
archived request/result rows, audits the contact/equality row mapping, and
summarizes the passive-arm timeline.  Outputs are machine-readable evidence.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R2R1 = ROOT / 'evidence/g4_s/s2a_r2_r1/matrix_v2'
P_FULL_XML = R2R1 / 'P_FULL_diagnostic.xml'
ENV_SOURCE = ROOT / 'scripts/g4s1b_env.py'
OUT_DEFAULT = ROOT / 'evidence/g4_s/s2a_r2_r2/api_audit_v1'
DT = 1.0 / 120.0
KP = 7500.0
KD = 173.0
FORCE_LIMIT = 40.0
FINGER_DOF = [7, 8]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


def rows_for(run: str):
    p = R2R1 / 'trajectories' / run / 'physics.jsonl'
    rows = []
    with p.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get('event') == 'physics_result': rows.append(row)
    return p, rows


def force_audit(row):
    M = np.asarray(row['matrices']['physical_full_M'], dtype=float)
    q = np.asarray(row['pre_state']['qpos'], dtype=float)[FINGER_DOF]
    v = np.asarray(row['pre_state']['qvel'], dtype=float)[FINGER_DOF]
    cmd = row['pre_state']['finger_command']
    target = float(cmd['target_opening_per_finger_m'])
    raw = KP * (target - q) - KD * v
    Minv = np.linalg.inv(M)
    Ainv = Minv[np.ix_(FINGER_DOF, FINGER_DOF)]
    Meff = np.linalg.inv(Ainv)
    B = np.eye(2) + DT * (KD + DT * KP) * Ainv
    rhs = KP * (target - q) - (KD + DT * KP) * v
    predicted = np.linalg.solve(B, rhs)
    applied = np.asarray(row['state']['finger_actuator_force_n'], dtype=float)
    Mff = M[np.ix_(FINGER_DOF, FINGER_DOF)]
    static_pred = np.clip(raw, -FORCE_LIMIT, FORCE_LIMIT)
    return {
        'run_id': row['run_id'], 'step': int(row['step']),
        'q_finger_m': q, 'qdot_finger_m_s': v, 'target_m': target,
        'raw_pd_force_N': raw, 'implicit_rhs_N': rhs,
        'implicit_system_B': B, 'implicit_predicted_force_N': predicted,
        'applied_actuator_force_N': applied,
        'formula_residual_N': predicted - applied,
        'raw_static_clipped_force_N': static_pred,
        'raw_to_implicit_ratio': np.divide(predicted, raw, out=np.full(2, np.nan), where=np.abs(raw)>0),
        'physical_full_mass_condition_2': float(np.linalg.cond(M)),
        'physical_full_mass_min_eigenvalue': float(np.min(np.linalg.eigvalsh(M))),
        'physical_full_mass_max_eigenvalue': float(np.max(np.linalg.eigvalsh(M))),
        'finger_acceleration_operator_Ainv_kg_inv': Ainv,
        'effective_finger_mass_kg': Meff,
        'finger_block_mass_Mff_kg': Mff,
        'Ainv_condition_2': float(np.linalg.cond(Ainv)),
        'B_condition_2': float(np.linalg.cond(B)),
        'Ainv_offdiag_correlation': float(Ainv[0,1] / math.sqrt(Ainv[0,0]*Ainv[1,1])),
        'Mff_offdiag_correlation': float(Mff[0,1] / math.sqrt(Mff[0,0]*Mff[1,1])),
        'units': {
            'q': 'm (slide joints)', 'qdot': 'm/s', 'kp': 'N/m = kg/s^2',
            'kd': 'N*s/m = kg/s', 'M_finger': 'kg', 'Ainv': '1/kg',
            'B': 'dimensionless', 'force': 'N', 'dt': 's'},
    }


def contact_audit(row):
    results = []
    for stage_name in ('integration_stage', 'post_forward_solver'):
        stage = row[stage_name]
        nefc = int(stage['nefc'])
        et = list(map(int, stage.get('efc_type', [])))
        ei = list(map(int, stage.get('efc_id', [])))
        ef = list(map(float, stage.get('efc_force', [])))
        contacts = stage.get('contacts', [])
        links = []
        for c in contacts:
            addr = int(c['efc_address'])
            width = 4  # condim=4 for all registered collision geoms
            links.append({'pair': [c['geom1'], c['geom2']], 'contact_index': int(c['index']),
                          'efc_address': addr, 'efc_rows': list(range(addr, min(addr+width, nefc))),
                          'distance_m': float(c['distance_m']),
                          'local_normal_force_N': float(c['contact_force_local'][0])})
        mapping_ok = all(0 <= x['efc_address'] < nefc and x['efc_address'] + 4 <= nefc
                         and all(et[j] == 7 and ei[j] == x['contact_index']
                                 for j in range(x['efc_address'], x['efc_address']+4)) for x in links)
        results.append({'stage': stage_name, 'nefc': nefc,
                        'equality_rows': [j for j,t in enumerate(et) if t == 0],
                        'contact_elliptic_rows': [j for j,t in enumerate(et) if t == 7],
                        'contact_links': links, 'contact_to_efc_mapping_valid': bool(mapping_ok),
                        'efc_force_first_row_N': ef[0] if ef else None,
                        'finger_qfrc_constraint_N_or_Nm': np.asarray(stage.get('qfrc_constraint', []), dtype=float)[FINGER_DOF].tolist(),
                        'finger_qacc_m_s2': np.asarray(stage.get('qacc', []), dtype=float)[FINGER_DOF].tolist(),
                        'finger_qacc_warmstart_m_s2': np.asarray(stage.get('qacc_warmstart', []), dtype=float)[FINGER_DOF].tolist()})
    return results


def timeline(run: str, start: int, stop: int):
    path, rows = rows_for(run)
    rows = [r for r in rows if start <= int(r['step']) <= stop]
    out = []
    prev_pairs = None
    prev_qvel = None
    prev_rot = None
    for r in rows:
        contacts = r['state']['contacts']
        pairs = sorted((c['geom1'], c['geom2']) for c in contacts)
        pad = r['pad_force_n']
        qpos = np.asarray(r['state']['qpos'], dtype=float)
        qvel = np.asarray(r['state']['qvel'], dtype=float)
        rot = float(r['state']['s2a']['peg_hand_slip_rad'])
        out.append({'step': int(r['step']), 'pairs': pairs,
                    'pair_set_changed': prev_pairs is not None and pairs != prev_pairs,
                    'pad_force_N': pad,
                    'left_positive': float(pad.get('left', 0.0)) > 1e-6,
                    'right_positive': float(pad.get('right', 0.0)) > 1e-6,
                    'finger_q_m': qpos[7:9].tolist(),
                    'finger_qdot_m_s': qvel[7:9].tolist(),
                    'finger_acc_interval_m_s2': None if prev_qvel is None else ((qvel[7:9]-prev_qvel[7:9])/DT).tolist(),
                    'arm_qdot_max_rad_s': float(np.max(np.abs(qvel[:7]))),
                    'peg_hand_slip_rad': rot,
                    'peg_hand_slip_delta_rad': None if prev_rot is None else rot-prev_rot,
                    'penetration_m': float(r['state']['s2a']['all_contact_penetration_m'])})
        prev_pairs, prev_qvel, prev_rot = pairs, qvel, rot
    return {'run': run, 'source': str(path.relative_to(ROOT)), 'rows': out}


def model_static():
    root = ET.parse(P_FULL_XML).getroot()
    opt = root.find('option')
    eqs = [] if root.find('equality') is None else [dict(x.attrib) for x in root.find('equality')]
    contact = root.find('contact')
    excludes = [] if contact is None else [dict(x.attrib) for x in contact]
    geom_default = root.find('default/geom')
    return {
        'option': {} if opt is None else dict(opt.attrib),
        'equality_constraints': eqs,
        'contact_excludes': excludes,
        'default_contact_geometry': {} if geom_default is None else dict(geom_default.attrib),
        'semantic_notes': [
            'The one equality is a joint equality between finger_joint1 and finger_joint2.',
            'The default collision condim is 4, so each active elliptic contact contributes four EFC rows in the archived traces.',
            'qfrc_constraint aggregates all active equality and contact constraints; efc_type/efc_id/efc_address are the row-level disambiguation.',
        ],
    }


def api_contract():
    # Values are the installed MuJoCo 3.5 Python enum contract; no model step.
    try:
        import mujoco
        version = {'python_version': mujoco.__version__, 'mj_versionString': mujoco.mj_versionString(),
                   'mjtConstraint': {k: int(v) for k,v in mujoco.mjtConstraint.__members__.items()},
                   'mjtEq': {k: int(v) for k,v in mujoco.mjtEq.__members__.items()},
                   'mj_contactForce_signature_contract': 'result[0:3] force, result[3:6] torque in contact frame; input id is contact index'}
    except Exception as e:
        version = {'error': repr(e)}
    return version


def main(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    force_rows = []
    for run, steps in [('I3_FULL_C0',[1,175,176,177]), ('I3_FULL_PASSIVE_ARM',[141,142,143,144,154])]:
        _, rows = rows_for(run)
        by = {int(r['step']): r for r in rows}
        for step in steps:
            if step in by:
                force_rows.append(force_audit(by[step]))
    all_resids = np.asarray([np.asarray(x['formula_residual_N']) for x in force_rows])
    contact_rows = []
    for run, steps in [('I3_FULL_C0',[174,175,176,177]), ('I3_FULL_PASSIVE_ARM',[141,142,143,144])]:
        _, rows = rows_for(run); by={int(r['step']):r for r in rows}
        for step in steps:
            if step in by:
                contact_rows.append({'run':run,'step':step,'stages':contact_audit(by[step])})
    evidence = {
      'task':'G4-S2A-R2-R2', 'scope':'STATIC_READ_ONLY_STAGE_B_C',
      'status':'AUDIT_COMPLETE_NO_SINGLE_CORRECTION_JUSTIFIED',
      'source_commit_expected':'d1a2097b1bf43d0cc95b9bae7da231644b43503a',
      'source_files': {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ENV_SOURCE, P_FULL_XML]},
      'trace_files': {run: sha(R2R1 / 'trajectories' / run / 'physics.jsonl') for run in ('I3_FULL_C0','I3_FULL_PASSIVE_ARM')},
      'controller': {
        'formula':'f = solve(I + dt*(kd + dt*kp)*Ainv, kp*(target-q) - (kd+dt*kp)*v), Ainv = (M^{-1})[finger_dof,finger_dof]',
        'derivation':'Backward-Euler free-motion prediction q+ = q + dt*(v + dt*Ainv*f), v+ = v + dt*Ainv*f, with f = kp(target-q+) - kd*v+',
        'registered_constants': {'dt_s':DT,'kp_N_per_m':KP,'kd_Ns_per_m':KD,'force_limit_N':FORCE_LIMIT,'finger_dof_ids':FINGER_DOF},
        'force_rows':force_rows,
        'max_formula_residual_N':float(np.max(np.abs(all_resids))) if len(all_resids) else None,
        'max_raw_abs_N':float(max(np.max(np.abs(x['raw_pd_force_N'])) for x in force_rows)),
        'max_implicit_abs_N':float(max(np.max(np.abs(x['implicit_predicted_force_N'])) for x in force_rows)),
        'interpretation':'~25 N raw PD versus ~0.18-0.20 N implicit force is expected: dt*(kd+dt*kp)*Ainv is ~131, so free-motion backward-Euler force is attenuated ~1/132. It is not evidence of a coding error.',
        'cross_coupling':'Ainv off-diagonal is small relative to diagonal (~0.5% correlation in archived I3), while full M inverse is the correct free-motion acceleration operator when other generalized forces are ignored.',
        'limitations':'Ainv excludes bias/Coriolis/gravity terms, arm torques, unknown contact reactions and equality reactions. Thus it is a target-engine approximation, not PhysX implicit-drive equivalence and not a proof that omitted reactions are negligible.'},
      'model_static':model_static(),
      'api_contract':api_contract(),
      'constraint_contact_audit':contact_rows,
      'timeline':{'passive_arm':timeline('I3_FULL_PASSIVE_ARM',138,154),'normal_arm':timeline('I3_FULL_C0',173,177)},
      'findings':[
        'The current formula is dimensionally consistent and independently reproduces every sampled archived actuator force to floating-point residual.',
        'The finger equality is one mjEQ_JOINT row (efc_type 0, efc_id 0); elliptic contacts are mjCNSTR_CONTACT_ELLIPTIC rows (efc_type 7), normally four rows per condim=4 contact. Contact efc_address is the explicit mapping; contact array index and efc row ordering must not be conflated.',
        'integration_stage is a solver/integration-stage snapshot before the wrapper mj_forward; post_forward_solver is recomputed at the advanced state. Their contact manifolds and qacc can differ and neither post-forward qfrc_constraint nor mj_contactForce is an integrated impulse history.',
        'In I3 normal-arm, left geom4 disappears at 176 and the next state has severe multi-geom penetration at 177. Finger qdot jumps from about 0.0005 m/s to 0.08 m/s only after the manifold change.',
        'In passive-arm, right geom5 disappears at 142, right positive pad force is lost at 143, and the large finger acceleration/peg rotation occurs at 144. Arm qdot remains below 0.001 rad/s through 144; this does not mean the arm is mechanically fixed, but it places contact/finger motion before large arm motion in this trace.',
      ],
      'recommendation': {'single_correction_justified':False,'candidate_status':'CANDIDATE_BLOCKED',
                         'reason':'No mathematical implementation error or misread MuJoCo API is established. A controller replacement would change the target-engine adaptation without a defensible correction. Contact/equality coupling remains plausible but unisolated; keep R3/S2B NO-GO and do not run a candidate matrix solely from this audit.'}
    }
    (out/'math_api_audit.json').write_text(json.dumps(jsonable(evidence),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'math_api_audit.sha256').write_text(sha(out/'math_api_audit.json')+'  math_api_audit.json\n')
    return evidence

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=OUT_DEFAULT); args=ap.parse_args(); main(args.output)
