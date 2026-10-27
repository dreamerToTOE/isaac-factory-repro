#!/usr/bin/env python3
"""Inspect pre-registered model geometry without integrating physics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
from scripts.g4s2a_gripper_v1_diagnostics import initial
from scripts.g4s2a_gripper_v1_r1_schedule import (
    DT_S, MAX_TARGET_DELTA_M, MAX_TARGET_RATE_M_S, INITIAL_OPENING_M,
    MAX_PREPARATION_STEPS, POSITIVE_CONTACT_N,
    ACQUISITION_BILATERAL_STEPS, HOLD_BILATERAL_STEPS,
)


def evaluate(name):
    env = NativeGripperEnv()
    initial(env, name)  # reset and mj_forward only; no mj_step
    m, d, mj = env.model, env.data, env.mj
    peg = m.geom('peg_collision').id
    sides = {}
    for side in ('left', 'right'):
        records = []
        for index in range(3, 8):
            geom_name = f'{side}_finger_geom_{index}'
            gid = m.geom(geom_name).id
            segment = np.zeros(6)
            distance = float(mj.mj_geomDistance(m, d, peg, gid, 1.0, segment))
            records.append({'geom': geom_name, 'compiled_type': int(m.geom_type[gid]),
                            'size_m': m.geom_size[gid].tolist(), 'distance_m': distance,
                            'nearest_points_world_m': segment.tolist()})
        sides[side] = {'closest_distance_m': min(r['distance_m'] for r in records), 'pad_geoms': records}
    return {'condition': name, 'registered_finger_openings_m': d.qpos[env.finger_qpos_ids].tolist(),
            'qvel_max_abs': float(np.max(np.abs(d.qvel))),
            'peg_geom': {'compiled_type': int(m.geom_type[peg]), 'size_m': m.geom_size[peg].tolist(),
                         'radius_m': float(m.geom_size[peg][0]), 'diameter_m': float(2*m.geom_size[peg][0])},
            'compiled_timestep_s': float(m.opt.timestep),
            'model_xml_sha256': hashlib.sha256(env.xml.encode()).hexdigest(), 'sides': sides}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cases = [evaluate(name) for name in ('I2', 'I3')]
    for case in cases:
        if not all(math.isclose(q, INITIAL_OPENING_M, rel_tol=0, abs_tol=1e-15) for q in case['registered_finger_openings_m']):
            raise ValueError('registered opening differs')
        if case['qvel_max_abs'] != 0 or case['compiled_timestep_s'] != DT_S:
            raise ValueError('invalid precondition')
        if not all(side['closest_distance_m'] > 0 for side in case['sides'].values()):
            raise ValueError('registered peg-pad overlap')
    import mujoco
    result = {'status': 'PASS', 'method': 'MuJoCo compiled geom distances after mj_forward; zero mj_step calls',
              'mujoco_version': mujoco.__version__, 'conditions': cases,
              'schedule': {'initial_opening_m': INITIAL_OPENING_M, 'reference_delta_limit_m_per_step': MAX_TARGET_DELTA_M,
                'rate_limit_m_s': MAX_TARGET_RATE_M_S, 'timestep_s': DT_S, 'horizon_steps': MAX_PREPARATION_STEPS,
                'contact_positive_strictly_greater_n': POSITIVE_CONTACT_N,
                'acquisition_consecutive_bilateral_observations': ACQUISITION_BILATERAL_STEPS,
                'hold_additional_consecutive_bilateral_observations': HOLD_BILATERAL_STEPS,
                'relative_to_existing_50um_pad_penetration_gate': MAX_TARGET_DELTA_M/50e-6,
                'method': 'Close reference by at most 10 micrometers/step; freeze permanently on first positive pad normal force. Positive bilateral forces at ten consecutive observations enter hold; ten additional positive bilateral observations complete hold. Lost contact resets streak, not reference.',
                'caveat': 'Reference bound is not a physical displacement, penetration, or contact stability guarantee.'}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')

if __name__ == '__main__':
    main()
