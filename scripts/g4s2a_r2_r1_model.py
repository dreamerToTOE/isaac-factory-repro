"""Versioned R2-R1 plant and arm-controller inertia selections.

No actor and no integration. The finger controller remains the frozen S1B
physical-plant predictor; C0/C1 selects only the arm controller matrix.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
import numpy as np

from scripts.g4s1b_env import AdaptedPegInsertEnv
from scripts.g4s2a_r1_viewer import XML_SHA
from scripts.g4s2a_r2_model import (ROBOT_BODIES, MAX_STEPS, THRESHOLDS,
    digest, fixed_reference, mass_at, model_identity, stop_reasons)

ARM_HAND_BODIES = tuple([f'link{i}' for i in range(8)] + ['hand'])
SELECTIONS = {'P_FULL': ROBOT_BODIES, 'P_ARM_HAND': ARM_HAND_BODIES}
RUNS = (
    ('I3_FULL_C0', 'I3', 'P_FULL', 'C0', False),
    ('I3_FULL_C1', 'I3', 'P_FULL', 'C1', False),
    ('I3_ARM_HAND_C0', 'I3', 'P_ARM_HAND', 'C0', False),
    ('I3_ARM_HAND_C1', 'I3', 'P_ARM_HAND', 'C1', False),
    ('I2_ARM_HAND_C0', 'I2', 'P_ARM_HAND', 'C0', False),
    ('I2_ARM_HAND_C1', 'I2', 'P_ARM_HAND', 'C1', False),
    ('I1_ARM_HAND_C0', 'I1', 'P_ARM_HAND', 'C0', False),
    ('I1_ARM_HAND_C1', 'I1', 'P_ARM_HAND', 'C1', False),
    ('I3_FULL_PASSIVE_ARM', 'I3', 'P_FULL', 'C0', True),
)

def adapted_xml(original, selection):
    if selection not in SELECTIONS:
        raise ValueError('UNKNOWN_PHYSICAL_INERTIA_SELECTION')
    if hashlib.sha256(original.encode()).hexdigest() != XML_SHA:
        raise ValueError('FROZEN_XML_CHANGED')
    tree = ET.fromstring(original)
    for name in SELECTIONS[selection]:
        body = tree.find(f'.//body[@name="{name}"]')
        inertia = None if body is None else body.find('inertial')
        if inertia is None:
            raise ValueError('MISSING_MAPPED_INERTIAL:' + name)
        key = 'fullinertia' if 'fullinertia' in inertia.attrib else 'diaginertia'
        values = np.asarray([float(v) for v in inertia.get(key, '').split()])
        if values.shape != ((6,) if key == 'fullinertia' else (3,)):
            raise ValueError('INVALID_SOURCE_INERTIA:' + name)
        values[:3] += .01
        inertia.set(key, ' '.join(format(float(v), '.17g') for v in values))
    return ET.tostring(tree, encoding='unicode')

def make_model(selection):
    if selection not in ('S1B', *SELECTIONS):
        raise ValueError('EXPLICIT_INERTIA_SELECTION_REQUIRED')
    env = AdaptedPegInsertEnv()
    if hashlib.sha256(env.xml.encode()).hexdigest() != XML_SHA:
        raise ValueError('FROZEN_XML_CHANGED')
    if selection != 'S1B':
        env.xml = adapted_xml(env.xml, selection)
        model = env.mj.MjModel.from_xml_string(env.xml)
        if (model.nq, model.nv, model.nu) != (env.model.nq, env.model.nv, env.model.nu):
            raise ValueError('MODEL_ORDER_CHANGED')
        env.model = model
        env.data = env.mj.MjData(model)
        env.reset()
    return env

class ModelPair:
    def __init__(self, physical, controller):
        if physical not in SELECTIONS or controller not in ('C0', 'C1'):
            raise ValueError('UNKNOWN_INERTIA_FACTOR')
        self.physical_selection, self.controller_selection = physical, controller
        self.plant = make_model(physical)
        self.shadows = {'C0': make_model('S1B'), 'C1': make_model(physical)}

    def matrices(self):
        env = self.plant
        q = env.data.qpos.copy()
        shadows = {key: mass_at(model, q) for key, model in self.shadows.items()}
        physical = np.zeros((env.model.nv, env.model.nv))
        env.mj.mj_fullM(env.model, physical, env.data.qM)
        if not np.allclose(physical, shadows['C1'], atol=2e-12, rtol=2e-12):
            raise ValueError('PHYSICAL_M_SEPARATION_FAILED')
        for m in (physical, *shadows.values()):
            if not np.isfinite(m).all() or not np.allclose(m, m.T, atol=1e-12, rtol=0):
                raise ValueError('INVALID_MASS_MATRIX')
            if np.min(np.linalg.eigvalsh(m)) <= 0:
                raise ValueError('MASS_MATRIX_NOT_POSITIVE_DEFINITE')
        ix = np.ix_(env.arm_dof_ids, env.arm_dof_ids)
        angular = []
        for name in ROBOT_BODIES:
            position_jac = np.zeros((3, env.model.nv))
            rotation_jac = np.zeros((3, env.model.nv))
            env.mj.mj_jacBody(env.model, env.data, position_jac, rotation_jac,
                              env.model.body(name).id)
            angular.append(rotation_jac[:, env.arm_dof_ids].copy())
        return {'same_qpos': q, 'physical_full_M': physical,
                'C0_full_M': shadows['C0'], 'C1_full_M': shadows['C1'],
                'physical_M': physical[ix], 'controller_M': shadows[self.controller_selection][ix],
                'same_q_M0': shadows['C0'][ix], 'same_q_M1': shadows['C1'][ix],
                'body_angular_jacobians': angular, 'body_angular_jacobian_names': ROBOT_BODIES,
                'controller_selection': self.controller_selection,
                'finger_predictor_mass_source': 'physical_full_M'}
