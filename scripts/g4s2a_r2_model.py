"""Explicit R2 diagnostic plant/controller selection; no integration or actor.

P1 changes body rotational tensors by 0.01 I in kg m^2, never armature.
Shadow models only evaluate M(q); the physical environment alone is integrated.
"""
from __future__ import annotations
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from scripts.g4s1b_env import AdaptedPegInsertEnv
from scripts.g4s2a_r1_viewer import XML_SHA

ROBOT_BODIES = tuple([f'link{i}' for i in range(8)]+['hand','left_finger','right_finger'])
VARIANTS = {'A':(0,0),'B':(1,0),'C':(0,1),'D':(1,1)}
MAX_STEPS = 235
THRESHOLDS = {
 'positive_pad_force_n':1e-6, 'accept_slip_m':.001,
 'accept_rotation_rad':.05, 'accept_arm_velocity_rad_s':3.,
 'accept_arm_limit_rad':1e-6, 'accept_finger_limit_m':1e-6,
 'accept_pad_penetration_m':50e-6,'accept_assembly_penetration_m':10e-6,
 'stop_slip_m':.005,'stop_rotation_rad':.2,'stop_arm_velocity_rad_s':15.,
 'stop_arm_limit_rad':1e-5,'stop_finger_limit_m':1e-4,
 'stop_penetration_m':.001,'stop_consecutive_contact_loss':3,
}

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def adapted_xml(original):
 if hashlib.sha256(original.encode()).hexdigest()!=XML_SHA:raise ValueError('FROZEN_XML_CHANGED')
 tree=ET.fromstring(original)
 for name in ROBOT_BODIES:
  body=tree.find(f'.//body[@name="{name}"]')
  inertial=None if body is None else body.find('inertial')
  if inertial is None:raise ValueError('MISSING_MAPPED_INERTIAL:'+name)
  key='fullinertia' if 'fullinertia' in inertial.attrib else 'diaginertia'
  values=np.array([float(x) for x in inertial.get(key,'').split()])
  if values.shape!=((6,) if key=='fullinertia' else (3,)):raise ValueError('INVALID_SOURCE_INERTIA')
  values[:3]+=.01
  inertial.set(key,' '.join(format(float(x),'.17g') for x in values))
 return ET.tostring(tree,encoding='unicode')

def make_model(adapted=False):
 if type(adapted) is not bool:raise ValueError('EXPLICIT_BOOLEAN_INERTIA_SELECTION_REQUIRED')
 env=AdaptedPegInsertEnv()
 if hashlib.sha256(env.xml.encode()).hexdigest()!=XML_SHA:raise ValueError('FROZEN_XML_CHANGED')
 if adapted:
  env.xml=adapted_xml(env.xml)
  model=env.mj.MjModel.from_xml_string(env.xml)
  if (model.nq,model.nv,model.nu)!=(env.model.nq,env.model.nv,env.model.nu):raise ValueError('MODEL_ORDER_CHANGED')
  env.model=model;env.data=env.mj.MjData(model);env.reset()
 return env

def model_identity(env):
 return {'xml_sha256':hashlib.sha256(env.xml.encode()).hexdigest(),
  'body_names':[env.model.body(i).name for i in range(env.model.nbody)],
  'body_mass_kg':env.model.body_mass.tolist(),'body_inertia_kg_m2':env.model.body_inertia.tolist(),
  'body_ipos_m':env.model.body_ipos.tolist(),'body_iquat_wxyz':env.model.body_iquat.tolist(),
  'dof_armature':env.model.dof_armature.tolist(),'force_limits':env.model.actuator_forcerange.tolist(),
  'arm_dof_ids':env.arm_dof_ids.tolist(),'arm_qpos_ids':env.arm_qpos_ids.tolist(),
  'finger_dof_ids':env.finger_dof_ids.tolist(),'finger_qpos_ids':env.finger_qpos_ids.tolist(),
  'arm_limits':env.model.jnt_range[env.arm_joint_ids].tolist(),
  'finger_limits':env.model.jnt_range[env.finger_joint_ids].tolist(),
  'arm_actuator_ids':env.arm_actuator_ids.tolist(),'finger_actuator_ids':env.finger_actuator_ids.tolist(),
  'mapped_bodies':list(ROBOT_BODIES),'unmapped_source_bodies':['force_sensor','fingertip_centered'],
  'missing_correspondence':'Source extra sensor/TCP bodies have no one-to-one target body. PhysX runtime M UNKNOWN.'}

def mass_at(shadow,qpos):
 q=np.asarray(qpos,dtype=float)
 if q.shape!=(shadow.model.nq,) or not np.isfinite(q).all():raise ValueError('INVALID_SAME_CONFIGURATION')
 peg_quat=q[shadow.peg_qpos_adr+3:shadow.peg_qpos_adr+7]
 if abs(np.linalg.norm(peg_quat)-1)>1e-6:raise ValueError('INVALID_FREE_PEG_QUATERNION')
 # Full q includes both fingers and free peg. Never integrate this data.
 shadow.data.qpos[:]=q
 shadow.mj.mj_forward(shadow.model,shadow.data)
 full=np.zeros((shadow.model.nv,shadow.model.nv))
 shadow.mj.mj_fullM(shadow.model,full,shadow.data.qM)
 return full

class ModelPair:
 def __init__(self,variant):
  if variant not in VARIANTS:raise ValueError('UNKNOWN_FACTORIAL_VARIANT')
  self.variant=variant;self.plant_selection,self.controller_selection=VARIANTS[variant]
  self.plant=make_model(bool(self.plant_selection))
  self.shadows=[make_model(False),make_model(True)]
 def matrices(self):
  env=self.plant;q=env.data.qpos.copy()
  full=[mass_at(s,q) for s in self.shadows]
  physical=np.zeros((env.model.nv,env.model.nv));env.mj.mj_fullM(env.model,physical,env.data.qM)
  if not np.allclose(physical,full[self.plant_selection],atol=2e-12,rtol=2e-12):raise ValueError('PHYSICAL_M_SEPARATION_FAILED')
  ix=np.ix_(env.arm_dof_ids,env.arm_dof_ids)
  angular=[]
  for name in ROBOT_BODIES:
   p,r=np.zeros((3,env.model.nv)),np.zeros((3,env.model.nv))
   env.mj.mj_jacBody(env.model,env.data,p,r,env.model.body(name).id)
   angular.append(r[:,env.arm_dof_ids].tolist())
  return {'physical_full_M':physical,'physical_M':physical[ix],
   'controller_M':full[self.controller_selection][ix],
   'same_q_M0':full[0][ix],'same_q_M1':full[1][ix],
   'same_qpos':q,'body_angular_jacobians':angular}

def stop_reasons(state,pad_force,consecutive_loss,warning_counts):
 q=np.asarray(state['qpos']);v=np.asarray(state['qvel']);m=state['s2a'];t=THRESHOLDS
 out=[]
 if set(pad_force)!={'left','right'} or not np.isfinite(list(pad_force.values())).all():out.append('INVALID_CONTACT_FORCE')
 if not np.isfinite(q).all() or not np.isfinite(v).all():out.append('NONFINITE_STATE')
 if any(warning_counts):out.append('NUMERICAL_SOLVER_WARNING')
 for key,limit,label in [('peg_hand_slip_m','stop_slip_m','PEG_ESCAPE_SLIP'),
                         ('peg_hand_slip_rad','stop_rotation_rad','PEG_ESCAPE_ROTATION'),
                         ('arm_joint_limit_violation_rad','stop_arm_limit_rad','ARM_LIMIT'),
                         ('finger_joint_limit_violation_m','stop_finger_limit_m','FINGER_LIMIT'),
                         ('all_contact_penetration_m','stop_penetration_m','SEVERE_PENETRATION')]:
  if not np.isfinite(m[key]) or m[key]>t[limit]:out.append(label)
 if np.max(abs(v[:7]))>t['stop_arm_velocity_rad_s']:out.append('IMPLAUSIBLE_ARM_VELOCITY')
 if consecutive_loss>=t['stop_consecutive_contact_loss']:out.append('SUSTAINED_CONTACT_LOSS')
 return out

def fixed_reference(step,initial_position,initial_quaternion,contract):
 if type(step) is not int or not 1<=step<=MAX_STEPS:raise ValueError('OUTSIDE_REGISTERED_HORIZON')
 if step<=16:return contract.targets()
 return {'position_target':np.asarray(initial_position).copy(),'quaternion_target':np.asarray(initial_quaternion).copy()}
