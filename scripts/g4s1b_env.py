#!/usr/bin/env python3
"""Independent, adapted Panda PegInsert world for bounded no-policy G4-S1B tests.

The Panda is the Apache-2.0 Menagerie model at the pinned commit. No NVIDIA
asset is read, converted, or distributed. Peg, hole and support are original
analytic constructions. This is intentionally NOT an exact Factory reconstruction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MENAGERIE = Path('/home/ubuntu2004/mujoco_menagerie')
MENAGERIE_COMMIT = '8161bba264d7fa7c99ca301e91e7fb44737676ad'
DEFAULT_Q = np.array([-1.3003, -.4015, 1.1791, -2.1493, .4001, 1.9425, .4754])
TCP_OFFSET = .11207100003957748


def vector(value, n, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (n,) or not np.isfinite(out).all():
        raise ValueError(f'INVALID_{name}')
    return out.copy()


@dataclass(frozen=True)
class EnvironmentConfig:
    timestep: float = 1/120
    decimation: int = 8
    peg_radius: float = .007986/2
    peg_height: float = .050
    peg_mass: float = .019
    bore_radius: float = .0081/2
    hole_height: float = .025
    hole_outer_radius: float = .020
    floor_thickness: float = .003
    hole_sectors: int = 256
    friction: float = .75
    robot_gravity_compensation: bool = True
    peg_gravity_compensation: bool = True
    contact_solref: tuple = (.005, 1.)
    contact_solimp: tuple = (.99, .999, .00001)
    solver_iterations: int = 100
    finger_kp: float = 7500.
    finger_kd: float = 173.
    finger_force_limit: float = 40.
    finger_actuation: str = 'step-implicit-pd'

    def validate(self):
        vals = asdict(self)
        for key, value in vals.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'INVALID_CONFIG_{key}')
        for key in ('robot_gravity_compensation','peg_gravity_compensation'):
            if not isinstance(vals[key],bool):raise ValueError(f'INVALID_CONFIG_{key}')
        for key in ('contact_solref','contact_solimp'):
            v=vals[key]
            expected=2 if key=='contact_solref' else 3
            if len(v)!=expected or any(not math.isfinite(x) or x<=0 for x in v):
                raise ValueError(f'INVALID_CONFIG_{key}')
        if self.finger_actuation not in ('step-implicit-pd','affine-pd'):
            raise ValueError('INVALID_FINGER_ACTUATION')
        if not isinstance(self.decimation,int) or isinstance(self.decimation,bool):
            raise ValueError('INVALID_DECIMATION')
        if self.timestep != 1/120 or self.decimation != 8:
            raise ValueError('FROZEN_TIMING_MISMATCH')
        if self.peg_radius != .007986/2 or self.bore_radius != .0081/2:
            raise ValueError('PRIMARY_NOMINAL_DIMENSION_MISMATCH')
        if not isinstance(self.hole_sectors, int) or isinstance(self.hole_sectors,bool) or not 64<=self.hole_sectors<=2048:
            raise ValueError('INSUFFICIENT_BORE_RESOLUTION')
        if self.bore_radius <= self.peg_radius or self.hole_outer_radius <= self.bore_radius:
            raise ValueError('INVALID_RADII')


def model_xml(config=EnvironmentConfig(), menagerie=MENAGERIE):
    """Read only independent licensed Panda; author analytic objects in memory."""
    config.validate()
    menagerie = Path(menagerie)
    revision = subprocess.check_output(['git', '-C', str(menagerie), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != MENAGERIE_COMMIT:
        raise ValueError('UNPINNED_MENAGERIE')
    if subprocess.check_output(['git','-C',str(menagerie),'status','--porcelain','--','franka_emika_panda'],text=True).strip():
        raise ValueError('DIRTY_PINNED_PANDA_ASSETS')
    panda = menagerie/'franka_emika_panda'
    root = ET.parse(panda/'panda.xml').getroot()
    root.set('model', 'g4_s1b_adapted_panda_peg_insert')
    root.find('compiler').set('meshdir', str(panda/'assets'))
    opt = root.find('option')
    opt.attrib.update(timestep=format(config.timestep, '.17g'), gravity='0 0 -9.81',
                      integrator='implicitfast', cone='elliptic', solver='Newton',
                      iterations=str(config.solver_iterations), tolerance='1e-10')
    default = root.find('default')
    ET.SubElement(default, 'geom', friction=f'{config.friction} .005 .0001',
                  margin='0', gap='0', condim='4',
                  solref=' '.join(map(str, config.contact_solref)),
                  solimp=' '.join(map(str, config.contact_solimp)))
    panda_default = default.find("default[@class='panda']")
    panda_default.find('joint').attrib.update(armature='0', damping='0', frictionloss='0')
    # Original task disables robot self collision, while all robot/object pairs remain enabled.
    collision_default = panda_default.find("default[@class='collision']/geom")
    collision_default.attrib.update(contype='1', conaffinity='2')
    for body in root.findall('.//worldbody//body'):
        body.set('gravcomp', '1' if config.robot_gravity_compensation else '0')
        for index, geom in enumerate(body.findall('geom')):
            geom.set('name', f'{body.get("name")}_geom_{index}')
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, 'site', name='factory_tcp', pos=f'0 0 {TCP_OFFSET:.17g}', size='.001', rgba='1 0 0 1')
    # Remove inherited affine position servos, split tendon and obsolete home dimensions.
    for tag in ('actuator', 'tendon', 'keyframe'):
        old = root.find(tag)
        if old is not None:
            root.remove(old)
    actuators = ET.SubElement(root, 'actuator')
    for i in range(1, 8):
        limit = 87 if i <= 4 else 12
        ET.SubElement(actuators, 'motor', name=f'arm_torque_{i}', joint=f'joint{i}',
                      gear='1', ctrlrange='-100 100', ctrllimited='true',
                      forcerange=f'-{limit} {limit}', forcelimited='true')
    for i in (1, 2):
        if config.finger_actuation=='affine-pd':
            ET.SubElement(actuators, 'general', name=f'finger_pd_{i}', joint=f'finger_joint{i}',
                          dyntype='none', biastype='affine', gainprm=str(config.finger_kp),
                          biasprm=f'0 {-config.finger_kp} {-config.finger_kd}',
                          ctrlrange='0 .04', ctrllimited='true',
                          forcerange=f'-{config.finger_force_limit} {config.finger_force_limit}', forcelimited='true')
        else:
            ET.SubElement(actuators,'motor',name=f'finger_pd_{i}',joint=f'finger_joint{i}',gear='1',
                          ctrlrange=f'-{config.finger_force_limit} {config.finger_force_limit}',ctrllimited='true',
                          forcerange=f'-{config.finger_force_limit} {config.finger_force_limit}',forcelimited='true')
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='independent_support', type='box',
                  pos='.6 0 .0235', size='.16 .16 .0235', contype='2', conaffinity='3', rgba='.4 .4 .4 1')
    hole = ET.SubElement(world, 'body', name='hole', pos='.6 0 .05')
    ET.SubElement(hole, 'geom', name='hole_floor', type='box',
                  pos=f'0 0 {-config.floor_thickness/2}',
                  size=f'{config.hole_outer_radius} {config.hole_outer_radius} {config.floor_thickness/2}',
                  contype='2', conaffinity='3', rgba='.1 .3 .6 1')
    # Each convex trapezoid spans a sector. Inner vertices lie on nominal circle;
    # its chord intrudes (never enlarges the bore), explicitly bounded below.
    assets = root.find('asset')
    n = config.hole_sectors
    rout = config.hole_outer_radius/math.cos(math.pi/n)
    for i in range(n):
        angles = (2*math.pi*i/n, 2*math.pi*(i+1)/n)
        planar = [(config.bore_radius*math.cos(a), config.bore_radius*math.sin(a)) for a in angles]
        planar += [(rout*math.cos(a), rout*math.sin(a)) for a in reversed(angles)]
        vertices = [(x,y,z) for z in (0.,config.hole_height) for x,y in planar]
        name = f'analytic_hole_sector_{i:03d}'
        ET.SubElement(assets, 'mesh', name=name, vertex=' '.join(format(v, '.17g') for p in vertices for v in p))
        ET.SubElement(hole, 'geom', name=f'hole_wall_{i:03d}', type='mesh', mesh=name,
                      contype='2', conaffinity='3', rgba='.1 .3 .6 1')
    peg = ET.SubElement(world, 'body', name='peg', pos='.6 0 .12',
                        gravcomp='1' if config.peg_gravity_compensation else '0')
    ET.SubElement(peg, 'freejoint', name='peg_free')
    r,h,m = config.peg_radius, config.peg_height, config.peg_mass
    ET.SubElement(peg, 'inertial', pos=f'0 0 {h/2}', mass=str(m),
                  diaginertia=f'{m*(3*r*r+h*h)/12} {m*(3*r*r+h*h)/12} {m*r*r/2}')
    ET.SubElement(peg, 'geom', name='peg_collision', type='cylinder', pos=f'0 0 {h/2}',
                  size=f'{r} {h/2}', contype='2', conaffinity='3', rgba='.8 .2 .1 1')
    return ET.tostring(root, encoding='unicode')


class AdaptedPegInsertEnv:
    """No actor is imported. External caller supplies explicit reference torques."""
    def __init__(self, config=EnvironmentConfig(), menagerie=MENAGERIE):
        import mujoco
        if mujoco.__version__ != '3.5.0' or mujoco.mj_versionString() != '3.5.0':
            raise RuntimeError('MUJOCO_3_5_0_REQUIRED')
        self.mj = mujoco
        self.config = config
        self.xml = model_xml(config, menagerie)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.arm_joint_ids = np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i}') for i in range(1,8)])
        self.arm_qpos_ids = self.model.jnt_qposadr[self.arm_joint_ids].copy()
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids].copy()
        self.arm_actuator_ids = np.array([self.model.actuator(f'arm_torque_{i}').id for i in range(1,8)])
        self.finger_joint_ids = np.array([self.model.joint(f'finger_joint{i}').id for i in (1,2)])
        self.finger_qpos_ids = self.model.jnt_qposadr[self.finger_joint_ids].copy()
        self.finger_dof_ids = self.model.jnt_dofadr[self.finger_joint_ids].copy()
        self.finger_actuator_ids = np.array([self.model.actuator(f'finger_pd_{i}').id for i in (1,2)])
        self.peg_joint_id = self.model.joint('peg_free').id
        self.peg_qpos_adr = int(self.model.jnt_qposadr[self.peg_joint_id])
        self.peg_dof_adr = int(self.model.jnt_dofadr[self.peg_joint_id])
        self.peg_body_id = self.model.body('peg').id
        self.hole_body_id = self.model.body('hole').id
        self.tcp_site_id = self.model.site('factory_tcp').id
        self.log = []
        self.last_requested_torque = np.zeros(7)
        self.last_software_torque = np.zeros(7)
        self.last_finger_command={}
        self.reset()

    def reset(self, seed=0, arm_qpos=None, peg_pos=None, peg_quat=None,
              hole_pos=None, hole_quat=None, finger_opening=.02):
        # Seed support is local and deterministic; no claim of paired PhysX draws.
        if not isinstance(seed, (int,np.integer)) or seed < 0:
            raise ValueError('INVALID_SEED')
        self.rng = np.random.default_rng(seed)
        q = vector(DEFAULT_Q if arm_qpos is None else arm_qpos,7,'ARM_QPOS')
        bounds = self.model.jnt_range[self.arm_joint_ids]
        if np.any(q < bounds[:,0]) or np.any(q > bounds[:,1]):
            raise ValueError('JOINT_LIMIT_RESET')
        if not math.isfinite(finger_opening) or not 0 <= finger_opening <= .04:
            raise ValueError('FINGER_LIMIT_RESET')
        p = vector([.6,0,.12] if peg_pos is None else peg_pos,3,'PEG_POSITION')
        quat = vector([1,0,0,0] if peg_quat is None else peg_quat,4,'PEG_QUATERNION')
        hpos = vector([.6,0,.05] if hole_pos is None else hole_pos,3,'HOLE_POSITION')
        hquat = vector([1,0,0,0] if hole_quat is None else hole_quat,4,'HOLE_QUATERNION')
        if not np.isclose(np.linalg.norm(quat),1,atol=1e-10,rtol=0) or not np.isclose(np.linalg.norm(hquat),1,atol=1e-10,rtol=0):
            raise ValueError('NONUNIT_RESET_QUATERNION')
        self.mj.mj_resetData(self.model,self.data)
        self.model.body_pos[self.hole_body_id] = hpos
        self.model.body_quat[self.hole_body_id] = hquat
        self.model.body_gravcomp[self.peg_body_id]=int(self.config.peg_gravity_compensation)
        self.data.qpos[self.arm_qpos_ids] = q
        self.data.qpos[self.finger_qpos_ids] = finger_opening
        self.data.qpos[self.peg_qpos_adr:self.peg_qpos_adr+7] = np.r_[p,quat]
        self.data.ctrl[:] = 0
        self.log = []
        self.last_requested_torque[:] = 0
        self.last_software_torque[:] = 0
        self.last_finger_command={}
        self.mj.mj_forward(self.model,self.data)
        return self.snapshot()

    def tcp_state(self):
        # Identity TCP orientation in hand frame: native xquat retains the model's
        # quaternion chain branch, without matrix conversion/canonicalization.
        quat=self.data.xquat[self.model.body('hand').id].copy()
        return self.data.site_xpos[self.tcp_site_id].copy(),quat

    def tcp_jacobian(self):
        p,r = np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
        self.mj.mj_jacSite(self.model,self.data,p,r,self.tcp_site_id)
        return np.vstack([p[:,self.arm_dof_ids],r[:,self.arm_dof_ids]])

    def controller_jacobian(self):
        values=[]
        for name in ('left_finger','right_finger'):
            p,r=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
            self.mj.mj_jacBody(self.model,self.data,p,r,self.model.body(name).id)
            values.append(np.vstack([p[:,self.arm_dof_ids],r[:,self.arm_dof_ids]]))
        return sum(values)/2

    def arm_mass_matrix(self):
        full=np.zeros((self.model.nv,self.model.nv))
        self.mj.mj_fullM(self.model,full,self.data.qM)
        return full[np.ix_(self.arm_dof_ids,self.arm_dof_ids)]

    def apply_arm_torque(self, requested):
        requested=vector(requested,7,'REQUESTED_TORQUE')
        self.last_requested_torque=requested
        self.last_software_torque=np.clip(requested,-100,100)
        self.data.ctrl[self.arm_actuator_ids]=self.last_software_torque
        return self.last_software_torque.copy()

    def physics_step(self, arm_torque=None, finger_target=0., record=True):
        if arm_torque is not None:
            self.apply_arm_torque(arm_torque)
        if not math.isfinite(finger_target) or not 0 <= finger_target <= .04:
            raise ValueError('INVALID_FINGER_TARGET')
        q=self.data.qpos[self.finger_qpos_ids]
        v=self.data.qvel[self.finger_dof_ids]
        raw=self.config.finger_kp*(finger_target-q)-self.config.finger_kd*v
        if self.config.finger_actuation=='affine-pd':
            self.data.ctrl[self.finger_actuator_ids]=finger_target
            force=np.clip(raw,-self.config.finger_force_limit,self.config.finger_force_limit)
        else:
            full=np.zeros((self.model.nv,self.model.nv))
            self.mj.mj_fullM(self.model,full,self.data.qM)
            # Backward-Euler free-motion prediction, before unknown contact and
            # equality reactions. This is a REGISTERED target-engine adaptation,
            # not the fully coupled PhysX implicit drive or identical static PD.
            ainv=np.linalg.inv(full)[np.ix_(self.finger_dof_ids,self.finger_dof_ids)]
            dt=self.config.timestep;kp=self.config.finger_kp;kd=self.config.finger_kd
            force=np.linalg.solve(np.eye(2)+dt*(kd+dt*kp)*ainv,
                                  kp*(finger_target-q)-(kd+dt*kp)*v)
            force=np.clip(force,-self.config.finger_force_limit,self.config.finger_force_limit)
            self.data.ctrl[self.finger_actuator_ids]=force
        self.last_finger_command={'mode':self.config.finger_actuation,'target_opening_per_finger_m':finger_target,
                                  'raw_pd_force_n':raw.tolist(),'commanded_force_n':force.tolist()}
        previous_warnings=np.array([x.number for x in self.data.warning])
        before_time=float(self.data.time)
        self.mj.mj_step(self.model,self.data)
        # The actuator values used by integration belong to its pre-step stage.
        # Keep them before refreshing derived quantities at the new state.
        integrated={'stage':'mj_step_preintegration_actuation',
                    'state_time_s':before_time,'step_end_time_s':float(self.data.time),
                    'arm_actuator_force_nm':self.data.actuator_force[self.arm_actuator_ids].tolist(),
                    'arm_qfrc_actuator_nm':self.data.qfrc_actuator[self.arm_dof_ids].tolist(),
                    'finger_actuator_force_n':self.data.actuator_force[self.finger_actuator_ids].tolist()}
        self.mj.mj_forward(self.model,self.data)
        new_warnings=np.array([x.number for x in self.data.warning])-previous_warnings
        if np.any(new_warnings):
            raise RuntimeError(f'MUJOCO_WARNING_COUNTS_{new_warnings.tolist()}')
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('NONFINITE_SIMULATION_STATE')
        result=self.snapshot()
        result['integration_actuation']=integrated
        result['snapshot_actuation_stage']='poststep_mj_forward_recomputed_at_time_s'
        if record:
            self.log.append(result)
        return result

    def contacts(self):
        result=[]
        for c in self.data.contact:
            force=np.zeros(6)
            # mj_contactForce requires a contact index, not the contact object.
            i=len(result)
            self.mj.mj_contactForce(self.model,self.data,i,force)
            result.append({'geom1':self.model.geom(int(c.geom1)).name,
                           'geom2':self.model.geom(int(c.geom2)).name,
                           'distance_m':float(c.dist),'normal_force_n':float(force[0]),
                           'position_world_m':c.pos.tolist()})
        return result

    def snapshot(self):
        p,q=self.tcp_state()
        return {'time_s':float(self.data.time),'qpos':self.data.qpos.tolist(),
                'qvel':self.data.qvel.tolist(),'tcp_position_world':p.tolist(),
                'tcp_quaternion_wxyz':q.tolist(),
                'peg_root_position':self.data.xpos[self.peg_body_id].tolist(),
                'peg_quaternion_wxyz':self.data.xquat[self.peg_body_id].tolist(),
                'hole_root_position':self.data.xpos[self.hole_body_id].tolist(),
                'requested_arm_torque_nm':self.last_requested_torque.tolist(),
                'software_limited_arm_torque_nm':self.last_software_torque.tolist(),
                'actuator_force_nm':self.data.actuator_force[self.arm_actuator_ids].tolist(),
                'generalized_actuator_contribution_nm':self.data.qfrc_actuator[self.arm_dof_ids].tolist(),
                'finger_actuator_force_n':self.data.actuator_force[self.finger_actuator_ids].tolist(),
                'finger_command':self.last_finger_command,
                'contacts':self.contacts()}

    def insertion_diagnostics(self, penetration_tolerance_m=1e-5):
        """Independent physical diagnostic; NEVER replaces original success test."""
        peg=self.data.xpos[self.peg_body_id]
        hole=self.data.xpos[self.hole_body_id]
        hole_rot=self.data.xmat[self.hole_body_id].reshape(3,3)
        rel=hole_rot.T@(peg-hole)
        axis=hole_rot.T@self.data.xmat[self.peg_body_id].reshape(3,3)[:,2]
        tilt=float(math.acos(np.clip(abs(axis[2]),-1,1)))
        contacts=[c for c in self.contacts() if c['geom1']=='peg_collision' or c['geom2']=='peg_collision']
        assembly=[c for c in contacts if c['geom1'].startswith('hole_') or c['geom2'].startswith('hole_')]
        minimum=min((c['distance_m'] for c in assembly),default=0.)
        radial=float(np.linalg.norm(rel[:2]))
        floor_contact=any('hole_floor' in (c['geom1'],c['geom2']) for c in assembly)
        wall_contact=any(any(g.startswith('hole_wall') for g in (c['geom1'],c['geom2'])) for c in assembly)
        # Conservative vertical-cylinder radial envelope, includes tilt of whole peg.
        envelope=radial+self.config.peg_radius+self.config.peg_height*math.sin(tilt)
        bore_apothem=self.config.bore_radius*math.cos(math.pi/self.config.hole_sectors)
        upper_center=rel+self.config.peg_height*axis
        radial_z_extent=self.config.peg_radius*math.sqrt(max(0,1-axis[2]*axis[2]))
        zmin=float(min(rel[2],upper_center[2])-radial_z_extent)
        zmax=float(max(rel[2],upper_center[2])+radial_z_extent)
        inside=bool(envelope < bore_apothem and zmin<self.config.hole_height and zmax>0 and zmin>=-penetration_tolerance_m)
        return {'scope':'adapted_ground_truth_diagnostic_not_original_success',
                'peg_root_in_hole_frame_m':rel.tolist(),'axis_tilt_rad':tilt,
                'axis_in_hole_frame':axis.tolist(),'other_axis_endpoint_in_hole_frame_m':upper_center.tolist(),
                'full_cylinder_z_extent_in_hole_frame_m':[zmin,zmax],
                'conservative_radial_envelope_m':envelope,'bore_apothem_m':bore_apothem,
                'inside_bore':inside,'floor_contact':floor_contact,'wall_or_rim_contact':wall_contact,
                'minimum_contact_distance_m':minimum,
                'interpenetration_beyond_tolerance':bool(minimum < -penetration_tolerance_m),
                'resting_above_opening':bool(rel[2]>=self.config.hole_height and assembly),
                'entirely_above_opening':bool(zmin>=self.config.hole_height),
                'physical_insertion_candidate':bool(inside and floor_contact and minimum>=-penetration_tolerance_m),
                'contacts':assembly}


def bounded_environment_checks():
    """All probes use explicit reference commands, with finite fixed step counts."""
    started=time.perf_counter()
    env=AdaptedPegInsertEnv()
    mj=env.mj
    report={'protocol':'ADAPTED_ZERO_SHOT_SIM2SIM_NO_POLICY','config':asdict(env.config),
            'mujoco_version':mj.__version__,'menagerie_commit':MENAGERIE_COMMIT,
            'xml_sha256':hashlib.sha256(env.xml.encode()).hexdigest(),
            'asset_scope':'Apache-2.0 Panda references; original analytic peg/hole/support; no NVIDIA assets',
            'adaptations':['Menagerie inertias, geometry and finger pads retained; no claim of Factory equivalence',
                'Robot gravity compensated and peg gravity compensated by default, matching source disable_gravity intent',
                'Implicitfast MuJoCo solver; independent nominal cylinder and polygonal blind bore',
                'Finger step-implicit free-motion PD prediction plus symmetric joint equality; not PhysX implicit drive parity; coefficients7500/173 and forcecap40 retained but effective contact stiffness differs',
                'Robot self collision disabled by group masks; all robot/object and peg/hole contacts enabled',
                'TCP quaternion uses native hand xquat chain branch; not measured PhysX raw-sign parity'],
            'structure':{'nq':env.model.nq,'nv':env.model.nv,'nu':env.model.nu,
                         'peg_joint_type':int(env.model.jnt_type[env.peg_joint_id]),
                         'equality_types':env.model.eq_type.tolist(),'has_weld':bool(any(env.model.eq_type==int(mj.mjtEq.mjEQ_WELD)))},
            'mapping':{'arm_qpos_ids':env.arm_qpos_ids.tolist(),'arm_dof_ids':env.arm_dof_ids.tolist(),
                       'finger_qpos_ids':env.finger_qpos_ids.tolist(),'finger_dof_ids':env.finger_dof_ids.tolist(),
                       'arm_joint_limits':env.model.jnt_range[env.arm_joint_ids].tolist(),
                       'tcp_hand_local_z_m':TCP_OFFSET},'probes':{}}
    panda=MENAGERIE/'franka_emika_panda'
    source_tree=ET.parse(panda/'panda.xml')
    asset_files=[panda/'panda.xml',panda/'LICENSE']+[panda/'assets'/x.get('file') for x in source_tree.findall('.//asset/mesh')]
    report['licensed_asset_manifest']=[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in asset_files]
    report['body_mass_inertia']=[{'name':env.model.body(i).name,'mass_kg':float(env.model.body_mass[i]),
        'com_local_m':env.model.body_ipos[i].tolist(),'principal_inertia_kg_m2':env.model.body_inertia[i].tolist(),
        'inertial_quaternion_wxyz':env.model.body_iquat[i].tolist(),'gravity_compensation':float(env.model.body_gravcomp[i])} for i in range(env.model.nbody)]
    c=env.config
    report['geometry']={'nominal_peg_diameter_m':2*c.peg_radius,'nominal_hole_diameter_m':2*c.bore_radius,
        'nominal_radial_clearance_m':c.bore_radius-c.peg_radius,'sectors':c.hole_sectors,
        'ideal_chord_intrusion_m':c.bore_radius*(1-math.cos(math.pi/c.hole_sectors)),
        'ideal_minimum_radial_clearance_m':c.bore_radius*math.cos(math.pi/c.hole_sectors)-c.peg_radius,
        'comparison':'New analytic nominal geometry; original raw straight-slab gap>=505.921um remains historical, cooked SDF unknown',
        'floor_thickness_m':c.floor_thickness,'peg_height_m':c.peg_height,'hole_height_m':c.hole_height,
        'margin_m':float(max(env.model.geom_margin)),'gap_m':float(max(env.model.geom_gap))}
    report['geometry']['positive_solref_timeconstant_floor_s']=2*c.timestep
    report['geometry']['refsafe_enabled']=not bool(int(env.model.opt.disableflags)&int(mj.mjtDisableBit.mjDSBL_REFSAFE))
    # Real compiled inner face vertex projection, check every authored sector.
    compiled=[]
    for i in range(c.hole_sectors):
        gid=env.model.geom(f'hole_wall_{i:03d}').id
        mid=int(env.model.geom_dataid[gid]);adr=int(env.model.mesh_vertadr[mid]);num=int(env.model.mesh_vertnum[mid])
        verts=np.asarray(env.model.mesh_vert[adr:adr+num],dtype=float)
        rot=np.zeros(9);mj.mju_quat2Mat(rot,env.model.geom_quat[gid])
        verts=verts@rot.reshape(3,3).T+env.model.geom_pos[gid]
        # Mesh pose corrected by compiler; all original inner vertices are near radius R.
        radial=np.linalg.norm(verts[:,:2],axis=1)
        inner=verts[np.argsort(radial)[:4]]
        normal=np.array([math.cos((2*i+1)*math.pi/c.hole_sectors),math.sin((2*i+1)*math.pi/c.hole_sectors)])
        compiled.append(float(np.min(inner[:,:2]@normal)))
    report['geometry']['compiled_measured_inner_apothem_range_m']=[min(compiled),max(compiled)]
    report['geometry']['compiled_measurement_is_float_not_rigorous_certificate']=True
    # FK independent finite differences for TCP and source-style averaged body-origin J.
    q0=env.data.qpos.copy();j=env.tcp_jacobian();body_j=env.controller_jacobian();eps=1e-7
    numeric=np.zeros((6,7));numeric_body=np.zeros((3,7))
    def mean_finger():
        return (env.data.xpos[env.model.body('left_finger').id]+env.data.xpos[env.model.body('right_finger').id])/2
    for k,idx in enumerate(env.arm_qpos_ids):
        env.data.qpos[:]=q0;env.data.qpos[idx]+=eps;mj.mj_forward(env.model,env.data)
        pplus,qplus=env.tcp_state();bplus=mean_finger().copy()
        env.data.qpos[:]=q0;env.data.qpos[idx]-=eps;mj.mj_forward(env.model,env.data)
        pminus,qminus=env.tcp_state();bminus=mean_finger().copy()
        numeric[:3,k]=(pplus-pminus)/(2*eps)
        dq=np.zeros(4);qc=qminus.copy();qc[1:]*=-1;mj.mju_mulQuat(dq,qplus,qc)
        vel=np.zeros(3);mj.mju_quat2Vel(vel,dq,2*eps);numeric[3:,k]=vel
        numeric_body[:,k]=(bplus-bminus)/(2*eps)
    env.data.qpos[:]=q0;mj.mj_forward(env.model,env.data)
    m=env.arm_mass_matrix()
    report['probes']['robot_kinematics']={'tcp_pose':env.tcp_state()[0].tolist(),
        'tcp_jacobian_fd_max_error':float(np.max(abs(j-numeric))),
        'controller_finger_origin_linear_jacobian_fd_max_error':float(np.max(abs(body_j[:3]-numeric_body))),
        'controller_and_tcp_jacobian_max_difference':float(np.max(abs(body_j-j))),
        'mass_symmetry_max_error':float(np.max(abs(m-m.T))),
        'mass_eigenvalues':np.linalg.eigvalsh(m).tolist(),'mass_matrix':m.tolist()}
    # No input torque and disabled robot gravity should hold the stationary arm.
    env.reset(peg_pos=[1,1,1],finger_opening=.04)
    hold_start=env.data.qpos[env.arm_qpos_ids].copy()
    for _ in range(24):env.physics_step(np.zeros(7),finger_target=.04,record=False)
    report['probes']['static_hold']={'steps':24,'time_s':float(env.data.time),
        'max_joint_displacement_rad':float(np.max(abs(env.data.qpos[env.arm_qpos_ids]-hold_start))),
        'robot_gravity_compensation':True,'snapshot':env.snapshot()}
    # One tiny positive torque per independent reset: qdd follows M inverse e_i.
    signs=[]
    for i in range(7):
        env.reset(peg_pos=[1,1,1],finger_opening=.04)
        torque=np.zeros(7);torque[i]=.001
        env.physics_step(torque,finger_target=.04,record=False)
        signs.append({'joint':i+1,'torque_nm':.001,'own_qvel_rad_s':float(env.data.qvel[env.arm_dof_ids[i]])})
    report['probes']['torque_direction']=signs
    env.reset(peg_pos=[1,1,1],finger_opening=.04)
    env.apply_arm_torque([200,-200,101,-101,20,-20,12.5]);mj.mj_forward(env.model,env.data)
    report['probes']['actuation_saturation']=env.snapshot()
    # Geometry static probes away from robot: no contacts in bore, contacts at wall/rim/floor.
    geometries={}
    for name,pos in [('open_bore',[.6,0,.060]),('wall',[.6,.0002,.060]),
                     ('rim',[.6,.003,.074]),('floor_penetration',[.6,0,.0498]),
                     ('above_opening',[.6,0,.080])]:
        env.reset(peg_pos=pos,finger_opening=.04)
        geometries[name]=env.insertion_diagnostics()
    report['probes']['static_geometry']=geometries
    # Free falling physical peg, explicitly turn off only its gravity compensation.
    env.reset(peg_pos=[.95,.25,.5],finger_opening=.04)
    env.model.body_gravcomp[env.peg_body_id]=0
    start_z=float(env.data.xpos[env.peg_body_id,2]);fall=[]
    for _ in range(12):
        env.physics_step(np.zeros(7),finger_target=.04,record=False)
        fall.append({'time_s':float(env.data.time),'z_m':float(env.data.xpos[env.peg_body_id,2]),'vz_m_s':float(env.data.qvel[env.peg_dof_adr+2])})
    report['probes']['freefall']={'steps':12,'initial_z_m':start_z,'trajectory':fall,
        'expected_velocity_m_s':-9.81*12*c.timestep,'observed_drop_m':start_z-fall[-1]['z_m'],
        'expected_semiimplicit_drop_m':9.81*c.timestep*c.timestep*12*13/2}
    # Basic insertion feasibility: aligned free peg is physically dropped into the
    # nominal bore, not robot manipulation or policy success. Robot is far away.
    env.reset(peg_pos=[.6,0,.080],finger_opening=.04)
    peak_pen=0.;history=[]
    for step in range(120):
        env.physics_step(np.zeros(7),finger_target=.04,record=False)
        diag=env.insertion_diagnostics();peak_pen=min(peak_pen,diag['minimum_contact_distance_m'])
        if step%10==0 or step==119:history.append({'step':step+1,'time_s':float(env.data.time),**diag})
    final=env.insertion_diagnostics()
    report['probes']['aligned_drop_insertion']={'steps':120,'scope':'direct placement plus freefall, not robot assembly or PPO',
        'peak_signed_contact_distance_m':peak_pen,'history':history,'final':final,
        'passed':bool(final['physical_insertion_candidate'])}
    # Gravity-compensated peg follows the original task gravity-disable choice.
    # A small explicit downward reference force tests nominal-clearance passage
    # and seating without the high-speed floor impact of the separate freefall.
    env.reset(peg_pos=[.6,0,.076],finger_opening=.04)
    env.data.xfrc_applied[env.peg_body_id,2]=-.001
    history=[];peak_pen=0.
    for step in range(240):
        env.physics_step(np.zeros(7),finger_target=.04,record=False)
        diag=env.insertion_diagnostics();peak_pen=min(peak_pen,diag['minimum_contact_distance_m'])
        if step%20==0 or step==239:history.append({'step':step+1,'time_s':float(env.data.time),**diag})
    final=env.insertion_diagnostics()
    report['probes']['gentle_reference_insertion']={'steps':240,'reference_force_world_n':[0,0,-.001],
        'scope':'independent aligned peg, direct placement plus bounded no-policy force, not robot assembly',
        'peg_gravity_compensated':True,'peak_signed_contact_distance_m':peak_pen,'history':history,'final':final,
        'passed':bool(final['physical_insertion_candidate'])}
    # Physical grasp: place an independent peg through existing licensed pads,
    # close with original-gain saturated PD while arm is gravity compensated.
    env.reset(peg_pos=[1,1,1],finger_opening=.02)
    p,tcpq=env.tcp_state();handrot=env.data.xmat[env.model.body('hand').id].reshape(3,3).copy()
    # Peg extends downward from the inner pad region. Its positive z points world
    # upward; choose a bottom root so its upper length crosses both pads.
    axis=-handrot[:,2];pegrot=np.column_stack([handrot[:,0],-handrot[:,1],axis])
    pegquat=np.zeros(4);mj.mju_mat2Quat(pegquat,pegrot.reshape(-1))
    pegroot=p-axis*.032392
    env.reset(peg_pos=pegroot,peg_quat=pegquat,finger_opening=.02)
    initial_root=env.data.xpos[env.peg_body_id].copy();grasp_history=[];both_contact_steps=0;both_window=[]
    initial_relative=handrot.T@(initial_root-p)
    for step in range(90):
        # First .25s closes under the original source-style disabled peg gravity;
        # later .5s is a separately labelled gravity-on stress diagnostic.
        if step==30:env.model.body_gravcomp[env.peg_body_id]=0
        env.physics_step(np.zeros(7),finger_target=0,record=False)
        snap=env.snapshot();pegcontacts=[x for x in snap['contacts'] if 'peg_collision' in (x['geom1'],x['geom2'])]
        bodies=set();normal_by_finger={'left_finger':0.,'right_finger':0.}
        for ct in pegcontacts:
            other=ct['geom2'] if ct['geom1']=='peg_collision' else ct['geom1']
            if other:
                gid=env.model.geom(other).id
                body=env.model.body(int(env.model.geom_bodyid[gid])).name
                if ct['normal_force_n']>1e-6:bodies.add(body)
                if body in normal_by_finger:normal_by_finger[body]+=ct['normal_force_n']
        both=bool({'left_finger','right_finger'}<=bodies);both_contact_steps+=both;both_window.append(both)
        if step%10==0 or step==89:grasp_history.append({'step':step+1,'both_fingers_contact':both,'finger_qpos_m':env.data.qpos[env.finger_qpos_ids].tolist(),
            'peg_root_position':snap['peg_root_position'],'peg_contacts':pegcontacts,'finger_forces_n':snap['finger_actuator_force_n'],
            'normal_force_by_finger_n':normal_by_finger,'finger_command':snap['finger_command']})
    drift=float(np.linalg.norm(env.data.xpos[env.peg_body_id]-initial_root))
    p,_=env.tcp_state();current_handrot=env.data.xmat[env.model.body('hand').id].reshape(3,3)
    relative_slip=float(np.linalg.norm(current_handrot.T@(env.data.xpos[env.peg_body_id]-p)-initial_relative))
    report['probes']['physical_grasp']={'steps':90,'no_weld':True,'gravity_compensated_close_steps':30,'gravity_on_stress_steps':60,
        'initial_peg_root':initial_root.tolist(),'root_displacement_m':drift,'both_finger_contact_steps':both_contact_steps,
        'final_30_steps_both_fingers_positive_force':all(both_window[-30:]),'hand_relative_slip_m':relative_slip,
        'history':grasp_history,'passed':bool(all(both_window[-30:]) and relative_slip<.001),
        'scope':'static close/hold of independently placed peg, not robot pickup or insertion'}
    # Three bounded explicit reference actions exercise the real MuJoCo bridge.
    # This is controller/observation engineering only: no policy, checkpoint or
    # neural network is imported. The final state is retained before any reset.
    from scripts.g4s1b_contract import ReferenceRunner
    reference=ReferenceRunner(env)
    initial=reference.reset(seed=0, observation_noise=np.zeros(3))
    reference_steps=[]
    for raw_action in ([0., 0., 0., 0., 0., 0.],
                       [1., 0., 0., 0., 0., 0.],
                       [-1., 1., 0., 0., 0., 0.]):
        result=reference.step_reference_action(raw_action)
        reference_steps.append({'episode_steps':result['episode_steps'],
                                'simulation_time_s':result['episode_simulation_time_s'],
                                'observation_size':int(result['observation'].size),
                                'max_arm_qvel_rad_s':float(np.max(np.abs(env.data.qvel[env.arm_dof_ids]))),
                                'official_geometric_success':result['official_geometric_success'],
                                'physical_insertion_candidate':result['physical_insertion_diagnostic']['physical_insertion_candidate'],
                                'substep_count':len(result['substeps']),
                                'state_is_pre_reset':result['state_is_pre_reset'],
                                'reset_required':result['reset_required']})
    report['probes']['reference_controller_bridge']={
        'scope':'three explicit bounded reference actions; no actor/checkpoint/policy inference',
        'initial_observation_size':int(initial['observation'].size),
        'lstm_reset_pending':initial['lstm_reset_pending'],
        'actions':reference_steps,
        'policy_updates':reference.contract.policy_updates,
        'physics_steps':len(reference_steps)*c.decimation,
        'target_rebuilt_each_physics_step':True,
        'official_predicate_and_physical_diagnostic_kept_separate':True,
    }
    # Independent negative control: naive force-saturated affine PD loses implicit
    # damping at saturation and oversteps the finger limits. Keep this failure.
    bad=AdaptedPegInsertEnv(EnvironmentConfig(finger_actuation='affine-pd'))
    bad.reset(peg_pos=[1,1,1],finger_opening=.02)
    bad.physics_step(np.zeros(7),finger_target=0,record=False)
    bad_q=bad.data.qpos[bad.finger_qpos_ids]
    report['probes']['naive_affine_finger_pd_negative_control']={'steps':1,
        'config':asdict(bad.config),'snapshot':bad.snapshot(),
        'joint_limit_violation_m':float(max(0,-min(bad_q),max(bad_q)-.04)),
        'rejected_for_final_configuration':True,
        'reason':'force clipping removes velocity derivative while saturated; source implicit drive not reproduced by naive affine actuator'}
    report['runtime_s']=time.perf_counter()-started
    report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['physics_steps_executed']=24+7+12+120+240+90+24+1
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError('EVIDENCE_ALREADY_EXISTS')
    report=bounded_environment_checks()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(path),'runtime_s':report['runtime_s'],'physics_steps':report['physics_steps_executed'],
        'gentle_insertion':report['probes']['gentle_reference_insertion']['passed'],
        'gravity_drop_insertion':report['probes']['aligned_drop_insertion']['passed'],'grasp':report['probes']['physical_grasp']['passed']}))


if __name__=='__main__':main()
