#!/usr/bin/env python3
"""G4-S2A-R1 static inertia diagnostics; never integrate physics or load policy.

Uses licensed S1B MuJoCo models plus a source-code-defined 0.01 kg m² body
rotational inertia sensitivity. This is DIAGNOSTIC_ONLY, not a reconstructed
Factory/PhysX runtime. USD assets are read locally only for scalar metadata.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s1a_robot_audit import BODY_MAP,JOINT_MAP,inertia_tensor,original_fk,quat_matrix
from scripts.g4s1b_env import model_xml,EnvironmentConfig
from scripts.g4s1b_contract import DEFAULT_Q,KP,KD

USD_LIB=Path('/home/ubuntu2004/isaacsim-4.5.0/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310')
USD_PYTHON='/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/envs/factory-g01/bin/python'
RUN=Path('/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g4_s/s2a/20261026T1403_frozen_smoke_v1')
EXPECTED_XML='c60c33ee11eb4806294aee259f97d987308b23e83da62f2219cbc9950fb195c9'
ARM_NAMES=tuple(f'joint{i}' for i in range(1,8))
FINGER_NAMES=('finger_joint1','finger_joint2')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def usd_readonly_probe():
    """Standalone USD only, never Kit/SimulationApp/PhysX, no asset export."""
    from pxr import Sdf,Usd
    archive=read(ROOT/'evidence/g4_s/s0/assets_audit.json')['original_usd_probe']['result']['assets'][0]
    path=Path(archive['file']['path'])
    if sha(path)!=archive['file']['sha256']:raise ValueError('ORIGINAL_USD_HASH_MISMATCH')
    layer=Sdf.Layer.FindOrOpen(str(path))
    if layer.GetExternalReferences():raise ValueError('UNEXPECTED_EXTERNAL_COMPOSITION')
    stage=Usd.Stage.Open(layer,load=Usd.Stage.LoadNone)
    found=[]
    for prim in stage.TraverseAll():
        mass=prim.GetAttribute('physics:mass')
        if not mass or not mass.HasAuthoredValueOpinion():continue
        fields={}
        for name in ('physics:mass','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes'):
            attr=prim.GetAttribute(name)
            if not attr or not attr.HasAuthoredValueOpinion():fields[name]=None;continue
            value=attr.Get()
            if hasattr(value,'GetReal'):value=[float(value.GetReal()),*map(float,value.GetImaginary())]
            elif hasattr(value,'__len__'):value=list(map(float,value))
            else:value=float(value)
            fields[name]=value
        found.append({'body':prim.GetName(),'path':str(prim.GetPath()),'metadata':fields})
    return {'status':'READONLY_LOCAL_USD_METADATA_VERIFIED','usd_version':list(Usd.GetVersion()),'usd_path':str(path),
            'usd_sha256':sha(path),'bodies':found,'no_original_or_derived_asset_export':True,
            'simulation_app':False,'physics_steps':0,'policy_calls':0,'physx_runtime_values':'UNKNOWN'}


def source_rule():
    source=read(ROOT/'evidence/g4_i/i0/checkpoint_audit.json')['source_files']['factory_env.py']
    if sha(source['path'])!=source['sha256']:raise ValueError('PINNED_FACTORY_SOURCE_CHANGED')
    tree=ast.parse(Path(source['path']).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FactoryEnv')
    rule=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_set_body_inertias')
    text=ast.get_source_segment(Path(source['path']).read_text(),rule)
    if 'offset[:, :, [0, 4, 8]] += 0.01' not in text:raise ValueError('SOURCE_INERTIA_RULE_CHANGED')
    cfg=read(ROOT/'evidence/g4_i/i0/checkpoint_audit.json')['source_files']['factory_env_cfg.py']
    if sha(cfg['path'])!=cfg['sha256']:raise ValueError('PINNED_FACTORY_CONFIGURATION_CHANGED')
    cfg_text=Path(cfg['path']).read_text()
    return {'configuration_source':cfg,'configuration_robot_range_lines':[120,183],
            'configuration_override_summary':{'robot_gravity_disabled':True,'robot_linear_angular_damping':0.,'arm_joint_damping':0.,'arm_joint_friction':0.,'robot_joint_armature':0.,'finger_stiffness':7500.,'finger_damping':173.,'finger_friction':.1,'finger_effort_limit_configured_N':40.,'configured_effort_is_not_measured_applied_force':True},
            'path':source['path'],'sha256':source['sha256'],'line_start':rule.lineno,'line_end':rule.end_lineno,
            'function_source':text,'quantity':'body rotational inertia tensor','offset_kg_m2':.01,
            'basis_invariance':'0.01I is isotropic, so adding0.01 to principal moments is equivalent in any rotated inertia basis.',
            'not_joint_armature':True,'body_scope':'all bodies returned by root_physx_view.get_inertias; actual returned/merged PhysX body set not measured',
            'physx_runtime_applied_torque':'UNKNOWN'}


def model_arrays(model):
    import mujoco
    return [{'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i),'id':i,'mass_kg':float(model.body_mass[i]),
             'CoM_body_m':model.body_ipos[i].tolist(),'principal_axes_wxyz':model.body_iquat[i].tolist(),
             'principal_inertia_kg_m2':model.body_inertia[i].tolist(),'inertia_tensor_body_kg_m2':inertia_tensor(model.body_inertia[i],model.body_iquat[i]).tolist(),
             'gravity_compensation':float(model.body_gravcomp[i])}
            for i in range(1,model.nbody)]


def matched_data(model,state):
    import mujoco
    data=mujoco.MjData(model)
    data.qpos[:]=state['qpos'];data.qvel[:]=state['qvel'];data.time=float(state['time_s'])
    mujoco.mj_forward(model,data)
    if data.time!=state['time_s']:raise AssertionError('Forward changed time')
    return data


def jacobians(model,data,arm_dof):
    import mujoco
    jac=[]
    for name in ('left_finger','right_finger'):
        linear,angular=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jacBody(model,data,linear,angular,model.body(name).id)
        jac.append(np.vstack([linear[:,arm_dof],angular[:,arm_dof]]))
    linear,angular=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,data,linear,angular,model.site('factory_tcp').id)
    return sum(jac)/2,np.vstack([linear[:,arm_dof],angular[:,arm_dof]])


def dynamics(model,data,torques):
    import mujoco
    arm=np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_NAMES])
    robot=np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_NAMES+FINGER_NAMES])
    full=np.zeros((model.nv,model.nv));mujoco.mj_fullM(model,full,data.qM)
    m=full[np.ix_(arm,arm)];robot_m=full[np.ix_(robot,robot)];mi=np.linalg.inv(m)
    j,tcp_j=jacobians(model,data,arm);task_inverse=j@mi@j.T;task_mass=np.linalg.inv(task_inverse)
    j_hash=task_mass@j@mi;projector=np.eye(7)-j.T@j_hash
    q=data.qpos[[model.jnt_qposadr[model.joint(n).id] for n in ARM_NAMES]]
    qdot=data.qvel[arm];wrapped=(DEFAULT_Q-q+np.pi)%(2*np.pi)-np.pi
    null_drive=m@(10*wrapped-6.3246*qdot);null_torque=projector@null_drive
    effects={}
    for name,tau in torques.items():
        tau=np.asarray(tau,dtype=float);acc=mi@tau
        effects[name]={'fixed_arm_torque_nm':tau.tolist(),'qddot_rad_s2':acc.tolist(),'joint7_qddot_rad_s2':float(acc[6]),'norm_qddot_rad_s2':float(np.linalg.norm(acc))}
    damping=j.T@np.diag(KD)@j
    damping_eigenvalues=np.linalg.eigvals(mi@damping)
    if np.max(np.abs(damping_eigenvalues.imag))>1e-10:raise ValueError('UNEXPECTED_COMPLEX_DAMPING_EIGENVALUE')
    damping_eigenvalues=damping_eigenvalues.real
    damping_indicator={'frozen_KP':KP.tolist(),'frozen_KD':KD.tolist(),'dt_s':1/120,
      'D_equals_J_transpose_KD_J':damping.tolist(),'eigenvalues_MinvD_per_s':damping_eigenvalues.tolist(),
      'maximum_dt_eigenvalue':float(np.max(damping_eigenvalues)/120),
      'damping_only_explicit_Euler_spectral_radius':float(np.max(np.abs(1-damping_eigenvalues/120))),
      'ideal_damped_modes_boundary':'0 < dt*lambda(M^-1*D) <2; J nullspace gives a neutral zero-eigenvalue mode.',
      'scope':'LOCAL_ALGEBRAIC_INDICATOR_ONLY: fixed q,J,M with instantaneous velocity feedback and unconstrained Euler damping update. Excludes actual FD delay, pose-target reconstruction, saturation, contact/equality constraints and MuJoCo integration. Not a measured stability certificate.'}
    return {'local_velocity_feedback_indicator':damping_indicator,'arm_dof_ids':arm.tolist(),'robot_dof_ids':robot.tolist(),'M7_kg_m2':m.tolist(),'M9_mixed_rotational_translational_units':robot_m.tolist(),
      'mass_eigenvalues':np.linalg.eigvalsh(m).tolist(),'M7_condition':float(np.linalg.cond(m)),
      'effective_arm_inertia_other_arm_coordinates_free_fingers_locked_kg_m2':(1/np.diag(mi)).tolist(),
      'effective_robot_coordinate_inertias_other_robot_coordinates_free_mixed_units':(1/np.diag(np.linalg.inv(robot_m))).tolist(),
      'controller_J_world_linear_angular_mean_finger_body_origins':j.tolist(),'TCP_J_world_linear_angular':tcp_j.tolist(),
      'TCP_vs_controller_J_Frobenius_difference':float(np.linalg.norm(tcp_j-j)),
      'task_inverse_condition':float(np.linalg.cond(task_inverse)),'task_space_effective_inertia':task_mass.tolist(),
      'dynamically_consistent_J_transpose_hash':j_hash.tolist(),'nullspace_torque_projector':projector.tolist(),
      'nullspace_dynamic_residual_norm':float(np.linalg.norm(j@mi@projector)),'q7':q.tolist(),'qdot7':qdot.tolist(),
      'nullspace_drive_unprojected_nm':null_drive.tolist(),'nullspace_torque_nm':null_torque.tolist(),
      'fixed_torque_incremental_response':effects,
      'fixed_torque_assumptions':'qddot=M7^-1 tau is an unconstrained incremental response with fingers locked. It excludes velocity/gravity bias, contacts, equality/joint-limit constraints and timestep/integrator effects; not a simulated or measured acceleration.',
      'qfrc_bias_generalized_full':data.qfrc_bias.tolist()}


def delta_mass_from_body_inertia(model,data,arm_dof,body_names):
    """Independent exact kinetic-energy pullback sum Jw.T * .01I * Jw."""
    import mujoco
    delta=np.zeros((7,7))
    for name in body_names:
        linear,angular=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jacBody(model,data,linear,angular,model.body(name).id)
        jw=angular[:,arm_dof];delta+=.01*(jw.T@jw)
    return delta


def original_metadata_kinematics(robot,state):
    """Only authored joint-frame FK/central derivatives; never source dynamics."""
    q=dict(zip(JOINT_MAP,state['qpos'][:9]));base=original_fk(robot,q);eps=1e-6;values=[]
    for source in JOINT_MAP:
        high=q.copy();low=q.copy();high[source]+=eps;low[source]-=eps
        hi,lo=original_fk(robot,high),original_fk(robot,low)
        per_body={}
        for body in ('panda_leftfinger','panda_rightfinger','panda_fingertip_centered'):
            rotation_derivative=(hi[body][:3,:3]-lo[body][:3,:3])/(2*eps)
            skew=rotation_derivative@base[body][:3,:3].T
            omega=np.array([skew[2,1]-skew[1,2],skew[0,2]-skew[2,0],skew[1,0]-skew[0,1]])/2
            per_body[body]=np.r_[(hi[body][:3,3]-lo[body][:3,3])/(2*eps),omega]
        values.append(per_body)
    mean=np.stack([(v['panda_leftfinger']+v['panda_rightfinger'])/2 for v in values[:7]],axis=1)
    tcp=np.stack([v['panda_fingertip_centered'] for v in values[:7]],axis=1)
    return {'controller_J_from_authored_FK_central_derivative':mean.tolist(),'TCP_J_from_authored_FK_central_derivative':tcp.tolist(),'central_difference_step_radian':eps,
            'source_TCP_world_transform':base['panda_fingertip_centered'].tolist(),
            'scope':'Conditional interpretation of archived USD joint frames with base identity; not measured PhysX runtime Jacobian or mass matrix.'}


def audit(output,run=RUN):
    import mujoco
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if mujoco.__version__!='3.5.0':raise ValueError('MUJOCO_VERSION_MISMATCH')
    xml=model_xml(EnvironmentConfig())
    if hashlib.sha256(xml.encode()).hexdigest()!=EXPECTED_XML:raise ValueError('FROZEN_XML_CHANGED')
    manifest=read(run/'artifact_manifest.json')
    for path,expected in manifest.items():
        if sha(path)!=expected:raise ValueError('ARCHIVED_TRACE_CHANGED:'+path)
    source=source_rule();save(output/'source_runtime_inertia_rule.json',source)
    archive=read(ROOT/'evidence/g4_s/s0/assets_audit.json');robot=archive['original_usd_probe']['result']['assets'][0]
    mapping=read(ROOT/'evidence/g4_s/s1a/robot_mapping.json')
    env=os.environ.copy();env['PYTHONPATH']=str(USD_LIB);env['LD_LIBRARY_PATH']=str(USD_LIB/'bin')
    child=subprocess.run([USD_PYTHON,str(Path(__file__).resolve()),'--usd-readonly-probe'],env=env,text=True,capture_output=True,timeout=30)
    usd={'command':[USD_PYTHON,str(Path(__file__).resolve()),'--usd-readonly-probe'],'returncode':child.returncode,'stderr':child.stderr}
    try:usd['result']=json.loads(child.stdout)
    except json.JSONDecodeError:usd['stdout']=child.stdout
    save(output/'usd_metadata_readonly.json',usd)
    if child.returncode!=0:raise RuntimeError('READONLY_USD_METADATA_PROBE_FAILED')
    for row in usd['result']['bodies']:
        original=next(b['attributes'] for b in robot['masses'] if b['path']==row['path'])
        for key,value in row['metadata'].items():
            if key not in original and value is None:continue
            if not np.array_equal(value,original[key]):raise ValueError('USD_METADATA_ARCHIVE_MISMATCH:'+row['path']+':'+key)
    models={};body_names=tuple(BODY_MAP.values());variants={
       'frozen_s1b':{'body_inertia_delta_kg_m2':0.,'joint_armature_delta':0.,'classification':'FROZEN_BASELINE'},
       'mapped_robot_body_plus_0p01I':{'body_inertia_delta_kg_m2':.01,'joint_armature_delta':0.,'classification':'DIAGNOSTIC_ONLY'},
       'robot_joint_armature_plus_0p01':{'body_inertia_delta_kg_m2':0.,'joint_armature_delta':.01,'classification':'DIAGNOSTIC_ONLY'}}
    for name,variant in variants.items():
        m=mujoco.MjModel.from_xml_string(xml)
        if variant['body_inertia_delta_kg_m2']:
            for body in body_names:m.body_inertia[m.body(body).id]+=.01
        if variant['joint_armature_delta']:
            for joint in ARM_NAMES+FINGER_NAMES:m.dof_armature[m.jnt_dofadr[m.joint(joint).id]]+=.01
        mujoco.mj_setConst(m,mujoco.MjData(m));models[name]=m
    table=[];compiled={row['name']:row for row in model_arrays(models['frozen_s1b'])}
    for row in mapping['bodies']:
        target=row['target'];match=compiled.get(target)
        table.append({'source_body':row['source'],'target_body':target,'source_authored_mass_kg':row['source_mass_kg'],
          'source_CoM_m_in_source_axes':row.get('source_CoM_body_m'),'source_principal_inertia_kg_m2':row.get('source_principal_inertia_kg_m2'),
          'source_tensor_in_target_body_axes_kg_m2':row.get('source_CoM_tensor_in_target_body_axes_kg_m2'),
          'conditional_source_tensor_plus0p01I_in_target_body_axes':row.get('conditional_runtime_tensor_in_target_body_axes_kg_m2'),
          'compiled_target':match,'source_runtime_status':'UNKNOWN_NOT_MEASURED_PHYSX','missing_target_body':target is None,
          'source_authored_inertia_missing':row['source_missing_authored_inertia']})
    authored_joint_properties=[{'source_joint':j['path'].split('/')[-1],'authored_properties':{k:v for k,v in j['attributes'].items() if any(term in k.lower() for term in ('friction','armature','damping','stiffness','maxforce'))}} for j in robot['joints']]
    save(output/'inertial_parameter_mapping.json',{'bodies':table,'mapped_body_count':len(body_names),'authored_joint_properties_prior_to_runtime_overrides':authored_joint_properties,
         'compiled_frozen_joint_parameters':{'armature':models['frozen_s1b'].dof_armature.tolist(),'damping':models['frozen_s1b'].dof_damping.tolist(),'frictionloss':models['frozen_s1b'].dof_frictionloss.tolist()},
         'limitations':['Source force_sensor rotational inertia is not authored; actual PhysX cooking or fixed-body merging unknown.','No source USD mesh/asset is exported or converted. Authored numeric metadata is compared only.','Mapped-body 0.01I variant lacks extra source sensor and centered-fingertip bodies; not a complete Factory reconstruction.']})
    initial=read(run/'seed_1001/interface_reset.json')['initial']['state'];physical=rows(run/'seed_1001/physics.jsonl')
    states={'admitted_initial_q':initial,'archived_after_physics_step4_q':physical[3]['state']}
    first_applied=physical[0]['state']['integration_actuation']['arm_actuator_force_nm'];step4_applied=physical[3]['state']['integration_actuation']['arm_actuator_force_nm']
    torques={'unit_joint7_1Nm':[0,0,0,0,0,0,1],'archived_first_applied_torque':first_applied,'archived_step4_applied_torque':step4_applied}
    records=[]
    for state_name,state in states.items():
        row={'state_id':state_name,'archive_state_time_s':state['time_s'],'source_qpos_full':state['qpos'],'source_qvel_full':state['qvel'],'variants':{}}
        datas={}
        for name,m in models.items():
            datas[name]=matched_data(m,state);row['variants'][name]=dynamics(m,datas[name],torques)
        base=row['variants']['frozen_s1b'];body=row['variants']['mapped_robot_body_plus_0p01I'];armature=row['variants']['robot_joint_armature_plus_0p01']
        theoretical=delta_mass_from_body_inertia(models['frozen_s1b'],datas['frozen_s1b'],np.array(base['arm_dof_ids']),body_names)
        observed=np.array(body['M7_kg_m2'])-np.array(base['M7_kg_m2'])
        row['body_inertia_delta_M_independent_Jw_pullback']=theoretical.tolist()
        row['body_delta_M_vs_Jw_pullback_max_abs_error']=float(np.max(np.abs(observed-theoretical)))
        row['armature_delta_M_vs0p01I_max_abs_error']=float(np.max(np.abs(np.array(armature['M7_kg_m2'])-np.array(base['M7_kg_m2'])-.01*np.eye(7))))
        row['body_delta_M_offdiagonal_Frobenius_norm']=float(np.linalg.norm(observed-np.diag(np.diag(observed))))
        row['source_authored_kinematics']=original_metadata_kinematics(robot,state)
        row['source_authored_vs_MJ_controller_J_max_abs_difference']=float(np.max(np.abs(np.array(row['source_authored_kinematics']['controller_J_from_authored_FK_central_derivative'])-np.array(base['controller_J_world_linear_angular_mean_finger_body_origins']))))
        row['unchanged_J_max_abs_difference_across_inertia_variants']=max(float(np.max(np.abs(np.array(v['controller_J_world_linear_angular_mean_finger_body_origins'])-np.array(base['controller_J_world_linear_angular_mean_finger_body_origins'])))) for v in row['variants'].values())
        row['minimum_body_delta_M_eigenvalue']=float(np.min(np.linalg.eigvalsh(observed)))
        if row['body_delta_M_vs_Jw_pullback_max_abs_error']>1e-12 or row['armature_delta_M_vs0p01I_max_abs_error']>1e-12:raise ValueError('INDEPENDENT_INERTIA_PULLBACK_CHECK_FAILED')
        if row['minimum_body_delta_M_eigenvalue']< -1e-12:raise ValueError('POSITIVE_BODY_INERTIA_INCREMENT_NOT_PSD')
        if row['unchanged_J_max_abs_difference_across_inertia_variants']!=0.:raise ValueError('INERTIA_VARIANT_CHANGED_KINEMATICS')
        for variant in row['variants'].values():
            if variant['nullspace_dynamic_residual_norm']>1e-10:raise ValueError('DYNAMIC_NULLSPACE_CHECK_FAILED')
        records.append(row)
    evidence={'task':'G4-S2A-R1','scope':'STATIC_DIAGNOSTIC_ONLY','mujoco_version':mujoco.__version__,'frozen_XML_sha256':EXPECTED_XML,
      'generated_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'source_rule':source,'variants':variants,'matched_configurations':records,
      'assumptions':['Arm7 principal mass block uses the same 9-coordinate Panda state; free peg dynamics are not included in the arm block.','Effective inertia1/diag(M^-1) lets other arm coordinates accelerate and locks finger coordinates; it is not the diagonal of M.','No contact/constraint/velocity-bias/gravity integration is used in fixed-torque incremental acceleration.','Nullspace operator is exactly I-J.T*(J*M^-1*J.T)^-1*J*M^-1.','Body isotropic inertia offsets leave geometric Jacobians unchanged and create dense generalized mass increments; joint armature creates a diagonal generalized mass increment.'],
      'original_runtime_mass_matrix':'UNKNOWN','original_runtime_applied_torque':'UNKNOWN','original_runtime_body_inertias':'UNKNOWN_NOT_DIRECTLY_MEASURED',
      'restrictions':{'physics_integration_steps':0,'actor_calls':0,'checkpoint_loads':0,'isaac_sim_launched':False,'usd_assets_exported':False,'historical_files_modified':False},
      'runtime_s':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    save(output/'matched_dynamics.json',evidence)
    receipt={'task':'G4-S2A-R1','status':'STATIC_DYNAMICS_AUDIT_COMPLETE','output':str(output),'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()},'physics_steps':0,'policy_calls':0}
    save(output/'manifest.json',receipt)
    for row in records:
        print(row['state_id'])
        for name,v in row['variants'].items():print(name,'Mcond',v['M7_condition'],'J7eff',v['effective_arm_inertia_other_arm_coordinates_free_fingers_locked_kg_m2'][6],'J7unitacc',v['fixed_torque_incremental_response']['unit_joint7_1Nm']['joint7_qddot_rad_s2'])
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--usd-readonly-probe',action='store_true');a=p.parse_args()
    if a.usd_readonly_probe:print(json.dumps(usd_readonly_probe(),allow_nan=False));return 0
    if a.output is None:p.error('--output required')
    audit(a.output,a.run);return 0

if __name__=='__main__':raise SystemExit(main())
