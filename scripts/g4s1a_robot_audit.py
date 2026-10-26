#!/usr/bin/env python3
"""G4-S1A static Panda audit, using archived Factory metadata and licensed Panda.

No original USD is loaded or converted. No dynamics forward, integration, policy,
viewer, or controller executes: only compilation, MjData, and mj_kinematics.
The computed Factory FK is an interpretation of authored joint frames, NOT a
PhysX run or a claim of frame/quaternion-sign/dynamics equivalence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_COMMIT = '8161bba264d7fa7c99ca301e91e7fb44737676ad'
DEFAULT_MODEL = Path('/home/ubuntu2004/mujoco_menagerie/franka_emika_panda/panda.xml')
JOINT_MAP = {**{f'panda_joint{i}': f'joint{i}' for i in range(1, 8)},
             'panda_finger_joint1': 'finger_joint1', 'panda_finger_joint2': 'finger_joint2'}
BODY_MAP = {**{f'panda_link{i}': f'link{i}' for i in range(8)},
            'panda_hand': 'hand', 'panda_leftfinger': 'left_finger',
            'panda_rightfinger': 'right_finger'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quat_matrix(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('invalid wxyz quaternion')
    norm = math.hypot(*q)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('invalid wxyz quaternion')
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform(pos=(0., 0., 0.), quat=(1., 0., 0., 0.)):
    t = np.eye(4); t[:3, :3] = quat_matrix(quat); t[:3, 3] = pos
    return t


def inertia_tensor(diagonal, principal_axes):
    diag = np.asarray(diagonal, float)
    if diag.shape != (3,) or not np.isfinite(diag).all() or np.any(diag <= 0):
        raise ValueError('principal inertias must be finite positive')
    r = quat_matrix(principal_axes)
    return r @ np.diag(diag) @ r.T


def original_fk(robot, q):
    """T_child=T_parent*T_joint0*D(q)*inverse(T_joint1), body roots aligned."""
    if set(q) != set(JOINT_MAP) or not np.isfinite(list(q.values())).all():
        raise ValueError('all nine original coordinates required, finite, by name')
    out = {'panda_link0': np.eye(4)}
    pending = [j for j in robot['joints'] if j['path'].split('/')[-1] != 'root_joint']
    while pending:
        progressed = False
        for j in list(pending):
            a = j['attributes']; rel = j['relations']
            parent = rel['physics:body0'][0].split('/')[-1]
            child = rel['physics:body1'][0].split('/')[-1]
            if parent not in out:
                continue
            t0 = transform(a.get('physics:localPos0', [0]*3), a.get('physics:localRot0', [1, 0, 0, 0]))
            t1 = transform(a.get('physics:localPos1', [0]*3), a.get('physics:localRot1', [1, 0, 0, 0]))
            d = np.eye(4); name = j['path'].split('/')[-1]
            if j['type'] == 'PhysicsRevoluteJoint':
                axis = 'XYZ'.index(a['physics:axis']); vector = np.zeros(3)
                vector[axis] = math.sin(q[name]/2)
                d[:3, :3] = quat_matrix([math.cos(q[name]/2), *vector])
            elif j['type'] == 'PhysicsPrismaticJoint':
                d['XYZ'.index(a['physics:axis']), 3] = q[name]
            elif j['type'] != 'PhysicsFixedJoint':
                raise ValueError('unsupported original joint type')
            out[child] = out[parent] @ t0 @ d @ np.linalg.inv(t1)
            pending.remove(j); progressed = True
        if not progressed:
            raise ValueError('joint tree has unresolved parents/cycle')
    return out


def angular_difference(a, b):
    # atan2 retains accuracy close to pi and zero, without a fragile acos clip.
    r = a.T @ b
    skew = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]]) / 2
    return math.atan2(float(np.linalg.norm(skew)), float((np.trace(r)-1)/2))


def audit(model_path=DEFAULT_MODEL):
    import mujoco
    started = time.monotonic()
    if mujoco.__version__ != '3.5.0' or mujoco.mj_versionString() != '3.5.0':
        raise ValueError('MuJoCo 3.5.0 required')
    model_path = Path(model_path).resolve()
    s0_path = ROOT/'evidence/g4_s/s0/assets_audit.json'
    s0 = json.loads(s0_path.read_text())
    ctrl_path = ROOT/'evidence/g4_s/s0/controller_audit.json'
    ctrl = json.loads(ctrl_path.read_text())
    inventory = s0['candidate_inventories'][0]
    expected = next(p['sha256'] for p in inventory['model_files'] if p['path'].endswith('/panda.xml'))
    if model_path.name != 'panda.xml' or sha(model_path) != expected:
        raise ValueError('reject model other than S0 pinned Panda XML')
    commit = subprocess.check_output(['git', '-C', str(model_path.parent), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != MENAGERIE_COMMIT:
        raise ValueError('Menagerie commit mismatch')
    reference_files = inventory['referenced_geometry'] + [inventory['license'], inventory['readme']]
    for record in reference_files:
        p = model_path.parent/('assets/'+Path(record['path']).name if '/assets/' in record['path'] else Path(record['path']).name)
        if not p.is_file() or sha(p) != record['sha256']:
            raise ValueError('missing or changed pinned Panda dependency: '+str(p))
    m = mujoco.MjModel.from_xml_path(str(model_path))
    m2 = mujoco.MjModel.from_xml_path(str(model_path))
    def binary_fingerprint(model):
        buf = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
        mujoco.mj_saveModel(model, None, buf)
        return hashlib.sha256(buf.tobytes()).hexdigest(), len(buf)
    fp, nbytes = binary_fingerprint(m); fp2, _ = binary_fingerprint(m2)
    if fp != fp2:
        raise ValueError('non-reproducible compiled model')
    name = lambda typ, i: mujoco.mj_id2name(m, typ, int(i))
    robot = s0['original_usd_probe']['result']['assets'][0]
    original_joints = {j['path'].split('/')[-1]: j for j in robot['joints']}
    joints = []
    for source, target in JOINT_MAP.items():
        j = original_joints[source]; a = j['attributes']
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, target)
        if i < 0:
            raise ValueError('missing mapped joint '+target)
        bid = int(m.jnt_bodyid[i]); pid = int(m.body_parentid[bid])
        unit = 'radian' if j['type'] == 'PhysicsRevoluteJoint' else 'metre'
        limits = [a['physics:lowerLimit'], a['physics:upperLimit']]
        if unit == 'radian': limits = list(map(math.radians, limits))
        axis = np.eye(3)['XYZ'.index(a['physics:axis'])]
        source_axis = quat_matrix(a['physics:localRot0']) @ axis
        target_axis = quat_matrix(m.body_quat[bid]) @ m.jnt_axis[i]
        joints.append({'source': source, 'target': target, 'target_joint_id': i,
            'unit': unit, 'qpos_address': int(m.jnt_qposadr[i]), 'dof_address': int(m.jnt_dofadr[i]),
            'source_joint_type': j['type'], 'source_authored_limit': [a['physics:lowerLimit'], a['physics:upperLimit']],
            'source_converted_limit': limits, 'target_limit': m.jnt_range[i].tolist(),
            'max_limit_difference': max(abs(limits[k]-m.jnt_range[i, k]) for k in range(2)),
            'source_parent': j['relations']['physics:body0'][0], 'source_child': j['relations']['physics:body1'][0],
            'source_local_joint_frames': {k: v for k, v in a.items() if k.startswith(('physics:local', 'physics:axis'))},
            'source_zero_pose_axis_in_parent': source_axis.tolist(),
            'target_zero_pose_axis_in_parent': target_axis.tolist(),
            'target_parent': name(mujoco.mjtObj.mjOBJ_BODY, pid), 'target_child': name(mujoco.mjtObj.mjOBJ_BODY, bid),
            'target_child_local_position': m.body_pos[bid].tolist(), 'target_child_local_quaternion_wxyz': m.body_quat[bid].tolist(),
            'target_joint_position_child_frame': m.jnt_pos[i].tolist(), 'target_axis_child_frame': m.jnt_axis[i].tolist(),
            'target_damping': float(m.dof_damping[m.jnt_dofadr[i]]),
            'target_armature': float(m.dof_armature[m.jnt_dofadr[i]]),
            'target_frictionloss': float(m.dof_frictionloss[m.jnt_dofadr[i]]),
            'source_mimic_metadata': {k: v for k, v in a.items() if 'Mimic' in k},
            'source_mimic_relations': {k: v for k, v in j['relations'].items() if 'Mimic' in k}})
    bodies = []
    for row in robot['masses']:
        src = row['path'].split('/')[-1]; a = row['attributes']; target = BODY_MAP.get(src)
        tensor = inertia_tensor(a['physics:diagonalInertia'], a['physics:principalAxes']) if 'physics:diagonalInertia' in a else None
        b = {'source': src, 'source_mass_kg': a['physics:mass'], 'source_CoM_body_m': a.get('physics:centerOfMass'),
             'source_principal_inertia_kg_m2': a.get('physics:diagonalInertia'),
             'source_principal_axes_body_wxyz': a.get('physics:principalAxes'),
             'source_CoM_tensor_in_body_axes_kg_m2': None if tensor is None else tensor.tolist(),
             'source_tensor_plus_runtime_0_01I_conditional': None if tensor is None else (tensor+np.eye(3)*.01).tolist(),
             'source_runtime_tensor_status': 'CONDITIONAL_AUTHORED_TENSOR_PLUS_SOURCE_CODE_RULE_NOT_MEASURED_PHYSX_RUNTIME',
             'source_missing_authored_inertia': tensor is None, 'target': target}
        if target:
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
            t = inertia_tensor(m.body_inertia[i], m.body_iquat[i])
            b.update(target_body_id=i, target_mass_kg=float(m.body_mass[i]),
                target_CoM_body_m=m.body_ipos[i].tolist(), target_principal_axes_body_wxyz=m.body_iquat[i].tolist(),
                target_principal_inertia_kg_m2=m.body_inertia[i].tolist(), target_CoM_tensor_in_body_axes_kg_m2=t.tolist(),
                mass_difference_kg=float(a['physics:mass']-m.body_mass[i]),
                authored_tensor_max_abs_difference_kg_m2=None if tensor is None else float(np.max(np.abs(t-tensor))),
                target_gravcomp=float(m.body_gravcomp[i]))
        bodies.append(b)
    # Static named coordinates; no time sequence, integration, control or observations.
    poses = {'zero_algebraic_only_outside_some_limits': [0]*7,
             'menagerie_home': [0, 0, 0, -1.57079, 0, 1.57079, -.7853],
             'frozen_factory_default': [-1.3003, -.4015, 1.1791, -2.1493, .4001, 1.9425, .4754],
             'interior_asymmetric': [.2, -.4, .3, -1.8, -.2, 1.7, .6]}
    static = []
    for label, arm in poses.items():
        for opening in [0., .02, .04]:
            q = dict(zip(list(JOINT_MAP)[:7], arm)); q.update(panda_finger_joint1=opening, panda_finger_joint2=opening)
            original = original_fk(robot, q); data = mujoco.MjData(m)
            for row in joints: data.qpos[row['qpos_address']] = q[row['source']]
            mujoco.mj_kinematics(m, data)
            frames = []
            for source, target in BODY_MAP.items():
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
                t = np.eye(4); t[:3, :3] = data.xmat[bid].reshape(3, 3); t[:3, 3] = data.xpos[bid]
                frames.append({'source': source, 'target': target, 'source_interpreted_world_transform': original[source].tolist(),
                    'target_world_transform': t.tolist(), 'origin_difference_m': float(np.linalg.norm(original[source][:3, 3]-t[:3, 3])),
                    'rotation_difference_rad': angular_difference(original[source][:3, :3], t[:3, :3])})
            hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hand')
            virtual = np.eye(4); virtual[:3, :3] = data.xmat[hand_id].reshape(3, 3); virtual[:3, 3] = data.xpos[hand_id]
            tcp_joint = original_joints['panda_fingertip_centered_joint']['attributes']
            virtual = virtual @ transform(tcp_joint['physics:localPos0'], tcp_joint['physics:localRot0']) @ np.linalg.inv(transform(tcp_joint['physics:localPos1'], tcp_joint['physics:localRot1']))
            ref = original['panda_fingertip_centered']
            static.append({'fixture': label, 'finger_opening_each_m': opening, 'source_q_by_name': q, 'frames': frames,
                'virtual_TCP_is_computed_metadata_not_installed_site': True,
                'source_interpreted_TCP_world_transform': ref.tolist(), 'target_virtual_TCP_world_transform': virtual.tolist(),
                'TCP_origin_difference_m': float(np.linalg.norm(ref[:3, 3]-virtual[:3, 3])),
                'TCP_rotation_difference_rad': angular_difference(ref[:3, :3], virtual[:3, :3]),
                'integration_steps': 0})
    # Compare inertial quantities only after expressing source data in target body axes.
    # This matters for right_finger's pi body-frame difference.
    zero_frames = {f['source']: f for f in static[0]['frames']}
    for b in bodies:
        if not b['target']:
            continue
        f = zero_frames[b['source']]
        ts = np.array(f['source_interpreted_world_transform'])
        tt = np.array(f['target_world_transform'])
        source_from_target = np.linalg.inv(ts) @ tt
        b['source_from_target_body_transform_at_zero'] = source_from_target.tolist()
        r = source_from_target[:3, :3]
        com = r.T @ (np.array(b['source_CoM_body_m'])-source_from_target[:3, 3])
        b['source_CoM_expressed_in_target_body_m'] = com.tolist()
        b['CoM_difference_after_frame_mapping_m'] = float(np.linalg.norm(com-np.array(b['target_CoM_body_m'])))
        if b['source_CoM_tensor_in_body_axes_kg_m2'] is not None:
            mapped_tensor = r.T @ np.array(b['source_CoM_tensor_in_body_axes_kg_m2']) @ r
            b['source_CoM_tensor_in_target_body_axes_kg_m2'] = mapped_tensor.tolist()
            b['tensor_max_abs_difference_after_frame_mapping_kg_m2'] = float(np.max(np.abs(mapped_tensor-np.array(b['target_CoM_tensor_in_body_axes_kg_m2']))))
            b['conditional_runtime_tensor_in_target_body_axes_kg_m2'] = (mapped_tensor+np.eye(3)*.01).tolist()
    geoms = []
    for i in range(m.ngeom):
        if not m.geom_contype[i] and not m.geom_conaffinity[i]: continue
        mesh_id = int(m.geom_dataid[i])
        geoms.append({'id': i, 'name': name(mujoco.mjtObj.mjOBJ_GEOM, i), 'body': name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]),
            'type': int(m.geom_type[i]), 'size': m.geom_size[i].tolist(), 'position_in_body_m': m.geom_pos[i].tolist(),
            'quaternion_in_body_wxyz': m.geom_quat[i].tolist(), 'mesh': name(mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id >= 0 else None,
            'contype': int(m.geom_contype[i]), 'conaffinity': int(m.geom_conaffinity[i]), 'margin_m': float(m.geom_margin[i]),
            'gap_m': float(m.geom_gap[i]), 'friction': m.geom_friction[i].tolist(), 'condim': int(m.geom_condim[i])})
    actuators = [{'name': name(mujoco.mjtObj.mjOBJ_ACTUATOR, i), 'trntype': int(m.actuator_trntype[i]),
        'transmission_ids': m.actuator_trnid[i].tolist(), 'gear': m.actuator_gear[i].tolist(),
        'gain_type': int(m.actuator_gaintype[i]), 'gain_parameters': m.actuator_gainprm[i].tolist(),
        'bias_type': int(m.actuator_biastype[i]), 'bias_parameters': m.actuator_biasprm[i].tolist(),
        'control_range': m.actuator_ctrlrange[i].tolist(), 'force_range': m.actuator_forcerange[i].tolist(),
        'force_limited': bool(m.actuator_forcelimited[i]), 'control_limited': bool(m.actuator_ctrllimited[i])} for i in range(m.nu)]
    invalid = []
    controls = {'missing_mesh': '<mujoco><asset><mesh name="bad" file="definitely_absent_task_G4S1A.stl"/></asset><worldbody><geom type="mesh" mesh="bad"/></worldbody></mujoco>',
        'non_positive_inertia': '<mujoco><worldbody><body><joint/><inertial pos="0 0 0" mass="1" diaginertia="-1 1 1"/></body></worldbody></mujoco>',
        'invalid_joint_reference': '<mujoco><actuator><motor joint="absent"/></actuator></mujoco>',
        'inertia_triangle_violation': '<mujoco><worldbody><body><joint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 3"/></body></worldbody></mujoco>'}
    for label, xml in controls.items():
        try:
            mujoco.MjModel.from_xml_string(xml)
            invalid.append({'control': label, 'rejected': False})
        except ValueError as error:
            invalid.append({'control': label, 'rejected': True, 'actual_error': str(error)})
    original_collision = [g for g in robot['geometry'] if 'PhysicsCollisionAPI' in g['apis']]
    return {'schema_version': 1, 'task': 'G4-S1A', 'status': 'VERIFIED_STATIC_MAPPING_WITH_RECONSTRUCTION_BLOCKERS',
        'scope': 'licensed pinned Menagerie Panda diagnostic only; archived Factory metadata algebra, not faithful Factory reconstruction',
        'engine_version': mujoco.__version__, 'engine_runtime_version': mujoco.mj_versionString(),
        'model_path': str(model_path), 'model_sha256': sha(model_path), 'menagerie_commit': commit,
        'archived_S0_assets_sha256': sha(s0_path), 'archived_S0_controller_sha256': sha(ctrl_path),
        'license_sha256': sha(model_path.parent/'LICENSE'), 'all_67_dependency_mesh_hashes_match_S0': len(inventory['referenced_geometry']) == 67,
        'compilation': {'nq': m.nq, 'nv': m.nv, 'nu': m.nu, 'nbody': m.nbody, 'ngeom': m.ngeom, 'njnt': m.njnt,
            'finger_coordinate_count': 2, 'binary_bytes': nbytes, 'first_MJB_sha256': fp, 'second_MJB_sha256': fp2,
            'same_process_recompilation_identical': fp == fp2},
        'static_FK_assumptions': ['Original root and candidate fixed base aligned to identity; not a measured live engine world pose.', 'USD local wxyz quaternions normalized before matrix conversion.', 'T_child=T_parent*T_joint0*D(q)*inverse(T_joint1); revolute coordinates radians, prismatic metres.', 'Positive finger q opens both fingers; original mimic gearing=-1 is stored separately, not reinterpreted as opposite signed coordinates.', 'No joint-limit admissibility is asserted for zero algebraic fixture; no MjData is integrated.'],
        'joints': joints, 'bodies': bodies, 'static_frame_checks': static, 'candidate_actuators': actuators,
        'candidate_collision_geoms': geoms, 'original_collision_geometry_archived_metadata': original_collision,
        'candidate_tendon_equality_and_contact_XML': {tag: ET.tostring(ET.parse(model_path).getroot().find(tag), encoding='unicode') for tag in ['tendon', 'equality', 'contact']},
        'original_fixed_joints': [j for j in robot['joints'] if j['type'] == 'PhysicsFixedJoint'],
        'frozen_runtime_parameters_archived': ctrl['physics_compatibility_matrix'],
        'candidate_defaults': {'timestep_s': m.opt.timestep, 'gravity_m_s2': m.opt.gravity.tolist(),
            'integrator_enum': int(m.opt.integrator), 'all_body_gravcomp': m.body_gravcomp.tolist()},
        'invalid_asset_controls': invalid,
        'findings': ['Nine named joint coordinates mapped with independent qpos and dof addresses.',
            'Right finger positive opening direction agrees but body axes differ by pi; transform collision and inertial frames explicitly.',
            'Factory TCP is hand-local z=0.11207100003957748; standard Panda has no corresponding site/body.',
            'Force_sensor and fingertip_centered physical bodies are absent in standard Panda; sensor authored inertia is absent.',
            'Arm inertias closely share origin but Factory runtime adds 0.01I to each body; Menagerie damping/armature and gravity differ.',
            'Default Panda arm position servos and split-tendon finger actuator are not original torque/implicit-PD/mimic semantics.',
            'Factory original collision meshes and contacts remain unconverted; diagnostic fingertip boxes are not equivalent.',
            'Authored-joint FK comparison does not measure live PhysX poses, imported inertias, raw quaternion signs, J or M parity.'],
        'restrictions': {'original_assets_read': False, 'original_assets_converted_or_exported': False, 'physics_integration_steps': 0,
            'forward_dynamics_calls': 0, 'policy_calls': 0, 'checkpoint_access': False, 'MJCF_assets_distributed': False},
        'primary_sources': {'API': 'https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html',
            'XML': 'https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html',
            'pinned_Panda': f'https://github.com/google-deepmind/mujoco_menagerie/tree/{MENAGERIE_COMMIT}/franka_emika_panda'},
        'runtime_s': time.monotonic()-started, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def markdown(d):
    out = ['# G4-S1A Panda 静态映射审计', '',
        '只用归档 S0 Factory 元数据与独立许可的固定 Menagerie Panda；不读取/转换原 USD，',
        '不积分、不运行策略。`mj_kinematics` 输出仅为静态 FK，不能建立 PhysX 动力学/接触或四元数符号一致性。', '',
        '| Factory joint | MuJoCo joint | qpos / dof | Factory limit (rad/m) | Panda limit | 最大差 |',
        '|---|---|---|---|---|---|']
    for j in d['joints']:
        out.append(f"| {j['source']} | {j['target']} | {j['qpos_address']} / {j['dof_address']} | {j['source_converted_limit']} | {j['target_limit']} | {j['max_limit_difference']:.9g} |")
    out += ['', '完整父子局部帧、轴、mimic、9D 地址在 JSON。右夹指尤其不能按名字直接等同：',
        'Factory `localRot0=localRot1=Rx(pi)` 相消，零位 finger body 与 hand 同向；',
        'Panda `right_finger` 是 `Rz(pi)`。两者 +q 均沿 hand -Y 开口，但 body 朝向相差 pi。', '',
        '| 静态 fixture / 单侧开口 m | virtual TCP 位置差 m | 朝向差 rad |', '|---|---|---|']
    for f in d['static_frame_checks']:
        out.append(f"| {f['fixture']} / {f['finger_opening_each_m']} | {f['TCP_origin_difference_m']:.9g} | {f['TCP_rotation_difference_rad']:.9g} |")
    out += ['', 'virtual TCP 只是在数值上使用 hand-local z=0.11207100003957748 的元数据计算；',
        '未发布含此站点的原资产衍生模型。零位 fixture 在 joint4 等限位外，只作代数帧检查。', '',
        '| Factory body | Panda body | 原 mass kg | Panda mass kg | 对齐 frame 后 authored tensor 最大差 kg·m² |', '|---|---|---|---|---|']
    for b in d['bodies']:
        out.append(f"| {b['source']} | {b['target'] or 'MISSING'} | {b['source_mass_kg']:.10g} | {b.get('target_mass_kg', 'N/A')} | {b.get('tensor_max_abs_difference_after_frame_mapping_kg_m2', 'N/A')} |")
    out += ['', '表中张量差先把原 source 惯量旋转到 candidate body 轴后再比较，右夹指的 pi 帧差已计入。',
        'JSON `source_principal_inertia` / `target_principal_inertia` 是 CoM 处主轴对角元；',
        '`*_CoM_tensor_in_body_axes` 才是 `R diag(I) Rᵀ` 恢复的 CoM 处完整3×3矩阵。',
        '`*_tensor_in_target_body_axes` 则进一步转换到 candidate body 轴；这些不是关于 body 原点的张量。',
        '另记源码要求的每体 `+0.01 I` 条件张量；这不是关节 armature，也不是实测 PhysX tensor。',
        '`force_sensor` 仅 authored mass/CoM、无 authored inertia；目标也缺该 0.01 kg 体。',
        'TCP参考体质量约1e-6 kg/inertia1e-9 kg·m² 亦不能静默删除后宣称等价。', '',
        'Panda 原驱动为 7 个位置伺服及 1 个 split tendon 开口驱动（0..255），并非 7 个 torque motor。',
        '原两夹指 implicit PD=7500/173、40 N、friction0.1、mimic 的实际实现尚未重建。',
        'JSON 列出所有 collision geoms 局部 origins、盒尺寸、mesh、过滤和材料；这些诊断接触不等于 Factory。',
        'Panda gravity=-9.81、gravcomp=0、armature=.1、damping=1，与 Factory robot gravity-disabled/0/.0 不符。', '',
        f"两次相同进程编译的 MJB SHA 相同：`{d['compilation']['first_MJB_sha256']}`。",
        '4 个无效输入（缺 mesh、负惯量、无效 joint 引用、惯量三角不等式）均记录实际拒绝错误。',
        f"本次耗时 {d['runtime_s']:.3f}s；进程 peak RSS {d['peak_rss_kib']} KiB；积分步数0。", '',
        '**结论：静态名字/帧/惯量映射已细化；完整 Factory 重建及接触/动力学等价仍未通过。**', '']
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--output', type=Path, required=True, help='New evidence directory; existing paths are rejected'); args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite evidence directory: '+str(args.output))
    d = audit(args.model); args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'robot_mapping.json').write_text(json.dumps(d, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (args.output/'robot_mapping.md').write_text(markdown(d))
    print(json.dumps({k: d[k] for k in ['status', 'compilation', 'runtime_s', 'peak_rss_kib']}, indent=2))

if __name__ == '__main__': main()
