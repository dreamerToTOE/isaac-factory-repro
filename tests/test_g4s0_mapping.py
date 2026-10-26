"""Independent synthetic geometry + pinned-source CPU probes; no physics runs."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from scripts import g4s0_mapping as m
from scripts.g4i0_interface import normalize_observation,RNNStateSpec,smooth_action,clip_policy_action

REPO=Path(__file__).resolve().parents[1]


def test_all_19_fields_and_quaternion_sign_are_preserved():
    obs=m.observation(np.array([3.,4.,5.]),np.array([-1.,0.,0.,0.]),np.array([1.,1.,1.]),
                      np.array([.1,.2,.3]),np.array([6.,7.,8.]),np.array([9.,10.,11.]),np.arange(12.,18.))
    expected=np.array([1.9,2.8,3.7,-1,0,0,0,6,7,8,9,10,11,12,13,14,15,16,17],dtype=np.float32)
    assert obs.dtype==np.float32 and obs.shape==(19,)
    np.testing.assert_array_equal(obs,expected)
    # A q/-q rewrite would affect the actor despite identical physical rotation.
    assert obs[3]==-1.


def test_world_relative_translation_is_not_rotated_into_hole_frame():
    root=np.array([1.,2.,3.]);q=np.array([math.sqrt(.5),0.,0.,math.sqrt(.5)])
    tip=m.world_point(root,q,np.array([.025,0.,0.]))
    np.testing.assert_allclose(tip,[1.,2.025,3.],atol=1e-15)
    obs=m.observation(np.array([1.1,2.025,3.]),q,tip,np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(6))
    np.testing.assert_allclose(obs[:3],[.1,0.,0.],atol=1e-7)


def test_quaternion_order_conversion_against_known_90deg_rotation():
    q=np.array([math.sqrt(.5),0.,0.,math.sqrt(.5)])
    np.testing.assert_array_equal(m.wxyz_to_xyzw(q),[0.,0.,q[3],q[0]])
    np.testing.assert_array_equal(m.xyzw_to_wxyz(m.wxyz_to_xyzw(q)),q)
    np.testing.assert_allclose(m.world_point(np.zeros(3),q,np.array([1.,0.,0.])),[0.,1.,0.],atol=1e-15)


def test_velocity_uses_one_physics_step_world_axes_and_reset_zeroes():
    theta=.01;q=np.array([math.cos(theta/2),0.,0.,math.sin(theta/2)])
    v,w=m.fd_velocity(np.array([1/120,0.,0.]),np.zeros(3),q,np.array([1.,0.,0.,0.]))
    np.testing.assert_allclose(v,[1,0,0]);np.testing.assert_allclose(w,[0,0,1.2],atol=1e-14)
    v,w=m.fd_velocity(np.ones(3),np.zeros(3),q,-q,reset=True)
    np.testing.assert_array_equal(v,np.zeros(3));np.testing.assert_array_equal(w,np.zeros(3))


def test_original_exact_half_turn_sign_zero_edge_is_documented_not_repaired():
    _,omega=m.fd_velocity(np.zeros(3),np.zeros(3),np.array([0.,1.,0.,0.]),np.array([1.,0.,0.,0.]))
    np.testing.assert_array_equal(omega,np.zeros(3))


def test_noncommuting_world_rotation_increment_is_left_multiplication():
    # 90deg world-X applied to an existing90deg world-Z: ordered rotations differ.
    s=math.sqrt(.5);current=np.array([s,0.,0.,s]);delta=np.array([s,s,0.,0.])
    left=m.quat_mul(delta,current);right=m.quat_mul(current,delta)
    np.testing.assert_allclose(left,[.5,.5,-.5,.5],atol=1e-15)
    np.testing.assert_allclose(right,[.5,.5,.5,.5],atol=1e-15)
    assert not np.allclose(left,right)


def test_action_clipping_ema_scaling_and_no_reclip_have_independent_numbers():
    out=m.translation_action(np.array([2.,-2.,0.,1.,0.,-1.]),np.array([2.,0.,0.,0.,0.,0.]),
                             np.array([.049,0.,0.]),np.zeros(3))
    np.testing.assert_allclose(out['ema'],[1.8,-.2,0,.2,0,-.2])
    np.testing.assert_allclose(out['position_target'],[.05,-.004,0])
    np.testing.assert_allclose(out['world_rotvec_increment'],[.0194,0,-.0194])
    np.testing.assert_allclose(out['ema'],smooth_action(clip_policy_action(np.array([2.,-2.,0.,1.,0.,-1.],np.float32)),np.array([2.,0.,0.,0.,0.,0.],np.float32)),atol=1e-7)
    np.testing.assert_allclose(m.reset_translation_cache(np.array([.1,-.05,0.]),np.zeros(3)),[2.,-1.,0.])


def test_target_is_rebuilt_from_new_state_not_integrated_from_old_target():
    action=np.array([1.,0.,0.,0.,0.,0.]);cache=action.copy()
    first=m.translation_action(action,cache,np.zeros(3),np.zeros(3))
    second=m.translation_action(action,cache,np.array([.001,0.,0.]),np.zeros(3))
    assert first['position_target'][0]==.02
    assert second['position_target'][0]==.021


def test_joint_mapping_is_explicit_and_independent_of_target_storage_order():
    target=('joint7','joint3','joint1','joint6','joint5','joint2','joint4','finger1','finger2')
    mapping={f'panda_joint{i}':f'joint{i}' for i in range(1,8)}
    assert m.joint_indices(target,mapping)==(2,5,1,6,4,3,0)
    for bad in ({}, {**mapping,'panda_joint7':'joint1'}, {**mapping,'panda_joint7':'absent'}):
        with pytest.raises(ValueError):m.joint_indices(target,bad)


@pytest.mark.parametrize('held,expected',[
    ([0.,0.,.000999],True),([0.,0.,.001],False),([.0025,0.,0.],False),
    ([.002499,0.,0.],True),([0.,0.,-1.],True),([.0025,.0025,-1.],False)])
def test_success_is_exact_strict_world_root_predicate_not_contact_or_absolute_z(held,expected):
    assert m.official_success(np.array(held),np.zeros(3)) is expected


def test_success_mapping_against_unmodified_original_source_at_fp32_boundaries():
    source=json.loads((REPO/'evidence/g4_i/i0/checkpoint_audit.json').read_text())['source_files']['factory_env.py']
    data=Path(source['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==source['sha256']
    cls=next(n for n in ast.parse(data).body if isinstance(n,ast.ClassDef) and n.name=='FactoryEnv')
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_curr_successes')
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module([function],type_ignores=[])),source['path'],'exec'),namespace)
    xy=np.float32(.0025);z=np.float32(.001)
    rows=np.array([[0,0,0],[xy,0,0],[np.nextafter(xy,np.float32(0)),0,0],
                   [0,0,z],[0,0,np.nextafter(z,np.float32(0))],[0,0,-1]],np.float32)
    proxy=SimpleNamespace(num_envs=len(rows),device='cpu',held_base_pos=torch.from_numpy(rows),
                          target_held_base_pos=torch.zeros_like(torch.from_numpy(rows)),
                          cfg_task=SimpleNamespace(name='peg_insert',fixed_asset_cfg=SimpleNamespace(height=.025)))
    official=namespace['_get_curr_successes'](proxy,.04,False).tolist()
    assert official==[True,False,True,False,True,True]
    assert official==[m.official_success(row,np.zeros(3)) for row in rows]


def test_fixed_rms_and_recurrent_reset_preserved_without_loading_actor():
    c=json.loads((REPO/'evidence/g4_i/i0/checkpoint_audit.json').read_text())['checkpoints']['g2_epoch200_primary']
    mean=np.array(c['rms']['running_mean'],np.float64);var=np.array(c['rms']['running_var'],np.float64)
    obs=np.linspace(-10.,10.,19,dtype=np.float32)
    expected=np.clip((obs-mean.astype(np.float32))/np.sqrt(var.astype(np.float32)+np.float32(1e-5)),-5.,5.)
    np.testing.assert_array_equal(normalize_observation(obs,mean,var),expected)
    assert c['rms']['count']==1638401.
    state=RNNStateSpec().initialize(2);state[0].fill_(1);state[1].fill_(2)
    reset=RNNStateSpec().reset(state,torch.tensor([True,False]))
    assert tuple(reset[0].shape)==(2,2,1024)
    assert reset[0][:,0].count_nonzero()==0 and reset[0][:,1].eq(1).all()
    assert reset[1][:,0].count_nonzero()==0 and reset[1][:,1].eq(2).all()


def test_original_torque_function_dimensions_and_nullspace_on_cpu():
    source=json.loads((REPO/'evidence/g4_i/i0/checkpoint_audit.json').read_text())['source_files']['factory_control.py']
    data=Path(source['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==source['sha256']
    tree=ast.parse(data)
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('compute_dof_torque','_apply_task_space_gains')]
    # Identity-orientation fixture: the independently known rotation error is zero.
    def pose_error(**kw):
        return kw['ctrl_target_fingertip_midpoint_pos']-kw['fingertip_midpoint_pos'],torch.zeros(1,3)
    namespace={'torch':torch,'math':math,'get_pose_error':pose_error}
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),source['path'],'exec'),namespace)
    cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=1),ctrl=SimpleNamespace(default_dof_pos_tensor=[0.]*6+[1.],kp_null=10.,kd_null=6.3246))
    J=torch.cat([torch.eye(6),torch.zeros(6,1)],dim=1)[None]
    kp=torch.tensor([[100.,100.,100.,30.,30.,30.]])
    args=dict(cfg=cfg,dof_pos=torch.zeros(1,9),dof_vel=torch.zeros(1,9),
              fingertip_midpoint_pos=torch.zeros(1,3),fingertip_midpoint_quat=torch.tensor([[1.,0.,0.,0.]]),
              fingertip_midpoint_linvel=torch.zeros(1,3),fingertip_midpoint_angvel=torch.zeros(1,3),
              jacobian=J,arm_mass_matrix=(torch.eye(7)*2)[None],ctrl_target_fingertip_midpoint_pos=torch.tensor([[.01,0.,0.]]),
              ctrl_target_fingertip_midpoint_quat=torch.tensor([[1.,0.,0.,0.]]),task_prop_gains=kp,task_deriv_gains=2*kp.sqrt(),device='cpu')
    torque,wrench=namespace['compute_dof_torque'](**args)
    assert torque.shape==(1,9) and wrench.shape==(1,6)
    torch.testing.assert_close(torque,torch.tensor([[1.,0.,0.,0.,0.,0.,20.,0.,0.]]))
    torch.testing.assert_close(wrench,torch.tensor([[1.,0.,0.,0.,0.,0.]]))
    with pytest.raises(torch.linalg.LinAlgError):namespace['compute_dof_torque'](**{**args,'jacobian':torch.zeros_like(J)})


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_inputs_rejected_before_interface_use(bad):
    with pytest.raises(ValueError):m.official_success(np.array([bad,0.,0.]),np.zeros(3))
    with pytest.raises(ValueError):m.translation_action(np.array([bad,0.,0.,0.,0.,0.]),np.zeros(6),np.zeros(3),np.zeros(3))


@pytest.mark.parametrize('dt,decimation',[(1/60,8),(1/120,4),(1/120,True)])
def test_invalid_timing_configuration_rejected(dt,decimation):
    with pytest.raises(ValueError):m.timing(dt,decimation)


def test_timing_simulation_clock_and_invalid_quaternions():
    assert m.timing()['policy_hz']==15
    assert m.timing()['timeout_policy_steps']==149
    assert m.timing()['timeout_simulation_seconds']==pytest.approx(9.933333333333334)
    for q in (np.zeros(4),np.ones(4),np.ones(3),np.array([1,0,0,0])):
        with pytest.raises(ValueError):m.quaternion(q)
