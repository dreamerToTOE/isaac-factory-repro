"""Negative controls with hand-calculated values; no MuJoCo integration."""
import copy
import numpy as np
import pytest
from scripts.analyze_g4s2a_gripper_v1 import (verify_transmission, summarize_preparation,
    raw_contact_metrics, step_stop_reasons, diagnose)


def input_values():
    # qleft=.010, qright=.012, vleft=.1, vright=-.05, target=.02.
    # L=.022, Ldot=.05; 3750*(-.002)-86.5*.05 = -11.825 N.
    return dict(qpos=[.010,.012],qvel=[.1,-.05],ctrl=.02,
                actuator_force=-11.825,generalized_force=[-11.825,-11.825],
                finger_qpos_ids=[0,1],finger_dof_ids=[0,1],moment=[1.,1.],
                tendon_length=.022,tendon_velocity=.05)


def metric():
    return {'step': 1,'relative_position_m':[0.,0.,0.],
            'relative_rotation':np.eye(3).tolist(), 'arm_limit_violation_rad':0.,
            'finger_limit_violation_m':0., 'warning_delta':[0], 'bilateral':True,
            'interference':[], 'pad_penetration_m':2e-5,
            'assembly_penetration_m':0., 'all_penetration_m':2e-5,
            'translation_slip_m':0., 'rotation_slip_rad':0., 'arm_speed_rad_s':0.}


def test_asymmetric_joint_coordinates_and_virtual_work():
    out=verify_transmission(**input_values())
    assert out['unclipped_output_n']==pytest.approx(-11.825)
    assert out['finger_generalized_n']==pytest.approx([-11.825,-11.825])
    assert not out['clipped']


def test_saturation_and_effective_individual_force():
    a=input_values();a.update(ctrl=.08,actuator_force=40.,generalized_force=[40.,40.])
    result=verify_transmission(**a)
    assert result['unclipped_output_n']==pytest.approx(213.175)
    assert result['clipped'] and result['finger_generalized_n']==[40.,40.]


@pytest.mark.parametrize('key,value', [('moment',[1.,-1.]),('tendon_length',.011),
                                      ('tendon_velocity',.025),('actuator_force',-12.),
                                      ('generalized_force',[-11.825,-22.]),('ctrl',.1)])
def test_corrupted_transmission_is_rejected(key,value):
    a=input_values();a[key]=value
    with pytest.raises(ValueError): verify_transmission(**a)


def test_contact_normals_count_pad_geometries_only():
    state={'contacts':[{'geom1':'peg_collision','geom2':'left_finger_geom_3','normal_force_n':3.,'distance_m':-.00003},
                       {'geom1':'right_finger_geom_7','geom2':'peg_collision','normal_force_n':5.,'distance_m':-.00002},
                       {'geom1':'hand_geom_0','geom2':'peg_collision','normal_force_n':200.,'distance_m':0.}]}
    force,depth,assembly,pad,interference=raw_contact_metrics(state)
    assert force == {'left':3.,'right':5.} and depth==3e-5 and pad==3e-5
    assert assembly == 0 and interference == ['hand_geom_0:peg_collision']


def test_preparation_must_record_30_steps():
    assert summarize_preparation([metric()]*29)['preparation_contact_quality_status']=='FAIL'


def test_preparation_transient_cannot_be_hidden_by_final_window():
    metrics=[copy.deepcopy(metric()) for _ in range(30)]
    metrics[1]['pad_penetration_m']=51e-6
    s=summarize_preparation(metrics)
    assert s['reset_admission_status']=='PASS'
    assert s['preparation_contact_quality_status']=='FAIL'
    assert s['pad_penetration_final10_max_m']==2e-5


def test_final_window_requires_positive_bilateral_and_no_warnings():
    metrics=[copy.deepcopy(metric()) for _ in range(30)]
    metrics[-5]['bilateral']=False
    assert summarize_preparation(metrics)['reset_admission_status']=='FAIL'
    metrics[-5]['bilateral']=True;metrics[0]['warning_delta']=[1]
    assert summarize_preparation(metrics)['preparation_contact_quality_status']=='FAIL'


def test_early_stop_requires_consecutive_loss_and_preserves_severe_depth():
    m=metric();m['all_penetration_m']=.0011
    assert step_stop_reasons(m,3)==['SEVERE_PENETRATION','SUSTAINED_CONTACT_LOSS']
    assert step_stop_reasons(metric(),2)==[]


def test_short_diagnostic_never_passes():
    metrics=[copy.deepcopy(metric()) for _ in range(234)]
    assert diagnose(metrics)['status']=='FAIL'
    metrics.append(copy.deepcopy(metric()))
    assert diagnose(metrics)['status']=='PASS'
    metrics[-1]['finger_limit_violation_m']=.000002
    assert diagnose(metrics)['status']=='FAIL'


def test_full_synthetic_row_with_zero_constraints_and_saturated_motor():
    from scripts.analyze_g4s2a_gripper_v1 import review_rows
    q=[0.]*7+[.01,.01]; v=[0.]*9
    state={'qpos':q,'qvel':v,'time_s':0.,'tcp_position_world':[0.,0.,0.],
           'tcp_quaternion_wxyz':[1.,0.,0.,0.],'peg_root_position':[0.,0.,.1],
           'peg_quaternion_wxyz':[1.,0.,0.,0.],'hole_root_position':[0.,0.,0.],
           'contacts':[]}
    before=copy.deepcopy(state)
    before['gripper']={'tendon_length_m':.02,'tendon_velocity_m_s':0.}
    after=copy.deepcopy(state);after['time_s']=1/120
    after['s2a']={'relative_peg_hand_m':[0.,0.,.1],
        'relative_peg_hand_rotation':np.eye(3).tolist(),
        'all_contact_penetration_m':0.,'peg_hole_penetration_m':0.,
        'arm_joint_limit_violation_rad':0.,'finger_joint_limit_violation_m':0.}
    joint_force=[0.]*7+[-40.,-40.]
    base_solver={'nefc':0,'ncon':0,'efc_type':[],'efc_id':[],'efc_force':[],
        'efc_vel':[],'efc_aref':[],'efc_pos':[],'efc_state':[],'efc_J_dense':[],
        'qfrc_constraint':[0.]*9,'qfrc_actuator':joint_force,'contacts':[],
        'qpos':q,'qvel':v,'cone':1}
    stage={**copy.deepcopy(base_solver),'stage':'immediately_after_mj_step_before_wrapper_mj_forward',
           'simulation_start_time_s':0.,'simulation_end_time_s':1/120,
           'ctrl':[0.]*8,'actuator_force':[0.]*7+[-40.],
           'gripper':{'control_m':0.,'actuator_force_n':-40.,
                      'moment_arm_by_dof':[0.]*7+[1.,1.],
                      'finger_generalized_force_n':[-40.,-40.]}}
    post={**copy.deepcopy(base_solver),'stage':'post_step_mj_forward'}
    row={'event':'physics_result','phase':'PREPARATION','step':1,'pre_state':before,
         'state':after,'integration_stage':stage,'post_forward_solver':post,
         'selected_arm_torque_nm':[0.]*7,'pad_force_n':{'left':0.,'right':0.},
         'pad_penetration_m':0.,'warning_delta':[0], 'stop_reasons':[]}
    identity={'finger_qpos_ids':[7,8], 'finger_dof_ids':[7,8], 'arm_qpos_ids':list(range(7)),
              'arm_dof_ids':list(range(7)), 'finger_actuator_id':7,'kp':3750.,'kv':86.5,
              'force_limit_n':40., 'timestep':1/120, 'finger_joint_range':[[0.,.04]]*2,
              'arm_joint_range':[[-1.,1.]]*7,'initial':state}
    out=review_rows(identity,[row],True)
    assert out[0]['actuation']['clipped_output_n']==-40.
    assert out[0]['finger_asymmetry_m']==0.
    bad=copy.deepcopy(row);bad['integration_stage']['gripper']['moment_arm_by_dof'][-1]=-1.
    with pytest.raises(ValueError,match='MOMENT_ARMS'):
        review_rows(identity,[bad],True)


def test_pre_step_reset_mismatch_is_not_valid_registered_input():
    from scripts.analyze_g4s2a_gripper_v1 import initial_differences
    expected={'qpos':[.1,.2], 'qvel':[0.,0.], 'tcp_position_world':[.1,.2,.3],
              'tcp_quaternion_wxyz':[1.,0.,0.,0.], 'peg_root_position':[.2,.3,.4],
              'peg_quaternion_wxyz':[1.,0.,0.,0.]}
    actual=copy.deepcopy(expected)
    assert initial_differences(actual,expected)=={}
    actual['qpos'][1]=0.; actual['peg_root_position'][0]=.6
    difference=initial_differences(actual,expected)
    assert difference['qpos']['max_abs_difference']==pytest.approx(.2)
    assert difference['qpos']['differing_indices']==[1]
    assert difference['peg_root_position']['actual_pre_step']==[.6,.3,.4]
