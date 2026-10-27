"""Synthetic adversarial traces only: no MuJoCo integration or policy."""
import copy
import pytest
from scripts.analyze_g4s2a_gripper_v1_r1 import (same_bits, check_vector,
    review_schedule, review_integrity, summarize_preparation_r1)


def snap():
    x={'time':[0.], 'qpos':[.2,.00499125,.00499125,.3,0.,0.,0.,1.],
       'qvel':[0.,0.], 'act':[], 'history':[], 'qacc_warmstart':[0.,0.],
       'ctrl':[0.]*8, 'qfrc_applied':[0.,0.], 'xfrc_applied':[0.], 'eq_active':[],
       'mocap_pos':[], 'mocap_quat':[], 'userdata':[], 'plugin':[],
       'model_fingerprint':'ab'}
    keys=('time','qpos','qvel','act','history','qacc_warmstart','ctrl',
          'qfrc_applied','xfrc_applied','eq_active','mocap_pos','mocap_quat','userdata','plugin')
    x['integration_state']=[v for k in keys for v in x[k]]
    return x


def test_vectors_independently_rebuilt():
    x=snap();check_vector(x)
    x['integration_state'][0]=.2
    with pytest.raises(ValueError,match='STATE_VECTOR'):check_vector(x)


@pytest.mark.parametrize('field,index,value',[
    ('qpos',0,.1),('qpos',1,.003),('qpos',3,.2),('qpos',4,.4),
    ('qvel',0,.1),('time',0,.1),('qfrc_applied',0,1.),('qacc_warmstart',0,2.),
    ('ctrl',0,.02)])
def test_independent_bitwise_mismatch(field,index,value):
    a=snap();b=copy.deepcopy(a);b[field][index]=value
    with pytest.raises(ValueError,match=field):same_bits(a,b,'GUARD')


def test_signed_zero_is_a_bitwise_change():
    a=snap();b=copy.deepcopy(a);b['qvel'][0]=-0.
    with pytest.raises(ValueError,match='qvel'):same_bits(a,b,'GUARD')


def contact(side,force):
    return {'geom1':'peg_collision','geom2':side+'_finger_geom_3','normal_force_n':force}


def row(n, before, after, force, target, acq=0,hold=0):
    prev=.00499125 if n==1 else target
    return {'gripper_schedule':{'phase_before':before,'phase_after':after,
       'observed_left_force_n':force[0],'observed_right_force_n':force[1],
       'positive_contact_threshold_n':1e-6,'bilateral_observed':min(force)>1e-6,
       'target_m':target,'target_delta_m':target-prev,'target_rate_m_s':(target-prev)*120,
       'tendon_target_m':2*target,'bilateral_streak':acq,'hold_streak':hold,
       'command_number':n,'complete':after=='COMPLETE'},
       'state':{'contacts':[contact('left',force[0]),contact('right',force[1])]}}


def test_approach_and_one_sided_contact_freeze():
    x=row(1,'FREE_SPACE_APPROACH','FREE_SPACE_APPROACH',(0,0),.00498125)
    x['state']['contacts']=[contact('left',.2)]
    y=row(2,'FREE_SPACE_APPROACH','CONTACT_ACQUISITION',(.2,0),.00498125)
    y['state']['contacts']=[contact('left',.2)]
    result=review_schedule([x,y],[])
    assert result['first_observed_contact_command']==2
    assert result['final_phase']=='CONTACT_ACQUISITION'
    bad=copy.deepcopy(y);bad['gripper_schedule']['target_m']-=10e-6
    with pytest.raises(ValueError,match='target_m'):review_schedule([x,bad],[])


def test_future_or_fabricated_contact_is_rejected():
    x=row(1,'FREE_SPACE_APPROACH','CONTACT_ACQUISITION',(.2,.2),.00499125,1)
    with pytest.raises(ValueError,match='phase_after'):review_schedule([x],[])


def test_no_false_short_prep_pass():
    assert summarize_preparation_r1([],{'final_phase':'COMPLETE'})['preparation_contact_quality_status']=='FAIL'
    assert summarize_preparation_r1([{}]*10,{'final_phase':'GRASP_HOLD'})['reset_admission_status']=='FAIL'


def test_acquisition_and_hold_need_twenty_observations():
    rows=[]
    for n in range(1,21):
        phase_before='FREE_SPACE_APPROACH' if n==1 else ('CONTACT_ACQUISITION' if n<=10 else 'GRASP_HOLD')
        phase_after='CONTACT_ACQUISITION' if n<10 else ('GRASP_HOLD' if n<20 else 'COMPLETE')
        rows.append(row(n,phase_before,phase_after,(.2,.2),.00499125,min(n,10),max(n-10,0)))
    result=review_schedule(rows,[contact('left',.2),contact('right',.2)])
    assert result['hold_entry_command']==10 and result['complete_command']==20
    assert result['final_phase']=='COMPLETE'
    bad=copy.deepcopy(rows);bad[9]['gripper_schedule']['bilateral_streak']=9
    with pytest.raises(ValueError,match='bilateral_streak'):review_schedule(bad,[contact('left',.2),contact('right',.2)])


def test_guard_independently_checks_previous_result_and_control_only():
    initial=snap();before=copy.deepcopy(initial);nextstate=copy.deepcopy(initial)
    expected=copy.deepcopy(initial);expected['ctrl']=[0.]*7+[.0099625]
    idx=len(expected['time'])+len(expected['qpos'])+len(expected['qvel'])+len(expected['qacc_warmstart'])
    expected['integration_state'][idx:idx+8]=expected['ctrl']
    data={'step':1,'expected_before_control':before,'actual_before_control':copy.deepcopy(before),
          'expected_before_step':expected,'actual_before_step':copy.deepcopy(expected),
          'actual_after_forward':nextstate,'viewer_sync_count':2,
          'pre_state':{'qpos':initial['qpos'],'qvel':initial['qvel'],'time_s':initial['time'][0]},
          'integration_stage':{'ctrl':expected['ctrl']},'finger_target_m':.00498125,'selected_arm_torque_nm':[0.]*7,
          'state':{'qpos':nextstate['qpos'],'qvel':nextstate['qvel'],'time_s':nextstate['time'][0]}}
    assert review_integrity({'initial_integrity':initial,'model_fingerprint':'ab','finger_actuator_id':7},[data])['status']=='PASS'
    data['actual_before_control']['qpos'][0]=.21
    with pytest.raises(ValueError,match='PRE_CONTROL'):review_integrity({'initial_integrity':initial,'model_fingerprint':'ab','finger_actuator_id':7},[data])


def metric():
    return {'relative_position_m':[0.,0.,.03], 'relative_rotation':[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        'arm_limit_violation_rad':0.,'finger_limit_violation_m':0.,'arm_speed_rad_s':0.,
        'bilateral':True,'pad_penetration_m':20e-6,'assembly_penetration_m':0.,
        'interference':[],'warning_delta':[0]}


def test_full_preparation_uses_all_transient_contacts_not_only_final_window():
    m=[metric() for _ in range(20)]
    assert summarize_preparation_r1(m,{'final_phase':'COMPLETE'})['preparation_contact_quality_status']=='PASS'
    m[0]['pad_penetration_m']=50.001e-6
    assert summarize_preparation_r1(m,{'final_phase':'COMPLETE'})['preparation_contact_quality_status']=='FAIL'


def test_final_contact_and_joint_limits_cannot_be_waived():
    m=[metric() for _ in range(20)];m[-1]['bilateral']=False
    assert summarize_preparation_r1(m,{'final_phase':'COMPLETE'})['reset_admission_status']=='FAIL'
    m=[metric() for _ in range(20)];m[2]['finger_limit_violation_m']=1.0001e-6
    assert summarize_preparation_r1(m,{'final_phase':'COMPLETE'})['reset_admission_status']=='FAIL'
