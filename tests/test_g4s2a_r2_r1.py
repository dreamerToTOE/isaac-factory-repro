"""CPU-only configuration safety and independent inertia contract tests.

Real 3.5.0 compiled models are checked separately by the guarded audit.
No policy, checkpoint load, viewer, or physics integration is performed here.
"""
import copy
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts import g4s2a_r2_model as r2
from scripts import g4s2a_r2_r1_model as model


def source_fixture():
    root = ET.Element('mujoco'); world = ET.SubElement(root, 'worldbody')
    for index, name in enumerate(r2.ROBOT_BODIES):
        body = ET.SubElement(world, 'body', name=name, pos='1 2 3')
        tensor = {'fullinertia': '.2 .3 .4 .001 .002 .003'} if index % 2 else {'diaginertia': '.2 .3 .4'}
        ET.SubElement(body, 'inertial', mass=str(index+1), pos='.1 .2 .3', **tensor)
        ET.SubElement(body, 'geom', type='sphere', size='.03')
    peg = ET.SubElement(world, 'body', name='peg')
    ET.SubElement(peg, 'inertial', mass='.019', pos='0 0 0', diaginertia='.001 .002 .003')
    return ET.tostring(root, encoding='unicode')


@pytest.mark.parametrize('selection, changed', [
    ('P_FULL', set([f'link{i}' for i in range(8)] + ['hand', 'left_finger', 'right_finger'])),
    ('P_ARM_HAND', set([f'link{i}' for i in range(8)] + ['hand']))])
def test_isolation_changes_exactly_selected_body_rotational_tensors(monkeypatch, selection, changed):
    text = source_fixture(); monkeypatch.setattr(model, 'XML_SHA', hashlib.sha256(text.encode()).hexdigest())
    before = ET.fromstring(text); expected = copy.deepcopy(before); actual = ET.fromstring(model.adapted_xml(text, selection))
    for name in changed:
        old = before.find(f'.//body[@name="{name}"]/inertial')
        target = expected.find(f'.//body[@name="{name}"]/inertial')
        obtained = actual.find(f'.//body[@name="{name}"]/inertial')
        key = 'fullinertia' if 'fullinertia' in old.attrib else 'diaginertia'
        values = np.asarray([float(v) for v in old.get(key).split()]); values[:3] += .01
        np.testing.assert_allclose([float(v) for v in obtained.get(key).split()], values, rtol=0, atol=1e-16)
        target.set(key, obtained.get(key))
    # Ensures unchanged fingers for ARM_HAND, plus mass/CoM/geometry/peg and off-diagonal inertia.
    assert ET.tostring(expected) == ET.tostring(actual)


@pytest.mark.parametrize('selection', ['', None, 'P1', 'p_full', True])
def test_invalid_selection_rejected_before_compilation(monkeypatch, selection):
    monkeypatch.setattr(model, 'AdaptedPegInsertEnv', lambda: pytest.fail('compiled invalid selection'))
    with pytest.raises(ValueError, match='EXPLICIT_INERTIA_SELECTION_REQUIRED'):
        model.make_model(selection)


@pytest.mark.parametrize('physical,controller', [('S1B','C1'), ('P_FULL','C2'), ('P_ARM_HAND',None)])
def test_invalid_controller_factor_rejected_before_compilation(monkeypatch, physical, controller):
    monkeypatch.setattr(model, 'make_model', lambda *_: pytest.fail('compiled invalid factor'))
    with pytest.raises(ValueError, match='UNKNOWN_INERTIA_FACTOR'):
        model.ModelPair(physical, controller)


def test_malformed_and_changed_sources_are_rejected_before_use(monkeypatch):
    with pytest.raises(ValueError, match='FROZEN_XML_CHANGED'):
        model.adapted_xml('<untrusted/>', 'P_FULL')
    text = source_fixture().replace('name="hand"', 'name="missing_hand"')
    monkeypatch.setattr(model, 'XML_SHA', hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError, match='MISSING_MAPPED_INERTIAL:hand'):
        model.adapted_xml(text, 'P_ARM_HAND')
    text = source_fixture().replace('diaginertia=".2 .3 .4"', 'diaginertia=".2 .3"', 1)
    monkeypatch.setattr(model, 'XML_SHA', hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError, match='INVALID_SOURCE_INERTIA'):
        model.adapted_xml(text, 'P_FULL')


def test_exact_eight_primary_and_one_separate_passive_control():
    expected = {
        ('I3_FULL_C0','I3','P_FULL','C0',False), ('I3_FULL_C1','I3','P_FULL','C1',False),
        ('I3_ARM_HAND_C0','I3','P_ARM_HAND','C0',False), ('I3_ARM_HAND_C1','I3','P_ARM_HAND','C1',False),
        ('I2_ARM_HAND_C0','I2','P_ARM_HAND','C0',False), ('I2_ARM_HAND_C1','I2','P_ARM_HAND','C1',False),
        ('I1_ARM_HAND_C0','I1','P_ARM_HAND','C0',False), ('I1_ARM_HAND_C1','I1','P_ARM_HAND','C1',False),
        ('I3_FULL_PASSIVE_ARM','I3','P_FULL','C0',True)}
    assert set(model.RUNS) == expected and len(model.RUNS) == 9
    assert sum(row[-1] for row in model.RUNS) == 1
    assert model.MAX_STEPS == 235 and len(model.RUNS)*model.MAX_STEPS == 2115


def test_r2_thresholds_retained_with_no_relaxation():
    assert model.THRESHOLDS is r2.THRESHOLDS
    assert model.THRESHOLDS == {
        'positive_pad_force_n':1e-6, 'accept_slip_m':.001, 'accept_rotation_rad':.05,
        'accept_arm_velocity_rad_s':3., 'accept_arm_limit_rad':1e-6, 'accept_finger_limit_m':1e-6,
        'accept_pad_penetration_m':50e-6, 'accept_assembly_penetration_m':10e-6,
        'stop_slip_m':.005, 'stop_rotation_rad':.2, 'stop_arm_velocity_rad_s':15.,
        'stop_arm_limit_rad':1e-5, 'stop_finger_limit_m':1e-4,
        'stop_penetration_m':.001, 'stop_consecutive_contact_loss':3}


@pytest.mark.parametrize('penetration, expected', [(.001,False), (np.nextafter(.001,np.inf), True), (np.nan, True)])
def test_severe_penetration_abort_boundary(penetration, expected):
    state = {'qpos':[0.]*16,'qvel':[0.]*15,'s2a':{
        'peg_hand_slip_m':0.,'peg_hand_slip_rad':0.,'arm_joint_limit_violation_rad':0.,
        'finger_joint_limit_violation_m':0.,'all_contact_penetration_m':penetration}}
    reasons = model.stop_reasons(state, {'left':1.,'right':1.}, 0, [])
    assert ('SEVERE_PENETRATION' in reasons) is expected


def test_fixed_reference_cannot_extend_horizon_and_keeps_first_16_actions():
    class Contract:
        def targets(self): return {'target':'archived'}
    pos=np.array([1.,2.,3.]); quat=np.array([1.,0.,0.,0.])
    assert model.fixed_reference(16,pos,quat,Contract()) == {'target':'archived'}
    np.testing.assert_array_equal(model.fixed_reference(17,pos,quat,Contract())['position_target'],pos)
    with pytest.raises(ValueError, match='OUTSIDE_REGISTERED_HORIZON'):
        model.fixed_reference(236,pos,quat,Contract())


def test_registry_rejects_tampered_matrix_schedule_thresholds_and_source_coverage(tmp_path, monkeypatch):
    from scripts import g4s2a_r2_r1_diagnostics as run
    initial = {'R2_registry': {'path':'historical.json','sha256':'a'}}
    monkeypatch.setattr(run, 'inputs', lambda: initial)
    monkeypatch.setattr(run, 'sha', lambda _: 'abc')
    monkeypatch.setattr(run, 'SOURCES', ['source_one', 'source_two'])
    schedule=run.read(run.R2 / 'preregistered_matrix.json')['input_schedule']
    forensic=tmp_path/'forensics.json'; forensic.write_text('{"passed": true}')
    registry={'task':'G4-S2A-R2-R1', 'matrix_order':[r[0] for r in model.RUNS],
        'runs':[{'id':r,'initial_id':i,'physical':p,'controller':c,'passive_arm':v,
                 'pool':'negative_control' if v else 'primary'} for r,i,p,c,v in model.RUNS],
        'steps_per_trajectory':235, 'maximum_new_integration_steps':2115, 'maximum_wall_s':900,
        'no_retry':True, 'policy_calls':0, 'cpu_receipt_required':True, 'thresholds':model.THRESHOLDS,
        'input_schedule':schedule, 'input_schedule_sha256':model.digest(schedule),
        'historical_inputs':initial,'forensics_receipt_path':str(forensic),'forensics_receipt_sha256':'abc',
        'source_sha256':{'source_one':'abc','source_two':'abc'}}
    import json
    path=tmp_path/'registry.json'; path.write_text(json.dumps(registry))
    assert run.verify_registry(path) == registry
    variants=[]
    for key,value in [('steps_per_trajectory',236),('maximum_new_integration_steps',2116),
                      ('maximum_wall_s',901),('no_retry',False),('policy_calls',1),('cpu_receipt_required',False),
                      ('source_sha256', {'source_one':'abc'}),('runs',registry['runs'][:-1]),
                      ('input_schedule',{'actions':[[999.]]}),('thresholds',{})]:
        bad=copy.deepcopy(registry); bad[key]=value; variants.append((key,bad))
    for key,bad in variants:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='REGISTRY|SOURCE'):
            run.verify_registry(path)


def fake_observer_env(monkeypatch, mode='once'):
    from types import SimpleNamespace
    from scripts import g4s2a_r2_r1_diagnostics as run
    data=SimpleNamespace(actuator_force=np.zeros(3), ctrl=np.zeros(3), qfrc_actuator=np.zeros(2), time=0.)
    calls=[]
    def native_step(m, d):
        calls.append('integrate'); d.actuator_force[:]=[2.,3.,4.]; d.qfrc_actuator[:]=[2.,3.]
        d.ctrl[:]=[1.,1.,1.]; d.time=.01
        if mode == 'native_exception': raise RuntimeError('native failure')
    mj=SimpleNamespace(mj_step=native_step)
    def fake_wrapper(torque, finger_target, record):
        calls.append(('command',torque,finger_target,record))
        if mode != 'no_step': mj.mj_step(None,data)
        if mode == 'twice': mj.mj_step(None,data)
        if mode == 'wrapper_exception': raise RuntimeError('wrapper failure')
        result={'integration_actuation': {'arm_actuator_force_nm':[2.,3.],
                 'finger_actuator_force_n':[4.], 'arm_qfrc_actuator_nm':[2.,3.]}}
        # Simulate the wrapper's subsequent forward computation, altering output.
        data.actuator_force[:]=99.; data.qfrc_actuator[:]=99.; data.ctrl[:]=99.
        return result
    env=SimpleNamespace(mj=mj, data=data, physics_step=fake_wrapper,
        arm_actuator_ids=np.array([0,1]), finger_actuator_ids=np.array([2]), arm_dof_ids=np.array([0,1]))
    monkeypatch.setattr(run, 'solver_and_contacts', lambda e:{'qfrc_actuator':e.data.qfrc_actuator.copy()})
    return run,env,native_step,calls


def test_observer_captures_single_integration_before_forward_without_changing_command(monkeypatch):
    run,env,original,calls=fake_observer_env(monkeypatch)
    torque=np.array([.2,-.5]); state,trace=run.observed_physics_step(env,torque)
    assert env.mj.mj_step is original and calls[0][0] == 'command' and calls[0][1] is torque
    assert calls[0][2:] == (0.,False) and calls.count('integrate') == 1
    np.testing.assert_array_equal(trace['actuator_force'],[2.,3.,4.])
    np.testing.assert_array_equal(trace['qfrc_actuator'],[2.,3.])
    np.testing.assert_array_equal(trace['ctrl'],[1.,1.,1.])
    assert trace['time_s'] == .01 and trace['stage'] == 'immediately_after_mj_step_before_wrapper_mj_forward'
    np.testing.assert_array_equal(env.data.actuator_force,[99.,99.,99.])
    assert state['integration_actuation']['finger_actuator_force_n'] == [4.]


@pytest.mark.parametrize('mode, message', [('no_step','EXACTLY_ONE_MJ_STEP_REQUIRED'),
    ('twice','MULTIPLE_MJ_STEP_CALLS_REJECTED'),('native_exception','native failure'),
    ('wrapper_exception','wrapper failure')])
def test_observer_restores_original_step_on_all_failures(monkeypatch, mode, message):
    run,env,original,calls=fake_observer_env(monkeypatch,mode)
    with pytest.raises(RuntimeError,match=message): run.observed_physics_step(env,np.zeros(2))
    assert env.mj.mj_step is original
    assert calls.count('integrate') == (0 if mode == 'no_step' else 1)


@pytest.mark.parametrize('field, value', [(None, 0.), ('qfrc_applied', 1e-12), ('xfrc_applied', -1e-12), ('xfrc_applied', np.nan)])
def test_gui_external_perturbations_are_rejected_without_integrating(field, value):
    from types import SimpleNamespace
    from scripts import g4s2a_r2_r1_diagnostics as run
    data=SimpleNamespace(qfrc_applied=np.zeros(15), xfrc_applied=np.zeros((12,6)))
    if field is not None: getattr(data,field).flat[0]=value
    env=SimpleNamespace(data=data)
    if field is None:
        run.reject_external_perturbation(env)
    else:
        with pytest.raises(RuntimeError, match='GUI_EXTERNAL_PERTURBATION_REJECTED'):
            run.reject_external_perturbation(env)
