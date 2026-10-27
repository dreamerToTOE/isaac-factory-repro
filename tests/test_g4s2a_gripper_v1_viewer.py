"""Saved-state viewer checks: data navigation and zero integration."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
pytest.importorskip('mujoco')
from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
from scripts.g4s2a_gripper_v1_viewer import apply_state, first_failure, read_run


def row(step, penetration=0., force=1.):
    return {'event':'physics_result', 'step':step, 'stop_reasons':[],
            'pad_penetration_m':penetration,'pad_force_n':{'left':force,'right':force},
            'state':{'qvel':[0.]*15,'s2a':{}}}


def test_first_failure_distinguishes_normal_closing_from_diagnostic_loss():
    rows=[row(1,force=0.),row(2,force=0.)]
    assert first_failure(rows,preparation=True) is None
    assert first_failure(rows,preparation=False)==1
    rows[1]['pad_penetration_m']=51e-6
    assert first_failure(rows,preparation=True)==2
    assert first_failure([row(1)],preparation=True,final_gate={'reset_admission':'FAIL','physical_quality':'FAIL'})==1


def test_read_trace_rejects_missing_or_out_of_order_frame(tmp_path):
    path=tmp_path/'runs'/'PREP_I2'/'output';path.mkdir(parents=True)
    (path/'identity.json').write_text(json.dumps({'run':'PREP_I2'}))
    (path/'physics.jsonl').write_text(json.dumps(row(2))+'\n')
    with pytest.raises(ValueError,match='INCONSISTENT'):read_run(tmp_path,'PREP_I2')
    (path/'physics.jsonl').write_text(json.dumps(row(1))+'\n')
    assert read_run(tmp_path,'PREP_I2')[3][0]['step']==1
    with pytest.raises(ValueError,match='UNREGISTERED'):read_run(tmp_path,'OTHER')


def test_apply_state_replays_original_preclosure_without_mj_step():
    env=NativeGripperEnv()
    path=Path('evidence/g4_s/s2a_r2/matrix_v2/initials/I2/admission.json')
    state=json.loads(path.read_text())['initial']
    with patch.object(env.mj,'mj_step',side_effect=AssertionError('physics forbidden')):
        apply_state(env,state)
    assert np.array_equal(env.data.qpos,state['qpos'])
    assert env.data.time==state['time_s']
    assert env.data.ctrl[env.finger_actuator_ids[0]]==0.
    state['qpos'][env.peg_qpos_adr+3:env.peg_qpos_adr+7]=[0.,0.,0.,0.]
    with pytest.raises(ValueError,match='INVALID_SAVED_STATE'):apply_state(env,state)
