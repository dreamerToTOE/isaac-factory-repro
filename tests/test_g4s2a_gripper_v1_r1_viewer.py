"""Playback is a saved-record state machine; these tests never integrate physics."""
import json
import pytest
from scripts.g4s2a_gripper_v1_r1_viewer import Playback, read_run, selection, sha

def test_pause_play_step_fault_speed_and_cameras():
    c=Playback(10)
    assert c.key('RIGHT') is None and c.index==1 and not c.playing
    c.key('LEFT');assert c.index==0
    c.key('END');assert c.index==10
    c.key('HOME');assert c.index==0
    c.key('L',7);assert c.index==7 and not c.playing
    assert c.key('V')=='overview' and c.key('Z')=='closeup'
    c.key('S');assert c.speed==.25
    c.key('SPACE');assert c.playing
    assert c.tick(10)==7
    assert c.tick(10.2)==10 and not c.playing
    c.key('SPACE');c.key('SPACE');assert not c.playing

def test_zero_step_trace_supported():
    c=Playback(0);c.key('SPACE');assert c.tick(1)==0;assert c.tick(2)==0 and not c.playing
    c.key('RIGHT');assert c.index==0
    assert selection('0,failure,last',0,0)==[0]

def test_selection_deduplicates_and_rejects_invalid():
    assert selection('0,3,last,failure',3,2)==[0,2,3]
    assert selection('failure,last',2,None)==[2]
    with pytest.raises(ValueError):selection('4',3,None)
    with pytest.raises(ValueError):Playback(-1)
    with pytest.raises(ValueError):Playback(1).tick(float('nan'))

def row(step,pen=0):
    return {'event':'physics_result','step':step,'pre_state':{'tag':'initial'},
      'state':{'tag':step,'s2a':{},'qvel':[0]*9},'pad_penetration_m':pen,'pad_force_n':{'left':0,'right':0},'stop_reasons':[]}

def create(tmp_path,run,rows,gate=None):
    directory=tmp_path/'runs'/run/'output';directory.mkdir(parents=True)
    (directory/'identity.json').write_text(json.dumps({'run':run,'initial':{'tag':'identity'}}))
    (directory/'physics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (directory/'result.json').write_text(json.dumps({'completed_steps':len(rows),'trace_sha256':sha(directory/'physics.jsonl')}))
    if gate is not None:(directory/'preparation_gate.json').write_text(json.dumps(gate))
    return directory

def test_reads_r1_run_and_finds_archived_breach(tmp_path):
    name='PREP_I2_RATE_LIMITED';create(tmp_path,name,[row(1),row(2,.000051)])
    run=read_run(tmp_path,name)
    assert run['first_failure']==2 and run['states'][0]=={'tag':'initial'} and len(run['states'])==3

def test_gate_failure_uses_final_saved_frame(tmp_path):
    name='PREP_I3_RATE_LIMITED';create(tmp_path,name,[row(1)],{'reset_admission':'FAIL','physical_quality':'FAIL'})
    assert read_run(tmp_path,name)['first_failure']==1

def test_zero_step_run_uses_identity_state(tmp_path):
    name='PREP_I2_RATE_LIMITED';d=create(tmp_path,name,[])
    (d/'result.json').write_text(json.dumps({'completed_steps':0,'trace_sha256':sha(d/'physics.jsonl'),'runtime_failure':{'status':'INVALID_INITIAL_STATE'}}))
    r=read_run(tmp_path,name);assert r['first_failure']==0 and r['states']==[{'tag':'identity'}]

def test_rejects_changed_trace_and_bad_identity(tmp_path):
    name='PREP_I2_RATE_LIMITED';d=create(tmp_path,name,[row(1)])
    (d/'physics.jsonl').write_text(json.dumps(row(2))+'\n')
    with pytest.raises(ValueError,match='INCONSISTENT'):read_run(tmp_path,name)
    with pytest.raises(ValueError,match='UNREGISTERED'):read_run(tmp_path,'PREP_I2')
    (d/'physics.jsonl').write_text(json.dumps(row(1,.1))+'\n')
    with pytest.raises(ValueError,match='HASH_MISMATCH'):read_run(tmp_path,name)
