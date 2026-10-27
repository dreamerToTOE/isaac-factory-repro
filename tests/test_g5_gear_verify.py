import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.g5_gear_verify import TASK, geometric, reference_point, verify


def trace_rows():
    q = [math.sqrt(.5), 0., 0., math.sqrt(.5)]  # +90 degrees about z
    origin = [10., 20., 3.]
    fixed = origin + q + [0.]*6
    target = reference_point(fixed, origin)
    rows = []
    for n in range(1, 301):
        delta = .001 if n in (10, 150) else .005
        held = [origin[0]+delta, origin[1], origin[2]] + q + [0.]*6
        done = n == 299
        rows.append({'task':TASK, 'num_envs':1, 'seed':0, 'decision':n,
                     'episode_step':n if n <= 299 else 1, 'physics_counter':8*n,
                     'terminated':done, 'truncated':done, 'reset_env_ids':[[0]] if done else [],
                     'post_reset_episode_counter':0 if done else (n if n < 299 else 1),
                     'official_success':n in (10,150), 'reward':0.25, 'keypoint_dist':0.1,
                     'policy_obs':[0.]*19, 'critic_obs':[0.]*43,
                     'joint_pos':[0.]*9, 'joint_vel':[0.]*9, 'held_root':held,
                     'fixed_root':fixed, 'small_root':fixed, 'large_root':fixed,
                     'env_origin':origin, 'held_base':reference_point(held, origin), 'target_base':target,
                     'episode_summary':{'env_id':0, 'episode_id':0, 'steps':299, 'first_success_step':10,
                                        'ever_success':True, 'terminal_success':False, 'terminal_kind':'timeout'} if done else None})
    return rows


def save(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_synthetic_complete_trace(tmp_path):
    result=verify(save(tmp_path/'trace', trace_rows()))
    assert result['accepted'], result['errors']
    assert result['decisions']==300 and result['policy_loop_physics_steps']==2400
    assert result['episode']=={'steps':299, 'ever_success':True, 'terminal_success':False, 'first_success_step':10}


def test_strict_thresholds_and_one_sided_height():
    origin = [0.,0.,0.]
    assert geometric([.00249, 0, .00099], origin)
    assert not geometric([.0025, 0, 0], origin)
    assert not geometric([0, 0, .001], origin)
    assert geometric([0, 0, -.1], origin)
    assert not geometric([.002, .002, 0], origin)


def test_world_pose_wxyz_and_environment_origin():
    q=[math.sqrt(.5),0,0,math.sqrt(.5)]
    assert reference_point([10,20,3]+q+[0]*6,[10,20,3]) == pytest.approx([0,.02025,0], abs=1e-12)
    q=[math.sqrt(.5),0,math.sqrt(.5),0]
    assert reference_point([10,20,3]+q+[0]*6,[10,20,3]) == pytest.approx([0,0,-.02025], abs=1e-12)


def mutate(rows, kind):
    if kind == 'empty': return []
    if kind == 'truncated': return rows[:-1]
    if kind == 'missing': return rows[:40]+rows[41:]
    if kind == 'duplicated': rows[40]=copy.deepcopy(rows[39])
    if kind == 'physics': rows[40]['physics_counter']+=1
    if kind == 'no_completion': rows[298]['terminated']=rows[298]['truncated']=False
    if kind == 'wrong_completion': rows[297]['terminated']=rows[297]['truncated']=True
    if kind == 'done_flags': rows[298]['truncated']=False
    if kind == 'reset_call': rows[298]['reset_env_ids']=[]
    if kind == 'reset_counter': rows[298]['post_reset_episode_counter']=299
    if kind == 'next_episode': rows[299]['episode_step']=300
    if kind == 'nan_reward': rows[4]['reward']=float('nan')
    if kind == 'inf_state': rows[4]['joint_pos'][0]=float('inf')
    if kind == 'nan_observation': rows[4]['policy_obs'][0]=float('nan')
    if kind == 'bad_observation_shape': rows[4]['critic_obs'].pop()
    if kind == 'malformed_geometry': rows[4]['held_root']=[0]*12
    if kind == 'bad_quaternion': rows[4]['held_root'][3:7]=[0]*4
    if kind == 'wrong_geometry': rows[4]['held_root'][0]+=.01
    if kind == 'predicate': rows[4]['official_success']=True
    if kind == 'summary_ever': rows[298]['episode_summary']['ever_success']=False
    if kind == 'summary_terminal': rows[298]['episode_summary']['terminal_success']=True
    if kind == 'summary_first': rows[298]['episode_summary']['first_success_step']=150
    if kind == 'task': rows[0]['task']='wrong'
    if kind == 'seed': rows[0]['seed']=1
    if kind == 'envs': rows[0]['num_envs']=2
    if kind == 'missing_origin': del rows[0]['env_origin']
    if kind == 'missing_field': del rows[0]['reward']
    return rows


@pytest.mark.parametrize('kind', ['empty','truncated','missing','duplicated','physics','no_completion',
    'wrong_completion','done_flags','reset_call','reset_counter','next_episode','nan_reward','inf_state','nan_observation','bad_observation_shape',
    'malformed_geometry','bad_quaternion','wrong_geometry','predicate','summary_ever','summary_terminal',
    'summary_first','task','seed','envs','missing_origin','missing_field'])
def test_invalid_evidence_is_rejected(tmp_path, kind):
    result=verify(save(tmp_path/'trace', mutate(trace_rows(),kind)))
    assert not result['accepted'] and result['errors']


def test_missing_and_malformed_json_fails_closed(tmp_path):
    path=tmp_path/'missing'
    assert not verify(path)['accepted']
    path.write_text('{bad json\n')
    assert not verify(path)['accepted']
    run=subprocess.run([sys.executable, str(Path(__file__).parents[1]/'scripts/g5_gear_verify.py'), str(path)],capture_output=True,text=True)
    assert run.returncode != 0 and not json.loads(run.stdout)['accepted']


def test_negative_z_success_is_accepted_in_complete_trace(tmp_path):
    rows=trace_rows()
    rows[4]['held_root'][0]=10.0
    rows[4]['held_root'][2]=2.9
    rows[4]['held_base']=reference_point(rows[4]['held_root'],rows[4]['env_origin'])
    rows[4]['official_success']=True
    rows[298]['episode_summary']['first_success_step']=5
    result=verify(save(tmp_path/'negative-z',rows))
    assert result['accepted'], result['errors']
    assert result['episode']['first_success_step']==5


def test_cli_empty_fails_and_complete_trace_succeeds(tmp_path):
    script=str(Path(__file__).parents[1]/'scripts/g5_gear_verify.py')
    empty=save(tmp_path/'empty', [])
    full=save(tmp_path/'complete', trace_rows())
    fail=subprocess.run([sys.executable,script,str(empty)],capture_output=True,text=True)
    ok=subprocess.run([sys.executable,script,str(full)],capture_output=True,text=True)
    assert fail.returncode != 0 and not json.loads(fail.stdout)['accepted']
    assert ok.returncode == 0 and json.loads(ok.stdout)['accepted']


def test_duplicate_json_keys_fail_closed(tmp_path):
    path=tmp_path/'duplicate'
    path.write_text('{"decision":1,"decision":2}\n')
    assert not verify(path)['accepted']


def test_huge_integer_fails_closed(tmp_path):
    rows=trace_rows()
    rows[0]['reward']=10**400
    assert not verify(save(tmp_path/'huge',rows))['accepted']
