"""Offline I1 acceptance: raw trace + frozen parity + actual child status.

A zero child exit alone is insufficient: a saved runtime error or missing final
result always rejects a run. This specifically guards the observed Kit shutdown
failure mode. No simulation, training, or GPU execution occurs in this checker.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.g4i1_parity import compare_stages, read_paired_npz, REQUIRED_CASES


def process_gate(runtime, supervision):
    runtime, supervision = Path(runtime), Path(supervision)
    reasons=[]
    if (runtime/'runtime_error.json').exists(): reasons.append('runtime_error artifact exists')
    if not (runtime/'result.json').is_file(): reasons.append('final result missing')
    try:
        sup=json.loads((supervision/'supervisor/status.json').read_text())
        guard=json.loads((supervision/'guard/status.json').read_text())
        waited=json.loads((supervision/'supervisor/child_exit.json').read_text())
        if sup['raw_wait_returncode'] != 0 or waited['raw_wait_returncode'] != 0:reasons.append('child wait nonzero')
        if sup['termination_reason']!='child_exit' or sup['error'] is not None:reasons.append('supervisor error/forced termination')
        if not guard['continuous_monitoring'] or guard['monitoring_gaps'] or guard['error'] is not None:reasons.append('guard coverage failed')
    except (OSError,KeyError,ValueError,TypeError) as exc:reasons.append('process evidence incomplete: '+str(exc))
    return {'passed':not reasons,'reasons':reasons}


def assess(run_root, phase='episode_verified'):
    run_root=Path(run_root);runtime=run_root/phase;supervision=run_root/(phase+'_supervision')
    gate=process_gate(runtime,supervision)
    if not gate['passed']:return {'status':'G4_I1_FAIL','passed':False,'process_gate':gate}
    result=json.loads((runtime/'result.json').read_text())
    rows=[json.loads(line) for line in (runtime/'trace.jsonl').open()]
    policies=[r for r in rows if r['phase']=='policy'];controls=[r for r in rows if r['phase']=='controller']
    successes=[r for r in rows if r['phase']=='pre_reset_success' and r['episode']==0]
    resets=[r for r in rows if r['phase']=='timeout_reset'];initial=[r for r in rows if r['phase']=='initial_reset']
    ref,actual=read_paired_npz(runtime/'paired_trace.npz')
    edge=json.loads((runtime/'edge_fixtures.json').read_text())
    observed_cases=[r['case'] for r in edge]
    if len(resets)==1:observed_cases.append('timeout')
    if len(policies)==150 and policies[-1]['episode']==1:observed_cases.append('post_reset_first_action')
    parity=compare_stages(ref,actual,device_kind='gpu',cases=observed_cases,require_cases=REQUIRED_CASES)
    states_zero=bool(len(resets)==1 and all(not np.asarray(resets[0][key]).any() for key in ['h_after','c_after']))
    post_reset=bool(len(resets)==1 and len(policies)==150 and policies[-1]['episode']==1
                    and np.array_equal(policies[-1]['observation'],resets[0]['post_reset_observation'])
                    and np.array_equal(policies[-1]['cache_before'],resets[0]['post_reset_cache']))
    ever=any(r['capture']['success'][0] for r in successes)
    terminal=bool(successes and successes[-1]['capture']['success'][0])
    first=next((r['capture']['step_numbers'][0] for r in successes if r['capture']['success'][0]),None)
    checks={'process_gate':gate['passed'],'identity':result['checkpoint_identity']['sha256']=='f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f',
       'mode':result['mode']=='episode','configuration':result['num_envs']==1 and result['seed']==1001 and result['torch_seed']==1001 and result['fp32'] and result['deterministic'],
       'no_runner_optimizer':not result['training_runner_instantiated'] and not result['optimizer_restored'] and not result['training_runner_module_loaded'],
       'parity':parity['passed'],'source_target_fixtures':len(edge)==4 and {r['case'] for r in edge}==set(REQUIRED_CASES[:4]) and all(r['result']['passed'] for r in edge),
       'complete_episode':len(successes)==149 and len(resets)==1 and result['denominator']==1 and result['episodes'][0]['steps']==149,
       'policy_physics_counts':len(policies)==150 and len(controls)==1200 and result['physics_substeps']==1200,
       'recurrent_reset':states_zero,'post_reset_first_action':post_reset,
       'pose_cache_reset':len(initial)==1 and initial[0]['pose_cache_check']['max_absolute_error']==0 and all(r['pose_cache_check']['max_absolute_error']==0 for r in resets),
       'rng':result['rng_comparisons']==150 and result['rng_all_equal'], 'rms':result['rms_unchanged'],
       'success_counts_match_trace':result['ever_success']==int(ever) and result['terminal_success']==int(terminal) and result['episodes'][0]['first_success_step']==first}
    return {'status':'G4_I1_PASS' if all(checks.values()) else 'G4_I1_FAIL','passed':all(checks.values()),'checks':checks,
            'process_gate':gate,'parity':parity,'raw_counts':{'policy':len(policies),'controller':len(controls),'first_episode_success_captures':len(successes),'resets':len(resets)},
            'independently_recomputed_success':{'denominator':1,'ever':int(ever),'terminal':int(terminal),'first_step':first,'first_seconds':None if first is None else first/15},
            'scope':'Runtime evidence gate only; CPU regression, source hashes, and CPU actor/edge parity are separately required'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--phase',default='episode_verified');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=assess(a.run_root,a.phase)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(r,stream,indent=2,allow_nan=False);stream.write('\n')
    print(r['status']);return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
