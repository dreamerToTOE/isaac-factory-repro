"""Independent scalar validation of the G5 pre-reset JSONL evidence."""
import json
import math
import sys
from pathlib import Path

def geometric(held, target):
    return math.hypot(held[0]-target[0], held[1]-target[1]) < .0025 and held[2]-target[2] < .001

def verify(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    failures=[]; episodes=[]; current=[]
    for n,r in enumerate(rows, 1):
        if r['decision'] != n: failures.append(f'noncontiguous decision {n}')
        if r['physics_counter'] != 8*n: failures.append(f'physics step counter {n}')
        x=geometric(r['held_base'],r['target_base'])
        if x != r['official_success']: failures.append(f'predicate disagreement {n}')
        if not math.isfinite(r['reward']): failures.append(f'nonfinite reward {n}')
        current.append(x)
        if r['terminated'] or r['truncated']:
            episodes.append({'decision':n,'steps':r['episode_step'],'ever':any(current),
                             'terminal':current[-1], 'first_success': next((i+1 for i,v in enumerate(current) if v),None),
                             'terminated':r['terminated'],'truncated':r['truncated'],
                             'post_reset_episode_counter':r['post_reset_episode_counter']})
            current=[]
    return {'decisions':len(rows),'policy_physics_steps':len(rows)*8,'predicate_mismatches_or_other_errors':failures,
            'episodes':episodes,'incomplete_episode_decisions':len(current),
            'all_rewards_finite':all(math.isfinite(r['reward']) for r in rows),
            'reward_min':min((r['reward'] for r in rows),default=None),'reward_max':max((r['reward'] for r in rows),default=None)}
if __name__=='__main__':
    result=verify(sys.argv[1]);print(json.dumps(result,indent=2,allow_nan=False));sys.exit(bool(result['predicate_mismatches_or_other_errors']))
