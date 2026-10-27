"""CPU-only fail-closed verifier for a registered 300-decision G5 preflight.

This verifies trace consistency, not process exit, GPU PhysX or contact quality.
Raw root states use Isaac Lab's world xyz + wxyz quaternion convention. Offsets
are transformed in world-aligned coordinates after subtracting the env origin.
"""
import json
import math
import sys
from pathlib import Path

TASK = 'Isaac-Factory-GearMesh-Direct-v0'
OFFSET = 0.02025
POSITION_TOLERANCE_M = 5e-6  # float32 pose arithmetic, including origin subtraction
QUAT_NORM_TOLERANCE = 1e-4


def geometric(held, target):
    return math.hypot(held[0]-target[0], held[1]-target[1]) < .0025 and held[2]-target[2] < .001


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'{name} must have {size} elements')
    return [number(v, name) for v in value]


def reference_point(root, origin):
    """Rotate local (+0.02025,0,0) by root wxyz without using simulator code."""
    root = vector(root, 13, 'root state')
    origin = vector(origin, 3, 'environment origin')
    w, x, y, z = root[3:7]
    if abs(math.sqrt(w*w+x*x+y*y+z*z)-1.0) > QUAT_NORM_TOLERANCE:
        raise ValueError('root quaternion must be approximately unit length')
    # First column of R(q), equivalent to q * (0, offset, 0, 0) * q^-1
    # for unit q. No re-normalization: preserve the input pose semantics.
    rot = [1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)]
    return [root[i]-origin[i]+OFFSET*rot[i] for i in range(3)]


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f'{name} must be a boolean')
    return value


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f'duplicate JSON key: {key}')
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def verify(path):
    errors = []
    try:
        with Path(path).open(encoding='utf-8') as f:
            rows = [json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant) for line in f]
    except (OSError, UnicodeError, ValueError) as exc:
        return {'accepted': False, 'errors': [f'unreadable or malformed trace: {exc}'], 'decisions': 0}
    if len(rows) != 300:
        errors.append(f'expected exactly 300 records, got {len(rows)}')
    success_values = []
    for n, r in enumerate(rows, 1):
        try:
            if not isinstance(r, dict):
                raise ValueError('record must be an object')
            if r['task'] != TASK or _integer(r['num_envs'], 'num_envs') != 1 or _integer(r['seed'], 'seed') != 0:
                raise ValueError('registered task/num_envs/seed mismatch')
            if _integer(r['decision'], 'decision') != n:
                raise ValueError('decision indices not contiguous')
            if _integer(r['physics_counter'], 'physics_counter') != n*8:
                raise ValueError('policy-loop physics counter mismatch')
            expected_step = n if n <= 299 else n-299
            if _integer(r['episode_step'], 'episode_step') != expected_step:
                raise ValueError('episode step mismatch')
            done = n == 299
            if _boolean(r['terminated'], 'terminated') != done or _boolean(r['truncated'], 'truncated') != done:
                raise ValueError('official dual done flags mismatch')
            if (r['reset_env_ids'] != ([[0]] if done else []) or
                (done and type(r['reset_env_ids'][0][0]) is not int)):
                raise ValueError('automatic reset invocation mismatch')
            if _integer(r['post_reset_episode_counter'], 'post_reset_episode_counter') != (0 if done else expected_step):
                raise ValueError('post-reset episode counter mismatch')
            number(r['reward'], 'reward')
            vector(r['policy_obs'], 19, 'policy_obs')
            vector(r['critic_obs'], 43, 'critic_obs')
            if number(r['keypoint_dist'], 'keypoint_dist') < 0:
                raise ValueError('negative keypoint distance')
            joint_pos, joint_vel = r['joint_pos'], r['joint_vel']
            if not isinstance(joint_pos, list) or len(joint_pos) < 7:
                raise ValueError('joint positions missing')
            vector(joint_pos, len(joint_pos), 'joint_pos')
            vector(joint_vel, len(joint_pos), 'joint_vel')
            for name in ('small_root', 'large_root'):
                vector(r[name], 13, name)
            origin = vector(r['env_origin'], 3, 'env_origin')
            held = reference_point(r['held_root'], origin)
            target = reference_point(r['fixed_root'], origin)
            for name, actual in [('held_base', held), ('target_base', target)]:
                recorded = vector(r[name], 3, name)
                if any(abs(a-b) > POSITION_TOLERANCE_M for a,b in zip(actual, recorded)):
                    raise ValueError(f'{name} disagrees with raw pose transform')
            success = geometric(held, target)
            if _boolean(r['official_success'], 'official_success') != success:
                raise ValueError('official geometric predicate disagreement')
            success_values.append(success)
            if not done and r['episode_summary'] is not None:
                raise ValueError('unexpected episode summary')
            if done:
                expected = {'env_id':0, 'episode_id':0, 'steps':299,
                            'first_success_step':next((i+1 for i,v in enumerate(success_values) if v), None),
                            'ever_success':any(success_values), 'terminal_success':success,
                            'terminal_kind':'timeout'}
                summary = r['episode_summary']
                if not isinstance(summary, dict) or set(summary) != set(expected):
                    raise ValueError('missing or malformed episode summary')
                for k,v in expected.items():
                    if type(summary[k]) is not type(v) or summary[k] != v:
                        raise ValueError(f'episode summary mismatch: {k}')
        except (KeyError, TypeError, ValueError, OverflowError, IndexError) as exc:
            errors.append(f'record {n}: {exc}')
    return {'accepted':not errors, 'errors':errors, 'decisions':len(rows),
            'policy_loop_physics_steps':len(rows)*8 if not errors else None,
            'episode': {'steps':299, 'ever_success':any(success_values[:299]),
                        'terminal_success':success_values[298],
                        'first_success_step':next((i+1 for i,v in enumerate(success_values[:299]) if v),None)}
                       if not errors else None,
            'physical_contact_quality':'not evaluated', 'actual_gear_engagement':'not evaluated'}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: g5_gear_verify.py TRACE.jsonl', file=sys.stderr)
        raise SystemExit(2)
    result = verify(sys.argv[1])
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['accepted'] else 1)
