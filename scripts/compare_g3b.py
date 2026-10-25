"""CPU-only comparison; historical and diagnostic episode pools stay separate."""
from __future__ import annotations
import copy


def _index(runs):
    episodes = {}
    checkpoints = set()
    per_seed = []
    for seed, result in sorted(runs.items()):
        seed = int(seed)
        if int(result['seed']) != seed:
            raise ValueError('seed provenance mismatch')
        checkpoints.add(result['checkpoint_sha256'])
        records = result['episodes']
        if result['episodes_completed'] != len(records):
            raise ValueError('completed denominator differs from episode records')
        for item in records:
            key = (seed, int(item['env_id']), int(item['episode_id']))
            if key in episodes:
                raise ValueError('duplicate completed episode')
            if not isinstance(item['ever_success'], bool) or not isinstance(item['terminal_success'], bool):
                raise ValueError('success flags must be explicit booleans')
            if item['terminal_success'] and not item['ever_success']:
                raise ValueError('terminal success without ever success')
            episodes[key] = copy.deepcopy(item)
        per_seed.append({'seed': seed, 'episodes_completed': len(records),
                         'ever_success_count': sum(x['ever_success'] for x in records),
                         'terminal_success_count': sum(x['terminal_success'] for x in records)})
    if len(checkpoints) != 1:
        raise ValueError('each pool must have one checkpoint identity')
    total = len(episodes)
    ever = sum(x['ever_success'] for x in episodes.values())
    terminal = sum(x['terminal_success'] for x in episodes.values())
    return episodes, {'episodes_completed': total, 'ever_success_count': ever,
                      'terminal_success_count': terminal, 'ever_success_rate': ever / total if total else None,
                      'terminal_success_rate': terminal / total if total else None, 'per_seed': per_seed,
                      'checkpoint_sha256': next(iter(checkpoints))}


def compare_evaluations(original, diagnostic):
    """Match identifiers/outcomes, never assert reconstruction of a missing path."""
    before, original_summary = _index(original)
    after, diagnostic_summary = _index(diagnostic)
    if original_summary['checkpoint_sha256'] != diagnostic_summary['checkpoint_sha256']:
        raise ValueError('checkpoint identity mismatch')
    def ids(keys):
        return [{'seed': k[0], 'env_id': k[1], 'episode_id': k[2]} for k in sorted(keys)]
    failures = {k for k, x in before.items() if not x['ever_success']}
    matched_failures = {k for k in failures & after.keys() if not after[k]['ever_success']}
    recovered = {k for k in failures & after.keys() if after[k]['ever_success']}
    additional = {k for k in before.keys() & after.keys() if before[k]['ever_success'] and not after[k]['ever_success']}
    mismatches = {k for k in before.keys() & after.keys()
                  if (before[k]['ever_success'], before[k]['terminal_success']) !=
                     (after[k]['ever_success'], after[k]['terminal_success'])}
    return {'original_primary': original_summary, 'g3b_diagnostic': diagnostic_summary,
            'identifier_matched_original_failure_count': len(matched_failures),
            'identifier_matched_original_failures_again_failed': ids(matched_failures),
            'original_failures_now_succeeded': ids(recovered),
            'original_failures_not_evaluated': ids(failures - after.keys()),
            'new_failures_at_previously_successful_identifiers': ids(additional),
            'outcome_mismatches': ids(mismatches), 'exact_replay_verified': False,
            'historical_failure_trajectories_available': False,
            'interpretation': 'Identifier/outcome recurrence only; seed/env/episode equality does not prove exact replay. Pools are never added together.'}
