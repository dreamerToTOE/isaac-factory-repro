import copy
import pytest
from scripts.compare_g3b import compare_evaluations


def run(episodes, seed=1001, checkpoint='trusted'):
    return {'seed': seed, 'checkpoint_sha256': checkpoint, 'episodes_completed': len(episodes),
            'episodes': [{'env_id': i, 'episode_id': 0, 'ever_success': ok,
                          'terminal_success': ok} for i, ok in enumerate(episodes)]}


def test_original_and_new_denominators_are_not_combined_or_overwritten():
    old = {1001: run([False, True])}; snapshot = copy.deepcopy(old)
    result = compare_evaluations(old, {1001: run([True])})
    assert result['original_primary']['episodes_completed'] == 2
    assert result['original_primary']['ever_success_count'] == 1
    assert result['g3b_diagnostic']['episodes_completed'] == 1
    assert result['g3b_diagnostic']['ever_success_count'] == 1
    assert result['original_failures_now_succeeded'] == [{'seed':1001,'env_id':0,'episode_id':0}]
    assert old == snapshot and result['exact_replay_verified'] is False


def test_failure_identifier_recurrence_is_not_exact_replay():
    result = compare_evaluations({1001:run([False, True])}, {1001:run([False, False])})
    assert result['identifier_matched_original_failure_count'] == 1
    assert result['new_failures_at_previously_successful_identifiers'] == [{'seed':1001,'env_id':1,'episode_id':0}]
    assert not result['exact_replay_verified']


def test_missing_diagnostic_episode_is_not_counted_as_failure():
    result = compare_evaluations({1001:run([True, False])}, {1001:run([True])})
    assert result['identifier_matched_original_failure_count'] == 0
    assert result['original_failures_not_evaluated'] == [{'seed':1001,'env_id':1,'episode_id':0}]


def test_duplicate_and_wrong_checkpoint_provenance_are_rejected():
    bad = run([False]); bad['episodes'] *= 2; bad['episodes_completed'] = 2
    with pytest.raises(ValueError, match='duplicate'):
        compare_evaluations({1001:run([False])}, {1001:bad})
    with pytest.raises(ValueError, match='checkpoint'):
        compare_evaluations({1001:run([False])}, {1001:run([False],checkpoint='wrong')})


def test_reported_denominator_and_seed_mismatch_are_rejected():
    bad = run([False]); bad['episodes_completed'] = 32
    with pytest.raises(ValueError, match='denominator'):
        compare_evaluations({1001:run([False])}, {1001:bad})
    with pytest.raises(ValueError, match='seed'):
        compare_evaluations({1001:run([False])}, {1001:run([False],seed=1002)})
