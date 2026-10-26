"""CPU analysis controls: full, interrupted, missing and contradictory evidence."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import analyze_g3cr4 as analysis

ROOT = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def supervisor(code=0, reason='child_exit'):
    return {'finalized': True, 'raw_wait_returncode': code,
            'child_exit_code': code if code >= 0 else None, 'child_signal': None,
            'termination_reason': reason, 'error': None, 'stop_signals': [],
            'received_supervisor_signals': [], 'guard_required': False, 'wall_seconds': 321.5}


def fixture(tmp_path, monkeypatch, epochs=400, complete=True):
    root = tmp_path / 'run'
    run_dir = tmp_path / 'logs'
    completion = root / 'completion'
    manifest = {'repository': str(ROOT), 'training': {'run_root': str(root), 'run_dir': str(run_dir)},
                'completion_output': str(completion)}
    put(root / 'launch.json', {'monitoring_policy': analysis.POLICY, 'gpu_guard_enabled': False,
                               'gpu_telemetry_required': False, 'gpu_telemetry_gaps_terminate_child': False})
    put(root / 'supervisor/status.json', supervisor() if complete else supervisor(1))
    put(root / 'supervisor/training_audit.json', {
        'epochs': [{'epoch': epoch, 'environment_transitions': 4096, 'epoch_loss_arrays_finite': True,
                    'official_total_seconds': 1.0, 'env_step_seconds': .5} for epoch in range(1, epochs + 1)],
        'runtime': {'num_actors': 32, 'horizon_length': 128}, 'instrumentation_errors': [],
        'optimizer_steps': {'actor': epochs * 32, 'central_value': epochs * 32}, 'checkpoint_events': []})
    put(root / 'supervisor/tensorboard_finalizer.json', {'status': 'complete', 'training_error': None,
                                                       'agents': [{'writer_expected': True, 'success': True}]})
    (root / 'supervisor/console.txt').write_text('MAX EPOCHS NUM!\n' if complete else 'Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n')
    monkeypatch.setattr(analysis, 'read_scalars', lambda *args: {
        'directory': str(run_dir / 'summaries'), 'files': [], 'all_logged_scalars_finite': True,
        'tags': {'losses/a_loss': [{'step': x, 'value': 1e6 if x % 2 else -1e6, 'finite': True} for x in range(epochs)]}})
    selected = {}
    for epoch in analysis.EPOCHS if complete else []:
        path = run_dir / 'nn' / f'last_Factory_ep_{epoch}_rew_0.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'CPU fixture checkpoint epoch{epoch}'.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        selected[str(epoch)] = {'path': str(path), 'sha256': digest, 'epoch': epoch, 'frame': epoch * 4096,
                                'all_tensors_and_numeric_arrays_finite': True, 'optimizer_step_min': 1,
                                'nonzero_adam_exp_avg_states': 1, 'actor_model_present': True, 'central_value_present': True}
        for seed in analysis.SEEDS:
            episodes_per_env = 3 if epoch == 400 else 1
            episodes = [{'seed': seed, 'env_id': env, 'episode_id': ep, 'steps': 149, 'terminal_kind': 'timeout',
                         'ever_success': env != 31, 'terminal_success': env != 31,
                         'first_success_step': 30 if env != 31 else None,
                         'first_success_time_seconds': 2.0 if env != 31 else None}
                        for env in range(32) for ep in range(episodes_per_env)]
            count = len(episodes); good = sum(row['ever_success'] for row in episodes)
            payload = {'status': 'complete', 'seed': seed, 'num_envs': 32, 'checkpoint_epoch': epoch,
                       'checkpoint': str(path), 'checkpoint_sha256': digest, 'deterministic': True,
                       'auto_reset_order_verified': True, 'step_dt_seconds': 1/15, 'official_success_capture': 'pre_reset_factory_get_curr_successes',
                       'episodes': episodes, 'episodes_completed': count,
                       'episodes_per_env_requested': episodes_per_env,
                       'aggregate': {'episodes_completed': count, 'ever_success_count': good,
                                     'ever_success_rate': good / count, 'terminal_success_count': good,
                                     'terminal_success_rate': good / count}}
            folder = completion / 'evaluation' / f'epoch{epoch:03d}' / f'seed{seed}'
            put(folder / 'metrics.json', payload)
            put(folder / 'supervisor/status.json', supervisor())
    put(completion / 'selected_checkpoints.json', selected)
    put(completion / 'status.json', {'phase': 'training_and_evaluation_complete' if complete else 'incomplete',
                                    'controller_exit_code': 0 if complete else 1,
                                    'training': {'tensorboard_tail_complete': complete},
                                    'selected_checkpoints': selected})
    return manifest


def test_complete_training_with_large_finite_losses_is_not_failed_as_nonconvergence(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch)
    report, _, _ = analysis.analyze(manifest)
    assert report['engineering_status'] == 'COMPLETE', report['completion_gates']
    assert report['scientific_status'] == 'ASSESSED_DESCRIPTIVELY'
    assert report['training']['verified_progress']['completed_epochs_verified'] == 400
    assert report['training']['environment_transitions'] == 1638400
    assert report['training']['verified_progress']['possibly_uncompleted_rollout_transitions'] == 0
    assert report['training']['launched_process_wall_seconds'] == 321.5
    assert report['training']['loss_stability']['observed_epoch_loss_arrays_finite'] is True
    assert report['training']['loss_stability']['scalar_statistics']['losses/a_loss']['max'] == 1e6
    assert report['completion_gates']['loss_magnitude_or_convergence_threshold_required'] is False
    assert report['monitoring_policy']['gpu_telemetry_required'] is False
    assert not any('GPU' in error for error in report['issues']['errors'])
    assert len(report['checkpoint_inventory']) == 4
    assert all(row['inspection_hash_matches'] for row in report['checkpoint_inventory'])


def test_evaluation_denominators_timing_and_historical_data_remain_separate(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch)
    report, _, _ = analysis.analyze(manifest)
    rows = report['evaluation']['checkpoints']
    assert [row['aggregate']['episodes_completed'] for row in rows] == [96, 96, 96, 288]
    assert rows[-1]['aggregate']['ever_success_count'] == 279
    assert rows[-1]['aggregate']['first_success_step_conditional']['n'] == 279
    assert rows[-1]['aggregate']['first_success_time_seconds_conditional']['median'] == 2.0
    comparison = report['historical_matched_transition_comparison']['rows']
    assert [row['historical_64env_epoch'] for row in comparison] == [50, 100, 150, 200]
    assert comparison[-1]['new_evaluation']['episodes_completed'] == 288
    assert comparison[-1]['historical_evaluation']['ever_success_count'] == 280
    assert all(row['denominators_pooled'] is False for row in comparison)


def test_incomplete_oom_preserves_progress_actual_error_and_unknown_extra_transitions(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch, epochs=10, complete=False)
    report, _, _ = analysis.analyze(manifest)
    assert report['engineering_status'] == 'INCOMPLETE'
    assert report['training']['environment_transitions'] == 40960
    assert report['training']['exit']['raw_wait_returncode'] == 1
    assert 'CUDA out of memory' in report['training']['console_evidence']['error_lines'][-1]
    assert report['training']['verified_progress']['possibly_uncompleted_rollout_transitions'].startswith('unknown')
    assert report['monitoring_policy']['gpu_monitoring_coverage'] == 'not measured; no continuous-coverage claim'
    assert [row['aggregate']['ever_success_rate'] for row in report['evaluation']['checkpoints']] == [None] * 4
    assert [row['aggregate']['episodes_completed'] for row in report['evaluation']['checkpoints']] == [0] * 4
    assert report['issues']['errors'] == []


def test_incomplete_audit_does_not_invent_missing_transitions(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch, epochs=10, complete=False)
    audit_path = Path(manifest['training']['run_root']) / 'supervisor/training_audit.json'
    audit = json.loads(audit_path.read_text())
    audit['epochs'].pop(4)
    put(audit_path, audit)
    report, _, _ = analysis.analyze(manifest)
    assert report['training']['verified_progress']['completed_epochs_verified'] == 4
    assert report['training']['environment_transitions'] == 16384
    assert report['training']['verified_progress']['all_audit_rows_in_contiguous_prefix'] is False
    train = report['training']
    assert train['end_to_end_transitions_per_wall_second'] == pytest.approx(16384 / 321.5)
    assert train['official_ppo_total_seconds'] == 4.0
    assert train['official_ppo_aggregate_fps'] == 4096
    assert train['environment_step_aggregate_fps'] == 8192
    assert train['throughput_scope']['excluded_audit_rows'] == 5
    assert train['stable_aggregate_total_fps'] is None
    assert train['stable_epoch_range'] is None


def test_nonfinite_epoch_is_an_engineering_failure_without_loss_threshold(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch)
    audit_path = Path(manifest['training']['run_root']) / 'supervisor/training_audit.json'
    audit = json.loads(audit_path.read_text()); audit['epochs'][10]['epoch_loss_arrays_finite'] = False
    put(audit_path, audit)
    report, _, _ = analysis.analyze(manifest)
    assert report['engineering_status'] == 'INCOMPLETE'
    assert report['training']['loss_stability']['observed_epoch_loss_arrays_finite'] is False
    assert any('loss arrays' in item for item in report['completion_gates']['failed_requirements'])


def test_changed_checkpoint_hash_is_not_reported_as_integrity_pass(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch)
    path = Path(manifest['training']['run_dir']) / 'nn/last_Factory_ep_100_rew_0.pth'
    path.write_bytes(b'changed')
    report, _, _ = analysis.analyze(manifest)
    assert report['engineering_status'] == 'INCOMPLETE'
    assert report['checkpoint_inventory'][0]['inspection_hash_matches'] is False
    assert any('changed after tensor inspection' in item for item in report['issues']['errors'])


def test_missing_one_eval_does_not_become_zero_success_or_complete(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch)
    folder = Path(manifest['completion_output']) / 'evaluation/epoch400/seed1003'
    (folder / 'metrics.json').unlink()
    report, _, _ = analysis.analyze(manifest)
    assert report['engineering_status'] == 'INCOMPLETE'
    row = report['evaluation']['checkpoints'][-1]
    assert row['aggregate']['episodes_completed'] == 192
    assert row['aggregate']['ever_success_count'] == 186
    assert row['per_seed'][-1]['ever_success_rate'] is None


def test_all_artifacts_missing_is_analyzable_and_does_not_require_gpu_files(tmp_path, monkeypatch):
    manifest = {'repository': str(ROOT), 'training': {'run_root': str(tmp_path / 'missing'), 'run_dir': str(tmp_path / 'logs')}}
    report, _, _ = analysis.analyze(manifest)
    assert report['engineering_status'] == 'INCOMPLETE'
    assert report['training']['environment_transitions'] == 0
    assert report['training']['loss_stability']['observed_epoch_loss_arrays_finite'] is None
    assert any('Launch evidence' in error for error in report['issues']['errors'])
    assert not any('guard missing' in error.lower() for error in report['issues']['errors'])


def test_csv_output_uses_each_checkpoint_denominator(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch)
    report, _, epochs = analysis.analyze(manifest)
    output = tmp_path / 'analysis'; output.mkdir()
    analysis.write_csv(output, report, epochs)
    import csv
    rows = list(csv.DictReader((output / 'evaluation_per_seed.csv').open()))
    assert len(rows) == 12
    assert rows[-1]['episodes_completed'] == '96'
    assert rows[0]['episodes_completed'] == '32'
    assert rows[-1]['conditional_first_success_n'] == '93'


def test_existing_analysis_output_is_rejected_before_read(tmp_path):
    with pytest.raises(SystemExit) as error:
        analysis.main(['--manifest', str(tmp_path / 'missing.json'), '--output', str(tmp_path)])
    assert error.value.code == 2


def test_controller_failure_before_training_gate_has_null_training_not_a_crash(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch, epochs=0, complete=False)
    path = Path(manifest['completion_output']) / 'status.json'
    status = json.loads(path.read_text()); status['training'] = None
    put(path, status)
    report, _, _ = analysis.analyze(manifest)
    assert report['engineering_status'] == 'INCOMPLETE'
    assert report['training']['environment_transitions'] == 0


def test_missing_prefix_timing_is_null_and_not_filled_from_later_epochs(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch, epochs=20, complete=False)
    audit_path = Path(manifest['training']['run_root']) / 'supervisor/training_audit.json'
    audit = json.loads(audit_path.read_text())
    audit['epochs'].pop(4)
    audit['epochs'][2].pop('official_total_seconds')
    audit['epochs'][1].pop('env_step_seconds')
    put(audit_path, audit)
    report, _, rows = analysis.analyze(manifest)
    train = report['training']
    assert train['environment_transitions'] == 16384
    assert train['official_ppo_total_seconds'] is None
    assert train['official_ppo_aggregate_fps'] is None
    assert train['environment_step_aggregate_fps'] is None
    output = tmp_path / 'csv'; output.mkdir()
    analysis.write_csv(output, report, rows)
    import csv
    csv_rows = list(csv.DictReader((output / 'completed_epochs.csv').open()))
    assert len(csv_rows) == 19
    assert [row['in_verified_contiguous_prefix'] for row in csv_rows[:5]] == ['True'] * 4 + ['False']


def test_unfinalized_wall_duration_does_not_produce_wall_throughput(tmp_path, monkeypatch):
    manifest = fixture(tmp_path, monkeypatch, epochs=10, complete=False)
    status_path = Path(manifest['training']['run_root']) / 'supervisor/status.json'
    status = json.loads(status_path.read_text()); status['finalized'] = False
    put(status_path, status)
    report, _, _ = analysis.analyze(manifest)
    assert report['training']['launched_process_wall_seconds'] is None
    assert report['training']['end_to_end_transitions_per_wall_second'] is None
    assert report['training']['official_ppo_aggregate_fps'] == 4096
