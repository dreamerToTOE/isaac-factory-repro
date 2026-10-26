#!/usr/bin/env python3
"""CPU-only G3C-R5 analysis, retaining partial evidence and separate denominators.

No GPU API, process-control API, simulation import, checkpoint unpickling, or
training launch is used here. TensorBoard extraction may use the registered
learning Python; plots use the current interpreter's Matplotlib installation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

try:
    from analyze_g2 import (descriptive, finite_number, identity, read_audit,
                            read_evaluations, read_scalars, scalar_values,
                            training_summary, write_json)
    from complete_g3cr5 import (validate_checkpoint, validate_evaluation,
                               validate_supervisor, validate_training)
except ImportError:
    from scripts.analyze_g2 import (descriptive, finite_number, identity, read_audit,
                                    read_evaluations, read_scalars, scalar_values,
                                    training_summary, write_json)
    from scripts.complete_g3cr5 import (validate_checkpoint, validate_evaluation,
                                       validate_supervisor, validate_training)

EPOCHS = [100, 200, 300, 400]
SEEDS = [1001, 1002, 1003]
POLICY = 'user-authorized-no-gpu-guard'


def load_optional(path, issues, label):
    path = Path(path)
    if not path.is_file():
        issues['warnings'].append(f'Missing {label}: {path}')
        return {}
    try:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError('expected JSON object')
        return result
    except (OSError, ValueError) as error:
        issues['errors'].append(f'Unreadable {label}: {path}: {error}')
        return {}


def locations(manifest):
    """Accept the frozen experiment manifest or a completion-controller manifest."""
    train = manifest.get('training', {})
    repo = Path(manifest.get('repository', Path(__file__).resolve().parents[1]))
    run_root = Path(train.get('run_root', manifest.get('training_supervisor_directory', '.')))
    supervisor = Path(manifest.get('training_supervisor_directory', run_root / 'supervisor'))
    run_dir = Path(train.get('run_dir', manifest.get('training_event_directory', '.')))
    completion = Path(manifest.get('completion_output', run_root / 'completion'))
    return {'repository': repo, 'run_root': run_root, 'supervisor': supervisor,
            'run_dir': run_dir, 'completion': completion,
            'audit': Path(manifest.get('training_audit', supervisor / 'training_audit.json')),
            'finalizer': Path(manifest.get('training_finalizer', supervisor / 'tensorboard_finalizer.json')),
            'events': Path(manifest.get('training_event_directory', run_dir / 'summaries')),
            'checkpoints': Path(manifest.get('checkpoint_directory', run_dir / 'nn')),
            'evaluations': Path(manifest.get('evaluation_root', completion / 'evaluation'))}


def completed_epoch_evidence(epochs, planned_epochs=400):
    """Count only a contiguous prefix with registered per-epoch transitions."""
    prefix = []
    for expected, row in enumerate(epochs, 1):
        if row.get('epoch') != expected or row.get('environment_transitions') != 4096:
            break
        prefix.append(row)
    count = len(prefix)
    return {'completed_epochs_verified': count,
            'completed_transitions_verified': sum(row['environment_transitions'] for row in prefix),
            'planned_epochs': planned_epochs, 'planned_transitions': planned_epochs * 4096,
            'completion_fraction': count / planned_epochs,
            'all_audit_rows_in_contiguous_prefix': len(prefix) == len(epochs),
            'possibly_uncompleted_rollout_transitions': 'unknown; not included in completed-epoch count',
            'count_rule': 'Sum actual audit transitions only for contiguous epochs starting at 1, each with 32*128 transitions; no interpolation or resumed-run concatenation.'}


def verified_throughput(train, epochs, supervisor):
    """Keep every throughput numerator and timing denominator on the same prefix."""
    count = train['verified_progress']['completed_epochs_verified']
    prefix = epochs[:count]
    transitions = train['verified_progress']['completed_transitions_verified']

    def seconds(rows, key):
        values = [row.get(key) for row in rows]
        return (sum(values) if values and all(finite_number(value) and value >= 0 for value in values)
                and sum(values) > 0 else None)

    official = seconds(prefix, 'official_total_seconds')
    environment = seconds(prefix, 'env_step_seconds')
    stable = [row for row in prefix if row['epoch'] >= 11]
    stable_seconds = seconds(stable, 'official_total_seconds')
    duration = supervisor.get('wall_seconds')
    reliable_wall = supervisor.get('finalized') is True and finite_number(duration) and duration > 0
    train.update({
        'official_ppo_total_seconds': official,
        'official_ppo_timing_source': 'verified_contiguous_epoch_audit_prefix' if official else None,
        'official_ppo_aggregate_fps': transitions / official if official else None,
        'environment_step_aggregate_fps': transitions / environment if environment else None,
        'launched_process_wall_seconds': duration if reliable_wall else None,
        'end_to_end_transitions_per_wall_second': transitions / duration if reliable_wall else None,
        'stable_epoch_range': [11, count] if stable_seconds else None,
        'stable_timing_source': 'verified_contiguous_epoch_audit_prefix_epochs_11_onward' if stable_seconds else None,
        'stable_aggregate_total_fps': sum(row['environment_transitions'] for row in stable) / stable_seconds if stable_seconds else None,
        'stable_epoch_seconds': descriptive([row['official_total_seconds'] for row in stable]) if stable_seconds else descriptive([]),
        'throughput_scope': {
            'completed_transition_numerator': 'verified contiguous audit prefix only',
            'official_and_environment_seconds_denominator': 'same verified contiguous audit prefix; no TensorBoard inference across audit gaps',
            'wall_seconds_denominator': 'entire finalized supervisor duration, including startup, excluded later audit rows, and any incomplete rollout',
            'wall_throughput_limit': 'Verified prefix transitions divided by whole finalized wall time; excludes all unverified or incomplete interactions, not total-interaction throughput.',
            'observed_audit_rows': len(epochs), 'included_audit_rows': count,
            'excluded_audit_rows': len(epochs) - count,
            'uncompleted_rollout_seconds': 'unknown; excluded from official completed-prefix PPO timing',
            'wall_duration_available': reliable_wall,
        },
    })
    return train


def checkpoint_records(directory, selected, issues):
    """Hash generated files; tensor inspection remains explicit separate evidence."""
    rows = []
    for path in sorted(directory.glob('*.pth')) if directory.is_dir() else []:
        record = identity(path)
        matches = [value for value in selected.values() if value.get('path') and
                   Path(value['path']).resolve() == path.resolve()]
        record['tensor_inspection'] = matches[0] if len(matches) == 1 else None
        record['inspection_hash_matches'] = (matches[0].get('sha256') == record['sha256']) if len(matches) == 1 else None
        if matches and record['inspection_hash_matches'] is not True:
            issues['errors'].append(f'Checkpoint changed after tensor inspection: {path}')
        rows.append(record)
    return rows


def amp_statistics(audit, epochs):
    """Summarize observed AMP/Adam counters without inferring gradients.

    Top-level counters are flushed by the read-only observer and may include
    work in an unfinished epoch.  Completed-epoch scale rows and cumulative
    optimizer counters are therefore reported separately.
    """
    def count(value):
        return value if type(value) is int and value >= 0 else None

    attempts = count(audit.get('actor_attempts'))
    optimizers = audit.get('optimizer_steps') or {}
    actor = count(optimizers.get('actor'))
    central = count(optimizers.get('central_value'))
    skips = audit.get('amp_skips')
    skip_events = skips if isinstance(skips, list) else None
    skipped = len(skip_events) if skip_events is not None else None
    scales = [{'epoch': row.get('epoch'), 'scale': row.get('amp_scale')}
              for row in epochs if finite_number(row.get('amp_scale'))]
    finite_scale_values = [row['scale'] for row in scales]
    missing_scale_epochs = [row.get('epoch') for row in epochs if not finite_number(row.get('amp_scale'))]
    last_cumulative = (epochs[-1].get('optimizer_steps_cumulative') if epochs else None)
    difference = attempts - actor if attempts is not None and actor is not None else None
    return {'mixed_precision': audit.get('runtime', {}).get('mixed_precision'),
            'actor_attempts': attempts, 'actor_optimizer_steps': actor,
            'central_value_optimizer_steps': central,
            'amp_skip_count': skipped, 'amp_skip_events': skip_events,
            'actor_attempts_minus_actor_steps': difference,
            'skip_count_matches_attempt_step_difference':
                difference == skipped if difference is not None and skipped is not None else None,
            'actor_applied_step_fraction': actor / attempts if attempts else None,
            'counter_scope': 'Actual observer top-level counters, including any recorded unfinished-epoch work; no count is inferred from expected minibatches.',
            'completed_epoch_amp_scale': {'statistics': descriptive(finite_scale_values),
                                          'first': scales[0] if scales else None,
                                          'last': scales[-1] if scales else None,
                                          'per_epoch': scales,
                                          'missing_or_nonfinite_scale_epochs': missing_scale_epochs},
            'last_completed_epoch_optimizer_steps_cumulative': last_cumulative,
            'gradient_finiteness_measured': False,
            'interpretation': 'AMP skips are observed GradScaler steps without an actor Adam update. Finite losses or scaler values do not establish per-gradient finiteness or scientific success.'}


def matched_comparison(current, seed0):
    """Compare identical 32-env checkpoints from training seeds 1 and 0.

    Evaluation seeds (1001/1002/1003) stay as per-seed rows and are never
    treated as additional training replicates.  The historical 64-env G2
    report is deliberately not loaded by this R5 analyzer.
    """
    past = {row['checkpoint_epoch']: row for row in (seed0 or {}).get('evaluation', {}).get('checkpoints', [])}
    rows = []
    for row in current['checkpoints']:
        epoch = row['checkpoint_epoch']
        if epoch not in EPOCHS:
            continue
        old = past.get(epoch)
        now_aggregate = row['aggregate']
        old_aggregate = old['aggregate'] if old else None
        difference = {}
        for key in ('ever_success_rate', 'terminal_success_rate'):
            a = now_aggregate.get(key)
            b = old_aggregate.get(key) if old_aggregate else None
            difference[key + '_percentage_points'] = (a - b) * 100 if a is not None and b is not None else None
        current_per_seed = {int(item['seed']): item for item in row.get('per_seed', []) if int(item.get('seed', -1)) in SEEDS}
        old_per_seed = {int(item['seed']): item for item in (old or {}).get('per_seed', []) if int(item.get('seed', -1)) in SEEDS}
        per_seed = []
        for seed in SEEDS:
            now = current_per_seed.get(seed, {})
            before = old_per_seed.get(seed, {})
            per_seed.append({'evaluation_seed': seed, 'seed1': now, 'seed0': before or None,
                             'ever_success_rate_difference_percentage_points':
                                 ((now.get('ever_success_rate') - before.get('ever_success_rate')) * 100
                                  if now.get('ever_success_rate') is not None and before.get('ever_success_rate') is not None else None),
                             'terminal_success_rate_difference_percentage_points':
                                 ((now.get('terminal_success_rate') - before.get('terminal_success_rate')) * 100
                                  if now.get('terminal_success_rate') is not None and before.get('terminal_success_rate') is not None else None)})
        rows.append({'checkpoint_epoch': epoch, 'training_seed1_epoch': epoch,
                     'training_seed0_epoch': epoch,
                     'matched_training_transitions': epoch * 4096,
                     'seed1_evaluation': now_aggregate, 'seed0_evaluation': old_aggregate,
                     'observed_rate_differences': difference,
                     'per_evaluation_seed': per_seed,
                     'denominators_pooled': False})
    return {'rows': rows,
            'seed0_protocol': 'G3C-R4 32 environments, training seed 0, 400 epochs, horizon128',
            'seed1_protocol': 'G3C-R5 32 environments, training seed 1, 400 epochs, horizon128',
            'training_seed_comparison': {'seed0': 0, 'seed1': 1, 'same_num_envs': True,
                                         'same_horizon': True, 'same_planned_transitions': True},
            'denominator_rule': 'Each training seed/checkpoint keeps its original per-evaluation-seed denominator; no seed pooling is performed.',
            'evaluation_seed_rule': 'Evaluation seeds 1001/1002/1003 are initial-condition seeds shared by both training runs, not training replicates.',
            'limits': ['Equal 32-env transitions do not imply identical stochastic trajectories; training seed is intentionally the only preregistered change.',
                       'Success-rate differences are descriptive and do not establish a population-level training-seed distribution from two runs.',
                       'The same evaluation seeds are paired protocol inputs, but their episode outcomes remain separate denominators.',
                       'Historical 64-env G2 evidence is excluded from this 32-env seed comparison.']}


def load_seed0_baseline(manifest, paths, issues):
    """Read the immutable R4 analysis report used for paired seed comparison."""
    configured = manifest.get('seed0_analysis_report') or manifest.get('seed0_baseline_report')
    candidate = Path(configured) if configured else paths['repository'] / 'evidence/g3c_r4/analysis/report.json'
    if not candidate.is_file():
        issues['warnings'].append('Seed-0 G3C-R4 analysis report is unavailable: ' + str(candidate))
        return {}, {'path': str(candidate), 'available': False}
    try:
        baseline = json.loads(candidate.read_text())
        if not isinstance(baseline, dict):
            raise ValueError('expected JSON object')
    except (OSError, ValueError, json.JSONDecodeError) as error:
        issues['errors'].append(f'Unreadable seed-0 baseline report: {candidate}: {error}')
        return {}, {'path': str(candidate), 'available': False}
    return baseline, {'path': str(candidate.resolve()), 'available': True,
                      'sha256': identity(candidate)['sha256']}


def completion_gates(paths, supervisor, audit, finalizer, completion, selected, evaluations, issues):
    failures = []

    def gate(label, action):
        try:
            action()
        except (ValueError, KeyError, TypeError, OSError) as error:
            failures.append(f'{label}: {error}')

    console = paths['supervisor'] / 'console.txt'
    console_text = console.read_text(errors='replace') if console.is_file() else ''
    gate('training', lambda: validate_training(audit, supervisor, finalizer, console_text))
    if completion.get('phase') != 'training_and_evaluation_complete' or completion.get('controller_exit_code') != 0:
        failures.append('completion controller has not confirmed training and all evaluations complete')
    if (completion.get('training') or {}).get('tensorboard_tail_complete') is not True:
        failures.append('strict final TensorBoard tail has not been confirmed')
    for epoch in EPOCHS:
        checkpoint = selected.get(str(epoch), {})
        gate(f'checkpoint epoch{epoch}', lambda checkpoint=checkpoint, epoch=epoch: validate_checkpoint(checkpoint, epoch))
        if not checkpoint.get('path') or not Path(checkpoint['path']).is_file():
            failures.append(f'checkpoint epoch{epoch}: artifact unavailable')
        elif checkpoint.get('sha256') != identity(Path(checkpoint['path']))['sha256']:
            failures.append(f'checkpoint epoch{epoch}: current SHA256 differs from inspection')
        for seed in SEEDS:
            folder = paths['evaluations'] / f'epoch{epoch:03d}' / f'seed{seed}'
            metrics_path = folder / 'metrics.json'
            status_path = folder / 'supervisor/status.json'
            if not metrics_path.is_file() or not status_path.is_file():
                failures.append(f'evaluation epoch{epoch}/seed{seed}: metrics or supervisor status unavailable')
                continue
            try:
                metrics = json.loads(metrics_path.read_text())
                status = json.loads(status_path.read_text())
            except (ValueError, OSError) as error:
                failures.append(f'evaluation epoch{epoch}/seed{seed}: {error}')
                continue
            gate(f'evaluation supervisor epoch{epoch}/seed{seed}', lambda status=status: validate_supervisor(status))
            gate(f'evaluation epoch{epoch}/seed{seed}', lambda metrics=metrics, checkpoint=checkpoint, epoch=epoch, seed=seed:
                 validate_evaluation(metrics, checkpoint, epoch, seed))
    if not all(row['complete_preregistered_seeds'] for row in evaluations['checkpoints'] if row['checkpoint_epoch'] in EPOCHS):
        failures.append('one or more preregistered evaluation denominators are incomplete')
    if issues['errors']:
        failures.append('analysis input errors remain')
    return {'complete': not failures, 'failed_requirements': failures,
            'scientific_success_rate_threshold_required': False,
            'loss_magnitude_or_convergence_threshold_required': False,
            'gpu_monitoring_coverage_required': False}


def analyze(manifest, *, tensorboard_python=None):
    issues = {'errors': [], 'warnings': []}
    paths = locations(manifest)
    configured_training_seed = manifest.get('training_seed', 1)
    if configured_training_seed != 1:
        issues['errors'].append('R5 analysis requires training_seed=1 in the frozen manifest.')
    supervisor = load_optional(paths['supervisor'] / 'status.json', issues, 'training supervisor status')
    launch = load_optional(paths['run_root'] / 'launch.json', issues, 'launch policy record')
    finalizer = load_optional(paths['finalizer'], issues, 'TensorBoard finalizer')
    completion = load_optional(paths['completion'] / 'status.json', issues, 'completion controller status')
    selected = load_optional(paths['completion'] / 'selected_checkpoints.json', issues, 'selected checkpoint inspection')
    if not selected:
        selected = completion.get('selected_checkpoints', {})
    try:
        scalars = read_scalars(paths['events'], Path(tensorboard_python) if tensorboard_python else None, issues)
    except (OSError, ValueError, RuntimeError) as error:
        issues['errors'].append(f'TensorBoard extraction failed: {error}')
        scalars = {'directory': str(paths['events']), 'files': [], 'tags': {}, 'all_logged_scalars_finite': None}
    try:
        audit, epochs = read_audit(paths['audit'], issues)
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues['errors'].append(f'Audit unreadable: {error}')
        audit, epochs = {}, []
    duration = supervisor.get('wall_seconds')
    monitor = {'wall_duration_seconds': duration,
               'wall_evidence': {'source': str(paths['supervisor'] / 'status.json'),
                                 'duration_scope': 'supervisor start through actual child wait and finalization'}}
    train = training_summary(scalars, audit, epochs, monitor, 32, 128, issues)
    train['training_seed'] = configured_training_seed
    train['evaluation_seed_set'] = list(SEEDS)
    train['seed_roles'] = {'training_seed': 'seed controlling the fresh PPO training run',
                           'evaluation_seeds': 'independent evaluator initial-condition seeds; not additional training runs'}
    # Keep the user-facing duration explicit.  ``training_summary`` retains
    # the established ``launched_process_wall_seconds`` field, but a direct
    # name avoids making readers infer which timing is the actual supervisor
    # duration when comparing runs.
    train['training_duration_seconds'] = supervisor.get('wall_seconds')
    train['training_duration_scope'] = 'durable supervisor start through actual child wait and finalization'
    train['verified_progress'] = completed_epoch_evidence(epochs)
    train['environment_transitions'] = train['verified_progress']['completed_transitions_verified']
    train['transition_count_source'] = 'verified_contiguous_32x128_epoch_audit_prefix'
    train = verified_throughput(train, epochs, supervisor)
    train['exit'] = {key: supervisor.get(key) for key in (
        'raw_wait_returncode', 'child_exit_code', 'child_signal', 'termination_reason',
        'error', 'stop_signals', 'received_supervisor_signals')}
    train['tensorboard_finalizer'] = finalizer
    console_path = paths['supervisor'] / 'console.txt'
    console_text = console_path.read_text(errors='replace') if console_path.is_file() else ''
    train['console_evidence'] = {'identity': identity(console_path) if console_path.is_file() else None,
        'last_120_lines_bounded_16000_chars': '\n'.join(console_text.splitlines()[-120:])[-16000:],
        'error_lines': [line for line in console_text.splitlines() if any(token in line for token in
                       ('Traceback (', 'CUDA out of memory', 'OutOfMemoryError', 'RuntimeError:', 'Error executing job'))][-30:]}
    loss_tags = {tag: values for tag, values in scalars['tags'].items()
                 if tag.startswith('losses/') or tag in ('info/kl', 'info/last_lr', 'info/cval_lr')}
    train['loss_stability'] = {
        'observed_epoch_loss_arrays_finite': (all(row.get('epoch_loss_arrays_finite') is True for row in epochs) if epochs else None),
        'logged_loss_and_optimization_scalars_finite': (all(point.get('finite') is True for values in loss_tags.values() for point in values) if loss_tags else None),
        'scalar_statistics': {tag: descriptive(scalar_values(scalars['tags'], tag)) for tag in loss_tags},
        'optimizer_steps': audit.get('optimizer_steps'), 'gradient_finiteness_measured': False,
        'limitation': 'Finite observed losses/parameters support numerical stability only. Loss magnitude does not establish convergence or task success; applied gradients are not measured.'}
    train['amp_statistics'] = amp_statistics(audit, epochs)
    evaluation = read_evaluations(paths['evaluations'], EPOCHS, SEEDS, 32, 3, issues)
    seed0, seed0_identity = load_seed0_baseline(manifest, paths, issues)
    inventory = checkpoint_records(paths['checkpoints'], selected, issues)
    policy_ok = (launch.get('monitoring_policy') == POLICY and launch.get('gpu_guard_enabled') is False
                 and launch.get('gpu_telemetry_required') is False
                 and launch.get('gpu_telemetry_gaps_terminate_child') is False)
    if not policy_ok:
        issues['errors'].append('Launch evidence does not establish the explicitly authorized no-GPU-guard policy.')
    gates = completion_gates(paths, supervisor, audit, finalizer, completion, selected, evaluation, issues)
    train['verified_progress']['possibly_uncompleted_rollout_transitions'] = (0 if gates['complete'] else 'unknown; not included in completed-epoch count')
    report = {
        'schema_version': 'g3cr5_analysis_v1',
        'engineering_status': 'COMPLETE' if gates['complete'] else 'INCOMPLETE',
        'scientific_status': ('ASSESSED_DESCRIPTIVELY' if gates['complete'] else 'NOT_ASSESSED_AS_COMPLETED_400_EPOCH_RUN'),
        'monitoring_policy': {'selected': launch.get('monitoring_policy'),
                              'explicit_user_authorization_recorded': policy_ok,
                              'independent_gpu_guard_enabled': False,
                              'gpu_telemetry_required': False,
                              'gpu_monitoring_coverage': 'not measured; no continuous-coverage claim',
                              'telemetry_missing_or_gap_is_failure': False,
                              'normal_cuda_error_handling': 'unchanged'},
        'training': train, 'training_seed': configured_training_seed,
        'evaluation_seeds': list(SEEDS),
        'training_seed_vs_evaluation_seed': 'Training seed 1 is the sole R5 training replicate; evaluation seeds 1001/1002/1003 are reused per checkpoint and remain separate denominators.',
        'checkpoint_inventory': inventory,
        'checkpoint_inspections': selected, 'completion_gates': gates,
        'evaluation': evaluation, 'seed0_comparison': matched_comparison(evaluation, seed0),
        'seed0_baseline_identity': seed0_identity,
        'epoch_csv_scope': 'completed_epochs.csv retains all observed audit rows; in_verified_contiguous_prefix marks rows included in verified transition totals and PPO timing.',
        'historical_input_issues': {'errors': [], 'warnings': ['Historical G2 64-env evidence intentionally excluded from G3C-R5 statistics.']}, 'issues': issues,
        'audit_identity': audit.get('input_identity'), 'audit_runtime': audit.get('runtime'),
        'audit_exit_error': audit.get('exit_error'), 'audit_instrumentation_errors': audit.get('instrumentation_errors'),
        'limits': ['No automatic training retry, resume, or concatenation is performed by this analyzer.',
                   'Unfinished evaluation episodes are excluded; missing results are null, not zero.',
                   'First-success timing is conditional on successful completed episodes.',
                   'G3B diagnostic episodes and prior interrupted runs remain outside all new denominators.',
                   'Hashing confirms artifact identity; tensor finiteness is known only for explicitly inspected checkpoints.']}
    return report, scalars, epochs


def write_csv(output, report, epochs):
    with (output / 'completed_epochs.csv').open('w', newline='') as stream:
        fields = ['epoch', 'environment_transitions', 'official_total_seconds', 'total_fps', 'epoch_loss_arrays_finite', 'in_verified_contiguous_prefix']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        prefix_count = report['training']['verified_progress']['completed_epochs_verified']
        writer.writerows({**row, 'in_verified_contiguous_prefix': index < prefix_count} for index, row in enumerate(epochs))
    with (output / 'evaluation_per_seed.csv').open('w', newline='') as stream:
        fields = ['checkpoint_epoch', 'seed', 'status', 'episodes_completed', 'ever_success_count', 'ever_success_rate',
                  'terminal_success_count', 'terminal_success_rate', 'conditional_first_success_n',
                  'conditional_first_success_step_median', 'conditional_first_success_seconds_median']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in report['evaluation']['checkpoints']:
            for seed in row['per_seed']:
                writer.writerow({'checkpoint_epoch': row['checkpoint_epoch'],
                    **{key: seed.get(key) for key in fields[1:8]},
                    'conditional_first_success_n': seed['first_success_step_conditional']['n'],
                    'conditional_first_success_step_median': seed['first_success_step_conditional']['median'],
                    'conditional_first_success_seconds_median': seed['first_success_time_seconds_conditional']['median']})
    with (output / 'seed0_comparison.csv').open('w', newline='') as stream:
        fields = ['checkpoint_epoch', 'training_seed1_epoch', 'training_seed0_epoch', 'matched_training_transitions',
                  'seed1_episodes', 'seed0_episodes', 'seed1_ever_rate', 'seed0_ever_rate',
                  'seed1_terminal_rate', 'seed0_terminal_rate']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in report['seed0_comparison']['rows']:
            now, old = row['seed1_evaluation'], row['seed0_evaluation'] or {}
            writer.writerow({'checkpoint_epoch': row['checkpoint_epoch'],
                'training_seed1_epoch': row['training_seed1_epoch'],
                'training_seed0_epoch': row['training_seed0_epoch'],
                'matched_training_transitions': row['matched_training_transitions'],
                'seed1_episodes': now['episodes_completed'], 'seed0_episodes': old.get('episodes_completed'),
                'seed1_ever_rate': now['ever_success_rate'], 'seed0_ever_rate': old.get('ever_success_rate'),
                'seed1_terminal_rate': now['terminal_success_rate'], 'seed0_terminal_rate': old.get('terminal_success_rate')})


def plots(output, report, scalars, epochs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25, 'savefig.bbox': 'tight'})
    artifacts = []

    def save(fig, name):
        for suffix in ('png', 'pdf'):
            path = output / f'{name}.{suffix}'; fig.savefig(path, dpi=160); artifacts.append(path.name)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    groups = [('rewards/iter', 'Reward; not success'), ('losses/a_loss', 'Actor loss'), ('losses/cval_loss', 'Central value loss'),
              ('losses/entropy', 'Entropy'), ('info/kl', 'KL divergence'), ('info/last_lr', 'Adaptive actor learning rate')]
    for ax, (tag, title) in zip(axes.flat, groups):
        points = [point for point in scalars['tags'].get(tag, []) if point['finite']]
        if points:
            ax.plot([point['step'] for point in points], [point['value'] for point in points], linewidth=1)
        else:
            ax.text(.5, .5, 'No logged data', ha='center', transform=ax.transAxes)
        ax.set(title=title, xlabel='Official TensorBoard step')
    fig.suptitle('G3C-R5 training scalars; finite loss is not a convergence test')
    fig.tight_layout(); save(fig, 'training_scalars')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rows = report['seed0_comparison']['rows']
    for ax, metric, title in zip(axes, ('ever_success_rate', 'terminal_success_rate'), ('Ever-success', 'Terminal-success')):
        for field, label, marker in (('seed1_evaluation', 'G3C-R5 seed 1', 'o'), ('seed0_evaluation', 'G3C-R4 seed 0', 's')):
            visible = [row for row in rows if row.get(field) and row[field].get(metric) is not None]
            if visible:
                ax.plot([row['matched_training_transitions'] for row in visible], [row[field][metric] for row in visible], marker=marker, label=label)
        ax.set(title=title, xlabel='Matched training transitions', ylabel='Completed-episode fraction', ylim=(-.03, 1.03))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(6, 6))
        if ax.lines: ax.legend(fontsize=8)
    fig.suptitle('Paired 32-env training seeds; separate evaluation denominators')
    fig.tight_layout(); save(fig, 'matched_transition_success')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for field, label, marker in (('seed1_evaluation', 'G3C-R5 seed 1', 'o'), ('seed0_evaluation', 'G3C-R4 seed 0', 's')):
        visible = [row for row in rows if row.get(field) and row[field]['first_success_time_seconds_conditional']['n']]
        if visible:
            ax.plot([row['matched_training_transitions'] for row in visible],
                    [row[field]['first_success_time_seconds_conditional']['median'] for row in visible], marker=marker, label=label)
    if ax.lines: ax.legend(fontsize=8)
    ax.set(title='First success: conditional median among successful completed episodes',
           xlabel='Matched training transitions', ylabel='Simulated seconds')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6, 6)); save(fig, 'first_success_timing')
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensorboard-python', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error('--output already exists; use a fresh directory')
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('completion_manifest'):
        completion_manifest = json.loads(Path(manifest['completion_manifest']).read_text())
        manifest = {**manifest, **completion_manifest}
    args.output.mkdir(parents=True, exist_ok=False)
    python = args.tensorboard_python or manifest.get('python') or manifest.get('tensorboard_python')
    report, scalars, epochs = analyze(manifest, tensorboard_python=python)
    report['manifest_identity'] = identity(args.manifest)
    report['analysis_script_identity'] = identity(Path(__file__))
    report['plots'] = [] if args.no_plots else plots(args.output, report, scalars, epochs)
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'tensorboard_scalars.json', scalars)
    write_csv(args.output, report, epochs)
    print(json.dumps({'output': str(args.output.resolve()), 'engineering_status': report['engineering_status'],
                      'completed_epochs': report['training']['verified_progress']['completed_epochs_verified'],
                      'issues': report['issues']}, allow_nan=False))
    # An interrupted experiment is a valid analyzable result, not an analyzer crash.
    return 1 if report['issues']['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
