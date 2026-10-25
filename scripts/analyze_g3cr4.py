#!/usr/bin/env python3
"""CPU-only G3C-R4 analysis, retaining partial evidence and separate denominators.

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
    from complete_g3cr4 import (validate_checkpoint, validate_evaluation,
                               validate_supervisor, validate_training)
except ImportError:
    from scripts.analyze_g2 import (descriptive, finite_number, identity, read_audit,
                                    read_evaluations, read_scalars, scalar_values,
                                    training_summary, write_json)
    from scripts.complete_g3cr4 import (validate_checkpoint, validate_evaluation,
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


def matched_comparison(current, historical):
    past = {row['checkpoint_epoch']: row for row in historical['checkpoints']}
    rows = []
    for row in current['checkpoints']:
        epoch = row['checkpoint_epoch']
        if epoch not in EPOCHS:
            continue
        old = past.get(epoch // 2)
        now_aggregate = row['aggregate']
        old_aggregate = old['aggregate'] if old else None
        difference = {}
        for key in ('ever_success_rate', 'terminal_success_rate'):
            a = now_aggregate.get(key)
            b = old_aggregate.get(key) if old_aggregate else None
            difference[key + '_percentage_points'] = (a - b) * 100 if a is not None and b is not None else None
        rows.append({'new_32env_epoch': epoch, 'historical_64env_epoch': epoch // 2,
                     'matched_training_transitions': epoch * 4096,
                     'new_evaluation': now_aggregate, 'historical_evaluation': old_aggregate,
                     'observed_rate_differences': difference,
                     'denominators_pooled': False})
    return {'rows': rows, 'historical_protocol': 'G2 64 environments, training seed 0, 200 epochs, horizon128',
            'new_protocol': 'G3C-R4 32 environments, training seed 0, 400 epochs, horizon128',
            'denominator_rule': 'Each checkpoint/protocol keeps its original evaluation denominator; G3B diagnostic and interrupted runs are excluded.',
            'limits': ['Equal transitions do not imply identical training trajectories: rollout batches are 4096 versus 8192 transitions.',
                       'Keeping minibatch512 and four mini-epochs changes optimizer update grouping and on-policy data collection.',
                       'One training seed in each protocol does not quantify training-seed variability or isolate a causal environment-count effect.',
                       'Evaluation seeds 1001/1002/1003 are independent initial-condition seeds, not independent training runs.',
                       'Success-rate differences are descriptive; no pooled denominator or significance claim is made.']}


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
    train['verified_progress'] = completed_epoch_evidence(epochs)
    train['environment_transitions'] = train['verified_progress']['completed_transitions_verified']
    train['transition_count_source'] = 'verified_contiguous_32x128_epoch_audit_prefix'
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
    evaluation = read_evaluations(paths['evaluations'], EPOCHS, SEEDS, 32, 3, issues)
    historical_issues = {'errors': [], 'warnings': []}
    historical = read_evaluations(paths['repository'] / 'evidence/g2/evaluation', [50, 100, 150, 200], SEEDS, 32, 3, historical_issues)
    inventory = checkpoint_records(paths['checkpoints'], selected, issues)
    policy_ok = (launch.get('monitoring_policy') == POLICY and launch.get('gpu_guard_enabled') is False
                 and launch.get('gpu_telemetry_required') is False
                 and launch.get('gpu_telemetry_gaps_terminate_child') is False)
    if not policy_ok:
        issues['errors'].append('Launch evidence does not establish the explicitly authorized no-GPU-guard policy.')
    gates = completion_gates(paths, supervisor, audit, finalizer, completion, selected, evaluation, issues)
    train['verified_progress']['possibly_uncompleted_rollout_transitions'] = (0 if gates['complete'] else 'unknown; not included in completed-epoch count')
    report = {
        'schema_version': 'g3cr4_analysis_v1',
        'engineering_status': 'COMPLETE' if gates['complete'] else 'INCOMPLETE',
        'scientific_status': ('ASSESSED_DESCRIPTIVELY' if gates['complete'] else 'NOT_ASSESSED_AS_COMPLETED_400_EPOCH_RUN'),
        'monitoring_policy': {'selected': launch.get('monitoring_policy'),
                              'explicit_user_authorization_recorded': policy_ok,
                              'independent_gpu_guard_enabled': False,
                              'gpu_telemetry_required': False,
                              'gpu_monitoring_coverage': 'not measured; no continuous-coverage claim',
                              'telemetry_missing_or_gap_is_failure': False,
                              'normal_cuda_error_handling': 'unchanged'},
        'training': train, 'checkpoint_inventory': inventory,
        'checkpoint_inspections': selected, 'completion_gates': gates,
        'evaluation': evaluation, 'historical_matched_transition_comparison': matched_comparison(evaluation, historical),
        'historical_input_issues': historical_issues, 'issues': issues,
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
        fields = ['epoch', 'environment_transitions', 'official_total_seconds', 'total_fps', 'epoch_loss_arrays_finite']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(epochs)
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
    with (output / 'matched_transition_comparison.csv').open('w', newline='') as stream:
        fields = ['new_32env_epoch', 'historical_64env_epoch', 'matched_training_transitions',
                  'new_episodes', 'historical_episodes', 'new_ever_rate', 'historical_ever_rate',
                  'new_terminal_rate', 'historical_terminal_rate']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in report['historical_matched_transition_comparison']['rows']:
            now, old = row['new_evaluation'], row['historical_evaluation'] or {}
            writer.writerow({**{key: row[key] for key in fields[:3]},
                'new_episodes': now['episodes_completed'], 'historical_episodes': old.get('episodes_completed'),
                'new_ever_rate': now['ever_success_rate'], 'historical_ever_rate': old.get('ever_success_rate'),
                'new_terminal_rate': now['terminal_success_rate'], 'historical_terminal_rate': old.get('terminal_success_rate')})


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
    fig.suptitle('G3C-R4 training scalars; finite loss is not a convergence test')
    fig.tight_layout(); save(fig, 'training_scalars')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rows = report['historical_matched_transition_comparison']['rows']
    for ax, metric, title in zip(axes, ('ever_success_rate', 'terminal_success_rate'), ('Ever-success', 'Terminal-success')):
        for field, label, marker in (('new_evaluation', 'G3C-R4 32 env', 'o'), ('historical_evaluation', 'Historical G2 64 env', 's')):
            visible = [row for row in rows if row.get(field) and row[field].get(metric) is not None]
            if visible:
                ax.plot([row['matched_training_transitions'] for row in visible], [row[field][metric] for row in visible], marker=marker, label=label)
        ax.set(title=title, xlabel='Matched training transitions', ylabel='Completed-episode fraction', ylim=(-.03, 1.03))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(6, 6))
        if ax.lines: ax.legend(fontsize=8)
    fig.suptitle('Separate evaluation denominators; one training seed per protocol')
    fig.tight_layout(); save(fig, 'matched_transition_success')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for field, label, marker in (('new_evaluation', 'G3C-R4 32 env', 'o'), ('historical_evaluation', 'Historical G2 64 env', 's')):
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
