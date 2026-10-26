"""Streaming CPU-only timing export from completed registered G4-I2 runs.

Only timing records are retained; controller/action arrays are discarded after
one JSONL line. No Torch, GPU, simulator, policy or controller is imported.
Reference manifests contribute only A-events/A-minimal, never their B phases.
Output must be a new directory. Run only after all primary registered B runs end.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.g4i2_analysis import POLICY_INTERVAL_MS, statistics, summarize_timings

DECISION_FIELDS = ('run', 'role', 'seed', 'num_envs', 'policy_step', 'warmup',
                   'wall_command_monotonic_ns', 'control_loop_ms', 'sim_dt_s',
                   'timings_ms_json', 'step_timing_json', 'admission_json')
SCOPE_FIELDS = ('run', 'policy_step', 'warmup', 'scope_index', 'stage', 'wall_ns',
                'device_ms', 'succeeded', 'device_timing')


def _finite(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'nonfinite timing value: {label}')
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, label + '.' + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, label + f'[{index}]')


def _json(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def _run_specs(manifest_path, *, reference=False):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    result = []
    for name, phase in manifest['phases'].items():
        if reference and name not in ('a_events', 'a_minimal'):
            continue
        if not reference and not (name.startswith('b_') or name in ('a_events', 'a_minimal')):
            continue
        role = 'reference' if reference else ('evaluation' if name.startswith('b_') else 'bounded_validation')
        revision = str(manifest.get('revision', 'R0'))
        result.append({'run': ('reference_' if reference else '') + revision + '_' + name,
                       'phase': name, 'role': role, 'seed': phase['seed'],
                       'num_envs': phase['num_envs'], 'profiling': phase['profiling'],
                       'diagnostics': phase['diagnostics'],
                       'max_steps': phase['max_steps'],
                       'root': str(Path(phase['output']).resolve()),
                       'manifest': str(path), 'manifest_revision': revision})
    return result


def _require_completed(spec):
    root = Path(spec['root'])
    result = json.loads((root / 'result.json').read_text())
    outcome = json.loads((root / 'process_outcome.json').read_text())
    if result.get('status') != 'COMPLETED' or outcome.get('requested_exit_code') != 0:
        raise ValueError('run is not completed successfully: ' + spec['run'])
    if outcome.get('runtime_error_exists') or (root / 'runtime_error.json').exists():
        raise ValueError('run contains a runtime error: ' + spec['run'])
    actual_steps = result.get('policy_steps')
    maximum_steps = spec['max_steps']
    if type(maximum_steps) is not int or maximum_steps < 1:
        raise ValueError('registered max_steps must be a positive integer: ' + spec['run'])
    if type(actual_steps) is not int or not 0 < actual_steps <= maximum_steps:
        raise ValueError('actual result.policy_steps must be positive and within registered max_steps: ' + spec['run'])
    spec['actual_policy_steps'] = actual_steps
    if not (root / 'i2_trace.jsonl').is_file():
        raise ValueError('missing raw timing trace: ' + spec['run'])


def _read_timing_rows(spec, decision_writer, scope_writer):
    rows = []
    digest = hashlib.sha256()
    previous_step = -1
    with (Path(spec['root']) / 'i2_trace.jsonl').open('rb') as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get('kind') != 'timing':
                continue
            _finite(row, f'{spec["run"]}:line{line_number}')
            step = row.get('policy_step')
            if type(step) is not int or step <= previous_step:
                raise ValueError('timing steps must be distinct and increasing: ' + spec['run'])
            previous_step = step
            if type(row.get('warmup')) is not bool:
                raise ValueError('timing warmup must be bool')
            decision_writer.writerow({
                'run': spec['run'], 'role': spec['role'], 'seed': spec['seed'],
                'num_envs': spec['num_envs'], 'policy_step': step,
                'warmup': row['warmup'],
                'wall_command_monotonic_ns': row.get('wall_command_monotonic_ns'),
                'control_loop_ms': row.get('control_loop_ms'), 'sim_dt_s': row.get('sim_dt_s'),
                'timings_ms_json': _json(row.get('timings_ms', {})),
                'step_timing_json': _json(row.get('step_timing')),
                'admission_json': _json(row.get('admission')),
            })
            for index, section in enumerate(row.get('event_sections', [])):
                if type(section.get('succeeded')) is not bool:
                    raise ValueError('scope succeeded must be bool')
                wall = section.get('wall_ns')
                device = section.get('device_ms')
                if type(wall) is not int or wall < 0:
                    raise ValueError('scope wall_ns must be a nonnegative integer')
                if device is not None and (type(device) not in (int, float) or device < 0):
                    raise ValueError('scope device_ms must be finite nonnegative or null')
                if not isinstance(section.get('stage'), str):
                    raise ValueError('scope stage must be a string')
                if section.get('step', step) != step or section.get('warmup', row['warmup']) != row['warmup']:
                    raise ValueError('scope identity differs from decision identity')
                scope_writer.writerow({
                    'run': spec['run'], 'policy_step': step, 'warmup': row['warmup'],
                    'scope_index': index, 'stage': section['stage'], 'wall_ns': wall,
                    'device_ms': device, 'succeeded': section['succeeded'],
                    'device_timing': section.get('device_timing', 'UNKNOWN'),
                })
            rows.append(row)
    if len(rows) != spec['actual_policy_steps']:
        raise ValueError(f'{spec["run"]}: timing count {len(rows)} != completed result.policy_steps {spec["actual_policy_steps"]} (registered upper bound {spec["max_steps"]})')
    return rows, digest.hexdigest()


def _event_statistics(rows):
    wall, device = defaultdict(list), defaultdict(list)
    for row in rows:
        if row['warmup']:
            continue
        for section in row.get('event_sections', []):
            if not section['succeeded']:
                continue
            wall[section['stage']].append(section['wall_ns'] / 1e6)
            if section.get('device_ms') is not None:
                device[section['stage']].append(section['device_ms'])
    return {name: {'wall_ms': statistics(wall[name]), 'device_ms': statistics(device[name])}
            for name in sorted(set(wall) | set(device))}


def _admission_statistics(rows):
    decisions = [r['admission'] for r in rows if not r['warmup'] and r.get('admission') is not None]
    for row in decisions:
        for flag in ('allowed', 'deadline_missed', 'stop_requested'):
            if type(row.get(flag)) is not bool:
                raise ValueError('admission flags must be bool: ' + flag)
        for key in ('inference_wall_ms', 'observation_age_ms', 'deadline_ms'):
            if type(row.get(key)) not in (int, float) or row[key] < 0:
                raise ValueError('admission timing must be nonnegative: ' + key)
    misses = sum(row['deadline_missed'] for row in decisions)
    return {'checks': len(decisions), 'miss_count': misses,
            'miss_rate': misses / len(decisions) if decisions else None,
            'rejected_count': sum(not row['allowed'] for row in decisions),
            'stop_requested_count': sum(row['stop_requested'] for row in decisions),
            'reasons': dict(Counter(row['reason'] for row in decisions)),
            'actions': dict(Counter(row['action'] for row in decisions)),
            'inference_preparation_wall_ms': statistics(row['inference_wall_ms'] for row in decisions),
            'observation_age_ms': statistics(row['observation_age_ms'] for row in decisions),
            'scope': 'software wrapper admission, decision preparation clock; not hardware safety'}


def _pooled_timing(run_rows):
    """Reuse existing summaries; combine intervals only within each run."""
    flattened = [{k: v for k, v in row.items() if k != 'wall_command_monotonic_ns'}
                 for rows in run_rows for row in rows]
    result = summarize_timings(flattened)
    intervals = []
    per_run = [summarize_timings(rows) for rows in run_rows]
    for rows in run_rows:
        measured = [row for row in rows if not row['warmup']]
        for left, right in zip(measured, measured[1:]):
            a, b = left.get('wall_command_monotonic_ns'), right.get('wall_command_monotonic_ns')
            if a is not None and b is not None:
                intervals.append((b-a)/1e6)
    interval_stats = statistics(intervals)
    misses = sum(item['command_deadline_misses']['count'] for item in per_run)
    result['command_intervals_ms'] = interval_stats
    result['command_deadline_misses'] = {
        'count': misses, 'n': len(intervals), 'rate': misses/len(intervals) if intervals else None}
    result['wall_clock_command_frequency_hz'] = 1000/interval_stats['mean'] if intervals and interval_stats['mean'] > 0 else None
    result['interval_pooling'] = 'same-run adjacent postwarmup observations only; no cross-seed or cross-run interval'
    result['run_count'] = len(run_rows)
    result['control_loop_scope'] = 'full intervening host command interval; each run first postwarmup loop may have a warmup endpoint'
    result['real_time_factor_scope'] = 'nominal policy-time / aligned control-loop wall; excludes auxiliary reset simulation advancement'
    return result


def export_timings(manifest_path, output_dir, reference_manifest_path=None):
    specs = _run_specs(manifest_path)
    if reference_manifest_path is not None:
        specs += _run_specs(reference_manifest_path, reference=True)
    if not specs or len({spec['run'] for spec in specs}) != len(specs):
        raise ValueError('missing or duplicate run identities')
    for spec in specs:
        _require_completed(spec)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results, data = {}, {}
    with (output / 'timings_decision.csv').open('x', newline='') as decisions, (output / 'scopes.csv.gz').open('xb') as raw_gz:
        writer = csv.DictWriter(decisions, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        # Reproducible gzip header: no filename, deterministic mtime.
        with gzip.GzipFile(fileobj=raw_gz, mode='wb', filename='', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as scopes:
                scope_writer = csv.DictWriter(scopes, fieldnames=SCOPE_FIELDS)
                scope_writer.writeheader()
                for spec in specs:
                    rows, trace_sha256 = _read_timing_rows(spec, writer, scope_writer)
                    data[spec['run']] = rows
                    results[spec['run']] = {
                        'registration': spec, 'trace_sha256': trace_sha256,
                        'timing': summarize_timings(rows),
                        'event_scopes': _event_statistics(rows),
                        'admission': _admission_statistics(rows)}
    evaluation = [data[spec['run']] for spec in specs if spec['role'] == 'evaluation']
    merged = [row for rows in evaluation for row in rows]
    report = {
        'schema': 'g4i2.timing_export.v1', 'status': 'COMPLETE',
        'source_manifest': str(Path(manifest_path).resolve()),
        'reference_manifest': str(Path(reference_manifest_path).resolve()) if reference_manifest_path else None,
        'runs': results,
        'pooled_evaluation': {
            'timing': _pooled_timing(evaluation),
            'event_scopes': _event_statistics(merged),
            'admission': _admission_statistics(merged)},
        'reference_comparison_scope': 'A-events and A-minimal are descriptive instrumentation/diagnostics modes; no causal attribution to CUDA-event overhead alone; reference B phases excluded',
        'measurement_limits': {
            'device': 'same-stream elapsed including possible launch/host idle; not isolated pure kernel time',
            'observation_to_command': 'observed construction start to first software effort-target callback; not actuator execution',
            'h2d': 'empty host scope only; no policy H2D transfer exists',
            'd2h': 'diagnostic .cpu().tolist scope, not pure DMA or policy input transfer',
            'clock': 'actual host intervals include diagnostics, parity, reset and logging overhead; frequency is not inverse forward latency',
            'real_time_factor': 'nominal policy-time numerator; auxiliary reset physics excluded',
            'statistics': 'existing g4i2_analysis.statistics and summarize_timings; postwarmup successful event scopes only',
            'off': 'no probe clock/CUDA calls, but I2 diagnostics and outer host clocks remain',
        },
    }
    _finite(report, 'timing_summary')
    with (output / 'timing_summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path)
    args = parser.parse_args()
    report = export_timings(args.manifest, args.output, args.reference_manifest)
    print(_json({'status': report['status'], 'runs': len(report['runs']),
                 'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
