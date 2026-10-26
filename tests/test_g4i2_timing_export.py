"""Small synthetic streaming-export controls; never read real evaluation output."""
import csv
import gzip
import json
from pathlib import Path

import pytest

from scripts.g4i2_timing_export import export_timings


def make_manifest(tmp_path, name, run_starts, *, reference=False):
    phases = {}
    for phase, start in run_starts.items():
        root = tmp_path / (name + '_' + phase)
        root.mkdir()
        (root/'result.json').write_text(json.dumps({'status': 'COMPLETED', 'policy_steps': 3}))
        (root/'process_outcome.json').write_text(json.dumps({'requested_exit_code': 0}))
        rows = []
        for i in range(3):
            # Controller rows may be large but must not enter timing data.
            rows.append({'kind': 'controller', 'irrelevant': [1, 2, 3]})
            rows.append({'kind': 'timing', 'policy_step': i + 1, 'warmup': i == 0,
                'wall_command_monotonic_ns': start + i*70_000_000,
                'control_loop_ms': 70, 'sim_dt_s': 1/15,
                'timings_ms': {'forward': 1, 'observation_to_command': 2},
                'event_sections': [{'stage': 'forward', 'wall_ns': 1_000_000,
                                    'device_ms': .25, 'succeeded': True}],
                'step_timing': {'wall_ns': 50_000_000},
                'admission': {'allowed': True, 'deadline_missed': False,
                    'stop_requested': False, 'reason': 'ON_TIME', 'action': 'submit',
                    'inference_wall_ms': 1, 'observation_age_ms': 2, 'deadline_ms': 1000/15}})
        (root/'i2_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        phases[phase] = dict(seed=1001, num_envs=1, profiling='events',
                             diagnostics='full', max_steps=3, output=str(root))
    path = tmp_path/(name + '.json')
    path.write_text(json.dumps({'revision': name, 'phases': phases}))
    return path


def test_pooled_intervals_never_cross_seed_or_run_and_full_raw_scopes_retained(tmp_path):
    manifest = make_manifest(tmp_path, 'R1', {'b_1001': 10_000_000_000, 'b_1002': 0})
    output = tmp_path/'export'
    report = export_timings(manifest, output)
    pooled = report['pooled_evaluation']
    assert pooled['timing']['command_intervals_ms']['n'] == 2
    assert pooled['timing']['command_intervals_ms']['mean'] == 70
    assert pooled['timing']['command_deadline_misses'] == {'count': 2, 'n': 2, 'rate': 1.}
    assert pooled['timing']['wall_clock_command_frequency_hz'] == pytest.approx(1000/70)
    assert pooled['timing']['control_loop_ms']['n'] == 4
    assert pooled['timing']['real_time_factor'] == pytest.approx((1/15)/.07)
    assert pooled['event_scopes']['forward']['device_ms']['n'] == 4
    with (output/'timings_decision.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert json.loads(rows[0]['timings_ms_json'])['forward'] == 1
    with gzip.open(output/'scopes.csv.gz', 'rt') as stream:
        scopes = list(csv.DictReader(stream))
    assert len(scopes) == 6
    assert sum(r['warmup'] == 'True' for r in scopes) == 2


def test_reference_skips_failed_b_even_when_its_output_is_missing(tmp_path):
    manifest = make_manifest(tmp_path, 'R1', {'b_1001': 0})
    reference = make_manifest(tmp_path, 'R0', {'a_events': 0, 'a_minimal': 0})
    data = json.loads(reference.read_text())
    data['phases']['b_failed'] = {'output': str(tmp_path/'NEVER_CREATED')}
    reference.write_text(json.dumps(data))
    report = export_timings(manifest, tmp_path/'export', reference)
    assert len(report['runs']) == 3
    assert report['pooled_evaluation']['timing']['measured_records'] == 2
    assert not any('failed' in key for key in report['runs'])


def test_existing_output_never_overwritten_and_running_input_rejected(tmp_path):
    manifest = make_manifest(tmp_path, 'R1', {'b_1001': 0})
    existing = tmp_path/'existing'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        export_timings(manifest, existing)
    phase = json.loads(manifest.read_text())['phases']['b_1001']
    (Path(phase['output'])/'result.json').write_text(json.dumps({'status': 'RUNNING'}))
    with pytest.raises(ValueError, match='not completed'):
        export_timings(manifest, tmp_path/'not_created')
    assert not (tmp_path/'not_created').exists()


def test_nonfinite_timing_is_rejected_not_exported_as_valid_summary(tmp_path):
    manifest = make_manifest(tmp_path, 'R1', {'b_1001': 0})
    phase = json.loads(manifest.read_text())['phases']['b_1001']
    trace = Path(phase['output'])/'i2_trace.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[1]['event_sections'][0]['device_ms'] = float('nan')
    trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='nonfinite'):
        export_timings(manifest, tmp_path/'export')
    assert not (tmp_path/'export/timing_summary.json').exists()


@pytest.mark.parametrize('phase', ['a_minimal', 'b_1001'])
def test_successful_early_completion_uses_actual_steps_with_registered_upper_bound(tmp_path, phase):
    # Registration is an execution cap, while the independent process result
    # records when the requested episode denominator was actually completed.
    manifest = make_manifest(tmp_path, 'R1', {phase: 0})
    registration = json.loads(manifest.read_text())
    registration['phases'][phase]['max_steps'] = 9
    manifest.write_text(json.dumps(registration))
    report = export_timings(manifest, tmp_path/'export')
    result = next(iter(report['runs'].values()))
    assert result['registration']['max_steps'] == 9
    assert result['registration']['actual_policy_steps'] == 3
    assert result['timing']['records_total'] == 3


@pytest.mark.parametrize('actual', [0, -1, 4, True, 1.5, None])
def test_actual_process_steps_must_be_positive_integer_within_registered_cap(tmp_path, actual):
    manifest = make_manifest(tmp_path, 'R1', {'b_1001': 0})
    phase = json.loads(manifest.read_text())['phases']['b_1001']
    (Path(phase['output'])/'result.json').write_text(
        json.dumps({'status': 'COMPLETED', 'policy_steps': actual}))
    with pytest.raises(ValueError, match='actual result.policy_steps'):
        export_timings(manifest, tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_timing_rows_must_match_actual_process_result_even_below_registered_cap(tmp_path):
    manifest = make_manifest(tmp_path, 'R1', {'b_1001': 0})
    registration = json.loads(manifest.read_text())
    registration['phases']['b_1001']['max_steps'] = 9
    manifest.write_text(json.dumps(registration))
    phase = registration['phases']['b_1001']
    (Path(phase['output'])/'result.json').write_text(
        json.dumps({'status': 'COMPLETED', 'policy_steps': 2}))
    with pytest.raises(ValueError, match='timing count 3 != completed result.policy_steps 2'):
        export_timings(manifest, tmp_path/'export')
    assert not (tmp_path/'export/timing_summary.json').exists()
