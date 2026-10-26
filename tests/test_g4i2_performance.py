"""CPU-only mock timing controls; these are not GPU performance measurements."""
from unittest.mock import Mock
import math

import pytest

from scripts.g4i2_performance import DeadlineGuard, StepProfiler, distribution


class Clock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        self.ns += 1_000_000
        return self.ns


class Events:
    def __init__(self):
        self.count = 0
        self.recorded = []
        self.synchronized = []

    def event(self):
        self.count += 1
        return self.count

    def record(self, event):
        self.recorded.append(event)

    def synchronize(self, event):
        self.synchronized.append(event)

    def elapsed_ms(self, start, end):
        assert self.synchronized, 'elapsed read must follow synchronization'
        return (end - start) * .25


def test_distribution_independent_percentiles_and_population_std():
    result = distribution([4, 0, 2, 1, 3])
    assert result == {'n': 5, 'mean': 2, 'p50': 2, 'p95': 3.8,
                      'p99': 3.96, 'max': 4, 'std': math.sqrt(2)}


def test_distribution_empty_and_singleton():
    assert distribution([])['n'] == 0
    assert distribution([])['p99'] is None
    assert distribution([3]) == {'n': 1, 'mean': 3, 'p50': 3, 'p95': 3,
                                  'p99': 3, 'max': 3, 'std': 0}


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -1])
def test_distribution_rejects_invalid_measurements(value):
    with pytest.raises(ValueError):
        distribution([value])


def test_disabled_profile_calls_no_clock_or_cuda():
    backend, clock = Mock(), Mock()
    profiler = StepProfiler('off', cuda_backend=backend, clock_ns=clock)
    with profiler.step(0):
        with profiler.section('forward', cuda=True):
            pass
    assert profiler.rows == profiler.steps == []
    assert profiler.summarize()['measured_steps'] == 0
    clock.assert_not_called()
    assert not backend.mock_calls


def test_events_resolve_after_last_event_single_sync_per_step():
    backend = Events()
    profiler = StepProfiler('events', cuda_backend=backend, clock_ns=Clock())
    with profiler.step(1):
        with profiler.section('normalization', cuda=True):
            pass
        with profiler.section('forward', cuda=True):
            pass
        assert not backend.synchronized
    assert backend.recorded == [1, 2, 3, 4]
    assert backend.synchronized == [4]
    assert [row['device_ms'] for row in profiler.rows] == [.25, .25]
    assert all(row['wall_ns'] == 1_000_000 for row in profiler.rows)
    assert profiler.steps[0]['synchronization_wall_ns'] == 1_000_000
    assert profiler.summarize()['stages']['forward']['device_ms']['n'] == 1


def test_nested_scope_events_use_last_end_in_record_order():
    backend = Events()
    profiler = StepProfiler('events', cuda_backend=backend, clock_ns=Clock())
    with profiler.step(1):
        with profiler.section('obs_to_command', cuda=True):
            with profiler.section('forward', cuda=True):
                pass
    assert backend.recorded == [1, 3, 4, 2]
    assert backend.synchronized == [2]
    assert [row['stage'] for row in profiler.rows] == ['forward', 'obs_to_command']


def test_wall_only_stage_never_initializes_cuda():
    backend = Mock()
    profiler = StepProfiler('events', cuda_backend=backend, clock_ns=Clock())
    with profiler.step(1):
        with profiler.section('physics_step'):
            pass
    assert not backend.mock_calls
    assert profiler.rows[0]['device_ms'] is None
    assert profiler.summarize()['stages']['physics_step']['device_ms']['n'] == 0


def test_warmup_raw_preserved_summary_excluded():
    profiler = StepProfiler('events', cuda_backend=Events(), clock_ns=Clock())
    for index in range(4):
        with profiler.step(index, warmup=index < 2):
            with profiler.section('forward', cuda=True):
                pass
    report = profiler.summarize()
    assert len(profiler.rows) == 4
    assert report['warmup_steps'] == report['measured_steps'] == 2
    assert report['stages']['forward']['wall_ms']['n'] == 2
    assert report['stages']['forward']['device_ms']['n'] == 2


def test_failed_step_preserves_trace_excludes_completed_inner_stages():
    profiler = StepProfiler('events', cuda_backend=Events(), clock_ns=Clock())
    with pytest.raises(RuntimeError, match='injected'):
        with profiler.step(1):
            with profiler.section('normalization'):
                pass
            with profiler.section('forward', cuda=True):
                raise RuntimeError('injected')
    assert len(profiler.rows) == 2
    assert profiler.steps[0]['succeeded'] is False
    assert profiler.summarize()['stages']['normalization']['wall_ms']['n'] == 0
    assert profiler.summarize()['failed_steps'] == 1
    with profiler.step(2):
        pass


@pytest.mark.parametrize('invalid', [math.nan, math.inf, -1])
def test_invalid_cuda_measurement_is_not_reported_as_timing(invalid):
    backend = Events()
    backend.elapsed_ms = lambda start, end: invalid
    profiler = StepProfiler('events', cuda_backend=backend, clock_ns=Clock())
    with pytest.raises(ValueError, match='CUDA event'):
        with profiler.step(1):
            with profiler.section('forward', cuda=True):
                pass
    assert profiler.rows[0]['device_ms'] is None
    assert profiler._active is None


def test_section_and_step_schema():
    with pytest.raises(ValueError):
        StepProfiler('wall')
    profiler = StepProfiler('events', clock_ns=Clock())
    with pytest.raises(RuntimeError):
        with profiler.section('forward'):
            pass
    with pytest.raises(ValueError):
        with profiler.section('invented'):
            pass
    with profiler.step(1):
        with pytest.raises(RuntimeError):
            with profiler.step(2):
                pass
    with pytest.raises(ValueError):
        with profiler.step(1):
            pass


@pytest.mark.parametrize('mode,allowed,stop', [('log_only', True, False),
                                             ('reject', False, False), ('stop', False, True)])
def test_delayed_inference_modes_are_explicit(mode, allowed, stop):
    guard = DeadlineGuard(deadline_ms=10, max_observation_age_ms=100, mode=mode)
    decision = guard.check(observation_ns=0, inference_started_ns=0, now_ns=11_000_000)
    assert decision.allowed is allowed
    assert decision.stop_requested is stop
    assert decision.deadline_missed
    assert decision.reason == 'INFERENCE_DEADLINE_MISSED'
    assert guard.summarize()['miss_count'] == 1
    assert guard.summarize()['miss_rate'] == 1


def test_exact_deadline_on_time_and_miss_rate_denominator():
    guard = DeadlineGuard(deadline_ms=10, max_observation_age_ms=100)
    assert guard.check(observation_ns=0, inference_started_ns=0,
                       now_ns=10_000_000).allowed
    guard.check(observation_ns=20_000_000, inference_started_ns=20_000_000,
                now_ns=31_000_000)
    report = guard.summarize()
    assert report['checks'] == 2 and report['miss_rate'] == .5


@pytest.mark.parametrize('mode', ['log_only', 'reject', 'stop'])
def test_stale_always_rejects_even_fresh_fast_inference(mode):
    guard = DeadlineGuard(deadline_ms=10, max_observation_age_ms=20, mode=mode)
    decision = guard.check(observation_ns=0, inference_started_ns=30_000_000,
                           now_ns=31_000_000)
    assert not decision.allowed
    assert not decision.deadline_missed
    assert not decision.stop_requested
    assert decision.reason == 'STALE_OBSERVATION'


def test_stop_latched_requires_new_guard():
    guard = DeadlineGuard(deadline_ms=10, max_observation_age_ms=100, mode='stop')
    guard.check(observation_ns=0, inference_started_ns=0, now_ns=11_000_000)
    later = guard.check(observation_ns=20_000_000, inference_started_ns=20_000_000,
                        now_ns=21_000_000)
    assert not later.allowed and later.stop_requested and later.reason == 'STOP_LATCHED'


@pytest.mark.parametrize('values', [(1, 0, 2), (0, 3, 2), (-1, 0, 2),
                                    (0., 1, 2), (True, 1, 2)])
def test_invalid_or_future_timestamps_rejected(values):
    with pytest.raises(ValueError):
        DeadlineGuard().check(observation_ns=values[0], inference_started_ns=values[1],
                              now_ns=values[2])


@pytest.mark.parametrize('value', [math.nan, math.inf, 0, -1, True])
def test_invalid_deadline_configuration(value):
    with pytest.raises(ValueError):
        DeadlineGuard(deadline_ms=value)
    with pytest.raises(ValueError):
        DeadlineGuard(max_observation_age_ms=value)


def test_no_deadline_observations_not_zero_miss_claim():
    report = DeadlineGuard().summarize()
    assert report['checks'] == 0 and report['miss_rate'] is None


def test_actual_command_intervals_frequency_and_warmup_endpoint_filter():
    from scripts.g4i2_performance import command_interval_summary
    result = command_interval_summary([0, 100_000_000, 150_000_000, 250_000_000],
                                       warmup_commands=1, expected_interval_ms=60)
    assert result['raw_intervals_ms'] == [50, 100]
    assert result['wall_command_frequency_hz'] == pytest.approx(2 / .15)
    assert result['miss_count'] == 1 and result['miss_rate'] == .5
    assert result['interval_ms']['n'] == 2


@pytest.mark.parametrize('stamps', [[], [0]])
def test_too_few_commands_have_no_measured_rate(stamps):
    from scripts.g4i2_performance import command_interval_summary
    report = command_interval_summary(stamps)
    assert report['wall_command_frequency_hz'] is None
    assert report['miss_rate'] is None
    assert report['interval_ms']['n'] == 0


@pytest.mark.parametrize('stamps', [[1, 1], [2, 1], [-1, 1], [0., 1], [False, 1]])
def test_invalid_command_timestamps(stamps):
    from scripts.g4i2_performance import command_interval_summary
    with pytest.raises(ValueError):
        command_interval_summary(stamps)
