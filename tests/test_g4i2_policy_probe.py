"""Optional timing does not change the existing actor; CPU mock probes only."""
from contextlib import contextmanager

import pytest
import torch

from scripts.g4i1_policy import FrozenPolicy
from scripts.g4i2_performance import StepProfiler
from test_g4i1_policy import MockModel
from test_g4i2_performance import Clock, Events


@pytest.mark.parametrize('consume_rng', [True, False])
def test_profiled_and_unprofiled_outputs_states_and_rng_exactly_equal(consume_rng):
    policy = FrozenPolicy(MockModel(), {'fixture': True})
    obs = torch.linspace(-1, 1, 19).reshape(1, 19)
    state = policy.initial_state()
    before_obs = obs.clone()
    torch.manual_seed(1001)
    before_rng = torch.get_rng_state()
    plain = policy.infer(obs, state, consume_rng=consume_rng)
    plain_rng = torch.get_rng_state()
    torch.set_rng_state(before_rng)
    profiler = StepProfiler('events', cuda_backend=Events(), clock_ns=Clock())
    with profiler.step(0):
        profiled = policy.infer(obs, state, consume_rng=consume_rng, profiler=profiler)
    assert torch.equal(plain_rng, torch.get_rng_state())
    for name in ('observation', 'normalized', 'mu', 'logstd', 'clipped_action', 'h', 'c'):
        assert torch.equal(getattr(plain, name), getattr(profiled, name))
    assert plain.consumed_discarded_sample is profiled.consumed_discarded_sample
    assert torch.equal(obs, before_obs)
    assert all(torch.count_nonzero(value) == 0 for value in state)
    assert policy.rms_unchanged()
    assert [row['stage'] for row in profiler.rows] == ['normalization', 'forward', 'action_postprocess']
    for name in ('normalization', 'forward', 'action_postprocess'):
        assert profiler.summarize()['stages'][name]['device_ms']['n'] == 1


def test_default_and_disabled_profiler_keep_behavior_without_cuda_access():
    from unittest.mock import Mock
    backend, clock = Mock(), Mock()
    profiler = StepProfiler('off', cuda_backend=backend, clock_ns=clock)
    policy = FrozenPolicy(MockModel(), {'fixture': True})
    obs = torch.zeros(1, 19)
    result = policy.infer(obs, policy.initial_state(), consume_rng=False, profiler=profiler)
    assert result.mu.shape == (1, 6)
    assert not backend.mock_calls
    clock.assert_not_called()


def test_probe_order_surrounds_existing_network_forward_without_reimplementing_it():
    events = []
    class Probe:
        @contextmanager
        def section(self, name, *, cuda):
            assert cuda is True
            events.append(('enter', name))
            try:
                yield
            finally:
                events.append(('exit', name))
    policy = FrozenPolicy(MockModel(), {'fixture': True})
    handle = policy.model.a2c_network.register_forward_hook(lambda *unused: events.append(('actor', 'forward')))
    try:
        policy.infer(torch.zeros(1, 19), policy.initial_state(), consume_rng=False, profiler=Probe())
    finally:
        handle.remove()
    assert events == [('enter', 'normalization'), ('exit', 'normalization'),
                      ('enter', 'forward'), ('actor', 'forward'), ('exit', 'forward'),
                      ('enter', 'action_postprocess'), ('exit', 'action_postprocess')]
