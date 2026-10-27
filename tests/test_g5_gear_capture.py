import pytest
from scripts.g5_gear_capture import ResetCapture


class FakeEnv:
    def __init__(self):
        self.calls = []
    def _reset_idx(self, ids):
        self.calls.append(ids)
        if ids == ['fail']:
            raise RuntimeError('reset failed')
        return 'unchanged result'


def test_reset_tap_delegates_records_and_restores():
    env=FakeEnv()
    with ResetCapture(env, list) as tap:
        tap.begin_decision()
        assert env._reset_idx([0]) == 'unchanged result'
        assert env.calls == [[0]] and tap.events == [[0]]
        tap.begin_decision()
        assert tap.events == []
        with pytest.raises(RuntimeError):
            env._reset_idx(['fail'])
        assert tap.events == []
    assert env._reset_idx([1]) == 'unchanged result'
    assert env.calls == [[0], ['fail'], [1]] and tap.events == []
