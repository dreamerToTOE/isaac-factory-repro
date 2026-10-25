from types import SimpleNamespace

import pytest

from scripts.success_metrics import PreResetSuccessCapture, SuccessTracker


def test_transient_success_is_not_terminal_success():
    tracker = SuccessTracker(1)
    tracker.update([True], [False], [False], [3])
    completed = tracker.update([False], [True], [True], [5])
    assert completed[0].first_success_step == 3
    assert completed[0].ever_success is True
    assert completed[0].terminal_success is False
    assert completed[0].terminal_kind == "timeout"


def test_success_on_timeout_is_terminal_success():
    tracker = SuccessTracker(1)
    completed = tracker.update([True], [True], [True], [7])
    assert completed[0].first_success_step == 7
    assert completed[0].ever_success is True
    assert completed[0].terminal_success is True


def test_episode_reset_clears_success_state():
    tracker = SuccessTracker(1)
    tracker.update([True], [True], [True], [4])
    completed = tracker.update([False], [True], [True], [2])
    assert completed[0].episode_id == 1
    assert completed[0].first_success_step is None
    assert completed[0].ever_success is False
    assert completed[0].terminal_success is False


def test_vector_envs_finalize_independently():
    tracker = SuccessTracker(2)
    completed = tracker.update([True, False], [True, False], [True, False], [5, 1])
    assert [(x.env_id, x.terminal_success) for x in completed] == [(0, True)]
    completed = tracker.update([False, True], [True, True], [True, True], [2, 3])
    assert [(x.env_id, x.episode_id, x.first_success_step) for x in completed] == [(0, 1, None), (1, 0, 3)]


def test_terminated_without_timeout_is_distinct():
    tracker = SuccessTracker(1)
    completed = tracker.update([False], [True], [False], [2])
    assert completed[0].terminal_kind == "terminated"


class AutoResetControl:
    """Synthetic order control; it does not model robotics or success geometry."""

    def __init__(self, success_history):
        self.cfg_task = SimpleNamespace(name="peg_insert", success_threshold=0.04, engage_threshold=0.9)
        self.episode_length_buf = [0]
        self.history = success_history
        self.current_mask = [False]
        self.predicate_calls = []

    def _get_curr_successes(self, success_threshold, check_rot=False):
        self.predicate_calls.append((success_threshold, check_rot))
        return self.current_mask

    def step(self):
        self.episode_length_buf[0] += 1
        self.current_mask[0] = self.history[self.episode_length_buf[0] - 1]
        # Mirrors the official reward-stage keyword calls, including a second
        # invocation for engagement. Capture must retain only the first mask.
        reward_success = self._get_curr_successes(success_threshold=0.04, check_rot=False)
        reward_identity = reward_success is self.current_mask
        self._get_curr_successes(success_threshold=0.9, check_rot=False)
        done = self.episode_length_buf[0] == len(self.history)
        if done:
            self.episode_length_buf[0] = 0
            self.current_mask[0] = False
        return self.current_mask, done, reward_identity


@pytest.mark.parametrize("history,ever,terminal,first", [
    ([False, True, False], True, False, 2),
    ([False, False, True], True, True, 3),
    ([False, False, False], False, False, None),
])
def test_capture_survives_auto_reset_and_excludes_engagement(history, ever, terminal, first):
    env = AutoResetControl(history)
    original = env._get_curr_successes
    tracker = SuccessTracker(1)
    with PreResetSuccessCapture(env) as capture:
        for _ in history:
            returned_mask, done, original_result_identity = env.step()
            snapshot = capture.consume()
            tracker.update(snapshot["success"], [done], [done], snapshot["step_numbers"])
            assert original_result_identity  # instrumentation returned the original object
        assert capture.captures_consumed == len(history)
    assert env._get_curr_successes == original
    assert returned_mask == [False]  # would produce a wrong terminal metric after reset
    assert tracker.completed[0].ever_success is ever
    assert tracker.completed[0].terminal_success is terminal
    assert tracker.completed[0].first_success_step == first
    assert tracker.completed[0].steps == len(history)
    assert len(env.predicate_calls) == 2 * len(history)


def test_capture_fails_closed_if_official_call_missing():
    env = AutoResetControl([False])
    original = env._get_curr_successes
    with pytest.raises(RuntimeError, match="expected one"):
        with PreResetSuccessCapture(env) as capture:
            capture.consume()
    assert env._get_curr_successes == original


def test_threshold_and_task_rotation_selection_are_not_changed():
    env = AutoResetControl([True])
    env.cfg_task.name = "nut_thread"
    with PreResetSuccessCapture(env) as capture:
        result = env._get_curr_successes(success_threshold=0.04, check_rot=True)
        assert result is env.current_mask
        env._get_curr_successes(success_threshold=0.9, check_rot=False)
        assert capture.consume()["check_rot"] is True
    assert env.predicate_calls == [(0.04, True), (0.9, False)]


def test_optional_geometry_callback_runs_before_reset_and_does_not_change_result():
    env = AutoResetControl([True])
    calls = []

    def snapshot(instance, record):
        calls.append((instance.episode_length_buf[:], record["success"][:]))
        return [{"episode_step": instance.episode_length_buf[0], "synthetic": True}]

    with PreResetSuccessCapture(env, snapshot=snapshot) as capture:
        returned, done, original_identity = env.step()
        result = capture.consume()
    assert original_identity and done
    assert returned == [False]
    assert calls == [([1], [True])]
    assert result["success"] == [True]
    assert result["geometric_snapshots"] == [{"episode_step": 1, "synthetic": True}]
