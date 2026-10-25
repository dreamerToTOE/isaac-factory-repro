"""Pure-Python episode success accounting for Factory checkpoint evaluation.

This module deliberately contains no Isaac Sim imports.  The evaluator feeds it
the official success mask captured inside ``FactoryEnv._get_rewards`` before the
parent DirectRLEnv performs its automatic reset.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Callable, Iterable, Sequence


class PreResetSuccessCapture:
    """Observe the official predicate result without reimplementing it.

    Factory calls this predicate for success and for engagement. Only the
    configured success threshold with the official task's rotation flag is
    selected. ``convert`` makes a detached snapshot; it must not mutate input.
    The default supports pure-Python synthetic tests; the evaluator supplies a
    Tensor-to-CPU converter. The original method's return value is unchanged.
    """

    def __init__(self, env, convert: Callable = copy.deepcopy, snapshot: Callable | None = None):
        self.env = env
        self.convert = convert
        self.original = env._get_curr_successes
        self.threshold = float(env.cfg_task.success_threshold)
        self.check_rot = env.cfg_task.name == "nut_thread"
        self.pending: list[dict] = []
        self.captures_consumed = 0
        self.snapshot = snapshot
        tap = self

        def instrumented(instance, success_threshold, check_rot=False):
            result = tap.original(success_threshold=success_threshold, check_rot=check_rot)
            if float(success_threshold) == tap.threshold and bool(check_rot) == tap.check_rot:
                record = {
                    "success": tap.convert(result),
                    "step_numbers": [int(x) for x in tap.convert(instance.episode_length_buf)],
                    "check_rot": bool(check_rot),
                }
                if tap.snapshot is not None:
                    record["geometric_snapshots"] = tap.snapshot(instance, record)
                tap.pending.append(record)
            return result

        self.instrumented = MethodType(instrumented, env)

    def __enter__(self):
        self.env._get_curr_successes = self.instrumented
        return self

    def consume(self) -> dict:
        if len(self.pending) != 1:
            raise RuntimeError(f"expected one official pre-reset success call, got {len(self.pending)}")
        self.captures_consumed += 1
        return self.pending.pop()

    def __exit__(self, exc_type, exc_value, traceback):
        self.env._get_curr_successes = self.original


@dataclass(frozen=True)
class EpisodeSummary:
    """Metrics for one completed environment episode."""

    env_id: int
    episode_id: int
    steps: int
    first_success_step: int | None
    ever_success: bool
    terminal_success: bool
    terminal_kind: str

    def to_dict(self) -> dict:
        return asdict(self)


class SuccessTracker:
    """Track transient and terminal success without changing task semantics.

    ``success`` must be the mask returned by the official
    ``_get_curr_successes(success_threshold=cfg_task.success_threshold, ...)``
    call for the current pre-reset step.  ``done`` is the combined terminated /
    truncated mask returned by the environment.  The tracker only finalizes an
    episode on ``done``; it never treats success itself as a terminal condition.
    """

    def __init__(self, num_envs: int):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self._episode_ids = [0] * self.num_envs
        self._steps = [0] * self.num_envs
        self._first_success = [None] * self.num_envs
        self._ever_success = [False] * self.num_envs
        self.completed: list[EpisodeSummary] = []

    @staticmethod
    def _bools(values: Iterable[bool], name: str, n: int) -> list[bool]:
        result = [bool(x) for x in values]
        if len(result) != n:
            raise ValueError(f"{name} length {len(result)} != num_envs {n}")
        return result

    def update(
        self,
        success: Sequence[bool],
        done: Sequence[bool],
        timeout: Sequence[bool] | None = None,
        step_numbers: Sequence[int] | None = None,
    ) -> list[EpisodeSummary]:
        """Add one vector environment step and return episodes finalized now.

        ``step_numbers`` are the official pre-reset episode counters when
        available.  Supplying them prevents reset-order ambiguity; otherwise
        the tracker uses a local 1-based count.
        """

        success_b = self._bools(success, "success", self.num_envs)
        done_b = self._bools(done, "done", self.num_envs)
        timeout_b = self._bools(timeout if timeout is not None else [False] * self.num_envs, "timeout", self.num_envs)
        if step_numbers is not None:
            if len(step_numbers) != self.num_envs:
                raise ValueError("step_numbers length does not match num_envs")
            steps = [int(x) for x in step_numbers]
            if any(x < 1 for x in steps):
                raise ValueError("step_numbers must be 1-based positive values")
        else:
            steps = [step + 1 for step in self._steps]

        finalized: list[EpisodeSummary] = []
        for env_id in range(self.num_envs):
            self._steps[env_id] = steps[env_id]
            if success_b[env_id]:
                self._ever_success[env_id] = True
                if self._first_success[env_id] is None:
                    self._first_success[env_id] = steps[env_id]
            if not done_b[env_id]:
                continue
            kind = "timeout" if timeout_b[env_id] else "terminated"
            finalized.append(
                EpisodeSummary(
                    env_id=env_id,
                    episode_id=self._episode_ids[env_id],
                    steps=steps[env_id],
                    first_success_step=self._first_success[env_id],
                    ever_success=self._ever_success[env_id],
                    terminal_success=success_b[env_id],
                    terminal_kind=kind,
                )
            )
            self.completed.append(finalized[-1])
            self._episode_ids[env_id] += 1
            self._steps[env_id] = 0
            self._first_success[env_id] = None
            self._ever_success[env_id] = False
        return finalized

    def aggregate(self, requested_episodes: int | None = None) -> dict:
        """Return aggregate rates over completed episodes."""

        n = len(self.completed)
        return {
            "episodes_completed": n,
            "episodes_requested": requested_episodes,
            "ever_success_count": sum(x.ever_success for x in self.completed),
            "ever_success_rate": (sum(x.ever_success for x in self.completed) / n if n else None),
            "terminal_success_count": sum(x.terminal_success for x in self.completed),
            "terminal_success_rate": (sum(x.terminal_success for x in self.completed) / n if n else None),
            "transient_only_count": sum(x.ever_success and not x.terminal_success for x in self.completed),
            "first_success_count": sum(x.first_success_step is not None for x in self.completed),
            "first_success_rate": (sum(x.first_success_step is not None for x in self.completed) / n if n else None),
            "mean_first_success_step": (
                sum(x.first_success_step for x in self.completed if x.first_success_step is not None)
                / sum(x.first_success_step is not None for x in self.completed)
                if any(x.first_success_step is not None for x in self.completed)
                else None
            ),
        }
