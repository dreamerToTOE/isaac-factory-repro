"""Optional G4-I2 timing probes and software deadline admission.

No simulator or policy is constructed here. ``off`` makes no timer/CUDA calls;
``events`` records host wall scopes and optional same-stream CUDA event scopes.
CUDA events are resolved once at the end of each step, after synchronizing the
last recorded end event. This synchronization changes runtime scheduling: event
runs are profiled runs, never an uninstrumented-performance claim. Host section
measurements include event creation/record overhead and any blocking inside the
section, but exclude deferred step-end synchronization. Device elapsed time is
stream elapsed time between events (including possible host-induced idle gaps),
not a sum of isolated kernel execution durations. Nested stages overlap.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import statistics
from time import perf_counter_ns
from typing import Callable

STAGES = frozenset({
    "observation_construction", "normalization", "cpu_gpu_transfer_h2d",
    "cpu_gpu_transfer_d2h", "forward", "action_postprocess", "ema",
    "controller_target", "physics_step", "obs_to_command", "control_loop",
})


def distribution(values):
    """Finite-only population statistics; empty distributions have null metrics.

    Percentiles use linear interpolation at rank (n - 1) * p, matching NumPy's
    default linear convention. The standard deviation is population (ddof=0).
    """
    data = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in data):
        raise ValueError("timing samples must be finite and nonnegative")
    result = {key: None for key in ("mean", "p50", "p95", "p99", "max", "std")}
    result["n"] = len(data)
    if not data:
        return result

    def percentile(p):
        rank = (len(data) - 1) * p
        lower, upper = math.floor(rank), math.ceil(rank)
        return data[lower] + (data[upper] - data[lower]) * (rank - lower)

    result.update(mean=statistics.fmean(data), p50=percentile(.50),
                  p95=percentile(.95), p99=percentile(.99), max=data[-1],
                  std=statistics.pstdev(data))
    return result


class TorchCudaEvents:
    """Real CUDA backend; imported/created lazily only for CUDA-enabled scopes."""

    def __init__(self, device):
        import torch
        self.torch, self.device = torch, torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA event profiling requires an available CUDA device")

    def event(self):
        with self.torch.cuda.device(self.device):
            return self.torch.cuda.Event(enable_timing=True)

    def record(self, event):
        event.record(self.torch.cuda.current_stream(self.device))

    def synchronize(self, event):
        event.synchronize()

    @staticmethod
    def elapsed_ms(start, end):
        return start.elapsed_time(end)


class StepProfiler:
    """Per-step optional profiler, injectable for CPU-only deterministic tests.

    Usage: ``with probe.step(1, warmup=True):`` then
    ``with probe.section('forward', cuda=True): ...``. Only a single calling
    thread and CUDA stream per step are supported. All raw rows, including warmup
    and failed sections, remain available; summaries exclude warmup/failed rows.
    The probe never copies tensors or changes their dtype/device.
    """

    def __init__(self, mode="off", *, device="cuda:0", cuda_backend=None,
                 clock_ns: Callable[[], int] = perf_counter_ns):
        if mode not in ("off", "events"):
            raise ValueError("profiling mode must be off or events")
        self.mode, self.device = mode, device
        self._backend, self._clock = cuda_backend, clock_ns
        self.rows, self.steps = [], []
        self._active = None
        self._pending = []
        self._open_sections = 0
        self._seen_steps = set()

    def _cuda(self):
        if self._backend is None:
            self._backend = TorchCudaEvents(self.device)
        return self._backend

    @contextmanager
    def step(self, step_id, *, warmup=False):
        if self.mode == "off":
            yield self
            return
        if self._active is not None:
            raise RuntimeError("profiling steps cannot be nested")
        if type(step_id) is not int or step_id < 0 or step_id in self._seen_steps:
            raise ValueError("step_id must be a unique nonnegative integer")
        if type(warmup) is not bool:
            raise ValueError("warmup must be bool")
        self._seen_steps.add(step_id)
        self._active = {"step": step_id, "warmup": warmup}
        self._pending = []
        start = self._clock()
        succeeded = False
        try:
            yield self
            succeeded = True
        finally:
            sync_start = self._clock()
            sync_ns = 0
            try:
                if self._pending:
                    backend = self._cuda()
                    # End records append in completion order, including nested
                    # scopes; the last end event follows all earlier events.
                    backend.synchronize(self._pending[-1][2])
                    sync_ns = self._clock() - sync_start
                    if sync_ns < 0:
                        raise ValueError("monotonic profiling clock moved backwards")
                    for row, begin, end in self._pending:
                        elapsed = float(backend.elapsed_ms(begin, end))
                        if not math.isfinite(elapsed) or elapsed < 0:
                            raise ValueError("CUDA event elapsed time must be finite and nonnegative")
                        row["device_ms"] = elapsed
                end = self._clock()
                if end < start:
                    raise ValueError("monotonic profiling clock moved backwards")
                self.steps.append({**self._active, "wall_ns": end - start,
                                   "synchronization_wall_ns": sync_ns,
                                   "succeeded": succeeded})
            finally:
                self._active = None
                self._pending = []

    @contextmanager
    def section(self, name, *, cuda=False):
        if name not in STAGES:
            raise ValueError("unknown timing stage: " + str(name))
        if type(cuda) is not bool:
            raise ValueError("cuda must be bool")
        if self.mode == "off":
            yield
            return
        if self._active is None:
            raise RuntimeError("timing section requires an active step")
        start = self._clock()
        begin = end = None
        if cuda:
            backend = self._cuda()
            begin, end = backend.event(), backend.event()
            backend.record(begin)
        succeeded = False
        self._open_sections += 1
        try:
            yield
            succeeded = True
        finally:
            self._open_sections -= 1
            if cuda:
                backend.record(end)
            wall_ns = self._clock() - start
            if wall_ns < 0:
                raise ValueError("monotonic profiling clock moved backwards")
            row = {**self._active, "stage": name, "wall_ns": wall_ns,
                   "device_ms": None, "succeeded": succeeded,
                   "device_timing": "same_stream_cuda_events" if cuda else "not_requested"}
            self.rows.append(row)
            if cuda:
                self._pending.append((row, begin, end))

    def summarize(self):
        result = {}
        valid_steps = {row["step"] for row in self.steps if row["succeeded"]}
        for stage in sorted(STAGES):
            selected = [row for row in self.rows if row["stage"] == stage and
                        not row["warmup"] and row["succeeded"] and row["step"] in valid_steps]
            result[stage] = {
                "wall_ms": distribution(row["wall_ns"] / 1e6 for row in selected),
                "device_ms": distribution(row["device_ms"] for row in selected
                                          if row["device_ms"] is not None),
            }
        return {"mode": self.mode, "units": "milliseconds", "stages": result,
                "warmup_steps": sum(row["warmup"] for row in self.steps),
                "measured_steps": sum(not row["warmup"] and row["succeeded"] for row in self.steps),
                "failed_steps": sum(not row["succeeded"] for row in self.steps),
                "deferred_synchronization_ms": distribution(
                    row["synchronization_wall_ns"] / 1e6 for row in self.steps
                    if not row["warmup"] and row["succeeded"]),
                "scope": "profiling overhead included; not uninstrumented performance"}


@dataclass(frozen=True)
class DeadlineDecision:
    allowed: bool
    deadline_missed: bool
    stop_requested: bool
    reason: str
    action: str
    observation_age_ms: float
    inference_wall_ms: float
    deadline_ms: float

    def to_dict(self):
        return asdict(self)


class DeadlineGuard:
    """Software admission only; it does not hold a robot or certify safety.

    Default log_only preserves scientific simulator parity on a late, fresh
    inference. reject blocks only that command; stop requests caller shutdown and
    latches rejection. Stale/future observations always reject. The runtime owns
    monotonic timestamps, command submission and actual shutdown. No synthetic
    delay is inserted here. A deadline is missed only when elapsed > deadline.
    """

    def __init__(self, *, deadline_ms=1000 / 15, max_observation_age_ms=1000 / 15,
                 mode="log_only"):
        if mode not in ("log_only", "reject", "stop"):
            raise ValueError("unknown deadline policy")
        if any(type(value) is bool or not math.isfinite(float(value)) or float(value) <= 0
               for value in (deadline_ms, max_observation_age_ms)):
            raise ValueError("deadline and observation age limit must be finite positive milliseconds")
        self.deadline_ms = float(deadline_ms)
        self.max_observation_age_ms = float(max_observation_age_ms)
        self.mode, self.stopped = mode, False
        self.decisions = []

    def check(self, *, observation_ns, inference_started_ns, now_ns):
        if any(type(value) is not int or value < 0
               for value in (observation_ns, inference_started_ns, now_ns)):
            raise ValueError("timestamps must be nonnegative integer monotonic nanoseconds")
        if inference_started_ns > now_ns or observation_ns > inference_started_ns:
            raise ValueError("observation/inference timestamps are future or unordered")
        age_ms = (now_ns - observation_ns) / 1e6
        elapsed_ms = (now_ns - inference_started_ns) / 1e6
        missed = elapsed_ms > self.deadline_ms
        stale = age_ms > self.max_observation_age_ms
        if self.stopped:
            allowed, stop, reason, action = False, True, "STOP_LATCHED", "stop"
        elif stale:
            allowed, stop, reason, action = False, False, "STALE_OBSERVATION", "reject"
        elif missed and self.mode == "stop":
            self.stopped = True
            allowed, stop, reason, action = False, True, "INFERENCE_DEADLINE_MISSED", "stop"
        elif missed and self.mode == "reject":
            allowed, stop, reason, action = False, False, "INFERENCE_DEADLINE_MISSED", "reject"
        else:
            allowed, stop = True, False
            reason = "INFERENCE_DEADLINE_MISSED" if missed else "ON_TIME"
            action = "log_only" if missed else "submit"
        decision = DeadlineDecision(allowed, missed, stop, reason, action,
                                    age_ms, elapsed_ms, self.deadline_ms)
        self.decisions.append(decision.to_dict())
        return decision

    def summarize(self):
        count = len(self.decisions)
        misses = sum(row["deadline_missed"] for row in self.decisions)
        return {"mode": self.mode, "deadline_ms": self.deadline_ms,
                "max_observation_age_ms": self.max_observation_age_ms,
                "checks": count, "miss_count": misses,
                "miss_rate": misses / count if count else None,
                "rejected_count": sum(not row["allowed"] for row in self.decisions),
                "stop_requested_count": sum(row["stop_requested"] for row in self.decisions),
                "scope": "software admission; no real hardware safety claim"}


def command_interval_summary(command_timestamps_ns, *, warmup_commands=0,
                             expected_interval_ms=1000 / 15):
    """Actual host command intervals; never infer control Hz from forward time.

    An interval is retained only when BOTH endpoints are outside warmup. These
    timestamps must be taken at the same submission boundary; they are neither
    actuator execution timestamps nor simulated physics timestamps. Deadline
    misses mean interval > expected interval. No commands means no measured rate.
    """
    stamps = list(command_timestamps_ns)
    if type(warmup_commands) is not int or warmup_commands < 0:
        raise ValueError("warmup_commands must be a nonnegative integer")
    if type(expected_interval_ms) is bool or not math.isfinite(float(expected_interval_ms)) or expected_interval_ms <= 0:
        raise ValueError("expected interval must be finite positive milliseconds")
    if any(type(stamp) is not int or stamp < 0 for stamp in stamps):
        raise ValueError("command timestamps must be nonnegative integer monotonic nanoseconds")
    if any(right <= left for left, right in zip(stamps, stamps[1:])):
        raise ValueError("command timestamps must be distinct and increasing")
    selected = stamps[warmup_commands:]
    intervals_ms = [(right-left)/1e6 for left, right in zip(selected, selected[1:])]
    misses = sum(interval > expected_interval_ms for interval in intervals_ms)
    measured_ns = selected[-1] - selected[0] if len(selected) >= 2 else None
    return {"interval_ms": distribution(intervals_ms), "raw_intervals_ms": intervals_ms,
            "warmup_commands": warmup_commands, "command_count": len(stamps),
            "expected_interval_ms": float(expected_interval_ms), "miss_count": misses,
            "miss_rate": misses / len(intervals_ms) if intervals_ms else None,
            "wall_command_frequency_hz": len(intervals_ms)*1e9/measured_ns if measured_ns else None,
            "scope": "host software submission intervals; not actuator execution or hard real time"}
