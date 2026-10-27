#!/usr/bin/env python3
"""Durable child/event accounting for G4-S2A-R2-R2 diagnostics.

This module contains no MuJoCo imports and never starts a policy or a physics
step.  It is intentionally additive: the frozen R2/R2-R1 runners remain
unchanged.  A diagnostic child can use :func:`durable_integration` around its
existing one-step call, while :class:`ChildSupervisor` owns one child process
and records the wait result exactly once.

The event ledger is append-only JSONL.  Every event is flushed and fsynced
before the call returns; ``last_event.json`` is atomically replaced after the
same fsync.  A request therefore cannot be mistaken for a completed result.
The supervisor performs no retry.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

try:  # Linux is the supported execution host; keep import failure explicit.
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX hosts
    fcntl = None


SCHEMA = "g4s2a_r2_r2_durable_supervision_v1"
CRASH_SIGNALS = frozenset(
    signal_number for signal_number in (
        getattr(signal, "SIGABRT", None), getattr(signal, "SIGBUS", None),
        getattr(signal, "SIGFPE", None), getattr(signal, "SIGILL", None),
        getattr(signal, "SIGSEGV", None), getattr(signal, "SIGSYS", None),
        getattr(signal, "SIGTRAP", None),
    ) if signal_number is not None
)


def _clean(value: Any) -> Any:
    """Return JSON-safe values without silently encoding non-finite floats."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NONFINITE_EVENT_FIELD")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    # Numpy arrays and scalar objects are not needed by the generic ledger.  A
    # child should explicitly convert diagnostic state before recording it.
    raise TypeError(f"UNSUPPORTED_EVENT_VALUE:{type(value).__name__}")


def _fsync_directory(path: Path) -> None:
    """Synchronize a containing directory where the platform supports it."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(_clean(value), stream, ensure_ascii=False, indent=2,
                  allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class DurableEventLog:
    """Durable event ledger for one diagnostic child and its supervisor.

    The child calls ``record_request`` immediately before
    the existing integration function, then ``record_result`` only after it
    returns successfully.  If the child crashes between those calls, the
    durable ledger ends with a request (or an exception), never a fabricated
    result.
    """

    def __init__(self, directory: str | os.PathLike[str], *, run_id: str | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self.last_path = self.directory / "last_event.json"
        self.lock_path = self.directory / "events.lock"
        self.run_id = run_id
        previous = _read_json(self.last_path)
        self._sequence = int(previous.get("sequence", 0)) + 1 if previous else 1
        # Refuse to append to a completed ledger: a retry would concatenate
        # two attempts and make request/result accounting ambiguous.
        if previous and previous.get("terminal") is True:
            raise FileExistsError("DURABLE_LEDGER_ALREADY_TERMINAL")

    def record(self, event: str, *, terminal: bool = False, **fields: Any) -> dict[str, Any]:
        if not event or not isinstance(event, str):
            raise ValueError("EVENT_NAME_REQUIRED")
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "sequence": self._sequence,
            "event": event,
            "terminal": bool(terminal),
            "pid": os.getpid(),
            "utc": datetime.now(timezone.utc).isoformat(),
            "unix_seconds": time.time(),
            "monotonic_seconds": time.monotonic(),
        }
        if self.run_id is not None:
            value["run_id"] = self.run_id
        value.update(fields)
        value = _clean(value)
        if fcntl is None:
            raise RuntimeError("DURABLE_LEDGER_REQUIRES_POSIX_FILE_LOCK")
        # The supervisor and child intentionally share one ledger.  Refresh
        # the sequence while holding an interprocess lock so a stale cached
        # sequence can never duplicate an event after the other process writes.
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            previous = _read_json(self.last_path)
            self._sequence = int(previous.get("sequence", 0)) + 1 if previous else self._sequence
            value["sequence"] = self._sequence
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _atomic_json(self.last_path, value)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        self._sequence += 1
        return value

    def record_request(self, step: int, **fields: Any) -> dict[str, Any]:
        return self.record("integration_request", step=int(step), **fields)

    def record_result(self, step: int, **fields: Any) -> dict[str, Any]:
        return self.record("physics_result", step=int(step), **fields)

    def record_exception(self, step: int, exc: BaseException, **fields: Any) -> dict[str, Any]:
        return self.record("integration_exception", step=int(step),
                           exception_type=type(exc).__name__,
                           exception_message=str(exc), **fields)

    def record_terminal(self, reason: str, **fields: Any) -> dict[str, Any]:
        return self.record("child_terminal", terminal=True, reason=str(reason), **fields)


def durable_integration(log: DurableEventLog, step: int, operation: Callable[[], Any],
                        **request_fields: Any) -> Any:
    """Run one existing integrator with durable request/result boundaries.

    ``record_request`` completes (including ``fsync``) before ``operation``
    is called.  Exceptions are recorded and re-raised; no retry is attempted.
    """
    log.record_request(step, **request_fields)
    try:
        value = operation()
    except BaseException as exc:
        try:
            log.record_exception(step, exc)
        finally:
            raise
    log.record_result(step)
    return value


def classify_wait_returncode(returncode: int | None, *, timed_out: bool = False) -> str:
    """Classify observed process state; ``None`` remains unknown.

    Fatal native signals are separated as ``CHILD_CRASH``; ordinary signals
    such as SIGTERM remain ``CHILD_SIGNAL``.
    """
    if timed_out:
        return "TIMEOUT"
    if returncode is None:
        return "UNKNOWN_TERMINATION"
    if returncode < 0 and -returncode in CRASH_SIGNALS:
        return "CHILD_CRASH"
    if returncode < 0:
        return "CHILD_SIGNAL"
    return "CHILD_EXIT"


def status_from_returncode(returncode: int | None, *, timed_out: bool = False,
                           reason: str | None = None) -> dict[str, Any]:
    classification = classify_wait_returncode(returncode, timed_out=timed_out)
    return {
        "schema": SCHEMA,
        "termination_class": classification,
        "termination_reason": reason or classification.lower(),
        "raw_wait_returncode": returncode,
        "child_exit_code": returncode if returncode is not None and returncode >= 0 else None,
        "child_signal": -returncode if returncode is not None and returncode < 0 else None,
        "timed_out": bool(timed_out),
        "retry": False,
    }


class ChildSupervisor:
    """One-shot supervisor for a synthetic or diagnostic child command."""

    def __init__(self, directory: str | os.PathLike[str], command: Sequence[str], *,
                 cwd: str | os.PathLike[str] | None = None,
                 env: Mapping[str, str] | None = None,
                 timeout_s: float = 900.0,
                 terminate_grace_s: float = 1.0):
        self.directory = Path(directory)
        self.command = [str(part) for part in command]
        self.cwd = None if cwd is None else str(cwd)
        self.env = None if env is None else {str(k): str(v) for k, v in env.items()}
        self.timeout_s = float(timeout_s)
        self.terminate_grace_s = float(terminate_grace_s)
        if not self.command:
            raise ValueError("CHILD_COMMAND_REQUIRED")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("INVALID_CHILD_TIMEOUT")
        if not math.isfinite(self.terminate_grace_s) or self.terminate_grace_s < 0:
            raise ValueError("INVALID_TERMINATE_GRACE")

    def _write_status(self, value: Mapping[str, Any]) -> None:
        _atomic_json(self.directory / "supervisor_status.json", dict(value))

    def run(self) -> dict[str, Any]:
        # mkdir(exist_ok=False) is the no-retry/output-exclusivity boundary.
        self.directory.mkdir(parents=True, exist_ok=False)
        ledger = DurableEventLog(self.directory / "events", run_id=self.directory.name)
        started = time.monotonic()
        base = {
            "schema": SCHEMA,
            "command": self.command,
            "cwd": self.cwd,
            "timeout_s": self.timeout_s,
            "retry": False,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        ledger.record("supervisor_start", command=self.command, timeout_s=self.timeout_s)
        self._write_status({**base, "state": "STARTING", "raw_wait_returncode": None})
        console_path = self.directory / "console.txt"
        child: subprocess.Popen[str] | None = None
        timed_out = False
        launch_error: dict[str, Any] | None = None
        returncode: int | None = None
        try:
            child_env = os.environ.copy()
            if self.env is not None:
                child_env.update(self.env)
            child_env.setdefault("FACTORY_DURABLE_EVENT_DIR", str(self.directory / "events"))
            with console_path.open("x", encoding="utf-8") as console:
                child = subprocess.Popen(self.command, cwd=self.cwd, env=child_env,
                                         stdout=console, stderr=subprocess.STDOUT,
                                         start_new_session=True, text=True)
                ledger.record("child_spawn", pid=child.pid, pgid=child.pid)
                self._write_status({**base, "state": "RUNNING", "pid": child.pid,
                                    "pgid": child.pid, "raw_wait_returncode": None})
                try:
                    returncode = child.wait(timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    ledger.record("timeout", pid=child.pid, timeout_s=self.timeout_s)
                    self._terminate_group(child.pid, ledger)
                    returncode = child.wait()
        except OSError as exc:
            launch_error = {"type": type(exc).__name__, "message": str(exc)}
            ledger.record("launch_error", exception_type=type(exc).__name__,
                          exception_message=str(exc), terminal=True)
        except BaseException as exc:
            # This is a supervisor-side failure.  Do not synthesize a child
            # exit code; preserve the last durable event and mark unknown.
            launch_error = {"type": type(exc).__name__, "message": str(exc)}
            if child is not None and child.poll() is None:
                # Status persistence failed after spawn; do not leave an
                # unowned diagnostic child running.  Cleanup is best effort,
                # and a missing wait result remains UNKNOWN by contract.
                try:
                    self._terminate_group(child.pid, ledger)
                    child.wait(timeout=max(1.0, self.terminate_grace_s + 1.0))
                except BaseException:
                    pass
            try:
                ledger.record("supervisor_error", exception_type=type(exc).__name__,
                              exception_message=str(exc), terminal=True)
            except BaseException:
                # The original supervisor error is already represented by
                # the final UNKNOWN receipt if the ledger itself is broken.
                pass
        if launch_error is not None:
            status = {**base, **status_from_returncode(None, reason="launch_or_supervisor_error"),
                      "state": "FAILED", "error": launch_error,
                      "wall_s": time.monotonic() - started}
        else:
            status = {**base, **status_from_returncode(returncode, timed_out=timed_out),
                      "state": "FINISHED", "pid": child.pid if child else None,
                      "pgid": child.pid if child else None,
                      "wall_s": time.monotonic() - started,
                      "error": None}
            ledger.record_terminal(status["termination_class"],
                                   raw_wait_returncode=returncode,
                                   timed_out=timed_out)
        last_event = _read_json(ledger.last_path)
        status["last_durable_event"] = last_event
        self._write_status(status)
        return status

    def _terminate_group(self, pid: int, ledger: DurableEventLog) -> None:
        """Terminate only the child process group owned by this supervisor."""
        try:
            os.killpg(pid, signal.SIGTERM)
            ledger.record("signal_sent", signal="SIGTERM", pgid=pid)
        except ProcessLookupError:
            ledger.record("signal_target_absent", signal="SIGTERM", pgid=pid)
            return
        except OSError as exc:
            ledger.record("signal_error", signal="SIGTERM", pgid=pid,
                          exception_type=type(exc).__name__, exception_message=str(exc))
            return
        deadline = time.monotonic() + self.terminate_grace_s
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            except OSError:
                return
            time.sleep(min(.02, max(0., deadline - time.monotonic())))
        try:
            os.killpg(pid, signal.SIGKILL)
            ledger.record("signal_sent", signal="SIGKILL", pgid=pid)
        except ProcessLookupError:
            ledger.record("signal_target_absent", signal="SIGKILL", pgid=pid)
        except OSError as exc:
            ledger.record("signal_error", signal="SIGKILL", pgid=pid,
                          exception_type=type(exc).__name__, exception_message=str(exc))


def unknown_status(directory: str | os.PathLike[str], *, reason: str) -> dict[str, Any]:
    """Write an explicit unknown receipt when the supervisor cannot wait.

    This helper is for an external collector or post-mortem tool.  It never
    invents an exit code and refuses to overwrite an existing final receipt.
    """
    directory = Path(directory)
    target = directory / "supervisor_status.json"
    if target.exists():
        raise FileExistsError("SUPERVISOR_STATUS_ALREADY_EXISTS")
    value = {"schema": SCHEMA, **status_from_returncode(None, reason=reason),
             "state": "UNKNOWN", "error": None,
             "unknown_reason": str(reason), "retry": False}
    _atomic_json(target, value)
    return value


__all__ = [
    "SCHEMA", "DurableEventLog", "durable_integration", "ChildSupervisor",
    "classify_wait_returncode", "status_from_returncode", "unknown_status",
]
