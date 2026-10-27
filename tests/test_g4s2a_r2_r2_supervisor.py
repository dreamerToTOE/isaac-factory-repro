"""Synthetic CPU-only checks for G4-S2A-R2-R2 lifecycle accounting.

These tests deliberately never import MuJoCo, open a viewer, load a policy or
perform a physics step.  The child programs are tiny subprocesses that only
exercise the durable request/result boundary and process-status classifier.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from scripts.g4s2a_r2_r2_supervisor import (
    ChildSupervisor,
    DurableEventLog,
    classify_wait_returncode,
    durable_integration,
    status_from_returncode,
    unknown_status,
)


ROOT = Path(__file__).resolve().parents[1]


def read_events(path: Path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def child_code(body: str) -> list[str]:
    # The parent starts from ROOT, so the repository is importable without a
    # package installation.  No physics package is imported by this module.
    return [sys.executable, "-c", body]


def test_durable_integration_orders_request_before_operation_and_result(tmp_path):
    ledger = DurableEventLog(tmp_path / "events", run_id="synthetic")
    seen = []

    def operation():
        seen.append("operation")
        events = read_events(tmp_path / "events")
        assert [item["event"] for item in events] == ["integration_request"]
        seen.append("durable_request_seen")
        return 17

    assert durable_integration(ledger, 3, operation, pre_state_hash="abc") == 17
    assert seen == ["operation", "durable_request_seen"]
    events = read_events(tmp_path / "events")
    assert [item["event"] for item in events] == ["integration_request", "physics_result"]
    assert [item["sequence"] for item in events] == [1, 2]
    assert json.loads((tmp_path / "events/last_event.json").read_text())["event"] == "physics_result"


def test_durable_integration_records_exception_without_fabricating_result(tmp_path):
    ledger = DurableEventLog(tmp_path / "events")

    def operation():
        raise RuntimeError("synthetic integration failure")

    with pytest.raises(RuntimeError, match="synthetic integration failure"):
        durable_integration(ledger, 8, operation)
    events = read_events(tmp_path / "events")
    assert [item["event"] for item in events] == ["integration_request", "integration_exception"]
    assert not any(item["event"] == "physics_result" for item in events)


def test_child_exit_code_and_last_event_are_durable(tmp_path):
    child = (
        "from scripts.g4s2a_r2_r2_supervisor import DurableEventLog,durable_integration;"
        "import os;"
        "l=DurableEventLog(os.environ['FACTORY_DURABLE_EVENT_DIR'],run_id='child');"
        "durable_integration(l,1,lambda:None,marker='before');"
        "raise SystemExit(7)"
    )
    status = ChildSupervisor(tmp_path / "run", child_code(child), cwd=ROOT, timeout_s=5).run()
    assert status["termination_class"] == "CHILD_EXIT"
    assert status["raw_wait_returncode"] == 7
    assert status["child_exit_code"] == 7
    assert status["child_signal"] is None
    assert status["retry"] is False
    assert status["last_durable_event"]["event"] == "child_terminal"
    events = read_events(tmp_path / "run/events")
    assert [item["event"] for item in events] == [
        "supervisor_start", "child_spawn", "integration_request", "physics_result", "child_terminal"
    ]
    assert [item["sequence"] for item in events] == list(range(1, len(events) + 1))
    assert (tmp_path / "run/console.txt").exists()


def test_signal_exit_is_not_reported_as_normal_child_exit(tmp_path):
    # SIGTERM exercises the exact wait(-signal) path without generating a
    # host coredump that would make this CPU-only test unnecessarily slow.
    child = "import os,signal; os.kill(os.getpid(),signal.SIGTERM)"
    status = ChildSupervisor(tmp_path / "run", child_code(child), cwd=ROOT, timeout_s=5).run()
    assert status["termination_class"] == "CHILD_SIGNAL"
    assert status["raw_wait_returncode"] == -signal.SIGTERM
    assert status["child_exit_code"] is None
    assert status["child_signal"] == signal.SIGTERM


def test_timeout_terminates_only_owned_child_group_and_records_timeout(tmp_path):
    child = "import time; time.sleep(10)"
    status = ChildSupervisor(tmp_path / "run", child_code(child), cwd=ROOT,
                             timeout_s=.1, terminate_grace_s=.05).run()
    assert status["termination_class"] == "TIMEOUT"
    assert status["timed_out"] is True
    assert status["raw_wait_returncode"] is not None
    assert status["retry"] is False
    events = read_events(tmp_path / "run/events")
    assert any(item["event"] == "timeout" for item in events)
    assert any(item["event"] == "signal_sent" and item["signal"] == "SIGTERM" for item in events)


def test_crash_after_request_leaves_unresolved_request_and_signal_receipt(tmp_path):
    child = (
        "from scripts.g4s2a_r2_r2_supervisor import DurableEventLog;"
        "import os,signal,resource;"
        "l=DurableEventLog(os.environ['FACTORY_DURABLE_EVENT_DIR']);"
        "l.record_request(4,pre_state_hash='x');"
        "resource.setrlimit(resource.RLIMIT_CORE,(0,0));"
        "os.kill(os.getpid(),signal.SIGABRT)"
    )
    status = ChildSupervisor(tmp_path / "run", child_code(child), cwd=ROOT, timeout_s=5).run()
    assert status["termination_class"] == "CHILD_CRASH"
    assert status["child_signal"] == signal.SIGABRT
    events = read_events(tmp_path / "run/events")
    assert [item["event"] for item in events] == [
        "supervisor_start", "child_spawn", "integration_request", "child_terminal"
    ]
    assert events[2]["step"] == 4
    assert status["last_durable_event"]["event"] == "child_terminal"
    assert not any(item["event"] == "physics_result" for item in events)


def test_unknown_status_never_invents_exit_code_and_does_not_overwrite(tmp_path):
    status = unknown_status(tmp_path / "run", reason="supervisor_process_lost")
    assert status["termination_class"] == "UNKNOWN_TERMINATION"
    assert status["raw_wait_returncode"] is None
    assert status["child_exit_code"] is None
    assert status["child_signal"] is None
    with pytest.raises(FileExistsError, match="SUPERVISOR_STATUS_ALREADY_EXISTS"):
        unknown_status(tmp_path / "run", reason="second_attempt")


@pytest.mark.parametrize("returncode, expected", [
    (0, "CHILD_EXIT"), (7, "CHILD_EXIT"), (-signal.SIGTERM, "CHILD_SIGNAL"),
    (-signal.SIGABRT, "CHILD_CRASH"),
    (None, "UNKNOWN_TERMINATION"),
])
def test_status_classifier_distinguishes_observed_and_unknown_states(returncode, expected):
    assert classify_wait_returncode(returncode) == expected
    assert status_from_returncode(returncode)["termination_class"] == expected
    assert classify_wait_returncode(0, timed_out=True) == "TIMEOUT"
