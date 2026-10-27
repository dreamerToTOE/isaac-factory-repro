"""Additional no-physics R2-R3 supervision integration tests."""
import json
import multiprocessing
import os
from pathlib import Path
import sys

import pytest

from scripts.g4s2a_r2_r2_supervisor import DurableEventLog
from scripts.g4s2a_r2_r3_supervisor import R3ChildSupervisor

ROOT = Path(__file__).resolve().parents[1]


def _write_many(directory, worker):
    ledger = DurableEventLog(directory)
    for i in range(20):
        ledger.record("synthetic", worker=worker, index=i)


def test_interprocess_concurrent_ledger_sequence_is_complete(tmp_path):
    directory = tmp_path / "events"
    directory.mkdir()
    processes = [multiprocessing.Process(target=_write_many, args=(directory, i)) for i in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join(15)
        assert p.exitcode == 0
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert len(events) == 80
    assert [e["sequence"] for e in events] == list(range(1, 81))
    assert {(e["worker"], e["index"]) for e in events} == {(i, j) for i in range(4) for j in range(20)}
    assert json.loads((directory / "last_event.json").read_text()) == events[-1]


def test_child_uses_reserved_absolute_ledger_even_with_stale_inherited_env(tmp_path, monkeypatch):
    stale = tmp_path / "wrong"
    monkeypatch.setenv("FACTORY_DURABLE_EVENT_DIR", str(stale))
    body = ("import os; from scripts.g4s2a_r2_r2_supervisor import DurableEventLog, durable_integration; "
            "l=DurableEventLog(os.environ['FACTORY_DURABLE_EVENT_DIR']); durable_integration(l,1,lambda:None)")
    run = tmp_path / "run"
    s = R3ChildSupervisor(run, [sys.executable, "-c", body], cwd=ROOT, timeout_s=5).run()
    events = [json.loads(x) for x in (run / "events/events.jsonl").read_text().splitlines()]
    assert s["raw_wait_returncode"] == 0
    assert [x["event"] for x in events] == ["supervisor_start", "child_spawn", "integration_request", "physics_result", "child_terminal"]
    assert not stale.exists()
    with pytest.raises(FileExistsError):
        R3ChildSupervisor(run, [sys.executable, "-c", body], cwd=ROOT, timeout_s=5).run()
