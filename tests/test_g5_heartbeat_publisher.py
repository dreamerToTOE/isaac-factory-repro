"""CPU-only tests for the bounded heartbeat publisher."""
import threading
import time
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from scripts.gpu_guard import HeartbeatPublisher
import scripts.gpu_guard as gpu_guard

ROOT = Path(__file__).parents[1]
PY = sys.executable
GUARD = ROOT / "scripts/gpu_guard.py"
SUPERVISOR = ROOT / "scripts/durable_supervisor.py"


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_fast_publisher_publishes_latest_and_finishes(tmp_path):
    writes = []

    def write(path, payload, durable=False):
        writes.append((path, payload, durable))

    publisher = HeartbeatPublisher(tmp_path / "heartbeat.json", write_fn=write, stale_seconds=1)
    publisher.submit({"sample": 1}, 1.0)
    assert wait_until(lambda: len(writes) == 1)
    status = publisher.close()
    assert status["failure"] is None
    assert status["published_count"] == 1
    assert status["coalesced_count"] == 0
    assert writes[0][1] == {"sample": 1} and writes[0][2] is False


def test_slow_heartbeat_does_not_block_sampling_submissions(tmp_path):
    started = threading.Event()
    release = threading.Event()
    writes = []

    def write(path, payload, durable=False):
        writes.append(payload)
        started.set()
        release.wait(1)

    publisher = HeartbeatPublisher(tmp_path / "heartbeat.json", write_fn=write, stale_seconds=10)
    publisher.submit({"sample": 1}, 1.0)
    assert started.wait(1)
    begin = time.monotonic()
    for sample in range(2, 12):
        publisher.submit({"sample": sample}, float(sample))
    submit_seconds = time.monotonic() - begin
    assert submit_seconds < 0.2
    release.set()
    assert wait_until(lambda: len(writes) >= 2)
    status = publisher.close()
    # A bounded latest-value slot records coalescing; it never grows with the
    # number of samples while the filesystem operation is blocked.
    assert len(writes) <= 2
    assert status["published_count"] >= 2


def test_publisher_failure_is_visible_to_sampler(tmp_path):
    def write(path, payload, durable=False):
        raise OSError("synthetic heartbeat failure")

    publisher = HeartbeatPublisher(tmp_path / "heartbeat.json", write_fn=write, stale_seconds=1)
    publisher.submit({"sample": 1}, 1.0)
    assert wait_until(lambda: publisher.health() == "failed")
    status = publisher.close()
    assert "synthetic heartbeat failure" in status["failure"]


def test_stale_publication_is_fail_closed(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def write(path, payload, durable=False):
        started.set()
        release.wait(1)

    publisher = HeartbeatPublisher(tmp_path / "heartbeat.json", write_fn=write, stale_seconds=0.03)
    publisher.submit({"sample": 1}, time.monotonic())
    assert started.wait(1)
    time.sleep(0.05)
    assert publisher.health() == "stale"
    release.set()
    publisher.close()


def test_stale_publication_detected_even_while_new_samples_arrive(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def write(path, payload, durable=False):
        started.set()
        release.wait(1)

    publisher = HeartbeatPublisher(tmp_path / "heartbeat.json", write_fn=write, stale_seconds=0.03)
    publisher.submit({"sample": 1}, time.monotonic())
    assert started.wait(1)
    for sample in range(2, 8):
        publisher.submit({"sample": sample}, time.monotonic())
        time.sleep(0.01)
    assert publisher.health() == "stale"
    release.set()
    publisher.close()


def test_repeated_cpu_guard_supervisor_lifecycle(tmp_path):
    results = []
    for index in range(5):
        mock = tmp_path / f"mock-{index}.json"
        mock.write_text(json.dumps([{"free_MiB": 4096}]))
        sup = tmp_path / f"sup-{index}"
        guard = tmp_path / f"guard-{index}"
        g = subprocess.Popen(
            [PY, str(GUARD), "--output", str(guard), "--supervisor-directory", str(sup),
             "--mock-samples", str(mock)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        s = subprocess.Popen(
            [PY, str(SUPERVISOR), "--output", str(sup), "--guard-directory", str(guard),
             "--startup-timeout", "3", "--lock-file", str(tmp_path / f"lock-{index}"), "--",
             PY, "-c", "import time; time.sleep(.15)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        sout, serr = s.communicate(timeout=20)
        gout, gerr = g.communicate(timeout=20)
        assert not serr and not gerr, (sout, serr, gout, gerr)
        supervisor = json.loads((sup / "status.json").read_text())
        monitor = json.loads((guard / "status.json").read_text())
        results.append({"supervisor": supervisor["raw_wait_returncode"],
                        "guard": monitor["continuous_monitoring"],
                        "samples": monitor["sample_count"]})
    assert all(item["supervisor"] == 0 and item["guard"] for item in results)
    assert min(item["samples"] for item in results) >= 1


def test_publisher_failure_terminates_bound_child(tmp_path, monkeypatch):
    mock = tmp_path / "mock.json"
    mock.write_text(json.dumps([{"free_MiB": 4096}]))
    supervisor_dir = tmp_path / "supervisor"
    output = tmp_path / "guard"
    supervisor_dir.mkdir()
    child = subprocess.Popen([PY, "-c", "import time; time.sleep(20)"], start_new_session=True)
    supervisor_identity = gpu_guard.identity()
    child_identity = gpu_guard.identity(child.pid)
    (supervisor_dir / "identity.json").write_text(json.dumps({"identity": supervisor_identity}))
    (supervisor_dir / "child_identity.json").write_text(json.dumps({"identity": child_identity, **child_identity}))

    class FailedPublisher:
        def __init__(self, path, stale_seconds):
            self.status = {"failure": "synthetic", "thread_alive_after_close": False,
                           "published_count": 1}
        def submit(self, payload, sample_monotonic):
            return None
        def health(self):
            return "failed"
        def close(self):
            return self.status

    monkeypatch.setattr(gpu_guard, "HeartbeatPublisher", FailedPublisher)
    args = SimpleNamespace(output=output, supervisor_directory=supervisor_dir,
                           interval=.01, gap_limit=.5, stale_seconds=1,
                           startup_timeout=1, mock_samples=mock)
    try:
        result = gpu_guard.run(args)
        assert result == 124
        status = json.loads((output / "status.json").read_text())
        assert status["stop_reason"] == "heartbeat_publisher_failed"
    finally:
        if child.poll() is None:
            gpu_guard.stop_owned(child_identity)
        child.wait(timeout=5)
    assert child.poll() is not None
