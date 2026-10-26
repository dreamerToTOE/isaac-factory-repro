"""CPU-only collection tests; fixtures contain no simulator or GPU process."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import collect_g3cr5 as collector


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fingerprint(path):
    return collector.identity(path)


@pytest.fixture
def frozen(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    evidence = repo / "evidence/g3c_r5"
    evidence.mkdir(parents=True)
    tracked = repo / "README.md"
    tracked.write_text("historical readme")
    source = repo / "source.py"
    source.write_text("frozen scientific source")
    g3v = repo / "evidence/g3v/README.md"
    g3v.parent.mkdir(parents=True)
    g3v.write_text("unrelated untracked evidence")
    old_raw = tmp_path / "r4_raw/checkpoint.pth"
    old_raw.parent.mkdir()
    old_raw.write_bytes(b"historical R4 checkpoint")
    older_raw = tmp_path / "r2_failure/console.txt"
    older_raw.parent.mkdir()
    older_raw.write_text("historical failed-run traceback")
    historical = write(repo / "evidence/g3c_r4/historical_fingerprints.json",
                       {"untracked_g3v": [fingerprint(g3v)]})
    prior_audit = write(repo / "evidence/g3c_r4/postrun_preservation_audit.json",
                        {"historical_artifacts": [{"scope": "checkpoint", "expected": fingerprint(older_raw)},
                                                   {"scope": "unrelated_g3v", "expected": fingerprint(g3v)}]})
    snapshot = write(evidence / "r5_preflight_snapshot.json", {
        "repo_tracked_files": [{**fingerprint(tracked), "relative_path": "README.md"}],
        "r4_evidence_files": [fingerprint(historical), fingerprint(prior_audit)],
        "untracked_preexisting": ["evidence/g3v/"]})
    runtime = write(evidence / "r4_runtime_integrity.json", {
        "files": [{"path": str(old_raw), "expected_sha256": collector.sha(old_raw)}]})
    launch_root = tmp_path / "r5_supervision"
    sup = launch_root / "supervisor"
    write(sup / "status.json", {"finalized": True, "raw_wait_returncode": 0,
                                "termination_reason": "child_exit"})
    write(sup / "training_audit.json", {"epochs": [1, 2, 3]})
    (sup / "console.txt").write_text("real child output\n")
    (sup / "events.jsonl").write_text('{"event":"exec_released"}\n')
    run = tmp_path / "r5_artifacts"
    (run / "nn").mkdir(parents=True)
    (run / "nn/epoch100.pth").write_bytes(b"new checkpoint")
    (run / "nn/epoch400.pth").write_bytes(b"new final checkpoint")
    (run / "summaries").mkdir()
    (run / "summaries/events.out.tfevents.test").write_bytes(b"tensorboard event")
    (run / "params").mkdir()
    (run / "params/agent.yaml").write_text("seed: 1\n")
    (run / "params/env.yaml").write_text("num_envs: 32\n")
    completion = tmp_path / "r5_completion"
    write(completion / "status.json", {"phase": "training_and_evaluation_complete",
                                       "controller_exit_code": 0, "ended": {"utc": "now"}})
    write(completion / "evaluation/epoch400/seed1001/metrics.json", {"status": "complete"})
    manifest = {"repository": str(repo), "starting_commit": "seed0-starting-commit",
                "training_supervisor_directory": str(sup), "completion_output": str(completion),
                "training": {"run_root": str(launch_root), "run_dir": str(run)},
                "preflight_files": {p.name: fingerprint(p) for p in (snapshot, runtime)},
                "frozen_source_sha256": {str(source): collector.sha(source)}}
    path = write(evidence / "manifest_frozen.json", manifest)
    return {"manifest": path, "data": manifest, "repo": repo, "output": evidence / "raw_collect",
            "tracked": tracked, "source": source, "r4_raw": old_raw, "older_raw": older_raw,
            "g3v": g3v, "sup": sup, "completion": completion, "run": run,
            "snapshot": snapshot, "runtime": runtime}


def test_collects_small_evidence_and_hashes_every_raw_binary_without_copying(frozen):
    before = {path: path.read_bytes() for path in (frozen["tracked"], frozen["source"],
              frozen["r4_raw"], frozen["older_raw"], frozen["g3v"])}
    result = collector.collect(frozen["manifest"], frozen["output"])
    out = frozen["output"]
    assert result["status"] == "complete" and result["cpu_only"] is True
    assert result["control_actions"] == [] and result["checkpoint_count"] == 2
    assert result["tensorboard_event_file_count"] == 1
    assert (out / "run/supervisor/console.txt").read_text() == "real child output\n"
    assert (out / "completion/evaluation/epoch400/seed1001/metrics.json").is_file()
    assert (out / "resolved_configs/agent.yaml").read_text() == "seed: 1\n"
    assert not list(out.rglob("*.pth"))
    assert not list(out.rglob("events.out.tfevents*"))
    inventory = collector.load(out / "raw_artifact_inventory.json")
    assert inventory["file_count"] == len(inventory["files"])
    assert all(len(row["sha256"]) == 64 for row in inventory["files"])
    preservation = collector.load(out / "postrun_preservation_audit.json")
    assert preservation["passed"] is True
    assert preservation["summary"]["starting_tracked_files"]["count"] == 1
    assert preservation["summary"]["r4_evidence_files"]["count"] == 2
    assert preservation["summary"]["r4_raw_artifacts"]["count"] == 1
    assert preservation["summary"]["prior_historical_artifacts"]["count"] == 2
    assert preservation["summary"]["preexisting_g3v"]["count"] == 1
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("state", ["missing_supervisor", "unfinalized_supervisor", "missing_completion", "active_completion"])
def test_requires_finalized_training_and_terminal_controller_before_output(frozen, state):
    if state == "missing_supervisor":
        (frozen["sup"] / "status.json").unlink()
    elif state == "unfinalized_supervisor":
        write(frozen["sup"] / "status.json", {"finalized": False})
    elif state == "missing_completion":
        (frozen["completion"] / "status.json").unlink()
    else:
        write(frozen["completion"] / "status.json", {"phase": "evaluating"})
    with pytest.raises(ValueError):
        collector.collect(frozen["manifest"], frozen["output"])
    assert not frozen["output"].exists()


@pytest.mark.parametrize("name,scope", [("tracked", "starting_tracked_files"), ("source", "frozen_sources"),
                                       ("r4_raw", "r4_raw_artifacts"), ("older_raw", "prior_historical_artifacts"),
                                       ("g3v", "preexisting_g3v")])
def test_changed_baseline_gets_exact_failure_evidence_without_raw_copy(frozen, name, scope):
    frozen[name].write_text("changed after preflight")
    with pytest.raises(ValueError, match="Baseline preservation failed"):
        collector.collect(frozen["manifest"], frozen["output"])
    audit = collector.load(frozen["output"] / "postrun_preservation_audit.json")
    assert audit["passed"] is False
    assert any(row["scope"] == scope and row["path"] == str(frozen[name]) for row in audit["failures"])
    status = collector.load(frozen["output"] / "collection_status.json")
    assert status["status"] == "failed" and str(frozen[name]) in status["error"]["message"]
    assert not (frozen["output"] / "run").exists()


def test_changed_preflight_reference_is_not_trusted(frozen):
    write(frozen["snapshot"], {"repo_tracked_files": []})
    with pytest.raises(ValueError, match="Registered preflight evidence changed"):
        collector.collect(frozen["manifest"], frozen["output"])
    assert collector.load(frozen["output"] / "collection_status.json")["status"] == "failed"


def test_existing_output_is_never_reused_or_modified(frozen):
    frozen["output"].mkdir()
    marker = frozen["output"] / "keep.txt"
    marker.write_text("existing evidence")
    with pytest.raises(ValueError, match="refusing overwrite"):
        collector.collect(frozen["manifest"], frozen["output"])
    assert list(frozen["output"].iterdir()) == [marker]
    assert marker.read_text() == "existing evidence"


def test_failed_training_and_completion_can_be_collected_without_claiming_success(frozen):
    write(frozen["sup"] / "status.json", {"finalized": True, "raw_wait_returncode": 1,
                                           "termination_reason": "child_exit"})
    write(frozen["completion"] / "status.json", {"phase": "incomplete", "controller_exit_code": 1,
                                                 "ended": {"utc": "now"}, "error": "CUDA OOM"})
    result = collector.collect(frozen["manifest"], frozen["output"])
    assert result["status"] == "complete"  # Collection success does not mean training success.
    assert result["terminal_records"]["raw_wait_returncode"] == 1
    assert result["terminal_records"]["completion_phase"] == "incomplete"


def test_early_runtime_failure_without_training_artifacts_preserves_console_and_missing_root(frozen):
    write(frozen["sup"] / "status.json", {"finalized": True, "raw_wait_returncode": 1,
                                           "termination_reason": "child_exit"})
    (frozen["sup"] / "console.txt").write_text("RuntimeError: CUDA out of memory before run directory\n")
    write(frozen["completion"] / "status.json", {"phase": "incomplete", "controller_exit_code": 1,
                                                 "ended": {"utc": "now"}, "error": "Child exited 1"})
    manifest = dict(frozen["data"])
    manifest["training"] = {**manifest["training"], "run_dir": str(frozen["run"].parent / "never_created")}
    write(frozen["manifest"], manifest)
    result = collector.collect(frozen["manifest"], frozen["output"])
    assert result["status"] == "complete" and result["raw_artifact_status"] == "incomplete"
    assert result["terminal_records"]["raw_wait_returncode"] == 1
    assert result["checkpoint_count"] == result["tensorboard_event_file_count"] == 0
    assert result["missing_raw_roots"] == [{"scope": "training_artifacts",
        "path": manifest["training"]["run_dir"],
        "reason": "Failed training ended before creating its artifact directory"}]
    assert "CUDA out of memory" in (frozen["output"] / "run/supervisor/console.txt").read_text()
    inventory = collector.load(frozen["output"] / "raw_artifact_inventory.json")
    assert inventory["all_raw_roots_present"] is False
    assert inventory["all_existing_raw_files_hashed"] is True


def test_successful_training_cannot_silently_omit_training_artifact_root(frozen):
    manifest = dict(frozen["data"])
    manifest["training"] = {**manifest["training"], "run_dir": str(frozen["run"].parent / "never_created")}
    write(frozen["manifest"], manifest)
    with pytest.raises(ValueError, match="Missing raw artifact directory"):
        collector.collect(frozen["manifest"], frozen["output"])
    assert collector.load(frozen["output"] / "collection_status.json")["status"] == "failed"


def test_large_json_is_inventoried_and_skipped_without_checkpoint_copy(frozen, monkeypatch):
    monkeypatch.setattr(collector, "SMALL_LIMIT", 100)
    large = frozen["sup"] / "large.json"
    large.write_text('{"evidence":"' + "a" * 200 + '"}')
    collector.collect(frozen["manifest"], frozen["output"])
    copies = collector.load(frozen["output"] / "copy_inventory.json")
    assert any(row["path"] == str(large) for row in copies["skipped_large_records"])
    inventory = collector.load(frozen["output"] / "raw_artifact_inventory.json")
    assert any(row["path"] == str(large) for row in inventory["files"])
    assert not (frozen["output"] / "run/supervisor/large.json").exists()


def test_output_cannot_overlap_or_be_inside_original_raw_root(frozen):
    with pytest.raises(ValueError, match="separate from raw artifact roots"):
        collector.collect(frozen["manifest"], frozen["run"] / "nested_new_collection")
    assert not (frozen["run"] / "nested_new_collection").exists()
