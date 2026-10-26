#!/usr/bin/env python3
"""CPU-only, one-shot G3C-R5 evidence collection after all processes ended.

Read finalized supervision/controller records, copy small records and resolved
YAML into a fresh output directory, and hash every raw R5 file.  The collector
also checks every fingerprint registered before training: starting tracked
files, the complete R4 evidence tree, R4 raw artifacts, frozen sources, and
pre-existing G3V files when their recorded fingerprints are available.  It
never starts, stops, retries, restores, deletes, or modifies a training run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import sys

try:
    from supervision_common import atomic_json, sha
except ImportError:
    from scripts.supervision_common import atomic_json, sha


SMALL_LIMIT = 5 * 2**20
SMALL_SUFFIXES = {".json", ".txt", ".jsonl"}
TERMINAL_CONTROLLER_PHASES = {"training_and_evaluation_complete", "incomplete"}


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_record(row, *, default_root=None, expected_key="sha256"):
    raw = row.get("path")
    if raw is None and default_root is not None:
        raw = Path(default_root) / row["relative_path"]
    require(raw is not None, "Fingerprint record has no path")
    expected = row.get(expected_key, row.get("sha256", row.get("expected_sha256")))
    require(isinstance(expected, str) and len(expected) == 64,
            f"Missing expected SHA256 for {raw}")
    path = Path(raw)
    try:
        actual = identity(path) if path.is_file() else None
    except OSError as error:
        return {"path": str(path), "expected_sha256": expected, "actual": None,
                "matches": False, "error": {"type": type(error).__name__, "message": str(error)}}
    return {"path": str(path), "expected_sha256": expected, "actual": actual,
            "matches": actual is not None and actual["sha256"] == expected,
            **({"relative_path": row["relative_path"]} if "relative_path" in row else {})}


def _registered_json(manifest, manifest_path, filename):
    """Use the frozen evidence copy if present; verify before trusting paths."""
    registration = manifest.get("preflight_files", {}).get(filename)
    require(isinstance(registration, dict), f"Missing frozen preflight registration: {filename}")
    local = Path(manifest_path).parent / filename
    path = local if local.is_file() else Path(registration["path"])
    require(path.is_file(), f"Missing registered preflight evidence: {path}")
    require(sha(path) == registration["sha256"], f"Registered preflight evidence changed: {path}")
    return load(path), identity(path)


def verify_preservation(manifest, manifest_path):
    """Return full evidence with actual match counts; no historical writes."""
    snapshot, snapshot_identity = _registered_json(manifest, manifest_path, "r5_preflight_snapshot.json")
    runtime, runtime_identity = _registered_json(manifest, manifest_path, "r4_runtime_integrity.json")
    repo = Path(manifest["repository"])
    require(bool(snapshot.get("repo_tracked_files")), "No starting tracked-file fingerprints")
    require(bool(snapshot.get("r4_evidence_files")), "No R4 evidence-file fingerprints")
    require(bool(runtime.get("files")), "No R4 raw-artifact fingerprints")
    groups = {
        "starting_tracked_files": [verify_record(row, default_root=repo)
                                    for row in snapshot["repo_tracked_files"]],
        "r4_evidence_files": [verify_record(row) for row in snapshot["r4_evidence_files"]],
        "r4_raw_artifacts": [verify_record(row, expected_key="expected_sha256")
                             for row in runtime["files"]],
        "frozen_sources": [verify_record({"path": path, "sha256": digest})
                           for path, digest in manifest.get("frozen_source_sha256", {}).items()],
    }
    require(bool(groups["frozen_sources"]), "No frozen source fingerprints")
    # R4's complete evidence list authenticates its fingerprint manifest too.
    historical = repo / "evidence/g3c_r4/historical_fingerprints.json"
    authenticated = next((record for record in groups["r4_evidence_files"]
                          if Path(record["path"]).resolve() == historical.resolve()), None)
    g3v = []
    g3v_basis = "No authenticated pre-existing G3V fingerprints available"
    if authenticated and authenticated["matches"]:
        historical_data = load(historical)
        g3v = historical_data.get("untracked_g3v", [])
        g3v_basis = str(historical)
    groups["preexisting_g3v"] = [verify_record(row) for row in g3v]
    prior_audit_path = repo / "evidence/g3c_r4/postrun_preservation_audit.json"
    prior_audit_identity = next((record for record in groups["r4_evidence_files"]
                                if Path(record["path"]).resolve() == prior_audit_path.resolve()), None)
    require(prior_audit_identity is not None,
            "R4 postrun preservation audit is absent from registered evidence")
    if prior_audit_identity["matches"]:
        prior_audit = load(prior_audit_path)
        prior_artifacts = prior_audit.get("historical_artifacts")
        require(isinstance(prior_artifacts, list) and bool(prior_artifacts),
                "R4 postrun preservation audit has no historical artifact fingerprints")
        groups["prior_historical_artifacts"] = [
            {"historical_scope": row.get("scope"), **verify_record(row["expected"])}
            for row in prior_artifacts]
    else:
        # The changed R4 audit is already a failure in r4_evidence_files.  Do not
        # trust its content as another set of reference fingerprints.
        groups["prior_historical_artifacts"] = []
    summaries = {name: {"count": len(rows), "match_count": sum(row["matches"] for row in rows),
                        "passed": all(row["matches"] for row in rows)}
                 for name, rows in groups.items()}
    failures = [{"scope": name, **row} for name, rows in groups.items() for row in rows if not row["matches"]]
    return {"schema_version": "g3cr5_postrun_preservation_v1", "verified_utc": utc_now(),
            "cpu_only": True, "control_actions": [], "starting_commit": manifest.get("starting_commit"),
            "registered_preflight_identities": [snapshot_identity, runtime_identity],
            "groups": groups, "summary": summaries,
            "preexisting_g3v_fingerprint_basis": g3v_basis,
            "failures": failures, "passed": not failures}


def terminal_records(manifest):
    supervisor_path = Path(manifest["training_supervisor_directory"]) / "status.json"
    completion_path = Path(manifest["completion_output"]) / "status.json"
    require(supervisor_path.is_file(), f"Training supervisor status is not available: {supervisor_path}")
    supervisor = load(supervisor_path)
    require(supervisor.get("finalized") is True, "Training supervisor has not finalized")
    require(completion_path.is_file(), f"Completion controller status is not available: {completion_path}")
    completion = load(completion_path)
    require(completion.get("phase") in TERMINAL_CONTROLLER_PHASES
            and type(completion.get("controller_exit_code")) is int
            and isinstance(completion.get("ended"), dict),
            "Completion controller has not published a terminal status")
    return {"supervisor_status": identity(supervisor_path),
            "raw_wait_returncode": supervisor.get("raw_wait_returncode"),
            "training_termination_reason": supervisor.get("termination_reason"),
            "completion_status": identity(completion_path),
            "completion_phase": completion["phase"],
            "controller_exit_code": completion["controller_exit_code"]}


def raw_roots(manifest):
    training = manifest["training"]
    return [("training_supervision", Path(training["run_root"]), Path("run")),
            ("training_artifacts", Path(training["run_dir"]), Path("training_artifacts")),
            ("completion_and_evaluation", Path(manifest["completion_output"]), Path("completion"))]


def inventory_raw(manifest, *, allow_missing_training_artifacts=False):
    rows, missing = [], []
    for label, root, _ in raw_roots(manifest):
        if not root.is_dir() and label == "training_artifacts" and allow_missing_training_artifacts:
            missing.append({"scope": label, "path": str(root.resolve()),
                            "reason": "Failed training ended before creating its artifact directory"})
            continue
        require(root.is_dir(), f"Missing raw artifact directory: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append({"scope": label, "relative_path": str(path.relative_to(root)), **identity(path)})
    checkpoints = [row for row in rows if Path(row["path"]).suffix == ".pth"]
    events = [row for row in rows if Path(row["path"]).name.startswith("events.out.tfevents")]
    return {"schema_version": "g3cr5_raw_artifact_inventory_v1", "generated_utc": utc_now(),
            "cpu_only": True, "control_actions": [], "files": rows, "file_count": len(rows),
            "checkpoint_count": len(checkpoints), "tensorboard_event_file_count": len(events),
            "all_existing_raw_files_hashed": True, "all_raw_roots_present": not missing,
            "missing_raw_roots": missing, "large_files_remain_at_original_paths": True}


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    shutil.copystat(source, target)
    require(sha(source) == sha(target), f"Copied artifact differs: {source}")
    return {"source": str(Path(source).resolve()), "destination": str(target.resolve()),
            "sha256": sha(target), "bytes": target.stat().st_size}


def copy_small(manifest, output):
    copied, skipped = [], []
    for label, root, relative in raw_roots(manifest):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SMALL_SUFFIXES:
                continue
            if path.stat().st_size >= SMALL_LIMIT:
                skipped.append({"scope": label, "path": str(path), "reason": "small_record_limit", **identity(path)})
                continue
            # Configs have their own simple destination; raw binary/event files
            # stay at their immutable paths and are represented in the inventory.
            copied.append(_copy(path, output / relative / path.relative_to(root)))
    params = Path(manifest["training"]["run_dir"]) / "params"
    if params.is_dir():
        for path in sorted(params.glob("*.yaml")):
            copied.append(_copy(path, output / "resolved_configs" / path.name))
    return {"small_record_limit_bytes": SMALL_LIMIT, "copied_files": copied,
            "copied_count": len(copied), "skipped_large_records": skipped}


def collect(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    require(not output.exists(), f"Output directory already exists; refusing overwrite: {output}")
    manifest = load(manifest_path)
    terminal = terminal_records(manifest)
    for _, root, _ in raw_roots(manifest):
        root = root.resolve()
        require(output != root and output not in root.parents and root not in output.parents,
                f"Output must be separate from raw artifact roots: {root}")
    # Require actual terminal states before creating or copying anything.
    output.mkdir(parents=True, exist_ok=False)
    try:
        preservation = verify_preservation(manifest, manifest_path)
        atomic_json(output / "postrun_preservation_audit.json", preservation, replace=False)
        require(preservation["passed"],
                "Baseline preservation failed: " + "; ".join(
                    f"{row['scope']}: {row['path']}" for row in preservation["failures"]))
        incomplete_failure = (terminal["completion_phase"] == "incomplete" and
                              (terminal["raw_wait_returncode"] != 0 or
                               terminal["training_termination_reason"] != "child_exit"))
        inventory = inventory_raw(manifest, allow_missing_training_artifacts=incomplete_failure)
        atomic_json(output / "raw_artifact_inventory.json", inventory, replace=False)
        copies = copy_small(manifest, output)
        atomic_json(output / "copy_inventory.json", copies, replace=False)
        result = {"schema_version": "g3cr5_collection_v1", "status": "complete",
                  "created_utc": utc_now(), "manifest": identity(manifest_path),
                  "terminal_records": terminal, "preservation_passed": True,
                  "raw_file_count": inventory["file_count"],
                  "checkpoint_count": inventory["checkpoint_count"],
                  "tensorboard_event_file_count": inventory["tensorboard_event_file_count"],
                  "raw_artifact_status": "complete" if inventory["all_raw_roots_present"] else "incomplete",
                  "missing_raw_roots": inventory["missing_raw_roots"],
                  "copied_count": copies["copied_count"], "cpu_only": True, "control_actions": []}
        atomic_json(output / "collection_status.json", result, replace=False)
        return result
    except Exception as error:
        atomic_json(output / "collection_status.json", {"schema_version": "g3cr5_collection_v1",
                    "status": "failed", "created_utc": utc_now(), "cpu_only": True,
                    "control_actions": [], "error": {"type": type(error).__name__, "message": str(error)}},
                    replace=False)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = collect(args.manifest, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": {"type": type(error).__name__,
              "message": str(error)}}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
