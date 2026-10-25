"""[ENGINEERING] Execute one fresh G2 run, then its preregistered evaluations.

The pipeline never resumes training and never changes learning configuration.
Run in a detached session with stdout/stderr retained; status.json is atomic.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["preflight_passed"] is True
    workspace, repo = Path(manifest["workspace"]), Path(manifest["repository"])
    run_name = manifest["run_name"]
    root = Path(manifest["monitor_root"])
    root.mkdir(parents=True, exist_ok=False)
    status = {"run_name": run_name, "manifest_sha256": digest(args.manifest),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "pipeline_pid": os.getpid(), "phase": "starting", "evaluations": []}
    start = time.monotonic()

    def write():
        temp = root / "status.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(root / "status.json")

    base_env = dict(os.environ, FACTORY_VENV=str(workspace / "work/envs/factory-g01"), OMP_NUM_THREADS="4")
    for key in ("FACTORY_AUDIT_OUTPUT", "FACTORY_G2_AUDIT_OUTPUT", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        base_env.pop(key, None)
    python = workspace / "work/envs/factory-g01/bin/python"
    official = workspace / "work/upstream/IsaacLab/scripts/reinforcement_learning/rl_games/train.py"
    run_dir = workspace / "work/runs/logs/rl_games/Factory" / run_name
    assert not run_dir.exists(), "Refusing to reuse an experiment directory"
    for path, sha in manifest["frozen_script_sha256"].items():
        assert digest(repo / path) == sha, f"Script changed after preflight: {path}"
    upstream = workspace / "work/upstream/IsaacLab"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip() == manifest["isaac_lab_sha"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=upstream, text=True)

    def bounded(command, output, timeout, env):
        full = [sys.executable, str(repo / "scripts/run_bounded.py"), "--output", str(output),
                "--timeout", str(timeout), "--min-free-mib", "1024", "--", *command]
        status["active_command"] = full
        process = subprocess.Popen(full, cwd=workspace / "work/runs", env=env)
        status["active_monitor_pid"] = process.pid
        write()
        code = process.wait()
        status["active_returncode"] = code
        write()
        if code:
            raise RuntimeError(f"Bounded subprocess failed with code {code}; see {output}")
        result = json.loads((output / "result.json").read_text())
        assert result["stop_reason"] is None and not result["monitor_errors"]
        return result

    try:
        status["phase"] = "training"
        write()
        train_command = ["bash", str(repo / "scripts/run_sim.sh"), str(official), "--headless", "--device", "cuda:0",
                         "--task", "Isaac-Factory-PegInsert-Direct-v0", "--num_envs", "64", "--seed", "0",
                         "--max_iterations", "200", f"agent.params.config.full_experiment_name={run_name}",
                         "agent.params.config.save_frequency=50"]
        assert "--checkpoint" not in train_command
        train_env = dict(base_env, FACTORY_G2_AUDIT_OUTPUT=str(root / "train/training_audit.json"))
        status["training_result"] = bounded(train_command, root / "train", 43200, train_env)
        audit = json.loads((root / "train/training_audit.json").read_text())
        assert not audit["instrumentation_errors"], audit["instrumentation_errors"]
        assert len(audit["epochs"]) == 200 and audit["epochs"][-1]["epoch"] == 200
        assert all(e["epoch_loss_arrays_finite"] for e in audit["epochs"])
        assert not any(e["operation"] == "restore" for e in audit["checkpoint_events"])
        status["training_complete_uninterrupted"] = True
        status["phase"] = "checkpoint_inventory"
        write()
        inventory = []
        for checkpoint in sorted((run_dir / "nn").glob("*.pth")):
            inventory.append({"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": digest(checkpoint)})
        (root / "checkpoint_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
        selected = {}
        for epoch in (50, 100, 150, 200):
            candidates = sorted((run_dir / "nn").glob(f"last_Factory_ep_{epoch}_rew_*.pth"))
            assert candidates, f"No milestone checkpoint at epoch {epoch}"
            # At the last epoch, prefer the unconditional final-save file.
            checkpoint = next((p for p in candidates if "_rew__" in p.name), candidates[0])
            checked = subprocess.run([str(python), str(repo / "scripts/inspect_checkpoint.py"), str(checkpoint)],
                                     env=dict(base_env, OMP_NUM_THREADS="2"), text=True, capture_output=True, check=True)
            details = json.loads(checked.stdout)
            assert details["epoch"] == epoch and details["frame"] == epoch * 64 * 128
            assert details["all_tensors_finite"]
            selected[str(epoch)] = details
        (root / "selected_checkpoints.json").write_text(json.dumps(selected, indent=2) + "\n")
        status["phase"] = "evaluation"
        write()
        for epoch in (50, 100, 150, 200):
            checkpoint = Path(selected[str(epoch)]["path"])
            assert digest(checkpoint) == selected[str(epoch)]["sha256"]
            episodes = 3 if epoch == 200 else 1
            for seed in manifest["evaluation"]["seeds"]:
                output = root / "evaluation" / f"epoch{epoch:03d}" / f"seed{seed}"
                for path, sha in manifest["frozen_evaluation_sha256"].items():
                    assert digest(repo / path) == sha, f"Evaluator changed after preregistration: {path}"
                command = ["bash", str(repo / "scripts/run_sim.sh"), str(repo / "scripts/evaluate_checkpoint.py"),
                           "--headless", "--device", "cuda:0", "--task", "Isaac-Factory-PegInsert-Direct-v0",
                           "--checkpoint", str(checkpoint), "--num_envs", "32", "--episodes", str(episodes),
                           "--seed", str(seed), "--max_steps", str(160 * episodes), "--snapshot-limit", "12",
                           "--progress-interval", "10", "--output", str(output / "metrics.json")]
                status["current_evaluation"] = {"epoch": epoch, "seed": seed, "requested_episodes": 32 * episodes}
                write()
                bounded(command, output, 1800, dict(base_env, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"))
                metrics = json.loads((output / "metrics.json").read_text())
                assert metrics["episodes_completed"] == 32 * episodes
                status["evaluations"].append({"epoch": epoch, "seed": seed, "output": str(output / "metrics.json"),
                                              "aggregate": metrics["aggregate"]})
                write()
        status["phase"] = "training_and_evaluation_complete"
    except BaseException as error:
        status["phase"] = "incomplete"
        status["error"] = repr(error)
        traceback.print_exc()
        raise
    finally:
        status["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status["elapsed_seconds"] = time.monotonic() - start
        write()


if __name__ == "__main__":
    main()
