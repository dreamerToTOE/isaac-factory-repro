"""CPU-only controls for conditional evaluation; no simulator/GPU is launched."""
import copy
import json
from pathlib import Path

import pytest

from scripts import complete_g3cr5 as completion


def successful_supervisor():
    return {"finalized": True, "raw_wait_returncode": 0, "termination_reason": "child_exit",
            "error": None, "stop_signals": [], "received_supervisor_signals": [],
            "child_signal": None, "guard_required": False}


def successful_training():
    audit = {"epochs": [{"epoch": epoch, "environment_transitions": 4096,
                         "epoch_loss_arrays_finite": True} for epoch in range(1, 401)],
             "runtime": {"num_actors": 32, "horizon_length": 128},
             "instrumentation_errors": [], "checkpoint_events": [{"operation": "save"}]}
    finalizer = {"status": "complete", "training_error": None,
                 "agents": [{"writer_expected": True, "success": True}]}
    return audit, successful_supervisor(), finalizer, "MAX EPOCHS NUM!"


def checkpoint(epoch, path="/trusted/checkpoint.pth", digest="abc"):
    return {"path": path, "sha256": digest, "epoch": epoch, "frame": epoch * 4096,
            "all_tensors_and_numeric_arrays_finite": True, "optimizer_step_min": 1,
            "nonzero_adam_exp_avg_states": 1, "actor_model_present": True,
            "central_value_present": True}


def evaluation(epoch, seed, digest="abc"):
    per_env = 3 if epoch == 400 else 1
    rows = [{"env_id": env, "episode_id": ep, "ever_success": env != 0,
             "terminal_success": env > 1, "steps": 149, "terminal_kind": "timeout",
             "first_success_step": 80 if env else None,
             "first_success_time_seconds": 80 * .1 if env else None}
            for ep in range(per_env) for env in range(32)]
    count = len(rows)
    return {"status": "complete", "seed": seed, "num_envs": 32, "checkpoint_sha256": digest,
            "deterministic": True, "auto_reset_order_verified": True,
            "official_success_capture": "pre_reset_factory_get_curr_successes",
            "step_dt_seconds": .1, "episodes_per_env_requested": per_env,
            "episodes_completed": count, "episodes": rows,
            "aggregate": {"episodes_completed": count, "ever_success_count": 31 * per_env,
                          "terminal_success_count": 30 * per_env,
                          "ever_success_rate": 31 / 32, "terminal_success_rate": 30 / 32}}


def test_budget_requires_actual_contiguous_epochs():
    data = successful_training()
    assert completion.validate_training(*data)["completed_transitions"] == 1_638_400
    data[0]["epochs"].pop(100)
    with pytest.raises(ValueError, match="contiguous"):
        completion.validate_training(*data)


@pytest.mark.parametrize("mutate,match", [
    (lambda a, s, f: a["epochs"][0].update(environment_transitions=8192), "transitions"),
    (lambda a, s, f: a["epochs"][-1].update(epoch_loss_arrays_finite=False), "nonfinite"),
    (lambda a, s, f: a["checkpoint_events"].append({"operation": "restore"}), "restore"),
    (lambda a, s, f: f.update(status="error"), "finalizer"),
    (lambda a, s, f: s.update(stop_signals=["SIGTERM"]), "stopped"),
    (lambda a, s, f: s.update(raw_wait_returncode=1), "wait"),
    (lambda a, s, f: s.update(guard_required=True), "no-guard"),
])
def test_incomplete_or_altered_training_cannot_trigger_evaluation(mutate, match):
    audit, supervisor, finalizer, console = successful_training()
    mutate(audit, supervisor, finalizer)
    with pytest.raises(ValueError, match=match):
        completion.validate_training(audit, supervisor, finalizer, console)


def test_checkpoint_must_have_correct_actual_epoch_frame_and_finite_optimization():
    record = checkpoint(100)
    completion.validate_checkpoint(record, 100)
    record["frame"] = 100 * 8192
    with pytest.raises(ValueError, match="frame"):
        completion.validate_checkpoint(record, 100)
    record = checkpoint(400)
    record["all_tensors_and_numeric_arrays_finite"] = False
    with pytest.raises(ValueError, match="Nonfinite"):
        completion.validate_checkpoint(record, 400)


def test_evaluation_denominators_and_first_success_are_independent():
    early = completion.validate_evaluation(evaluation(100, 1001), checkpoint(100), 100, 1001)
    late = completion.validate_evaluation(evaluation(400, 1001), checkpoint(400), 400, 1001)
    assert (early["episodes_completed"], late["episodes_completed"]) == (32, 96)
    invalid = evaluation(400, 1001)
    invalid["episodes"][1]["first_success_time_seconds"] = 9
    with pytest.raises(ValueError, match="First-success"):
        completion.validate_evaluation(invalid, checkpoint(400), 400, 1001)
    invalid = evaluation(100, 1001)
    invalid["episodes"][-1] = copy.deepcopy(invalid["episodes"][0])
    with pytest.raises(ValueError, match="identifiers"):
        completion.validate_evaluation(invalid, checkpoint(100), 100, 1001)


def test_changed_sources_cannot_be_used(tmp_path):
    source = tmp_path / "official.py"
    source.write_text("original")
    frozen = {str(source): completion.sha(source)}
    completion.verify_sources(frozen)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        completion.verify_sources(frozen)


def test_periodic_and_final_epoch400_selects_last_audited_final_preserving_both(tmp_path):
    periodic = tmp_path / "last_Factory_ep_400_rew_17.0.pth"
    final = tmp_path / "last_Factory_ep_400_rew__17.0_.pth"
    periodic.write_bytes(b"periodic")
    final.write_bytes(b"final")
    saves = [{"operation": "save", "epoch": 400, "frame": 400 * 4096,
              "filename": str(path.with_suffix(""))} for path in (periodic, final)]
    selected, provenance = completion.select_checkpoint(tmp_path, {"checkpoint_events": saves}, 400)
    assert selected == final and provenance["selected_save_event"] == saves[-1]
    assert len(provenance["all_milestone_candidate_paths"]) == 2
    assert periodic.read_bytes() == b"periodic" and final.read_bytes() == b"final"
    with pytest.raises(ValueError, match="maximum-epoch final"):
        completion.select_checkpoint(tmp_path, {"checkpoint_events": saves[:1]}, 400)
    with pytest.raises(ValueError, match="No existing audited"):
        completion.select_checkpoint(tmp_path, {"checkpoint_events": []}, 400)
    relative = copy.deepcopy(saves)
    for row in relative:
        row["filename"] = Path(row["filename"]).name
    selected_relative, _ = completion.select_checkpoint(tmp_path, {"checkpoint_events": relative}, 400,
                                                         training_cwd=tmp_path)
    assert selected_relative == final


def manifest_fixture(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("immutable")
    sup = tmp_path / "training" / "supervisor"
    sup.mkdir(parents=True)
    audit, supervisor, finalizer, console = successful_training()
    (sup / "status.json").write_text(json.dumps(supervisor))
    (sup / "console.txt").write_text(console)
    audit_path, finalizer_path = tmp_path / "audit.json", tmp_path / "finalizer.json"
    audit_path.write_text(json.dumps(audit))
    finalizer_path.write_text(json.dumps(finalizer))
    nn = tmp_path / "training_checkpoints"
    nn.mkdir()
    audit["checkpoint_events"] = []
    for epoch in completion.EPOCHS:
        checkpoint_path = nn / f"last_Factory_ep_{epoch}_rew_12.5.pth"
        checkpoint_path.write_bytes(b"trusted local test artifact")
        audit["checkpoint_events"].append({"operation": "save", "epoch": epoch,
            "frame": epoch * 4096, "filename": str(checkpoint_path.with_suffix(""))})
    final_path = nn / "last_Factory_ep_400_rew__12.5_.pth"
    final_path.write_bytes(b"trusted local final test artifact")
    audit["checkpoint_events"].append({"operation": "save", "epoch": 400, "frame": 400 * 4096,
                                        "filename": str(final_path.with_suffix(""))})
    audit_path.write_text(json.dumps(audit))
    manifest = {"repository": str(tmp_path / "repository"), "workspace": str(tmp_path),
                "training_seed": 1, "training_num_envs": 32, "training_epochs": 400,
                "training_horizon": 128, "automatic_resume": False, "seed0_baseline": "G3C-R4",
                "training_supervisor_directory": str(sup), "training_audit": str(audit_path),
                "training_finalizer": str(finalizer_path), "training_event_directory": str(tmp_path / "events"),
                "checkpoint_directory": str(nn), "completion_output": str(tmp_path / "completion"),
                "python": "/venv/bin/python", "launcher": "/repository/scripts/g3cr5_launch.py",
                "evaluation_cwd": str(tmp_path), "evaluation_unit_prefix": "test-r4-eval",
                "isaacsim_path": "/sim", "launcher_arguments": ["--monitoring-policy", "user-authorized-no-gpu-guard"],
                "frozen_source_sha256": {str(source): completion.sha(source)}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return manifest, path


def test_controller_evaluates_once_in_registered_order_after_complete_training(tmp_path, monkeypatch):
    manifest, path = manifest_fixture(tmp_path)
    calls = []

    def fake_command(command, prefix, **kwargs):
        prefix = Path(prefix)
        calls.append(command)
        result = prefix.with_suffix(".stdout.txt")
        if "--inspect-checkpoint" in command:
            ckpt_path = Path(command[-1])
            epoch = int(ckpt_path.name.split("_ep_")[1].split("_")[0])
            result.write_text(json.dumps(checkpoint(epoch, str(ckpt_path), completion.sha(ckpt_path))))
        elif "--expected-epochs" in command:
            result.write_text(json.dumps({"status": "complete"}))
        else:
            output = Path(command[command.index("--output") + 1])
            (output / "supervisor").mkdir(parents=True)
            (output / "supervisor/status.json").write_text(json.dumps(successful_supervisor()))
            seed = int(command[command.index("--seed") + 1])
            ckpt_path = Path(command[command.index("--checkpoint") + 1])
            epoch = int(ckpt_path.name.split("_ep_")[1].split("_")[0])
            (output / "metrics.json").write_text(json.dumps(evaluation(epoch, seed, completion.sha(ckpt_path))))
            result.write_text("")
        return result

    monkeypatch.setattr(completion, "recorded_command", fake_command)
    assert completion.main(["--manifest", str(path)]) == 0
    final = completion.load(Path(manifest["completion_output"]) / "status.json")
    assert final["phase"] == "training_and_evaluation_complete"
    assert [(row["epoch"], row["seed"]) for row in final["evaluations"]] == [
        (epoch, seed) for epoch in completion.EPOCHS for seed in completion.SEEDS]
    assert len(calls) == 1 + 4 + 12
    for command in calls[5:]:
        assert "user-authorized-no-gpu-guard" in command and "--stochastic" not in command
    with pytest.raises(FileExistsError):
        completion.main(["--manifest", str(path)])


def test_failed_training_stops_without_checkpoint_or_evaluation_commands(tmp_path, monkeypatch):
    manifest, path = manifest_fixture(tmp_path)
    status = Path(manifest["training_supervisor_directory"]) / "status.json"
    failed = successful_supervisor()
    failed["raw_wait_returncode"] = 1
    status.write_text(json.dumps(failed))
    monkeypatch.setattr(completion, "recorded_command", lambda *args, **kwargs: pytest.fail("Must not evaluate"))
    assert completion.main(["--manifest", str(path)]) == 1
    final = completion.load(Path(manifest["completion_output"]) / "status.json")
    assert final["phase"] == "incomplete" and final["evaluations"] == []
    assert final["error"]["message"] == "Training child wait was not zero"
