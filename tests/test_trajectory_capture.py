import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.trajectory_capture import SCHEMA_VERSION, VectorTrajectoryRecorder, save_trajectory_npz


class TinyFactory:
    """Independent CPU control: only the original method changes task state."""

    def __init__(self):
        self.num_envs = 2
        self.device = "cpu"
        self.scene = SimpleNamespace(env_origins=torch.tensor([[10., 20., 0.], [-10., 0., 0.]]))
        self.actions = torch.zeros(2, 6)
        self.pre_physics_calls = 0
        self.episode_length_buf = torch.tensor([149, 149])
        self.reset_buf = torch.tensor([True, True])
        self.reset_time_outs = torch.tensor([True, True])
        self.held_pos = torch.tensor([[.003, .004, .02], [0., 0., .03]])
        self.fixed_pos = torch.zeros(2, 3)
        self.held_base_pos = self.held_pos.clone()
        self.target_held_base_pos = torch.tensor([[0., 0., .01], [0., 0., .01]])
        self.fingertip_midpoint_pos = torch.tensor([[0., 0., .1], [0., 0., .1]])
        self.fingertip_midpoint_linvel = torch.zeros(2, 3)
        self.fingertip_midpoint_angvel = torch.zeros(2, 3)
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos.clone()
        for name in ("held_quat", "fixed_quat", "held_base_quat", "target_held_base_quat",
                     "fingertip_midpoint_quat", "ctrl_target_fingertip_midpoint_quat"):
            setattr(self, name, torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]))

    def _pre_physics_step(self, action):
        self.pre_physics_calls += 1
        self.actions = .25 * action + .75 * self.actions
        return "unchanged-return"


def test_pre_physics_tap_preserves_behavior_and_clones_distinct_command_stages():
    env = TinyFactory()
    recorder = VectorTrajectoryRecorder(env, max_steps=2, seed=1001)
    original = env._pre_physics_step
    recorder.install_action_tap()
    action = torch.ones(2, 6)
    assert env._pre_physics_step(action) == "unchanged-return"
    assert env.pre_physics_calls == 1
    assert torch.equal(env.actions, torch.full((2, 6), .25))
    assert torch.equal(action, torch.ones(2, 6))
    action.zero_()
    assert torch.equal(recorder.policy_actions, torch.ones(2, 6))
    recorder.restore_action_tap()
    assert env._pre_physics_step == original


def test_pre_reset_terminal_data_world_frames_and_episode_ids_survive_reset():
    env = TinyFactory()
    recorder = VectorTrajectoryRecorder(env, max_steps=2, seed=1001)
    recorder.install_action_tap()
    env._pre_physics_step(torch.ones(2, 6))
    success = torch.tensor([False, True])
    recorder.record_success(env, success)
    recorder.record_engagement(torch.tensor([True, True]))
    # Imitate automatic reset after the reward phase. Capture must retain 149
    # and terminal geometry, while the next sample belongs to episode 1.
    success.zero_()
    env.held_pos.zero_()
    env.episode_length_buf.fill_(1)
    env.reset_buf.zero_()
    env.reset_time_outs.zero_()
    recorder.record_success(env, torch.tensor([False, False]))
    recorder.record_engagement(torch.tensor([False, True]))
    arrays = recorder.arrays()
    assert arrays["policy_step"].tolist() == [[149, 149], [1, 1]]
    assert arrays["episode_id"].tolist() == [[0, 0], [1, 1]]
    assert arrays["done"].tolist() == [[True, True], [False, False]]
    assert arrays["official_success"].tolist() == [[False, True], [False, False]]
    assert arrays["engaged"].tolist() == [[True, True], [False, True]]
    assert arrays["peg_pos_w"][0, 0].tolist() == pytest.approx([10.003, 20.004, .02])
    assert arrays["hole_pos_w"][0, 0].tolist() == pytest.approx([10., 20., 0.])
    assert arrays["xy_error_m"][0].tolist() == pytest.approx([.005, 0.])
    assert arrays["axial_displacement_m"][0].tolist() == pytest.approx([.01, .02])
    assert arrays["peg_quat_w"][0, 0].tolist() == [1., 0., 0., 0.]


def test_capture_only_mutates_its_own_buffers_and_never_task_tensors():
    env = TinyFactory()
    recorder = VectorTrajectoryRecorder(env, max_steps=1, seed=1003)
    recorder.policy_actions = torch.ones(2, 6)
    before = {name: value.clone() for name, value in vars(env).items() if torch.is_tensor(value)}
    success = torch.tensor([False, True])
    recorder.record_success(env, success)
    recorder.record_engagement(torch.tensor([False, True]))
    for name, value in before.items():
        assert torch.equal(getattr(env, name), value), name
    assert success.tolist() == [False, True]
    with pytest.raises(RuntimeError, match="fixed policy-step bound"):
        recorder.record_success(env, success)


def test_capture_rejects_missing_action_or_wrong_engagement_order():
    recorder = VectorTrajectoryRecorder(TinyFactory(), max_steps=2, seed=1001)
    with pytest.raises(RuntimeError, match="call order"):
        recorder.record_engagement(torch.tensor([True, False]))
    with pytest.raises(RuntimeError, match="no associated policy action"):
        recorder.record_success(recorder.env, torch.tensor([True, False]))
    recorder.policy_actions = torch.zeros(2, 6)
    recorder.record_success(recorder.env, torch.tensor([True, False]))
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.arrays()
    recorder.record_engagement(torch.tensor([True, False]))
    with pytest.raises(RuntimeError, match="call order"):
        recorder.record_engagement(torch.tensor([True, False]))


def test_serialization_is_pickle_free_and_retains_identity_provenance(tmp_path):
    path = tmp_path / "trajectory.npz"
    metadata = {"schema_version": SCHEMA_VERSION, "seed": 1001, "checkpoint_sha256": "a" * 64,
                "quaternion_order": "wxyz", "position_frame": "world"}
    result = save_trajectory_npz(path, {"policy_step": [[1, 1], [2, 2]],
                                        "official_success": [[False, False], [True, False]]}, metadata)
    with np.load(path, allow_pickle=False) as loaded:
        assert json.loads(str(loaded["metadata_json"])) == metadata
        assert loaded["policy_step"].tolist() == [[1, 1], [2, 2]]
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        save_trajectory_npz(path, {"policy_step": [[1, 1]]}, metadata)


@pytest.mark.parametrize("arrays", [{}, {"scalar": [1, 2]},
    {"a": [[1, 2]], "b": [[1], [2]]}, {"a": [[float("nan")]]},
    {"a": np.array([[object()]], dtype=object)}])
def test_serialization_rejects_invalid_or_unbounded_missing_data(tmp_path, arrays):
    with pytest.raises(ValueError):
        save_trajectory_npz(tmp_path / "invalid.npz", arrays, {"schema_version": SCHEMA_VERSION})
    assert not (tmp_path / "invalid.npz").exists()


def test_production_wrapper_persists_before_close_and_keeps_predicate_returns(tmp_path, monkeypatch):
    from scripts import diagnose_checkpoint, success_metrics

    class ControlledFactory(TinyFactory):
        def __init__(self):
            super().__init__()
            self.cfg_task = SimpleNamespace(name="peg_insert", success_threshold=.04, engage_threshold=.9,
                                            fixed_asset_cfg=SimpleNamespace(height=.025),
                                            held_asset_cfg=SimpleNamespace(height=.05))
            self.step_dt = 1 / 15
            self.max_episode_length = 150
            self.last_result = None
            self.closed = False

        def _get_curr_successes(self, success_threshold, check_rot=False):
            # An independent controlled result: success [False, True], engagement
            # [True, True]. The production observer must return this SAME object.
            self.last_result = torch.tensor([success_threshold > .1, True])
            return self.last_result

        def close(self):
            self.closed = True

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"controlled-checkpoint")
    monkeypatch.setattr(diagnose_checkpoint, "EXPECTED_CHECKPOINT_SHA256",
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    trajectory = tmp_path / "trajectories.npz"
    metadata_path = tmp_path / "trajectory_metadata.json"
    result_path = tmp_path / "metrics.json"
    env = ControlledFactory()
    original_predicate = env._get_curr_successes
    original_pre_physics = env._pre_physics_step
    original_capture = success_metrics.PreResetSuccessCapture

    def controlled_evaluator(path, run_name):
        assert path.endswith("evaluate_checkpoint.py")
        with success_metrics.PreResetSuccessCapture(env, convert=lambda value: value.tolist()) as capture:
            env._pre_physics_step(torch.ones(2, 6))
            result = env._get_curr_successes(.04)
            assert result is env.last_result
            assert result.tolist() == [False, True]
            engagement = env._get_curr_successes(.9)
            assert engagement is env.last_result
            assert capture.consume()["step_numbers"] == [149, 149]
        # No evaluator metrics or close have occurred yet: shutdown cannot lose
        # the already serialized trajectory and identity metadata.
        assert trajectory.is_file() and metadata_path.is_file()
        assert not env.closed
        assert env._get_curr_successes == original_predicate
        assert env._pre_physics_step == original_pre_physics
        with np.load(trajectory, allow_pickle=False) as loaded:
            assert loaded["official_success"].tolist() == [[False, True]]
            assert loaded["engaged"].tolist() == [[True, True]]
            assert loaded["policy_step"].tolist() == [[149, 149]]
        result_path.write_text(json.dumps({"status": "complete", "aggregate": {"episodes_completed": 2},
                                           "episodes_completed": 2}))
        env.close()
        # The close hook links final primary metrics BEFORE the original close.
        assert json.loads(metadata_path.read_text())["evaluation_output"]["episodes_completed"] == 2
        assert env.closed
        return {}

    monkeypatch.setattr(diagnose_checkpoint.runpy, "run_path", controlled_evaluator)
    monkeypatch.setattr(sys, "argv", ["diagnose_checkpoint.py", "--trajectory-output", str(trajectory),
        "--diagnostic-metadata", str(metadata_path), "--checkpoint", str(checkpoint),
        "--output", str(result_path), "--task", "Isaac-Factory-PegInsert-Direct-v0",
        "--seed", "1001", "--num_envs", "32", "--episodes", "1", "--max_steps", "149"])
    diagnose_checkpoint.main()
    assert success_metrics.PreResetSuccessCapture is original_capture
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "complete"
    assert metadata["trajectory"]["sha256"] == hashlib.sha256(trajectory.read_bytes()).hexdigest()
    assert metadata["contact_signals"]["available"] is False
    assert metadata["contact_signals"]["applied_wrench_used"] is False
