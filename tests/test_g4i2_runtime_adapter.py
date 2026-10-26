"""CPU synthetic adapter integration, not simulator timing or policy evidence.

A minimal fake Factory module and CPU tensors exercise Telemetry's public call
sequence. No Isaac module, actor forward, CUDA event, or robot command is used.
"""
import csv
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from scripts.g4i1_policy import PolicyStep
from scripts.g4i2_controller_probe import TorqueProbe
from scripts.g4i2_runtime import Telemetry
from scripts.success_metrics import EpisodeSummary


def mock_torque(requested):
    # The actual probe executes this CPU mock code object once, intercepting the
    # same one-clamp signature used by the frozen upstream torque function.
    torque = torch.clamp(requested, min=-100.0, max=100.0)
    wrench = torch.ones((requested.shape[0], 6), dtype=requested.dtype)
    return torque, wrench


@pytest.fixture
def synthetic_runtime(monkeypatch, tmp_path):
    fc = ModuleType("isaaclab_tasks.direct.factory.factory_control")
    fc.compute_dof_torque = mock_torque
    package_names = ["isaaclab_tasks", "isaaclab_tasks.direct", "isaaclab_tasks.direct.factory"]
    packages = {name: ModuleType(name) for name in package_names}
    for package in packages.values(): package.__path__ = []
    packages[package_names[0]].direct = packages[package_names[1]]
    packages[package_names[1]].factory = packages[package_names[2]]
    packages[package_names[2]].factory_control = fc
    for name, package in packages.items(): monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, fc.__name__, fc)

    cuda_rng = Mock(return_value=torch.tensor([4, 5], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "get_rng_state", cuda_rng)
    # Any accidental CUDA use outside the mocked state receipt fails this CPU
    # test; profiler off must never allocate events or synchronize a device.
    monkeypatch.setattr(torch.cuda, "Event", Mock(side_effect=AssertionError("unexpected CUDA event")))
    monkeypatch.setattr(torch.cuda, "synchronize", Mock(side_effect=AssertionError("unexpected CUDA synchronization")))
    saved = []
    def save_cpu_receipt(value, path):
        saved.append((value, path))
        path.write_text("CPU synthetic RNG receipt\n")
    save = Mock(side_effect=save_cpu_receipt)
    monkeypatch.setattr(torch, "save", save)

    obs = torch.arange(19, dtype=torch.float32).reshape(1, 19) / 10
    state = (torch.zeros(2, 1, 8), torch.zeros(2, 1, 8))
    robot_data = SimpleNamespace(root_state_w=torch.zeros(1, 13), joint_pos=torch.zeros(1, 9),
                                 joint_vel=torch.zeros(1, 9), joint_effort_limits=torch.full((1, 9), 87.))
    asset_data = lambda: SimpleNamespace(root_state_w=torch.zeros(1, 13),
                                        joint_pos=torch.zeros(1, 0), joint_vel=torch.zeros(1, 0))
    observe = Mock(return_value={"obs": obs.clone()})
    physics = Mock(return_value="physics_cpu_mock")
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=1/15,
        _get_observations=observe, sim=SimpleNamespace(step=physics),
        _robot=SimpleNamespace(data=robot_data), _held_asset=SimpleNamespace(data=asset_data()),
        _fixed_asset=SimpleNamespace(data=asset_data()),
        cfg=SimpleNamespace(ctrl=SimpleNamespace(pos_action_bounds=[.04, .04, .04])),
        joint_pos=robot_data.joint_pos, joint_vel=robot_data.joint_vel,
        fingertip_midpoint_pos=torch.tensor([[.1, .2, .3]]),
        fingertip_midpoint_quat=torch.tensor([[1., 0., 0., 0.]]),
        ee_linvel_fd=torch.zeros(1, 3), ee_angvel_fd=torch.zeros(1, 3),
        held_pos=torch.tensor([[.2, .3, .4]]), held_quat=torch.tensor([[1., 0., 0., 0.]]),
        held_base_pos=torch.tensor([[.2, .3, .35]]), target_held_base_pos=torch.tensor([[.2, .3, .34]]),
        fixed_pos=torch.tensor([[.2, .3, .3]]), fixed_quat=torch.tensor([[1., 0., 0., 0.]]),
        fixed_pos_obs_frame=torch.tensor([[.2, .3, .3]]), fixed_pos_action_frame=torch.tensor([[.1, .2, .3]]),
        init_fixed_pos_obs_noise=torch.tensor([[.001, -.002, .003]]), actions=torch.zeros(1, 6),
        task_prop_gains=torch.ones(1, 6), task_deriv_gains=torch.ones(1, 6),
        pos_threshold=torch.full((1, 3), .02), rot_threshold=torch.full((1, 3), .1),
        ctrl_target_fingertip_midpoint_pos=torch.tensor([[.1, .2, .3]]),
        ctrl_target_fingertip_midpoint_quat=torch.tensor([[1., 0., 0., 0.]]), joint_torque=torch.zeros(1, 9))
    return SimpleNamespace(fc=fc, env=env, obs=obs, state=state, observe=observe,
                           physics=physics, cuda_rng=cuda_rng, save=save, saved=saved, output=tmp_path)


@pytest.mark.parametrize("diagnostics", ["full", "minimal"])
@pytest.mark.parametrize("terminal_success", [True, False])
def test_complete_public_api_and_json_csv_schema(synthetic_runtime, diagnostics, terminal_success):
    f = synthetic_runtime
    args = SimpleNamespace(profiling="off", diagnostics=diagnostics, device="cpu",
                           output=f.output, warmup=0, seed=1001)
    policy = SimpleNamespace(identity={"scope": "CPU mock, not actual inference"})
    telemetry = Telemetry(args, f.env, policy, f.state, f.obs)
    assert isinstance(telemetry.torque, TorqueProbe) == (diagnostics == "full")
    assert f.fc.compute_dof_torque is (telemetry.torque if diagnostics == "full" else mock_torque)
    assert len(f.saved) == 1
    f.cuda_rng.assert_called_once_with("cpu")
    assert set(f.saved[0][0]) == {"cpu", "cuda"}
    assert f.saved[0][1] == f.output / "initial_rng.pt"
    original_initial = f.env.held_pos.clone()
    raw = torch.tensor([[1.2, -.5, .4, -.3, .2, -.1]])
    step = PolicyStep(f.obs.clone(), f.obs.clone(), raw, torch.zeros_like(raw), f.state,
                      torch.clamp(raw, -1., 1.), False)
    cache = torch.zeros(1, 6)
    submitted = []
    try:
        telemetry.begin(1)
        telemetry.action(step, cache)
        def original_submission(actions):
            submitted.append(actions)
            # Simulate the original runtime callback order: targets, torque,
            # command admission marker, physics, then returned observation.
            f.env.actions = .2 * actions + .8 * cache
            telemetry.target_begin()
            f.env.ctrl_target_fingertip_midpoint_pos = f.env.fingertip_midpoint_pos + f.env.actions[:, :3] * f.env.pos_threshold
            telemetry.target_end(f.env)
            f.env.joint_torque, _wrench = f.fc.compute_dof_torque(torch.full((1, 9), 120.))
            telemetry.controller(f.env)
            f.env.sim.step(render=False)
            return f.env._get_observations()
        returned = telemetry.submit(original_submission)(step.clipped_action)
        assert submitted == [step.clipped_action]
        assert len(telemetry.command_stamps) == 1 and telemetry.command_stamps[0] > 0
        assert telemetry.admission["allowed"] is True
        assert returned["obs"].shape == (1, 19)
        f.observe.assert_called_once_with()
        f.physics.assert_called_once_with(render=False)
        episode = EpisodeSummary(env_id=0, episode_id=0, steps=1,
            first_success_step=1 if terminal_success else None,
            ever_success=terminal_success, terminal_success=terminal_success, terminal_kind="timeout")
        cap = {"success": [terminal_success], "step_numbers": [1],
               "geometric_snapshots": telemetry.snapshot()}
        telemetry.episodes([episode], cap, torch.tensor([True]), returned["obs"], f.state)
        # A deterministic synthetic reset makes preservation of the distinct
        # episode-0 initial state independently observable in episode-1 receipt.
        f.env.held_pos = original_initial + .01
        telemetry.end(returned["obs"], f.state)
        result = telemetry.finish(SimpleNamespace(completed=[episode]), {"status": "CPU_MOCK"})
        assert result["profiling"] == "off" and result["warmup"] == 0
        assert result["episodes"] == 1 and result["failures"] == int(not terminal_success)
        assert result["timing_finite"] is True
    finally:
        telemetry.close()
    assert f.fc.compute_dof_torque is mock_torque
    assert f.env._get_observations is f.observe and f.env.sim.step is f.physics
    assert telemetry.stream.closed

    records = [json.loads(line) for line in (f.output / "i2_trace.jsonl").read_text().splitlines()]
    by_kind = {}
    for record in records: by_kind.setdefault(record["kind"], []).append(record)
    assert set(by_kind) >= {"initial_state", "episode", "success_state", "timing", "pre_reset_state"}
    assert len(by_kind["initial_state"]) == 2
    assert [row["episode_id"] for row in by_kind["initial_state"]] == [0, 1]
    assert all(row["seed"] == 1001 for row in by_kind["initial_state"])
    assert by_kind["initial_state"][0]["randomized_state"]["held_pos"] == original_initial[0].tolist()
    assert by_kind["initial_state"][1]["randomized_state"]["held_pos"] == (original_initial + .01)[0].tolist()
    timing = by_kind["timing"][0]
    assert timing["warmup"] is False and timing["sim_dt_s"] == 1/15
    assert timing["wall_command_monotonic_ns"] == telemetry.command_stamps[0]
    assert timing["control_loop_ms"] >= 0 and timing["admission"]["allowed"]
    assert all(value >= 0 for value in timing["timings_ms"].values())
    assert timing["event_sections"] == [] and timing["step_timing"] is None
    summary = json.loads((f.output / "summary.json").read_text())
    assert set(summary) == {"timing", "actions", "controller", "episodes", "event_profile", "command_intervals", "admission"}
    assert summary["episodes"]["episodes_completed"] == 1
    assert summary["episodes"]["terminal_success_count"] == int(terminal_success)
    assert summary["timing"]["measured_records"] == 1
    assert summary["event_profile"]["mode"] == "off"
    assert summary["event_profile"]["measured_steps"] == 0
    assert summary["episodes"]["archived_comparison"]["kind"] == "descriptive_unpaired"
    if diagnostics == "full":
        assert len(by_kind["action"]) == len(by_kind["controller"]) == 1
        assert summary["actions"]["samples"] == summary["controller"]["samples"] == 1
        assert summary["actions"]["dimensions"][0]["raw_outside_unit_count"] == 1
        assert all(joint["software_changed_count"] == 1 for joint in summary["controller"]["joints"])
        assert summary["controller"]["applied_sim_effort"] == "UNKNOWN"
    else:
        assert "action" not in by_kind and "controller" not in by_kind
        assert summary["actions"]["status"] == "UNKNOWN_NO_ACTION_RECORDS"
        assert summary["controller"]["status"] == "UNKNOWN_NO_CONTROLLER_RECORDS"
    episodes = json.loads((f.output / "episodes.json").read_text())
    assert len(episodes) == 1
    assert episodes[0]["evaluation_seed"] == 1001
    assert episodes[0]["initial_randomized_configuration"]["seed"] == 1001
    assert episodes[0]["initial_randomized_configuration"]["randomized_state"]["held_pos"] == original_initial[0].tolist()
    assert episodes[0]["pre_reset_terminal_state"]["held_pos"] == original_initial[0].tolist()
    with (f.output / "episodes.csv").open(newline="") as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["evaluation_seed"] == "1001"
    assert rows[0]["initial_randomized_configuration"] != '"UNKNOWN"'
    assert json.loads(rows[0]["initial_randomized_configuration"])["seed"] == 1001
    assert rows[0]["pre_reset_terminal_state"] != '"UNKNOWN"'
    assert rows[0]["failure_category"] == ("terminal_success" if terminal_success else "never_success")
    assert len(json.loads((f.output / "failures.json").read_text())) == int(not terminal_success)
