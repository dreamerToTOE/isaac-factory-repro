"""Reproduce I0 checkpoint/RMS inspection on CPU; never run a policy forward.

Usage: CUDA_VISIBLE_DEVICES='' python -m scripts.audit_g4i0 --output NEW.json
Known trusted local files are SHA-256 checked BEFORE pickle deserialization.
The baseline identity file is the reviewed allowlist, not generated from input.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

import torch
import yaml

from scripts.g4i0_interface import (
    OBSERVATION_ORDER, normalize_observation, verify_checkpoint_identity,
    validate_checkpoint_compatibility,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--workspace", type=Path, default=repo.parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new evidence path; existing evidence is preserved")
    torch.set_num_threads(2)
    frozen = json.loads((repo / "evidence/g4_i/i0/checkpoint_and_source_identity.json").read_text())
    result = {"scope": "CPU hash/metadata/RMS inspection; no policy forward, simulator, or GPU",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "python": platform.python_version(), "packages": {}, "checkpoints": {}, "source_files": {}}
    for name in ("torch", "rl-games", "gymnasium", "numpy", "PyYAML"):
        result["packages"][name] = importlib.metadata.version(name)
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    for role, record in frozen["checkpoints"].items():
        path = args.workspace / record["workspace_relative_path"]
        digest = verify_checkpoint_identity(path, record["sha256"])
        # Only the three reviewed hashes above reach this trusted pickle loader.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = validate_checkpoint_compatibility(checkpoint, record["model_tensor_shapes"],
                                                   epoch=record["epoch"], frame=record["frame"])
        rms = {key: model["running_mean_std." + key] for key in ("running_mean", "running_var", "count")}
        official = RunningMeanStd((19,)).eval()
        official.load_state_dict(rms, strict=True)
        inputs = torch.stack([torch.linspace(-10, 10, 19), torch.linspace(1, -1, 19)])
        before = {key: value.clone() for key, value in official.state_dict().items()}
        with torch.inference_mode():
            expected = official(inputs)
            actual = normalize_observation(inputs, rms["running_mean"], rms["running_var"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(torch.equal(value, before[key]) for key, value in official.state_dict().items())
        agent_path, env_path = (path.parent.parent / "params" / name for name in ("agent.yaml", "env.yaml"))
        agent = yaml.safe_load(agent_path.read_text())
        assert agent["params"]["network"] == record["architecture"]["actor_critic"]
        assert agent["params"]["seed"] == record["training_seed"]
        result["checkpoints"][role] = {
            "path": str(path), "sha256": digest, "bytes": path.stat().st_size,
            "epoch": checkpoint["epoch"], "frame": checkpoint["frame"],
            "model_keys_shapes_dtypes_finite": "PASS", "rms": {k: v.tolist() for k, v in rms.items()},
            "rms_storage_dtype": "float64", "rms_calculation_dtype": "float32",
            "epsilon": official.epsilon, "normalized_clip": [-5.0, 5.0],
            "official_rms_cpu_parity_max_abs": (actual - expected).abs().max().item(),
            "rms_eval_statistics_unchanged": True, "rnn_runtime_state_in_checkpoint": False,
            "env_state": checkpoint.get("env_state"),
            "agent_config_path": str(agent_path), "agent_config_sha256": sha(agent_path),
            "env_config_path": str(env_path), "env_config_sha256": sha(env_path),
            "agent_configuration": agent,
            "num_actors_dump_caveat": "Pre-wrapper num_actors=128; actual env count comes from env config and historical runtime evidence",
        }
        del checkpoint, model
    lab = args.workspace / "work/upstream/IsaacLab"
    factory = lab / "source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py"
    tree = ast.parse(factory.read_text())
    class_cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FactoryEnvCfg")
    obs = next(ast.literal_eval(n.value) for n in class_cfg.body
               if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "obs_order")
    assert tuple(obs + ["prev_actions"]) == OBSERVATION_ORDER
    result["actual_source_observation_order_ast_verified"] = obs + ["prev_actions"]
    import rl_games
    rlg = Path(rl_games.__file__).parent
    paths = {
        "factory_env_cfg.py": factory,
        "factory_env.py": factory.with_name("factory_env.py"),
        "factory_control.py": factory.with_name("factory_control.py"),
        "direct_rl_env.py": lab / "source/isaaclab/isaaclab/envs/direct_rl_env.py",
        "direct_rl_env_cfg.py": lab / "source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py",
        "rl_games_wrapper.py": lab / "source/isaaclab_rl/isaaclab_rl/rl_games.py",
        "models.py": rlg / "algos_torch/models.py", "network_builder.py": rlg / "algos_torch/network_builder.py",
        "running_mean_std.py": rlg / "algos_torch/running_mean_std.py", "players.py": rlg / "algos_torch/players.py",
        "base_player.py": rlg / "common/player.py", "recurrent.py": rlg / "common/layers/recurrent.py",
        "evaluate_checkpoint.py": repo / "scripts/evaluate_checkpoint.py",
        "evaluation_runtime.py": repo / "scripts/evaluation_runtime.py",
    }
    for name, path in paths.items():
        result["source_files"][name] = {"path": str(path), "sha256": sha(path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "checkpoints_verified": len(result["checkpoints"]),
                      "source_files_hashed": len(paths), "official_rms_cpu_parity": "PASS"}))


if __name__ == "__main__":
    main()
