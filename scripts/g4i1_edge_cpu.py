"""CPU source-based clipping/EMA fixtures; no actor weights, simulator or GPU.

The original Player consumes injected synthetic raw mu through get_action. Only
the SHA-verified FactoryEnv._pre_physics_step AST is executed, on a tensor proxy.
This validates clipping and EMA at edge cases. Position/quaternion targets and
the physical controller remain covered by separate original-source GPU fixtures.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.g4i0_interface import clip_policy_action, smooth_action
from scripts.g4i1_parity import compare_stages


def _verified_source(path, expected_sha):
    path = Path(path)
    data = path.read_bytes()
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"frozen source SHA mismatch: {path}")
    return data.decode("utf-8"), {"path": str(path.resolve()), "sha256": actual_sha}


def extract_pre_step(path, expected_sha):
    """Execute exactly one known method after verifying the whole source file."""
    source, identity = _verified_source(path, expected_sha)
    parsed = ast.parse(source, filename=str(path))
    classes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "FactoryEnv"]
    methods = ([node for node in classes[0].body if isinstance(node, ast.FunctionDef)
                and node.name == "_pre_physics_step"] if len(classes) == 1 else [])
    if len(methods) != 1 or methods[0].decorator_list:
        raise ValueError("expected one undecorated FactoryEnv._pre_physics_step")
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {}
    exec(compile(module, filename=str(path), mode="exec"), namespace)
    return namespace["_pre_physics_step"], identity


def run_edge_check(*, repo=None, source_path=None):
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    identity = json.loads((repo / "evidence/g4_i/i0/checkpoint_and_source_identity.json").read_text())
    inventory = {item["label"]: item for item in identity["source_inventory"]}
    factory = inventory["factory_env.py"]
    factory_pre_step, factory_identity = extract_pre_step(source_path or factory["path"], factory["sha256"])
    from rl_games.algos_torch.players import PpoPlayerContinuous
    _source, player_identity = _verified_source(inspect.getfile(PpoPlayerContinuous),
                                               inventory["rl_games_players.py"]["sha256"])

    # Source-derived reset cache has no [-1,1] guarantee. This fixture represents
    # (position - fixed_frame)/.05, with an independently specified reset yaw.
    pose = torch.tensor([[.08, -.065, .025]], dtype=torch.float32)
    frame = torch.zeros((1, 3), dtype=torch.float32)
    cache = torch.zeros((1, 6), dtype=torch.float32)
    cache[:, :3] = (pose - frame) / .05
    cache[:, 5] = -.3
    fixture_mu = {
        "normal": [.2, -.1, .3, .1, .2, -.1],
        "action_clipping": [2., -2., 3., -3., 4., -4.],
        "near_zero_rotation": [.2, .1, -.1, 0., 0., 0.],
        "position_saturation": [1., 1., 1., .3, .1, .1],
    }
    results = []
    for case, values in fixture_mu.items():
        raw_mu = torch.tensor([values], dtype=torch.float32)
        def fake_model(inputs):
            return {"mus": raw_mu.clone(), "actions": -raw_mu.clone(), "rnn_states": inputs["rnn_states"]}
        proxy_player = SimpleNamespace(has_batch_dimension=True, _preproc_obs=lambda value: value,
            states=None, model=fake_model, clip_actions=True,
            actions_low=-torch.ones(6), actions_high=torch.ones(6))
        reference_clipped = PpoPlayerContinuous.get_action(proxy_player, torch.zeros((1, 19)), is_deterministic=True)
        proxy_env = SimpleNamespace(reset_buf=torch.zeros(1, dtype=torch.bool), device="cpu",
            actions=cache.clone(), cfg=SimpleNamespace(ctrl=SimpleNamespace(ema_factor=.2)))
        factory_pre_step(proxy_env, reference_clipped)
        actual_clipped = clip_policy_action(raw_mu)
        actual_ema = smooth_action(actual_clipped, cache)
        pairs_ref = {"clipped": reference_clipped.numpy(), "ema": proxy_env.actions.numpy()}
        pairs_actual = {"clipped": actual_clipped.numpy(), "ema": actual_ema.numpy()}
        parity = compare_stages(pairs_ref, pairs_actual, device_kind="cpu", cases=[case], require_all_stages=False)
        results.append({"case": case, "passed": parity["passed"], "parity": parity,
                        "raw_mu": raw_mu.tolist(), "pose_derived_cache": cache.tolist(),
                        "reference_clipped": reference_clipped.tolist(), "actual_clipped": actual_clipped.tolist(),
                        "reference_ema": proxy_env.actions.tolist(), "actual_ema": actual_ema.tolist(),
                        "synthetic_injected_mu": True, "simulator_commands_submitted": 0})
    return {"schema_version": "g4_i1_edge_cpu_v1", "passed": all(item["passed"] for item in results),
            "scope": "Original Player clipping and original Factory pre-step EMA only; target/controller GPU fixtures are separate",
            "actor_weights_loaded_or_mutated": False, "gpu_used": False,
            "source_identity": {"factory_env": factory_identity, "official_player": player_identity},
            "cases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a new output path; historical evidence is preserved")
    result = run_edge_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
