"""Inspect trusted locally generated RL-Games checkpoints on CPU; never fetch a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--compare", type=Path)
args = parser.parse_args()

def inspect(path):
    # RL-Games 1.6.1 stores numpy objects as well as tensors. Use only our own trusted files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(x) for x in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(x) for x in value)
        return True
    optim = state["optimizer"]["state"]
    steps = [int(value["step"]) for value in optim.values() if "step" in value]
    nonzero = sum(bool(torch.count_nonzero(value["exp_avg"])) for value in optim.values() if "exp_avg" in value)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return state, {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest,
                   "epoch": state.get("epoch"), "frame": state.get("frame"), "all_tensors_finite": finite(state),
                   "optimizer_states": len(optim), "optimizer_step_min": min(steps, default=0),
                   "optimizer_step_max": max(steps, default=0), "nonzero_adam_exp_avg_states": nonzero,
                   "central_value_present": "assymetric_vf_nets" in state}

state, result = inspect(args.checkpoint)
if args.compare:
    previous, previous_result = inspect(args.compare)
    differences = {}
    for group in ("model", "assymetric_vf_nets"):
        delta = {name: float((value.double() - previous[group][name].double()).abs().max())
                 for name, value in state[group].items()
                 if torch.is_tensor(value) and name in previous[group] and ("weight" in name or "bias" in name)}
        differences[group] = {"changed_parameter_tensors": sum(x > 0 for x in delta.values()),
                              "max_abs_parameter_change": max(delta.values(), default=0.)}
    result["comparison"] = {"previous": previous_result, "parameter_differences": differences}
result["optimization_evidence_passed"] = (result["all_tensors_finite"] and result["optimizer_step_min"] > 0
                                           and result["nonzero_adam_exp_avg_states"] > 0)
print(json.dumps(result, indent=2))
if not result["optimization_evidence_passed"]:
    raise SystemExit(1)
