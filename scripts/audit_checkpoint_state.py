"""Audit our trusted RL-Games checkpoints on CPU, without starting Isaac Sim.

The load is deliberately explicit weights_only=False: never run this on an
untrusted checkpoint. This is observation and a CPU restore test, not training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from rl_games.algos_torch import a2c_continuous, central_value, model_builder
from rl_games.common.ivecenv import IVecEnv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_restore_target(params):
    """Use the actual official networks and actual restore method on CPU.

    Skip the agent constructor, which requires a live environment; provide just
    the fields read by set_full_state_weights. No optimizer update is performed.
    """
    config = params["config"]
    target = a2c_continuous.A2CAgent.__new__(a2c_continuous.A2CAgent)
    target.model = model_builder.ModelBuilder().load(params).build(
        {"actions_num": 6, "input_shape": (19,), "num_seqs": 4,
         "value_size": 1, "normalize_value": True, "normalize_input": True}
    )
    target.optimizer = torch.optim.Adam(target.model.parameters(), lr=config["learning_rate"])
    cvconfig = config["central_value_config"]
    cvnetwork = model_builder.ModelBuilder().load(
        {"model": {"name": "central_value"}, "network": cvconfig["network"]}
    )
    target.central_value_net = central_value.CentralValueTrain(
        state_shape=(43,), value_size=1, ppo_device="cpu", num_agents=1,
        horizon_length=128, num_actors=4, num_actions=6, seq_length=128,
        normalize_value=True, network=cvnetwork, config=cvconfig, writter=None,
        max_epochs=200, multi_gpu=False, zero_rnn_on_done=True,
    )
    target.has_central_value = True
    target.normalize_input = target.normalize_value = True
    target.normalize_rms_advantage = False
    target.mixed_precision = True
    # AMP's serialized scalar state is device independent; this test uses CPU.
    target.scaler = torch.amp.GradScaler("cpu", enabled=True)
    target.vec_env = IVecEnv()
    target.last_lr = config["learning_rate"]
    target.entropy_coef = config["entropy_coef"]
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checksum = digest(args.checkpoint)
    if checksum != args.expected_sha256:
        raise SystemExit("Checkpoint SHA256 does not match the supplied trusted digest")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = yaml.safe_load(args.official_config.read_text())["params"]
    target = make_restore_target(params)
    target.set_full_state_weights(state)
    actor_equal = all(torch.equal(v, target.model.state_dict()[k]) for k, v in state["model"].items())
    critic_equal = all(torch.equal(v, target.central_value_net.state_dict()[k])
                       for k, v in state["assymetric_vf_nets"].items())
    actor_opt = target.optimizer.state_dict()
    critic_opt = target.central_value_net.optimizer.state_dict()
    fresh_critic_optimizer_empty = len(critic_opt["state"]) == 0
    source_actor_adam = state["optimizer"]["state"]
    actor_adam_equal = actor_opt["param_groups"] == state["optimizer"]["param_groups"] and len(source_actor_adam) == len(actor_opt["state"]) and all(
        all(torch.equal(v, actor_opt["state"][k][name]) if torch.is_tensor(v)
            else v == actor_opt["state"][k][name] for name, v in item.items())
        for k, item in source_actor_adam.items()
    )

    # A controlled check of the actual save method: an optimizer state and lr
    # attached to CentralValueTrain are excluded by inherited nn.Module.state_dict.
    p = next(target.central_value_net.model.parameters())
    target.central_value_net.optimizer.state[p] = {
        "step": torch.tensor(7.), "exp_avg": torch.ones_like(p),
        "exp_avg_sq": torch.ones_like(p),
    }
    target.last_lr = 2.0e-5
    saved_again = target.get_full_state_weights()
    critic_optimizer_omitted = not any("optim" in k for k in saved_again["assymetric_vf_nets"])
    last_lr_omitted = "last_lr" not in saved_again
    target.central_value_net.optimizer.state.clear()
    target.last_lr = params["config"]["learning_rate"]
    target.set_full_state_weights(saved_again)

    normalization = {}
    for name in ("model", "assymetric_vf_nets"):
        normalization[name] = {
            k: {"shape": list(v.shape), "finite": bool(torch.isfinite(v).all()),
                **({"value": float(v)} if v.numel() == 1 else {})}
            for k, v in state[name].items() if "mean_std" in k
        }
    checks = {
        "strict_official_actor_model_restore_equal": actor_equal,
        "strict_official_central_critic_model_restore_equal": critic_equal,
        "actor_optimizer_restore_equal": actor_adam_equal,
        "amp_scaler_restore_equal": target.scaler.state_dict() == state["scaler"],
        "epoch_frame_restore_equal": target.epoch_num == state["epoch"] and target.frame == state["frame"],
        "fresh_critic_optimizer_empty_after_real_restore": fresh_critic_optimizer_empty,
        "controlled_critic_optimizer_omitted_by_actual_save": critic_optimizer_omitted,
        "controlled_critic_optimizer_still_empty_after_restore": not target.central_value_net.optimizer.state,
        "controlled_last_lr_omitted_by_actual_save": last_lr_omitted,
        "controlled_last_lr_remains_config_after_restore": target.last_lr == params["config"]["learning_rate"],
    }
    result = {
        "scope": "CPU strict restore and controlled save/restore tests; no GPU or policy evaluation",
        "checkpoint": str(args.checkpoint.resolve()), "sha256": checksum,
        "checkpoint_top_level_keys": sorted(state),
        "epoch": state["epoch"], "frame": state["frame"],
        "model_state_tensor_count": len(state["model"]),
        "central_critic_state_tensor_count": len(state["assymetric_vf_nets"]),
        "actor_optimizer_state_count": len(source_actor_adam),
        "actor_optimizer_saved_lr": [x["lr"] for x in state["optimizer"]["param_groups"]],
        "actor_last_lr_after_restore": target.last_lr,
        "central_critic_optimizer_state_count_after_restore": len(critic_opt["state"]),
        "central_critic_epoch_frame_after_restore": [target.central_value_net.epoch_num, target.central_value_net.frame],
        "central_critic_scheduler_class": type(target.central_value_net.scheduler).__name__,
        "scaler": state["scaler"], "env_state": state.get("env_state"),
        "central_val_stats": state.get("central_val_stats"),
        "normalization_buffers": normalization,
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "all_checks_passed": result["all_checks_passed"]}))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
