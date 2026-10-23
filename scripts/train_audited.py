"""Run official RL-Games training with read-only optimization observations.

Use through run_sim.sh. This file does not replace a loss, gradient, optimizer,
environment method, seed, or hyperparameter. Tensor inspections synchronize CUDA
and therefore measured training times include audit overhead.
"""
import argparse
import atexit
import datetime
import functools
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--upstream", type=Path, required=True)
args, training_args = parser.parse_known_args()
upstream = args.upstream.resolve(strict=True)
output = Path(os.environ["FACTORY_AUDIT_OUTPUT"]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise SystemExit(f"Refusing to overwrite existing audit: {output}")

audit = {
    "schema_version": 1,
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "upstream_script": str(upstream),
    "upstream_arguments": training_args,
    "note": "Read-only tensor inspections synchronize CUDA; timings include audit overhead. "
            "Post-unscale gradients precede official clipping; optimizer gradients follow clipping. "
            "An AMP-skipped attempt is not an applied nonfinite optimizer update.",
    "instrumentation_errors": [],
    "runtime": {},
    "actor_attempts": [],
    "central_value_attempts": [],
    "gradient_events": [],
    "epochs": [],
    "checkpoint_events": [],
    "optimizer_steps": {"actor": 0, "central_value": 0},
}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def allocator():
    if not torch.cuda.is_initialized():
        return {}
    return {"device": torch.cuda.current_device(),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "memory_allocated_bytes": torch.cuda.memory_allocated(),
            "memory_reserved_bytes": torch.cuda.memory_reserved()}


def flush():
    audit["cuda_allocator"] = allocator()
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(plain(audit), indent=2, allow_nan=False) + "\n")
    temp.replace(output)


atexit.register(flush)


def observed(name, callback):
    """Keep an audit failure explicit without changing the official computation."""
    try:
        return callback()
    except Exception as error:
        audit["instrumentation_errors"].append({"point": name, "error": repr(error)})
        print(f"FACTORY_AUDIT_ERROR {name}: {error!r}", flush=True)
        return None


@torch.no_grad()
def tensor_stats(value):
    if not isinstance(value, torch.Tensor):
        return {"value": float(value), "finite": math.isfinite(float(value))}
    value = value.detach()
    if not value.numel():
        return {"count": 0, "finite": True}
    finite = torch.isfinite(value)
    result = {"count": value.numel(), "finite": bool(finite.all().item()),
              "nonfinite_count": int((~finite).sum().item())}
    if value.numel() == 1:
        result["value"] = value.item()
    else:
        result.update({"minimum": value.min().item(), "maximum": value.max().item()})
    return result


@torch.no_grad()
def gradient_stats(optimizer):
    gradients = [p.grad.detach() for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not gradients:
        return {"gradient_tensors": 0, "gradient_elements": 0, "finite": True,
                "nonfinite_elements": 0, "maximum_absolute_gradient": None}
    # One CPU transfer for all summaries, without retaining tensors in the audit.
    rows = torch.stack([torch.stack(((~torch.isfinite(g)).sum().to(torch.float64),
                                     g.abs().max().to(torch.float64))) for g in gradients]).cpu()
    nonfinite = int(rows[:, 0].sum().item())
    return {"gradient_tensors": len(gradients), "gradient_elements": sum(g.numel() for g in gradients),
            "finite": nonfinite == 0, "nonfinite_elements": nonfinite,
            "maximum_absolute_gradient": rows[:, 1].max().item()}


def gradient_event(role, phase, optimizer, **extra):
    audit["gradient_events"].append({"role": role, "phase": phase, **extra, **gradient_stats(optimizer)})


def register_optimizer(optimizer, role):
    def before_step(opt, positional, keyword):
        observed(f"{role}.optimizer_step_gradients", lambda: gradient_event(
            role, "optimizer_pre_step_post_clip", opt, actual_step=audit["optimizer_steps"][role] + 1))

    def after_step(opt, positional, keyword):
        audit["optimizer_steps"][role] += 1

    optimizer.register_step_pre_hook(before_step)
    optimizer.register_step_post_hook(after_step)


def install_observers():
    from rl_games.algos_torch.a2c_continuous import A2CAgent
    from rl_games.algos_torch.central_value import CentralValueTrain

    original_init = A2CAgent.__init__

    @functools.wraps(original_init)
    def audited_agent_init(self, *positional, **keyword):
        original_init(self, *positional, **keyword)
        register_optimizer(self.optimizer, "actor")
        if self.has_central_value:
            register_optimizer(self.central_value_net.optimizer, "central_value")
        audit["runtime"] = {
            "num_actors": self.num_actors, "num_agents": self.num_agents,
            "horizon_length": self.horizon_length, "batch_size": self.batch_size,
            "minibatch_size": self.minibatch_size, "mini_epochs": self.mini_epochs_num,
            "seq_length": self.seq_length, "observation_shape": self.obs_shape,
            "state_shape": self.state_shape, "actions_num": self.actions_num,
            "mixed_precision": self.mixed_precision,
            "config": plain(self.config), "actor_model": str(self.model),
            "actor_parameter_shapes": {n: list(p.shape) for n, p in self.model.named_parameters()},
            "actor_parameter_count": sum(p.numel() for p in self.model.parameters()),
            "central_value_model": str(self.central_value_net.model) if self.has_central_value else None,
            "central_value_parameter_shapes": {n: list(p.shape) for n, p in self.central_value_net.model.named_parameters()}
                if self.has_central_value else {},
        }
        original_unscale = self.scaler.unscale_

        @functools.wraps(original_unscale)
        def audited_unscale(optimizer):
            attempt = len(audit["actor_attempts"]) + 1
            observed("actor.raw_scaled_gradients", lambda: gradient_event(
                "actor", "before_unscale_scaled_amp", optimizer, attempt=attempt,
                scale=self.scaler.get_scale()))
            result = original_unscale(optimizer)
            observed("actor.unscaled_gradients", lambda: gradient_event(
                "actor", "after_unscale_before_clip", optimizer, attempt=attempt,
                scale=self.scaler.get_scale()))
            return result

        self.scaler.unscale_ = audited_unscale
        observed("initial_flush", flush)

    A2CAgent.__init__ = audited_agent_init
    original_actor = A2CAgent.train_actor_critic

    @functools.wraps(original_actor)
    def audited_actor(self, *positional, **keyword):
        before_steps = audit["optimizer_steps"]["actor"]
        before_scale = self.scaler.get_scale()
        result = original_actor(self, *positional, **keyword)

        def record():
            names = {0: "actor_loss", 1: "actor_value_loss", 2: "entropy", 3: "kl", 8: "bounds_loss"}
            audit["actor_attempts"].append({
                "attempt": len(audit["actor_attempts"]) + 1, "epoch": self.epoch_num,
                "optimizer_steps_applied": audit["optimizer_steps"]["actor"] - before_steps,
                "amp_skipped": audit["optimizer_steps"]["actor"] == before_steps,
                "scale_before": before_scale, "scale_after": self.scaler.get_scale(),
                "learning_rate": self.last_lr,
                "scalars": {name: tensor_stats(result[index]) for index, name in names.items()},
                "total_objective": tensor_stats(result[0] + 0.5 * result[1] * self.critic_coef
                    - result[2] * self.entropy_coef + result[8] * self.bounds_loss_coef),
                "policy_mean": tensor_stats(result[6]), "policy_sigma": tensor_stats(result[7]),
            })
        observed("actor.losses", record)
        return result

    A2CAgent.train_actor_critic = audited_actor
    original_central = CentralValueTrain.calc_gradients

    @functools.wraps(original_central)
    def audited_central(self, *positional, **keyword):
        before_steps = audit["optimizer_steps"]["central_value"]
        loss = original_central(self, *positional, **keyword)
        observed("central_value.loss", lambda: audit["central_value_attempts"].append({
            "attempt": len(audit["central_value_attempts"]) + 1,
            "epoch": self.epoch_num, "loss": tensor_stats(loss),
            "optimizer_steps_applied": audit["optimizer_steps"]["central_value"] - before_steps,
        }))
        return loss

    CentralValueTrain.calc_gradients = audited_central
    original_epoch = A2CAgent.train_epoch

    @functools.wraps(original_epoch)
    def audited_epoch(self, *positional, **keyword):
        start = time.perf_counter()
        result = original_epoch(self, *positional, **keyword)

        def record():
            step_time, play_time, update_time, total_time = result[:4]
            frames = self.curr_frames
            audit["epochs"].append({"epoch": self.epoch_num, "environment_transitions": frames,
                "env_step_seconds": step_time, "rollout_seconds": play_time,
                "optimization_seconds": update_time, "official_total_seconds": total_time,
                "observed_epoch_seconds": time.perf_counter() - start,
                "env_step_fps": frames / step_time, "rollout_fps": frames / play_time,
                "total_fps": frames / total_time, "cuda_allocator": allocator()})
            flush()
        observed("epoch", record)
        return result

    A2CAgent.train_epoch = audited_epoch
    for operation in ("save", "restore"):
        original = getattr(A2CAgent, operation)

        def make_wrapper(original_method, event):
            @functools.wraps(original_method)
            def wrapper(self, filename, *positional, **keyword):
                result = original_method(self, filename, *positional, **keyword)
                observed(event, lambda: audit["checkpoint_events"].append({
                    "operation": event, "filename": str(filename), "epoch": self.epoch_num,
                    "frame": self.frame, "success": True,
                }))
                observed("checkpoint_flush", flush)
                return result
            return wrapper
        setattr(A2CAgent, operation, make_wrapper(original, operation))

    audit["observers_installed"] = True
    print("FACTORY_TRAIN_AUDIT=" + str(output), flush=True)


# Install only after Kit initializes, matching the official import ordering.
original_launcher_init = AppLauncher.__init__


@functools.wraps(original_launcher_init)
def audited_launcher_init(self, *positional, **keyword):
    original_launcher_init(self, *positional, **keyword)
    install_observers()


AppLauncher.__init__ = audited_launcher_init
sys.argv = [str(upstream), *training_args]
try:
    runpy.run_path(str(upstream), run_name="__main__")
except BaseException as error:
    audit["exit_error"] = repr(error)
    raise
finally:
    audit["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    observed("final_flush", flush)
