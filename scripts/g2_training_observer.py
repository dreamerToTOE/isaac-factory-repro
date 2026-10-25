"""[ENGINEERING] Small read-only observations around the official training entry.

No gradient tensors are inspected. Optimizer hooks count actual calls; AMP skips
are inferred when GradScaler.step does not call Adam.step. Loss/timing summaries
are read once per epoch. The original methods receive and return unchanged data.
"""
import datetime
import functools
import json
import os
from pathlib import Path
import time

import torch


def install(launcher_class):
    output = Path(os.environ["FACTORY_G2_AUDIT_OUTPUT"]).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite G2 audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = {"engineering": "Read-only epoch summaries and actual optimizer/AMP call counters",
             "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "epochs": [], "amp_skips": [], "checkpoint_events": [],
             "optimizer_steps": {"actor": 0, "central_value": 0},
             "actor_attempts": 0, "instrumentation_errors": []}

    def flush():
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(audit, indent=2, default=str, allow_nan=False) + "\n")
        temp.replace(output)

    def observe(point, function):
        try:
            return function()
        except Exception as error:
            audit["instrumentation_errors"].append({"point": point, "error": repr(error)})
            print(f"G2_OBSERVATION_ERROR {point}: {error!r}", flush=True)

    original_launcher = launcher_class.__init__

    @functools.wraps(original_launcher)
    def launch(self, *args, **kwargs):
        original_launcher(self, *args, **kwargs)
        from rl_games.algos_torch.a2c_continuous import A2CAgent
        original_init = A2CAgent.__init__

        @functools.wraps(original_init)
        def initialize(agent, *args, **kwargs):
            original_init(agent, *args, **kwargs)
            for role, optimizer in (("actor", agent.optimizer), ("central_value", agent.central_value_net.optimizer)):
                def counted(opt, positional, keyword, role=role):
                    audit["optimizer_steps"][role] += 1
                optimizer.register_step_post_hook(counted)
            audit["runtime"] = {"num_actors": agent.num_actors, "horizon_length": agent.horizon_length,
                                "minibatch_size": agent.minibatch_size, "mini_epochs": agent.mini_epochs_num,
                                "obs_shape": agent.obs_shape, "state_shape": agent.state_shape,
                                "actions_num": agent.actions_num, "mixed_precision": agent.mixed_precision,
                                "config": agent.config, "actor_model": str(agent.model),
                                "central_value_model": str(agent.central_value_net.model)}
            original_step, original_update = agent.scaler.step, agent.scaler.update
            pending_skip = []

            def scaler_step(optimizer, *args, **kwargs):
                before = audit["optimizer_steps"]["actor"]
                audit["actor_attempts"] += 1
                result = original_step(optimizer, *args, **kwargs)
                if audit["optimizer_steps"]["actor"] == before:
                    event = {"epoch": agent.epoch_num, "attempt": audit["actor_attempts"],
                             "scale_before": agent.scaler.get_scale()}
                    audit["amp_skips"].append(event)
                    pending_skip.append(event)
                return result

            def scaler_update(*args, **kwargs):
                result = original_update(*args, **kwargs)
                if pending_skip:
                    pending_skip.pop()["scale_after"] = agent.scaler.get_scale()
                return result

            agent.scaler.step, agent.scaler.update = scaler_step, scaler_update
            audit["observers_installed"] = True
            observe("initialize", flush)

        A2CAgent.__init__ = initialize
        original_epoch = A2CAgent.train_epoch

        @functools.wraps(original_epoch)
        def epoch(agent, *args, **kwargs):
            began = time.perf_counter()
            result = original_epoch(agent, *args, **kwargs)

            def record():
                step, rollout, optimization, total = result[:4]
                values = [v.detach().reshape(-1) if torch.is_tensor(v) else torch.tensor([v], device=agent.ppo_device)
                          for group in result[4:9] for v in group]
                finite = bool(torch.isfinite(torch.cat(values)).all().item()) if values else None
                frames = agent.curr_frames
                audit["epochs"].append({"epoch": agent.epoch_num, "environment_transitions": frames,
                    "env_step_seconds": step, "rollout_seconds": rollout, "optimization_seconds": optimization,
                    "official_total_seconds": total, "observed_epoch_seconds": time.perf_counter()-began,
                    "env_step_fps": frames/step, "rollout_fps": frames/rollout, "total_fps": frames/total,
                    "epoch_loss_arrays_finite": finite, "amp_scale": agent.scaler.get_scale(),
                    "optimizer_steps_cumulative": dict(audit["optimizer_steps"]),
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                flush()
            observe("epoch", record)
            return result

        A2CAgent.train_epoch = epoch
        for operation in ("save", "restore"):
            original = getattr(A2CAgent, operation)
            def wrap(method, operation):
                @functools.wraps(method)
                def observed_method(agent, filename, *args, **kwargs):
                    result = method(agent, filename, *args, **kwargs)
                    audit["checkpoint_events"].append({"operation": operation, "filename": str(filename),
                                                       "epoch": agent.epoch_num, "frame": agent.frame})
                    observe(operation, flush)
                    return result
                return observed_method
            setattr(A2CAgent, operation, wrap(original, operation))
        print("[ENGINEERING] G2 lightweight observation: " + str(output), flush=True)

    launcher_class.__init__ = launch
