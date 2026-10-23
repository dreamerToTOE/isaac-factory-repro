# D001 training observer: measurement semantics

`scripts/train_audited.py` launches the unedited official training entry point with its arguments unchanged. Observers are installed after `AppLauncher` initializes Kit, keeping the official order of simulator and learning imports. The script changes no environment method, network, loss used for backpropagation, gradient value, clipping operation, random seed, optimizer update, scaler decision, or PPO configuration.

## Optimization evidence

- `actor_attempts` records each return from `A2CAgent.train_actor_critic`: actor loss, actor-side value loss, entropy, KL, bounds loss, policy mean/sigma, total objective, learning rate, and scaler values. In this asymmetric configuration the actor-side value loss can be zero; the independent value loss is separately recorded in `central_value_attempts`.
- `gradient_events.phase=before_unscale_scaled_amp` inspects the gradients already produced by the official scaled backward call. Nonfinite values here can be AMP overflow. This is not evidence that Adam applied a nonfinite update.
- `after_unscale_before_clip` inspects actual unscaled gradients immediately after the original `GradScaler.unscale_`, before the unchanged official gradient clipping call.
- `optimizer_pre_step_post_clip` is a Torch Adam pre-step hook. It observes the gradients reaching an actual optimizer invocation after unscale/clipping. The actor optimizer is not invoked when GradScaler skips an overflowing attempt. The central value optimizer is independent and has its own hook.
- `optimizer_steps` counts post-step hooks, so an attempted actor minibatch is distinguished from an executed Adam update. For each actor attempt, `amp_skipped` and scaler values distinguish skipped overflow from the finite updates actually applied. An AMP skip must still be reported when all applied updates are finite.
- Gradient summaries count gradient tensors/elements and nonfinite elements and report the largest absolute value; no gradients or tensors are retained in the JSON. Loss and policy summaries similarly report finite status explicitly.
- `checkpoint_events` records a successful return from the official save/restore method, not just that a filename was requested. Checkpoint contents and training-resume completeness require the separate checkpoint audit.

## Performance and memory evidence

- `environment_transitions` is the official `curr_frames` for that epoch.
- `env_step_fps = transitions / step_time` uses the time accumulated by RL-Games around environment stepping.
- `rollout_fps = transitions / play_time` includes rollout policy inference and collection.
- `total_fps = transitions / total_time` includes rollout and optimization. It excludes simulator startup, most epoch logging and checkpoint serialization. It is the appropriate training-throughput figure for a clearly labelled short-run epoch estimate.
- CUDA `max_memory_allocated` and `max_memory_reserved` are process allocator high-water marks since process startup. They exclude memory owned by PhysX/Kit outside the Torch allocator and other processes. The separate high-frequency NVML monitor provides whole-GPU/process memory evidence.
- Every gradient/loss inspection copies small summaries to CPU and synchronizes CUDA. The observed training times include this instrumentation overhead; its magnitude has not been separately benchmarked. Comparing environment counts with the same observer is useful, but these measurements must not be presented as uninstrumented production PPO speed. There is no justified numerical correction for observer overhead.
- Physics stability must be assessed alongside console errors/warnings and the existing environment probes. This observer does not establish long-horizon stability or successful task learning.

## Failure handling and constraints

Observation errors are recorded under `instrumentation_errors` and printed as `FACTORY_AUDIT_ERROR`. They are not silently converted to passing evidence. An output path that already exists is rejected to prevent accidental evidence overwrite. JSON is saved after each epoch/checkpoint and at exit. `exit_error` captures a training exception when Python reaches the wrapper; a hard kill can leave the last completed snapshot only.

The wrapper is intentionally for bounded G1 review measurements. A future baseline run must start fresh with the reviewed seed, official configuration, and no smoke-test checkpoint. A D001 throughput extrapolation is not an authorization to start that run.
