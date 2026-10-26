# G3C-R4 runtime configuration audit

Result: PASS. CPU-only read-only snapshot at 2026-10-25T14:09:01.754594+00:00.

The resolved R4 environment differs from G2 only in scene.num_envs (64→32) and is identical to R2. The R4 agent differs from G2 only in max_epochs (200→400), unique experiment name, and save_frequency (50→25); from R2 only in name and save interval. All other serialized task and PPO values match.

Actual runtime: {'num_actors': 32, 'horizon_length': 128, 'minibatch_size': 512, 'mini_epochs': 4, 'obs_shape': [19], 'state_shape': [43], 'actions_num': 6, 'mixed_precision': True}. Registered runtime max_epochs=400, save_frequency=25, fresh experiment name verified. Actor and central-value architecture strings exactly match R2. All 28 frozen source hashes match; pinned upstream is clean.

The pre-environment agent YAML contains num_actors=128; unchanged official train.py later assigns the actual32. This is confirmed by runtime, rather than inferred from the dumped YAML.

Supervisor mode: {'launch_policy': 'user-authorized-no-gpu-guard', 'gpu_guard_enabled': False, 'gpu_telemetry_required': False, 'gpu_telemetry_gaps_terminate_child': False, 'supervisor_guard_directory': None, 'training_argv_matches_manifest': True, 'no_guard_argv': True, 'only_supervisor_unit': True, 'passed': True}. No guard argv or guard unit exists. No restore events or instrumentation errors existed at this snapshot.

This audit verifies configuration and startup only. Completed epochs, loss stability, checkpoint integrity, finalization and evaluation must be judged from later outcome evidence. No GPU telemetry was queried.
