# G3C-R5 runtime configuration identity audit

CPU-only audit generated 2026-10-26T02:02:01.560442+00:00. **Overall result: PASS**. No GPU telemetry queries or process control were performed.

The R5 resolved configuration was compared recursively with the completed R4 resolved YAML. Allowed and actual differences are exactly:

- `env.yaml: seed`: 0 → 1.
- `agent.yaml: params.seed`: 0 → 1.
- `agent.yaml: params.config.full_experiment_name`: the unique R4 name → unique R5 name.

These two seed fields represent the single training-seed change; evaluation seeds remain 1001/1002/1003. All reward, task, physics, success, observations/actions, PPO/LSTM/network, horizon, update and checkpoint settings match R4.

Actual runtime dimensions are 32 actors, horizon 128, minibatch 512, four mini-epochs, observation shape 19, central state shape 43, six actions, and mixed precision enabled. Actor and central-value architecture strings exactly match R4. Resolved save frequency remains 25 and max epochs 400. No restore events or instrumentation errors were observed at this snapshot.

The runtime observer includes Python object repr strings with process-local addresses; only those addresses were normalized for its config comparison. The YAML comparison uses no normalization. The actual child command equals the frozen manifest, selects seed 1, has no checkpoint/resume argument, and runs under the approved no-GPU-guard policy.

| Check | Result |
|---|---|
| yaml_identity_only_allowed_seed_and_name_differences | PASS |
| runtime_dimensions_match | PASS |
| runtime_registered_config_match | PASS |
| runtime_config_after_process_address_normalization_only_name_diff | PASS |
| architecture_strings_exactly_match_r4 | PASS |
| no_restore_events | PASS |
| no_instrumentation_errors | PASS |
| frozen_sources_match | PASS |
| actual_command_matches_fresh_seed1_registration | PASS |
| supervision_policy_matches | PASS |
| upstream_pinned_clean | PASS |

At the snapshot, 2 complete epoch(s) were recorded; this is an initialization/configuration audit, not a final training-completion claim.
