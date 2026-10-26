# G4-I0 evidence (CPU/source audit only)

`environment_verification.json` records actual IsaacSim build, clean pinned IsaacLab and LayerNorm defaults.

Result: G4_I0_PASS for interface audit; I1 runtime NOT_RUN; learning gate PENDING.

- `interface_spec.json`: frozen actor/RMS/LSTM/action/controller/reset/timing contract and pseudocode.
- `checkpoint_and_source_identity.json`: initial allowlisted paths/hashes, all actor tensor shapes, architecture, pinned source identities.
- `checkpoint_audit.json`: successful reproducible SHA-before-load CPU inspection of all three checkpoints, full float64 RMS mean/variance/count, complete agent configs and env config hashes, 14 source hashes, exact official RMS CPU parity.
- `checkpoint_audit_console.txt`: successful collector console; initial AST-selector failure is preserved separately in `checkpoint_audit_initial_ast_error.txt`.
- `cpu_validation.json`: final 52 focused tests and unfiltered 272-test full regression PASS, exact receipts/hashes; prior failures and scoped diagnostics explicitly retained.
- `g4i0_acceptance_cpu.{txt,xml,json}`: final complete regression command, environment, actual child exit0, timing and JUnit.
- `g4i0_targeted_final_cpu.{txt,xml}`: intermediate46-case run; original14-case draft run remains as `g4i0_targeted_cpu.*`. Final52focused cases are included in the272-test acceptance JUnit. The earlier266-case full run is retained as `g4i0_full_final_cpu.*`.
- `g4i0_g3a_shm_venv_cpu.*` and `g4i0_g3a_shm_comparison.json`: unchanged G3A 16/16 rerun and causal limits. Initial symlink-resolved Python rerun also preserved, explicitly labelled.
- `g4i0_full_cpu.*`: initial 232passed/2failed run, not erased or relabelled. `g4i0_regression_excluding_preexisting_g3a.*` is a scoped diagnostic, not full-suite acceptance.
- `historical_preservation.json`: baseline tracked files, raw inventory and unrelated file identity checks (987 unique artifacts, no mismatch).

Reproduce CPU checkpoint audit from repository root with a NEW output:

```bash
CUDA_VISIBLE_DEVICES='' ../../work/envs/factory-g01/bin/python -m scripts.audit_g4i0 --output /tmp/g4i0_new_audit.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../work/envs/factory-g01/bin/python -m pytest -q tests --basetemp=/dev/shm/g4i0_new_tests
```

Do not invoke old training/evaluation commands. Checkpoint binaries, simulator logs and failed-run evidence remain unchanged at historical paths outside Git. I0 ran no actor forward, simulator, GPU or training workload; only the small official RMS module was executed on CPU. GPU/episode/quaternion numerical parity and latency checks are preregistered in `docs/G4_I1_ACCEPTANCE_PLAN.md` and the I2 roadmap.
