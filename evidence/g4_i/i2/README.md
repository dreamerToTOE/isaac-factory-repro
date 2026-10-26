# G4-I2 evidence index

Status: **G4_I2_PASS**. Learning: **PENDING**. Stop at I2.

- `experiment_manifest.json`: preregistration before initial GPU runs.
- `experiment_manifest_r1.json`, `engineering_revision_r1.json`: preserved initial B startup failure and corrected batch-diagnostic registration. No scientific changes.
- `i2_a_acceptance*.json`: actual guarded single-env gates, including repeated R1 parity before evaluation.
- `protocol_audit.json`, `scientific_protocol_review.json`: historical seed/allocation verification and unpaired comparison limits.
- `timing/`: complete decision-clock CSV, compressed raw wall/CUDA scopes, per-run and within-run-only pooled distributions. Warmup retained in raw.
- `results/episodes.json`, `results/episodes.csv`: all 288 identities, success, first-success time, terminal geometry and references to full state/statistics.
- `results/summary.json`, `results/plots/`: pooled action/controller/evaluation diagnostics and inspected PNG/SVG plots.
- `failures/`: all eight failures with complete 149-step official success/geometry/action CSV, full trace identities and descriptive classification.
- `episode_coverage.json`: 288/288 complete per-episode observation/action/controller/success/timing coverage.
- `executions/`, `execution_index.json`: exact argv/environment/preflight, real child/launcher exits and guard statuses, including the failed startup.
- `cpu/`: complete regression exits, frozen-checkpoint CPU parity, synthetic fault evidence and system-matplotlib tests.
- `scientific_source_preservation.json`, `runtime_source_snapshots.json`, `original_worktree_preservation.json`: fixed-source and unrelated-file checks.
- `external_artifact_inventory.json`: SHA256/size for all large raw traces, supervisor consoles, RNG snapshots, full per-episode JSON/CSV and source snapshots.
- `deliverable_manifest.json`: SHA256/size of committed I2 deliverables, excluding itself.

Large raw evidence remains outside Git under:

```
/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g4_i/i2/g4i2_20261026T094103Z
/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g4_i/i2/g4i2_20261026T094103Z_r1
```

The full 288-episode export is `..._r1/episode_statistics/episodes_with_statistics.json` and `.csv`. Each contains actual initial randomized state, pre-reset terminal state, complete per-episode action/controller/timing statistics and original raw trace provenance. Git compact indexes do not replace it. Preserve these external artifacts when moving the repository; hashes alone do not make raw data downloadable from GitHub.

GPU reproduction must create a new reviewed manifest revision and fresh output/tmpfs paths; never overwrite this evidence or inherit an unguarded exception. Exact commands for this run are in `executions/*_command.json`. Postprocessing commands below use the existing immutable raw data and **new** output directories.

```bash
CUDA_VISIBLE_DEVICES='' ../../work/envs/factory-g01/bin/python -m scripts.g4i2_timing_export \
  --manifest evidence/g4_i/i2/experiment_manifest_r1.json \
  --reference-manifest evidence/g4_i/i2/experiment_manifest.json --output NEW_TIMING_DIR

CUDA_VISIBLE_DEVICES='' ../../work/envs/factory-g01/bin/python -m scripts.g4i2_episode_statistics \
  --run-dir /absolute/raw_r1/b_1001 --run-dir /absolute/raw_r1/b_1002 \
  --run-dir /absolute/raw_r1/b_1003 --output NEW_FULL_EPISODE_DIR

CUDA_VISIBLE_DEVICES='' ../../work/envs/factory-g01/bin/python -m scripts.g4i2_failure_analysis \
  --run-dir /absolute/raw_r1/b_1001 --run-dir /absolute/raw_r1/b_1002 \
  --run-dir /absolute/raw_r1/b_1003 --output NEW_FAILURE_DIR

CUDA_VISIBLE_DEVICES='' python3 -m scripts.g4i2_report \
  --manifest evidence/g4_i/i2/experiment_manifest_r1.json --output NEW_RESULTS_DIR \
  --episode-export /absolute/full_export/episodes_with_statistics.json --plots
```

The last command uses the existing system NumPy/matplotlib; all other scientific commands use the pinned Factory Python. No package installation is required. No evidence claims continuous collision safety, real hardware safety, Sim2Sim success or original episode pairing.
