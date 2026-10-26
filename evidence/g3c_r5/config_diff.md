# G3C-R5 configuration identity diff (draft)

Read-only audit recorded 2026-10-26T01:40:36.192092+00:00. Baseline is the completed G3C-R4 experiment at repository commit `04af151c7b78c0a3fe4adaf7a646ff151feecc86`.

## Scientific identity

The R5 run must be a fresh training run with **exactly one scientific parameter change** from R4: `params.seed` / training seed **0 → 1**. All other scientific parameters and implementation identities remain frozen.

| Field | G3C-R4 baseline | G3C-R5 required | Allowed? |
|---|---|---|---|
| Task | `Isaac-Factory-PegInsert-Direct-v0` | unchanged | yes |
| Environment count | 32 | 32 | unchanged |
| Training seed | 0 | **1** | **sole scientific change** |
| Max epochs | 400 | 400 | unchanged |
| Horizon length | 128 | 128 | unchanged |
| Planned transitions | 1,638,400 | 1,638,400 | unchanged |
| PPO/LSTM/optimizer hyperparameters | official R4 values | exact R4 values | unchanged |
| Reward/physics/success predicate/obs/action/network | official R4 values | exact R4 values | unchanged |
| Initialization | fresh, no restore/resume | fresh, no restore/resume | unchanged |
| Evaluation checkpoints | 100, 200, 300, 400 | 100, 200, 300, 400 | unchanged |
| Evaluation seeds | 1001, 1002, 1003 | 1001, 1002, 1003 | unchanged; these are evaluation seeds |
| Supervision | user-authorized-no-gpu-guard | same | unchanged policy |
| Checkpoint frequency | 25 epochs | 25 epochs | unchanged logging policy |

## Required engineering identifiers

Only names and paths needed to isolate R5 may change:

- Experiment: `g3cr5_peginsert_32env_seed1_400ep_<UTCSTAMP>` (recommended concrete stamp: `20261026t014039z`).
- Supervisor output: `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g3c_r5/train_32env_seed1_400ep_<UTCSTAMP>`.
- rl-games output: `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/runs/logs/rl_games/Factory/g3cr5_peginsert_32env_seed1_400ep_<UTCSTAMP>`.
- Completion/evaluation evidence: `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g3c_r5/completion_32env_seed1_400ep_<UTCSTAMP>` and a new repository `evidence/g3c_r5/` directory.
- Use new systemd unit prefixes (for example `factory-g3cr5-s1-<UTCSTAMP>` and `factory-g3cr5-<UTCSTAMP>-completion`).

R5 must not use the R4 checkpoint directory, load path, experiment name, output path, or any historical run directory. It must not launch seed 2, GearMesh, or NutThread.

## R4 baseline references and hashes

- R4 manifest: `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/evidence/g3c_r4/manifest_frozen.json` SHA256 `d18f353c2e0719963194f63bdd667d9280f323c739f0dffa6ba6802029824686`.
- R4 result: `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/evidence/g3c_r4/result.json` SHA256 `8c1a3867418aa9f47576950d52c1d42d03136c71516e1624ed56dd3d5947576b`.
- R4 analysis: `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/evidence/g3c_r4/analysis/report.json` SHA256 `8c2797e056f34e6ec42abd80a91e73df51fd81e335dd12224ffc13fb44bad9bf`.
- R4 frozen source set: 28 entries; current verification `28/28` matches.
- R4 tracked evidence: 285 files; manifest digest `72cf9b779ee14028b0d15e4e4777bd9edf97da1dce6ecedbb903c35a8cd722a3`.
- R4 historical preservation record: `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/evidence/g3c_r4/historical_fingerprints.json` SHA256 `9177c4369149808a3494a8d63593e8533d89060a99957c74174823c6c7c0c7be`; recorded 645 tracked files, 40 prior checkpoints, and 302 prior raw evidence files.

## Pre-launch gate

Do not start training until the R5 manifest records the exact concrete UTC-stamped identifiers, the R4 source/evidence hashes above, `params.seed=1`, and all fields in the identity table. CPU regression and configuration-identity checks must pass first. This file is a plan/audit artifact only; it does not authorize seed 2 or any hyperparameter adjustment.
