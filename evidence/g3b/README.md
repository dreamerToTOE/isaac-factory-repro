# G3B evidence — bounded PegInsert failure diagnosis

This directory contains reviewable, small G3B artifacts. The original G2/G3A evidence is preserved and is not overwritten.

- [`protocol_frozen.json`](protocol_frozen.json): pre-run parameters, checkpoint/source hashes, original failure identifiers and commands.
- [`run_receipts.json`](run_receipts.json): actual child exit, guard coverage, GPU samples and output hashes for 1001/1002/1003. The first 1003 fail-closed startup is explicitly preserved as a non-sample.
- [`summaries/summary_compact.json`](summaries/summary_compact.json): original-vs-diagnostic comparison and eight failure rows.
- [`summaries/episode_summary.csv`](summaries/episode_summary.csv): 288 per-episode diagnostic rows.
- [`plots/`](plots/): six diagnostic figures in PNG/PDF formats.
- `g3b_source_audit_initial.json`, `g3b_source_review.md`, and `g3b_cpu_tests.*`: pre-run source preservation and test receipts.

The complete trajectory NPZ, evaluator metrics, metadata, supervisor console and NVML samples are intentionally outside Git at `work/logs/g3b/`; hashes are recorded above. The NPZ schema is `g3b_factory_trajectory_v1`, uses no pickle/object arrays, and records no measured contact because the baseline has no configured contact sensor. Commanded wrench/controller targets are not contact measurements.
