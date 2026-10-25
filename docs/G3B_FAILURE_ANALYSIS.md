# G3B PegInsert Failure Analysis

G3B is a bounded, read-only diagnosis of the preserved G2 checkpoint. It is not retraining and it does not redefine success.

## Reproduction and provenance

The frozen protocol is [`evidence/g3b/protocol_frozen.json`](../evidence/g3b/protocol_frozen.json). It fixes the checkpoint SHA256, three sequential evaluation seeds, 32 environments, 3 episodes per environment, 149-step official horizon, and the exploratory secondary descriptors (axis angle 5°, projected hole-axis gap range −1 mm to 25 mm). The valid run receipts, guard coverage, resource samples, and artifact hashes are in [`evidence/g3b/run_receipts.json`](../evidence/g3b/run_receipts.json).

The original G2 pool is 280/288 ever and terminal success. The G3B pool is independently 280/288. All eight original identifiers recurred as failures, with no original failure becoming a success and no new failure at an originally successful identifier. This is identifier/outcome recurrence only; complete historical failure trajectories were not available, so exact replay is false.

## Evidence interpretation

The official predicate remains the strict XY < 2.5 mm and one-sided world-Z < 1 mm test. The recorded `xy_error_m` and `axial_displacement_m` are the same held/target base geometry used by the task. Orientations use Isaac Lab wxyz quaternions; +Z axis angle is a secondary cylindrical-orientation descriptor and does not gate the official result.

The eight terminal failures have overlapping conditions A=3 (XY condition), B=8 (axial condition), C=0 (all max axis angles below 5°), D=0 (no independent measured contact), E=0 (no temporary official success), F=0. In this taxonomy, B describes an observed unmet predicate condition; it is not a causal diagnosis. Action and EE motion statistics are descriptive and cannot establish contact obstruction.

## Artifacts

- [`summary_compact.json`](../evidence/g3b/summaries/summary_compact.json): denominator-separated comparison, eight failure rows, secondary denominators and validation.
- [`episode_summary.csv`](../evidence/g3b/summaries/episode_summary.csv): one row per 288 episode.
- [`plots/`](../evidence/g3b/plots/): PNG and PDF XY, axial, orientation, actions, success/failure, and category figures.
- [`protocol_frozen.json`](../evidence/g3b/protocol_frozen.json): pre-run source hashes and commands.

Raw 447×32 NPZ and full metrics remain outside Git under `work/logs/g3b/diagnostic_20261025_01` and recovery `diagnostic_20261025_02`; their hashes are recorded in `run_receipts.json`. No checkpoint, video, secret, or unbounded raw log is committed.
