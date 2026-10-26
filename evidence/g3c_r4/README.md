# G3C-R4 evidence

The fresh, explicitly authorized no-independent-GPU-guard PegInsert run completed successfully.

- Registration and exact policy: [`manifest_frozen.json`](manifest_frozen.json)
- Compact result: [`result.json`](result.json)
- Full CPU analysis, milestone checks, success denominators, timing, and matched-transition comparison: [`analysis/report.json`](analysis/report.json)
- Raw post-run artifact inventory: [`raw_artifact_inventory.json`](raw_artifact_inventory.json)
- Preservation audit for the starting repository, historical checkpoints, raw evidence, and frozen sources: [`postrun_preservation_audit.json`](postrun_preservation_audit.json)
- CPU prelaunch regression: [`prelaunch_cpu_regression.xml`](prelaunch_cpu_regression.xml)
- Final CPU regression after post-run analysis: [`final_cpu_regression.xml`](final_cpu_regression.xml), 167 passed

Training used 32 environments, seed 0, 400 epochs, horizon 128, and 1,638,400 completed transitions. The supervisor exited with raw wait status 0 and no stop signals. The independent GPU guard was intentionally omitted under the user's authorization; no continuous GPU telemetry coverage is claimed. Historical G2/G3B/R2 evidence remains separate and was not overwritten or pooled. Large checkpoints and full runtime logs remain outside Git; their paths and hashes are recorded in the inventories.
