# G3C-1 evidence — independent training seed 1

G3C-1 was intentionally preserved as an incomplete engineering run. The frozen manifest used the G2 task/PPO settings with the only scientific difference `training seed 0 -> 1`. The independent G3A guard stopped the child after three consecutive NVML samples below the 1024 MiB free-memory floor.

- [`manifest_frozen.json`](manifest_frozen.json): pre-launch configuration, command and source identities.
- [`CONFIG_COMPARISON.md`](CONFIG_COMPARISON.md): G2 versus G3C-1 configuration difference.
- [`preflight.json`](preflight.json): preflight checks and decision.
- [`run_receipt.json`](run_receipt.json): supervisor/guard result and hashes of external raw evidence.
- [`runtime_config_snapshot.json`](runtime_config_snapshot.json): observed runtime dimensions and zero optimizer-step summary.
- [`evaluation_status.json`](evaluation_status.json): no checkpoint was available, so no evaluation was run.
- [`plots/gpu_guard_headroom.png`](plots/gpu_guard_headroom.png): bounded memory plot for the stop decision.

No checkpoint, raw NVML CSV, supervisor console, or TensorBoard event log is committed. They remain under the external `work/logs/g3c1/` path and are identified by SHA256 in `run_receipt.json`. G2/G3A/G3B evidence is preserved.
