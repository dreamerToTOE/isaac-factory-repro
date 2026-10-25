# G3C-R1 evidence — GPU capacity diagnosis

结论：**CAPACITY_NO_GO** for another full 64-environment PPO run under the observed current RTX 3070 occupancy.

- [`protocol_frozen.json`](protocol_frozen.json): read-only inventory protocol and unchanged guard policy.
- [`inventory_summary.json`](inventory_summary.json): corrected 60-second idle inventory, attribution limits, RAM/swap, G3C-1 and R1 smoke facts.
- [`idle_samples_1hz.csv`](idle_samples_1hz.csv): compact 1 Hz view of the corrected idle window.
- [`historical_capacity.csv`](historical_capacity.csv): D001/G2/G3A/G3B/G3C-1/G3C-R1 observed comparisons.
- [`plots/historical_capacity.png`](plots/historical_capacity.png): observed free-memory minima; no linear scaling fit.

The corrected idle window had 600 samples, 4,780 MiB minimum free. The fresh supervised 64-env 2-epoch smoke reached 6,798.75 MiB used and 1,018.875 MiB minimum free, then the unchanged guard stopped it after three low samples. No retry or long training was started. Raw inventory CSV, guard samples, supervisor console and smoke audit remain outside Git under `work/logs/g3c_r1/` and are hashed in `inventory_summary.json`.
