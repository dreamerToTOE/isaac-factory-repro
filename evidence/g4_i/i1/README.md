# G4-I1 evidence index

Conclusion: **G4_I1_PASS**. User learning remains PENDING by explicit scheduling update. No I2/Sim2Sim/other task/training was started.

- [Chinese engineering report](../../../docs/REVIEW_G4_I1.md) and [reproduction guide](../../../docs/G4_I1_USAGE.md).
- `integration.json`: complete engineering gate, 360 CPU tests including88 I1 cases,832800 exact-zero GPU-captured numerical comparisons.
- `run_manifest.json`: original preregistration, checkpoint SHA, seed1001,FP32,guard and tolerance policy.
- `engineering_correction_plan.json`: preserved first failure, validator layout/exit-status repairs; no scientific changes.
- `execution_index.json`: all4 executions distinguished: preflight, failed first full attempt with historical false0, intentional child1 exit probe, final verified episode.
- `cpu_actor_parity.json`: actual primaryG2/secondaryR4 CPU model vs officialPlayer, all stages zero error and RNG exact.
- `cpu_edge_parity.json`: SHA-verified originalPlayer clip and Factory pre-step EMA,4 synthetic injectedmu cases, no GPU or policy mutation.
- `cpu_final.{txt,xml,json}`: all360 tests, no skipped/excluded tests, child0. EarlierCPU receipts preserved.
- `runtime_acceptance_strict_coverage.json`: offline raw-trace/NPZ/process acceptance including observed case coverage, reset, RNG and independent success recomputation.
- `independent_parity_review.json`: separate CPU-only review linking NPZ to raw JSONL, source-based fixture checks, geometry/success and failed-attempt preservation.
- `scientific_source_preservation.json`: all14 frozen scientific/evaluator files unchanged.
- `historical_preservation.json`: 1397 baseline tracked files (only2authorized forward additions), I0 manifest37+itself,987 historical raw artifacts16.2GB,5unrelatedG3V files.
- Phase subdirectories contain small exact result/parity/error/invocation/guard/supervisor/config copies. Actual raw wait is authoritative; launcher124 in the intentional probe does not replace child1.
- `raw_artifact_inventory.json`:118 external raw files50,111,735bytes; full paths andSHA. Includes18MB traces, paired NPZ, fullconsole, sampling CSV, and SHA-matching first-attempt source snapshots. These large files remain local outsideGit, not remotely hosted.

The final episode has1complete149step timeout and the first action of the next episode. Ever/terminal1/1 and firstsuccess37steps=2.4666667simseconds are a path-validation result, not the historical288episode success rate. All scalar runtimes includevalidation/trace work and are not I2 latency benchmarks.
