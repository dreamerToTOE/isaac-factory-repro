# G3C-R5 证据索引

已完成唯一一次 fresh training seed1：32env、400epochs、1,638,400 transitions，child wait0；未重训、恢复或启动seed2。

首次评估在官方场景资产SSL连接故障后以child wait0结束，但没有生成metrics，因此初始controller正确标为INCOMPLETE，不能用exit0冒充科学评估完成。初始报告 `docs/REVIEW_G3C_R5.md`、`result.json`、`analysis/`、`raw_collect/` 均原样保留。

确认原始资产/metadata/list接口恢复200后，人工选择同协议的 evaluation-only recovery。`manifest_evaluation_recovery.json` 与 `evaluation_recovery_provenance.json` 记录只有评估输出目录/标识变化；训练、checkpoint、评估器、超参数、环境和episode分母保持不变。新评估和最终报告用独立路径，不合并失败attempt的数据。

- 原始训练预注册：`manifest_frozen.json`；启动前CPU回归203passed，configuration identity审计通过。
- 已完成训练及首次评估证据：`raw_collect/`；初始post-run回归220passed。
- 资产故障及恢复探针：`evaluation_network_diagnosis.json`、`evaluation_asset_failure_excerpt.txt`。
- R4完整保留审计：`r4_runtime_integrity.json` 和 `raw_collect/postrun_preservation_audit.json`。
- 正在完成同协议的12次独立评估，最终证据将在单独的 `*_evaluated` 路径保存。

正常CUDA错误处理及获批no-GPU-guard模式不变；没有driver/proxy改动、无关进程终止或其他训练任务。历史64-env G2不进入32-env种子比较。
