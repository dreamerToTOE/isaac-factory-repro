# G3C-R5 证据索引

工程完成；科学评估完成，结论仅作两个训练种子的描述性对照。唯一一次 fresh training seed 1 完成 32 env、400 epochs、1,638,400 transitions，child wait 0。未重训、恢复或启动 seed 2。

- 完整中文报告：[REVIEW_G3C_R5_FINAL.md](../../docs/REVIEW_G3C_R5_FINAL.md)。
- 最终机器结果：[result_evaluated.json](result_evaluated.json)；训练和 seed-0 对照：[analysis_evaluated/report.json](analysis_evaluated/report.json)。
- 最终 raw 小文件及 checkpoint/event 身份：[raw_collect_evaluated/](raw_collect_evaluated/)。大型原始文件仍保留在 manifest 的本地路径，未提交大模型文件。
- 最终 CPU 回归 **220 passed**：[JUnit](final_cpu_regression_evaluated.xml)、[实际命令与退出状态](artifact_execution_evaluated/status.json)。
- 独立科学审计 **139/139 passed**：[audit](final_scientific_audit.md)。
- 原始训练预注册：[manifest_frozen.json](manifest_frozen.json)；启动前 CPU 回归 203 passed，configuration identity 审计通过。
- R4 及更早历史保留：[postrun_preservation_audit.json](raw_collect_evaluated/postrun_preservation_audit.json)；在整合远端新增工程文档前全部通过。

最终 epoch 400 ever-success 与 terminal-success 均为 **278/288**；同协议 training seed 0 分别为 268/288、267/288。Evaluation seeds 1001/1002/1003 不是额外的训练重复。历史 64-env G2 不进入 32-env 种子比较。

首次评估在官方场景资产 SSL 连接故障后以 child wait 0 结束，但没有生成 metrics，因此初始 controller 正确标为 INCOMPLETE。初始 [中文报告](../../docs/REVIEW_G3C_R5.md)、[result.json](result.json)、[analysis/](analysis/)、[raw_collect/](raw_collect/) 原样保留。

确认原始资产/metadata/list 接口恢复 200 后，人工选择同协议的 evaluation-only recovery。见 [manifest_evaluation_recovery.json](manifest_evaluation_recovery.json)、[evaluation_recovery_provenance.json](evaluation_recovery_provenance.json)；仅评估输出目录/标识变化，训练、checkpoint、评估器、超参数、环境和 episode 分母保持不变。全部 12 次评估完成，控制器正常退出；没有合并最初失败 attempt 的数据。实际资产故障和探针见 [diagnosis](evaluation_network_diagnosis.json)、[excerpt](evaluation_asset_failure_excerpt.txt)。

用户授权的 `user-authorized-no-gpu-guard` 模式及 CUDA 正常错误处理不变。没有 driver/proxy 改动、无关进程终止、其他 task 训练或外部 GPU 租用。本任务停在 seed 1；seed 2 仅作为后续建议。
