# G3C-R2 证据索引

结果：64-env smoke **margin NO-GO**；32-env smoke **工程PASS**；32-env/400epoch seed0完整训练在10epochs后因0.512319s监控gap停止，**工程INCOMPLETE / 科学NOT ASSESSED**。已核实完成40,960 transitions；partial rollout未计数。未启动独立评估或额外训练。

| 文件 | 用途 |
|---|---|
| [manifest_frozen.json](manifest_frozen.json) | 按Git提交顺序冻结的分支参数；pending状态为训练前历史记录 |
| [registration_provenance.json](registration_provenance.json) | 各预注册提交与实际command文件索引 |
| [phase_a_inventory.json](phase_a_inventory.json) |600样本/60秒idle，minimum free5730MiB，通过5500门槛 |
| [smoke64_receipt.json](smoke64_receipt.json) / [smoke32_receipt.json](smoke32_receipt.json) | 容量门槛决策；32smoke完整，64smoke少78.875MiB margin |
| [result.json](result.json) | 当前正式状态、预算、停止原因、null评估率、尾日志限制 |
| [cpu_artifact_checks.json](cpu_artifact_checks.json) | post-run CPU工件检查，checkpoint和TensorBoard完整性，不运行策略 |
| runs/ | 三次attempt的小型原始launch/guard/supervisor/audit副本 |
| [guard_gap_tail.csv](guard_gap_tail.csv) | 正式run末20个原始样本，保留实际gap |
| [raw_artifact_inventory.json](raw_artifact_inventory.json) | 仓库外原始inventory/guard/console/checkpoint文件路径、大小和hash |
| [runtime_preservation_audit.md](runtime_preservation_audit.md) / [.json](runtime_preservation_audit.json) | 配置、pinned源码、历史数据完整性 |
| [cpu_regression.txt](cpu_regression.txt) / [.xml](cpu_regression.xml) |126 CPU tests passed |
| analysis/ | post-hoc CPU分析、训练早期曲线、guard图与独立分母比较；不是新的科学观察 |

原始数据在外部`work/logs/g3c_r2/`和`work/runs/logs/rl_games/Factory/`。checkpoint不入Git。没有测量逐梯度finite；有限loss/checkpoint不能替代此证据。没有预注册milestone checkpoint，因此ever/terminal success和first-success time均无可用统计，不以smoke或best-reward模型替代。

复现命令由`runs/<attempt>/launch.json`保存，包含真实systemd argv和source hashes。CPU复算命令见[审阅报告](../../docs/REVIEW_G3C_R2.md)和analysis/reproduction_command.json；复算输出须为新目录，不能覆盖历史。不会从这些记录自动恢复训练。
