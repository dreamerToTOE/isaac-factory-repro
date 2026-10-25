# G3A 证据索引

2026-10-25（Asia/Seoul）。最终 **PASS**，范围为监督与日志工程可靠性，只有2epoch短测；不包含G3B/seed1/seed2/200epoch实验。结论见[REVIEW_G3A](../../docs/REVIEW_G3A.md)，命令和语义见[运行协议](../../docs/G3_SUPERVISION_PROTOCOL.md)。

| 文件 | 用途 |
| --- | --- |
| [acceptance.json](acceptance.json) | 最终验收指标与已知限制 |
| [cpu_final.txt](cpu_final.txt)、[JUnit](cpu_final_junit.xml)、[cpu_cases/](cpu_cases/) | GPU前77项CPU测试；牺牲进程真实exit/故障/原子文件记录，均为测试数据 |
| [gated_preflight.json](gated_preflight.json) | 最终smoke前源码hash、版本、资源与G2/G3V保留清单 |
| [launch.json](smoke_gated/launch.json) | 实际训练argv、安全环境、cwd、两个systemd unit命令与源码hash |
| [supervisor status](smoke_gated/supervisor/status.json) | 实际wait0、exit0、signal=null与独立termination_reason |
| [guard status](smoke_gated/guard/status.json)、[samples.csv](smoke_gated/guard/samples.csv) | 829条约10Hz全卡采样、身份/健康、停止决定和gap数组 |
| [coverage](gated_coverage.json)、[运行中健康快照](gated_active_health_snapshot.json) | 首样本→spawn→release→wait→末样本交叉核验；双方和child存活、fresh且绑定一致 |
| [training audit](smoke_gated/supervisor/training_audit.json)、[console](smoke_gated/supervisor/console.txt) | 连续2epochs、16384transitions、优化观察和官方结束标记 |
| [TB源码审计](tensorboard_source_audit.json) | 缺显式flush/close的直接事实，不能证明G2唯一历史根因 |
| [TB tail](gated_tensorboard_tail.json)、[finalizer](smoke_gated/supervisor/tensorboard_finalizer.json) | flush/close成功、50tags有限、固定末epoch计数和step通过 |
| [checkpoint](gated_checkpoint.json)、[config identity](config_identity.json) | epoch/frame/有限性，配置仅允许覆盖，env与G2字节一致 |
| [preservation](preservation_verification.json) | G2证据/原事件、无关G3V文件hash保持，上游干净 |
| [attempts/](attempts/) | 手工迟启guard的两次开发尝试：首次失败、第二次监控PARTIAL；没有覆盖 |
| [raw inventory](raw_artifact_inventory.json) | 仓库外原始文件、事件、checkpoint路径/大小/SHA256 |

`cpu_reliability_tests.txt`、`script_hashes.txt`是早期schema1开发记录；最终验收使用`cpu_final*`和`gated_preflight.json`。`smoke_gated/`只提交小型记录，checkpoint/pickle/原事件不提交。实际YAML见[配置目录](../../configs/resolved/g3a_smoke_gated_64env_20261025_final/)。

完整覆盖按0.1秒采样/0.5秒gap线定义，不保证瞬时峰值或断电持久性。用户Linger=no，不保证跨完整注销/重启。缺失final永远不代表exit0。G2历史缺口、未知退出码与缺失标量保持不变。
