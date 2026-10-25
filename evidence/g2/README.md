# G2 证据索引

本目录归档 `g2_peginsert_64env_seed0_baseline_rerun` 的小型原始证据和离线派生结果。实验完成于 2026-10-23，交付整理于 2026-10-25。阅读结论请从 [RESULTS_G2.md](../../docs/RESULTS_G2.md) 开始，完整命令与重算限制见 [G2_COMMANDS.md](../../docs/G2_COMMANDS.md)。

| 证据 | 作用 |
|---|---|
| [g2_manifest.json](g2_manifest.json) | 训练前固定的任务、64 env、seed 0、200 epochs、checkpoint/评估预算及初始 source hash。 |
| [evaluation_amendment.json](evaluation_amendment.json) | 首次主评估前的工程修订：partial 元数据补全与监督续接；保留原 manifest hash，协议预算不变。 |
| [source_snapshots/evaluator_before_partial_metadata_fix.py](source_snapshots/evaluator_before_partial_metadata_fix.py) | 修复前 evaluator，hash 与原 manifest 一致；实际主评估使用 amendment 中的新 hash。 |
| [preflight/static_audit.json](preflight/static_audit.json)、[preflight/reverification.json](preflight/reverification.json) | 版本、上游源码、环境身份和当时资源状态。reverification 的 GPU 进程是已经启动的授权训练。 |
| `preflight/{gpu,factory64,observer64,known_checkpoint}/` | 本轮 GPU/PhysX、64-env Factory、只读 observer、已知 checkpoint 动态检查的 console、result、NVML 与相关指标。 |
| [initial_launch_status.json](initial_launch_status.json)、[launch_monitor_incident.json](launch_monitor_incident.json) | 零 epoch 的首次启动和正式 run 监督进程丢失事故；明确保留无法追回的信息。 |
| [orphaned_pipeline_status.json](orphaned_pipeline_status.json) | 原监督器最后状态及实际训练 argv；其 phase 停留在 training，不代表训练仍在运行。 |
| [continuation_status.json](continuation_status.json) | 同一训练 PID/birth 完成验证及 12 项主评估的实际状态；原 child 退出码为 null。 |
| [training/training_audit.json](training/training_audit.json)、[training/console.txt](training/console.txt) | 连续 epoch 1…200、optimizer/AMP、运行时形状、保存事件与官方结束标记。 |
| [training/reattached_guard_result.json](training/reattached_guard_result.json)、[monitor_segments.json](monitor_segments.json) | 重附 guard 的终态和分段时间轴；不能补回原监测缺口。 |
| [checkpoint_inventory.json](checkpoint_inventory.json)、[selected_checkpoints.json](selected_checkpoints.json) | 所有 6 个 checkpoint 的路径/hash，以及主要 50/100/150/200 checkpoint 内部 epoch/frame/optimizer/参数差异。 |
| `evaluation/epoch*/seed*/{metrics.json,result.json,console.txt}` | 12 次独立评估的完整 episode、pre-reset provenance、命令与退出状态。原始 GPU CSV 和 partial 快照在仓库外保留。 |
| [analysis/summary.json](analysis/summary.json)、[analysis/tensorboard_scalars.json](analysis/tensorboard_scalars.json) | 离线分析输入 hash、效率/显存/训练/评估摘要，以及全部已记录 scalar 的原始 step/wall_time。 |
| [analysis/evaluation_summary.json](analysis/evaluation_summary.json)、[analysis/evaluation_per_seed.csv](analysis/evaluation_per_seed.csv) | 从原始 episode 重算的实际分母、per-seed/aggregate 成功率、Wilson 区间与条件首次成功时间。 |
| `analysis/*.png`、`analysis/*.pdf` | reward、loss、throughput/memory、checkpoint success、条件首次成功时间的五组图；均为派生结果。 |
| [final_source_verification.json](final_source_verification.json) | 交付前 CPU-only 上游 HEAD/clean、7 个 Lab + 9 个 RL-Games 源文件、冻结脚本与配置差异复核。 |
| [playback/README.md](playback/README.md) | 补充1环境完整episode、pre-reset截图与149步几何；同时保留加载失败及显存guard停止的尝试，不加入主要评估分母。 |
| [scientific_review.md](scientific_review.md)、[unit_tests_final.txt](unit_tests_final.txt) | 只读交叉审阅、42项CPU测试实际输出与证据边界。 |
| [raw_artifact_inventory.json](raw_artifact_inventory.json) | 仓库外全部主训练/评估/动态 preflight 原始文件和 checkpoint 的绝对路径、大小、SHA256；已有 Git 内副本逐字节核对。 |

实际 resolved YAML 位于 [configs/resolved/g2_peginsert_64env_seed0_baseline_rerun](../../configs/resolved/g2_peginsert_64env_seed0_baseline_rerun)。checkpoint、完整 GPU 采样与 TensorBoard event 文件没有加入 Git；目录存在于本机并不表示已上传或可从远端下载。`raw_artifact_inventory.json` 明确记录存储范围，补充可视化 playback 若存在，由其自己的证据归档，不进入主评估分母。

## 保留的限制

- 初次零 epoch 启动的完整 manifest 在重新命名时被覆盖，仅其 hash/status 存留；没有把该启动计为训练数据。
- 正式训练的原 monitor/parent 消失，约 122.75 秒无 GPU 样本或显存 guard，OS child 退出码无法回收。训练预算完成证据与退出状态分开记录；显存极值只表示已观测样本。
- TensorBoard 没有完整保存最后 epoch 的所有 scalar；audit、最终 checkpoint 与官方结束标记用于预算完成核验，图表不插补缺失点。
- 200 epoch 存在 periodic/final 两个文件，hash 不同，按预注册规则使用 final-save 文件；best reward alias 不替代固定 milestone。
- 原始 partial 和最终 metrics 可能含相同 episodes，离线分析择优去重，不累加。每 seed 完成分母在 JSON 中保留；主评估固定三个 seeds，补充回放不混入。
- 只有一个训练 seed；Wilson 区间描述 episode 计数，不量化训练 seed 方差，也不保证向量环境中 episode 完全独立。

这些限制是结果的一部分。原始文件、manifest 与 amendment 保持不变；后续派生图或审阅文件不能替代原始观察记录。
