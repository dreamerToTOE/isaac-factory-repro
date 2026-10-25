# isaac-factory-repro
Reproduction and evaluation of contact-rich robotic assembly tasks using NVIDIA Isaac Lab, Franka, and PPO.

## G2 当前结果（训练 2026-10-23，审阅 2026-10-25）

**PASS：64环境、seed0、官方PPO全新连续200epochs完成，1,638,400 transitions。** 独立确定性评估最终ever/terminal均280/288（97.222%）；50/100/150里程碑分别69/96、81/96、92/96。约2小时32分53秒，PPO聚合180.18FPS，观测全卡显存峰值6909.69MiB。属于资源受限初步基线，样本量为官方128环境同200epochs的一半。

监督进程消失造成约123秒显存采样/guard缺口及原训练退出码未知，TensorBoard尾端未持久化epoch200；完成由连续逐轮audit、最终checkpoint和官方结束标记交叉确认。上述限制随结果一并保留。

见[完整中文报告](docs/RESULTS_G2.md)、[复现命令](docs/G2_COMMANDS.md)、[证据索引](evidence/g2/README.md)、[未决问题与G3提案](docs/OPEN_ISSUES.md)。大checkpoint和长原始日志留在仓库外，文件路径及SHA256有清单。未追加训练、调参或启动其它Factory任务。


## D001 审阅跟进（2026-10-23，历史记录）

R1–R4已完成；建议后续使用64环境、seed0全新训练（GO），当前停止等待审阅，未启动200 epochs。原G0/G1提交`a77d215b1244b120605eac5c0af59bf0f32c20c5`已通过既有SSH推送并独立核验，D001增量接续提交，保留全部历史。

- 16/32/64环境均完成3 epochs及checkpoint加载后额外1 epoch；PPO总FPS约67/125/227，全卡10Hz采样峰值约5669/5679/5725MiB。
- 独立成功指标15项CPU测试通过；两个seed共8个完整episode验证加载/自动reset链路，smoke checkpoint的ever/terminal均0/8，不代表正式性能。
- 全部损失和实际应用的梯度有限；各fresh run首次AMP overflow由官方GradScaler跳过，保留原AMP行为。
- 独立critic Adam、调度变量等缺失，checkpoint可推理和继续优化，不能声称完整可复现续训。
- 64环境200 epochs外推约2小时（建议排期2–2.5小时）；样本预算为官方默认128环境的一半。

详见[审阅报告](docs/REVIEW_D001.md)、[成功指标](docs/SUCCESS_METRICS.md)、[checkpoint恢复边界](docs/CHECKPOINT_REPRODUCIBILITY.md)及[D001证据](evidence/d001/README.md)。

## Isaac Factory — 原G0/G1记录（2026-10-23）

G0/G1实机验证已完成，没有启动正式长训练、没有修改官方PPO或成功条件。当时按用户决定仅本地提交；推送及后续审阅状态已由上方D001更新。

- 官方Isaac Lab：`v2.1.1` / `90b79bb2d44feb8d833f260f2bf37da3487180ba`。
- 实机：Ubuntu22.04.5、RTX3070 8GB、Isaac Sim4.5.0；Vulkan + GPU PhysX通过。
- 三个Factory ID注册并通过单环境初始化/reset/step；PegInsert额外验证16环境reset/step。
- 固定运行时Torch2.7.0+cu128 / RL-Games1.6.1：PegInsert4环境3epoch及checkpoint恢复、16环境1epoch完成真实PPO更新。
- PPO总FPS：4环境18–19，16环境71；全卡采样峰值分别5955/5659MiB。短测试不能证明长期收敛或256/512环境容量。

### 文档与证据

| 文件 | 内容 |
| --- | --- |
| [结果报告](docs/RESULTS_G0_G1.md) | 实测结果、失败修复、checkpoint与后续建议 |
| [REQUIREMENTS](docs/REQUIREMENTS.md) | OS/GPU/驱动/内存/设备权限与版本 |
| [SOURCE_AUDIT](docs/SOURCE_AUDIT.md) | 固定SHA逐项科学源码审计及行链接 |
| [SETUP](docs/SETUP.md) | 独立环境、安装与复测命令 |
| [EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md) | 基准/工程覆盖、样本预算与验收标准 |
| [OPEN_ISSUES](docs/OPEN_ISSUES.md) | 剩余限制、认证诊断和确切推送步骤 |
| [原始证据索引](evidence/2026-10-23/README.md) | 命令、console、显存CSV、checkpoint/事件校验 |

### 结构与使用

```text
docs/                     # 中文审计、协议与结果
scripts/                  # GPU/Factory探针、版本固定启动器、日志与checkpoint检查
configs/requirements-g01.lock
configs/resolved/         # 实测官方训练入口输出的配置
evidence/2026-10-23/       # 包含成功和失败的原始证据
```

从[SETUP](docs/SETUP.md)开始。上游Isaac Lab与Python环境均位于此仓库外，checkpoint大文件留在本地work/runs/logs，以SHA256追踪。README原始两行、原.gitignore全文与LICENSE均保留，Git历史从既有Initial commit延续。
