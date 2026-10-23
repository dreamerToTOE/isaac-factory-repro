# isaac-factory-repro
Reproduction and evaluation of contact-rich robotic assembly tasks using NVIDIA Isaac Lab, Franka, and PPO.


## Isaac Factory — G0/G1（2026-10-23）

G0/G1实机验证已完成，当前停止等待审阅；没有启动正式长训练、没有修改官方PPO或成功条件。本地提交后按用户决定暂不推送，GitHub认证单独处理。

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
