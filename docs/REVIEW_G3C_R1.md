# G3C-R1 GPU 显存容量诊断与安全恢复

日期：2026-10-25
起点：`306f884d1bb55d4a4b2b84abfab4b096d2da8488`
结论：**CAPACITY_NO_GO（当前 RTX 3070 资源状态不适合再次尝试完整 64-env PPO）**

## 根因判断

G3C-1 的停止不是 PPO、奖励、物理或 success criterion 问题。runtime 在初始化后达到约 6.9 GiB total used；全卡 free 连续三次低于 1024 MiB，独立 guard 按固定策略停止。G3C-R1 的新 64-env、2-epoch capacity smoke 在相同 guard 下再次触发：peak used **6798.75 MiB**、minimum free **1018.875 MiB**、无 monitoring gap。它在 epoch 完成前停止，因此没有被当作学习结果。

当前 GPU 的 60 秒 idle inventory 为 600 个 0.1 秒样本：RTX 3070 8 GiB，driver 580.178.04，`nvidia-smi` CUDA banner 13.0；idle used 峰值 3038 MiB、free 最低 4780 MiB、reserved 375 MiB、GPU utilization 2–61%。compute API 只报告 gnome-remote-desktop PID 2882 使用 157 MiB；其余 graphics/driver/桌面/未归因占用不能假设可回收。没有 Isaac Sim 或训练进程。

## 历史容量对照

实际观测如下，未拟合线性 env-to-memory 模型：

| 实验 | 环境数 | peak used MiB | minimum free MiB | 结果 |
|---|---:|---:|---:|---|
| D001 64-env short | 64 | 5724.94 | 2092.69 | 完成 |
| G2 observer64 preflight | 64 | 6134.44 | 1683.19 | 完成 |
| G3A 64-env 2-epoch smoke | 64 | 6458.00 | 1359.63 | 完成 |
| G2 64-env full training | 64 | 6909.69 | 907.94 | 完成但监控有缺口 |
| G3B 32-env diagnostic | 32 | 6308.00 | 1509.63 | 完成，推理不是训练 |
| G3C-1 64-env seed1 | 64 | 6899.69 | 917.94 | guard 停止 |
| G3C-R1 64-env smoke | 64 | 6798.75 | 1018.88 | guard 停止 |
| G3C-R1 idle | 0 | 3038.00 | 4780.00 | 只读 inventory |

D001 short、G3A smoke 和 G2 full 的差异来自运行阶段、桌面占用、初始化/优化器状态和时间范围；不能用 D001 的低峰值保证长跑。G2 长跑还存在约 123 秒监控空白，907.94 MiB 是观测极值而非完整无缺口证明。G3B 是 32-env 推理评估，不能代替 64-env 训练容量。

从本次 idle peak 到 G3C-1/G3C-R1 smoke peak 的观测增量约为 **3761–3862 MiB**。在此增量上要求运行中至少保留 1.5 GiB free，建议新的 idle preflight minimum free 至少 **5500 MiB**，并再做一次独立 2-epoch smoke。这个 5500 MiB 是工程容量门槛，不改变运行时 guard 的 1024 MiB×3 fail-closed 策略；短时峰值和未来桌面变化仍可能超出观测。

此前“启动前 free >1024 MiB”不足以证明容量，因为训练初始化、optimizer/AMP/RNN 状态和长时间阶段的分配会把约 4 GiB 的额外占用叠加到 idle 状态；1024 MiB 是最后一道运行时保护线，不是可持续训练容量保证。

## 安全恢复动作

本轮没有杀进程、停止桌面、停止 remote desktop、修改 driver/persistence/systemd、删除缓存或改变 guard/PPO/物理设置。唯一的 capacity smoke 使用了新的 run name、G3A supervisor、独立 guard、真实 child wait 和 finalizer 配置；它触发后立即停止，没有自动重试。历史 failed user units 被记录但没有清理，因为它们不是当前 GPU workload。

## 决策与下一步

当前判定 **CAPACITY_NO_GO**：在现有 RTX 3070 occupancy 下，不应再次尝试完整 64-env/200-epoch run。下一步按优先级为：

1. 在用户明确允许且不影响当前会话的前提下，安排无 remote-desktop/其它 GPU 图形占用的专用时段，重新做 60 秒 idle inventory；不自动终止当前桌面进程。
2. 使用更大显存 GPU，保持 Isaac Lab、RL-Games、任务和 PPO 配置不变。
3. 如需 32-env 诊断训练，必须单独注册新科学协议；它改变 rollout batch、每 epoch transitions 和优化预算，不能称为 G2 的 seed-only replication。

只有 idle minimum free 达到约 5500 MiB、2-epoch supervised smoke 观测 free 至少约 1.5 GiB 且无 guard 触发后，才建议重新预注册 fresh seed1 长训练。当前不启动 seed1 重试、seed2、contact-sensor、GearMesh 或 NutThread。

## 证据

- [`protocol_frozen.json`](../evidence/g3c_r1/protocol_frozen.json)
- [`inventory_summary.json`](../evidence/g3c_r1/inventory_summary.json)
- [`historical_capacity.csv`](../evidence/g3c_r1/historical_capacity.csv)
- [`historical_capacity.png`](../evidence/g3c_r1/plots/historical_capacity.png)
- [`G3C-1 run receipt`](../evidence/g3c1/run_receipt.json)
- [`G3C-R1 evidence README`](../evidence/g3c_r1/README.md)

完整 10 Hz inventory、guard CSV、supervisor console 和 smoke audit 保留在仓库外 `work/logs/g3c_r1/`，只在摘要中记录 SHA256。G3C-1 失败证据原样保留。
