# G5-GEAR-G0-R2：CPU-only 监护可靠性工程

**最终状态：`G5_GEAR_G0_R2_READY_FOR_REVIEW`。** 本阶段没有运行 Isaac Sim、GPU PhysX、CUDA workload 或 PPO，也没有执行新的 GearMesh GPU 预检。R2 针对 R1 的监护间隙完成了 CPU-only 诊断、最小架构加固和边界测试；G0-R1 的 GPU 运行结论仍保持 PARTIAL。

## 基线、证据和限制

R2 起点为 `966631dc3c8a67d20ec4c69cd2f5482c7f1784cf`，上游 Isaac Lab 仍为 `90b79bb2d44feb8d833f260f2bf37da3487180ba`。原始 G0、R0、R1 目录均保留未改动；源码、上游工作树和 checkpoint 完整性见 [source_and_checkpoint_integrity.json](../evidence/g5_gear/g0_r2/source_and_checkpoint_integrity.json)。PegInsert epoch 200 checkpoint 前后 SHA256 仍为 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`。

R2 的所有实验使用 sacrificial CPU child、mock memory source 或纯 Python heartbeat writer。CPU 结果不证明未来 GPU 运行一定成功，也不能替代新的 GPU PhysX 准入。

## R1 间隙复核和存储诊断

从 R1 原始 CSV 独立重算得到 68 个样本、67 个间隔：中位数 **99.999337 ms**，平均 **106.728409 ms**，P90 **100.049044 ms**，P95 **121.219314 ms**，P99 **276.371852 ms**，最大 **550.642873 ms**。超过 100/200/300/400/500 ms 分别为 **31/2/1/1/1** 次。最低空闲显存 **5481.625 MiB**，峰值采样 used **2336 MiB**。

R1 timing 记录中，最大 heartbeat 原子写入耗时 **550.085606 ms**，最大 memory query 约 **43.693 ms**；触发 gap 的那一轮 query 约 **45.5 µs**，前一轮 deadline lateness 约 **450.793 ms**。这确认了阻塞发生在 heartbeat 更新区域，但仍不能把它进一步归因于 ext4 元数据、调度抢占或内核等待。

仓库执行路径位于 ext4 `/dev/sdb3`（`rw,relatime,errors=remount-ro`）；`/run/user/1000` 是独立 tmpfs，约 4.9 GiB。R2 只读保存了 [storage_diagnostics.txt](../evidence/g5_gear/g0_r2/storage_diagnostics.txt)。它们说明两种存储类型不同，但没有证明 R1 的具体根因。

R2 开发期间还复现到一次未加固路径的 CPU 间歇：第 4 次重复生命周期的 gap **0.841449698 s**，其中 `processing_and_coordination_before_csv` **0.840930102 s**。该时间落在一次性 `attached_identity.json` 原子写入路径；临时目录在断言后自动清理，因此只保存了 [r1_forensics.json](../evidence/g5_gear/g0_r2/r1_forensics.json) 中的带来源说明摘要。该问题已移出采样循环，并作为回归风险保留。

## 设计选择

考虑过三种方向：

1. 只把 heartbeat 文件改到 tmpfs。实现最小，但不能保证所有部署路径都是 tmpfs，也不能隔离调度或文件系统长尾。
2. 建立独立 publisher 线程/进程。独立进程的身份和退出管理更复杂；线程足以释放采样主循环，同时保留有界状态和显式失败信号。
3. 丢弃或无限排队 heartbeat。前者可能隐藏 freshness 失效，后者会造成无界内存和无法审计的延迟，均不接受。

采用最小的单槽 `HeartbeatPublisher`：采样线程提交带**真实 sample monotonic timestamp** 的最新 heartbeat，后台 daemon 线程执行 `atomic_json(..., durable=False)`；单槽替换会计数 `coalesced_count`，不会无限增长。publisher failure、出版线程关闭失败或 heartbeat freshness 超过既有 `stale_seconds` 都会被采样器识别并 fail-closed。freshness 以最后一次实际发布（或首次提交）为基准，不能通过不断提交新时间戳伪造健康状态。

`attached_identity.json` 是一次性证据，不再在采样循环中同步写入；监护循环结束后再持久化。这样不改变身份绑定逻辑，也避免一次性持久写入制造新的监控 gap。最终 status 仍使用原有 durable atomic write；如果 publisher 线程在 bounded close 后仍存活，`continuous_monitoring` 为 false。

安全不变量未改变：0.1 秒目标、严格 `gap > 0.5 s`、1024 MiB 空闲下限、连续三次低显存、heartbeat freshness、身份检查、独占锁、信号停止、监督进程丢失检测和 fail-closed 语义均保留。每个 heartbeat 仍携带真实采样时间；没有伪造新时间戳。

## CPU-only 验证

R2 相关测试命令及原始输出见 [r2_relevant_tests.json](../evidence/g5_gear/g0_r2/r2_relevant_tests.json)：**99 项通过，28.15 秒，返回 0**。覆盖：

- 快速 publisher、慢 heartbeat writer、受控超过 0.5 秒的 writer 延迟、publisher failure 和 freshness stale；
- 真实 memory query 延迟超过 0.5 秒仍触发 sampling gap；
- 三次低显存、非连续低显存、mock NVML failure、guard/supervisor 丢失、身份校验、child 清理；
- atomic visibility、缺失 final evidence、完整系统生命周期；
- 空/截断/重复/错误 reset 的 GearMesh trace、原始 pose 几何重建和严格阈值。

新增的 publisher failure 集成测试直接运行 CPU guard，确认 publisher failure 会终止绑定 child。freshness 回归测试在 writer 阻塞期间持续提交新样本，仍能检测 stale；这覆盖了最初实现中“新提交时间戳掩盖旧 heartbeat 未发布”的竞态。

独立 bounded CPU 实验见 [cpu_experiments.json](../evidence/g5_gear/g0_r2/cpu_experiments.json)：

- ext4 仓库路径 50 次 `atomic_json`：median **121.0 µs**、P95 **286.8 µs**、max **612.3 µs**；
- `/run/user/1000` tmpfs 50 次：median **75.0 µs**、P95 **91.7 µs**、max **121.1 µs**；
- 两种路径的 publisher submit 都约 **0.9 µs median**，50 次中 49 次 coalesced、无 publisher failure；这是提交开销，不是文件发布耗时；
- 10 次 guard/supervisor CPU 生命周期全部通过，`continuous_monitoring=true`，supervisor raw return 0；wall median **1.067 s**、P95/max **2.717 s**。

这组 benchmark 只说明当前短测下 tmpfs 和 ext4 的可观测差异及后台提交的非阻塞性质，不能把 R1 的长尾归因于 ext4，也不能外推 GPU 运行。

排除所有历史 `test_g4*.py` 后的全套 CPU 测试为 **262 passed、1 failed**；唯一失败是既有 `test_trajectory_capture.py::test_production_wrapper_persists_before_close_and_keeps_predicate_returns`，不涉及 R2 guard/publisher，且 R2 相关 99 项全部通过。全仓库直接运行还会触发历史 G4 import-barrier/Sim2Sim 测试冲突；这些历史失败未被删除或修改。

## 代码和证据完整性

修改集中在 `scripts/gpu_guard.py`、新增 `scripts/g5_r2_cpu_stress.py`、`tests/test_g5_heartbeat_publisher.py`，以及本报告和 `evidence/g5_gear/g0_r2/`。没有修改 Isaac Lab、GearMesh reward、观测、动作、控制器、PhysX 参数、GPU 驱动或系统配置。没有遗留 R2 guard/supervisor/heartbeat 进程。

## R1 再运行建议与剩余风险

R2 只证明监护器在 CPU mock 下不再让 heartbeat 文件写入同步阻塞采样主循环，并且 publisher failure/stale 会安全停止。仍有三项风险：

- 每步 CSV 写入/flush 和 supervisor 的状态读取仍在采样路径；长尾仍可能制造真实 monitoring gap；
- final status 和一次性 attachment evidence 在监护结束阶段仍依赖文件系统；它们不会伪造“连续监控”，但可能造成最终证据不完整；
- CPU mock、ext4/tmpfs benchmark 和重复生命周期不能证明下一次 GPU 预检一定通过。

因此建议用户另行授权一次独立 G0-R1 GPU preflight，使用新的唯一输出目录，保留 0.5 秒阈值和现有 guard。若再次失败，应优先检查新的 `heartbeat_publisher`、CSV、协调和 deadline timing 字段，再决定是否继续工程修改。不得直接授权 G1 PPO 训练。
