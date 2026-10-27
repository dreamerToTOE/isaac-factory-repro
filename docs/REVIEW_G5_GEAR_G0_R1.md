# G5-GEAR-G0-R1：单次受监护 GPU PhysX 预检

**最终状态：`G5_GEAR_G0_R1_PARTIAL`。** 本次按授权只启动了一次新的 GearMesh GPU 预检，没有 PPO、checkpoint 加载或第二次 GPU 尝试。子进程在 Isaac Sim 预检轨迹产生前被现有 guard 因监控间隙安全终止；因此没有可验证的 GPU PhysX、环境初始化、完整 episode、自动 reset 或运行时指标轨迹。所有部分结果和原始退出证据均已保留。

## 注册身份与准入

实验前本地 HEAD、远端 `origin/main` 均为 `c5548697ef628ce32a2596a8c0bcac7884bc3fd3`，上游 Isaac Lab 为 `90b79bb2d44feb8d833f260f2bf37da3487180ba`；工作区干净，R0 CPU 检查通过 92 项。唯一执行计划、命令、unit 名称、路径、配置和源 hash 见 [registered_plan.json](../evidence/g5_gear/g0_r1/registered_plan.json)。输出目录为新的 `work/g5_gear_g0_r1_seed0_20261027`，trace 预定路径为 `evidence/g5_gear/g0_r1/trace.jsonl`；没有复用 G0/R0 目录。

启动前 RTX 3070 / 8192 MiB、驱动 580.178.04、CUDA compatibility 13.0；全卡约 2079 MiB 已用、5740 MiB 空闲、GPU utilization 37%，仅见已有桌面/远程进程，未杀死任何无关进程。系统内存约 46 GiB 总量、约 33 GiB available；当时 CPU pressure avg10 约 0，IO pressure avg10 约 1.56。该快照不是 GPU PhysX 成功证明，只是资源准入记录。

## 执行结果

guard 与 supervisor 成功启动，child 通过 gate release；但从 22:58:52 到 22:59:00 运行约 7.44 秒后，guard 在相邻样本间检测到 **0.550642873 s** 间隔，超过严格的 0.5 s 上限。guard 记录 `stop_reason=monitoring_gap`，随后向 child 发送 SIGTERM；supervisor 记录 raw wait `-15` / `SIGTERM`，launcher 和 systemd unit 返回 124。未触发 1024 MiB 空闲下限连续三次条件。该安全停止符合授权规则，不能作为环境初始化失败或 GPU 内存不足的结论。

child console 为 0 字节，trace 文件不存在。没有任何 policy-decision 记录，因此：

- 实际记录的 policy decisions：**0**；
- 已验证的 policy-loop physics steps：**0**；不能把初始化/reset 内部物理步误计为策略步；
- 完整 episode、299 步 timeout、双 done、自动 reset 和第 300 步新 episode：**未验证**；
- runtime metadata、任务注册、19/43 观测、6 维 action、GPU dynamics、CUDA PhysX view、reward/state finite checks：**未验证**。

未发现 `g5-gear-g0-r1`、`g5_gear_preflight` 或 `python_entry` 残留进程。无关项目进程未修改。

## guard 诊断

本次 guard 采集 68 个内存样本、67 个间隔；最低空闲显存 **5481.625 MiB**，峰值采样全卡 used **2336.0 MiB**，最大连续低显存计数为 0。memory-query 计时聚合最大约 **43.69 ms**，总约 **279.76 ms**；CSV write/flush 最大约 **0.085 ms**；heartbeat 原子写入最大 **550.086 ms**，总约 **649.54 ms**；heartbeat 后处理最大约 **0.252 ms**；sleep overshoot 最大约 **2.31 ms**；deadline lateness 最大约 **450.793 ms**。该 gap 的细节记录显示前一轮 heartbeat write 约 **550.086 ms**，本次 memory query 约 **0.046 ms**，因此本次可观测时间主要落在 heartbeat 原子更新区段；这仍不能区分文件系统阻塞、调度抢占或其它内核等待，也不能证明硬件或 GPU 是根因。原有 0.1 s 目标、0.5 s gap、1024 MiB floor、三次低显存、身份、心跳、超时和 fail-closed 规则均未放宽。

完整 guard 与 supervisor 原件：[guard](../evidence/g5_gear/g0_r1/guard/)、[supervisor](../evidence/g5_gear/g0_r1/supervisor/)。独立复核摘要见 [independent_runtime_review.json](../evidence/g5_gear/g0_r1/independent_runtime_review.json)。

## 独立验证与完整性

离线执行加固后的 `scripts/g5_gear_verify.py`，因 trace 缺失返回非零，结果 `accepted=false`；这是预期的 fail-closed 行为，不是手工补造的轨迹。结果见 [independent_verification_result.json](../evidence/g5_gear/g0_r1/independent_verification_result.json)。无法执行任何 episode-level ever-success、terminal-success 或 first-success 计算，也无法从 raw poses 重建运行时齿轮参考点。

R1 运行后，注册的源 hash 全部匹配；当前仓库仍为注册 commit，上游工作树干净。PegInsert epoch 200 checkpoint SHA256 仍为 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`。完整检查见 [source_and_checkpoint_integrity.json](../evidence/g5_gear/g0_r1/source_and_checkpoint_integrity.json)。原 G0/R0 证据未修改。

物理诊断状态：非有限状态、关节异常、齿轮异常运动、接触/穿透、`PHYSICAL_CONTACT_QUALITY` 和 `ACTUAL_GEAR_ENGAGEMENT` 均为 **NOT_MEASURED**；没有轨迹可供分析。参考动作也没有机会被解释为装配成功率。

## 结论与下一步

本次满足“单次授权执行”和“安全监护停止并保留证据”，但未满足 R1 PASS 的运行时验收门槛。状态保持 `G5_GEAR_G0_R1_PARTIAL`。不建议授权 G1 PPO 训练；应先由用户审阅本次 heartbeat/gap 证据，另行决定是否允许新的 G0-R1 运行。当前授权已消耗，本文不会自动重试，也不会开始训练或其它 GPU 实验。
