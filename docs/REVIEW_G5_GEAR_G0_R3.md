# G5-GEAR-G0-R3：单次受监护 GPU PhysX 验收运行

**最终状态：`G5_GEAR_G0_R3_PARTIAL`。** 本次按授权只执行了一次新的 GPU 运行，没有重试、没有 PPO、没有 checkpoint 加载，也没有其它 GPU 实验。Isaac Sim/Factory 场景完成了启动并持续占用 GPU，但 child 在环境初始 reset/初始化路径中运行到 900 秒超时，未产生 300 步 trace；因此不能宣称完整 episode、自动 reset 或独立轨迹验收通过。

## 注册与运行身份

运行前本地和远端 `main` 均为 `bf7ce69c87cc21642247ae60fabb73ab02f06844`，上游 Isaac Lab 为 `90b79bb2d44feb8d833f260f2bf37da3487180ba`。R2 CPU 相关测试 99 项通过。唯一运行计划、完整 argv、R2 源 hash、GPU 快照和唯一 systemd unit 记录于 [registered_plan.json](../evidence/g5_gear/g0_r3/registered_plan.json)。原始 guard/supervisor 文件见 [guard](../evidence/g5_gear/g0_r3/guard/) 和 [supervisor](../evidence/g5_gear/g0_r3/supervisor/)。G0、R0、R1、R2 证据未修改。

启动前 RTX 3070 为 8192 MiB、驱动 580.178.04；约 5625 MiB 空闲。没有终止无关进程。启动日志确认 Ubuntu/Isaac Sim、`cuda:0`、Factory 场景创建、`Number of environments: 1`、物理步长 1/120 和 simulation start 完成；GPU 进程在运行期间被 `nvidia-smi` 观察到。由于 preflight 在输出 `G5_RUNTIME_META` 前进入 reset/后续初始化，`is_gpu_dynamics_enabled()` 和 CUDA PhysX view 的专门断言结果没有被写入可验证 console，故“实际 GPU PhysX 完整验收”仍只能标为未完成。

## Episode 和 trace 结果

原始 trace 路径不存在，独立 verifier 对其返回非零和 `accepted=false`，结果见 [independent_verification.json](../evidence/g5_gear/g0_r3/independent_verification.json)。因此：

- 记录的 policy decisions：**0**；
- 已验证的 policy-loop physics steps：**0**；
- 完整 episode：**未完成**；
- decision 299 双 done：**未观察**；
- 自动 reset 和 post-reset counter：**未观察**；
- decision 300：**未观察**；
- 19/43/6 运行时接口、有限 states/rewards、raw pose 几何验证：**未完成**。

console 在场景启动后没有出现 `G5_RUNTIME_META` 或 `G5_RUNTIME_RESULT`。结合 trace 文件缺失，可以确定脚本没有进入其逐步 trace 写入阶段；不能把初始化期间的 simulator physics steps 计为 policy-loop steps。

## 监督和 R2 monitor effectiveness

R2 guard 在约 900 秒内完成 **9004** 个样本、9003 个间隔：

- sampling interval 中位数 **99.999670 ms**，P95 **100.072234 ms**；
- 最大间隔 **376.265 ms**，超过 500 ms **0 次**；
- 最低空闲显存 **1891.75 MiB**；峰值 sampled used **5925.875 MiB**；
- 连续低显存最大计数 **0**；
- heartbeat submit 最大 **3.596 ms**，总 **140.184 ms**；
- HeartbeatPublisher 实际写入最大 **534.122 ms**，发布 8927 次，coalesced 77 次，failure `null`，关闭后线程不存活；
- CSV write/flush 最大 **85.603 ms**；memory query 最大 **276.361 ms**；deadline lateness 最大 **177.298 ms**；heartbeat 后协调最大 **3.211 ms**。

R2 的关键目标在这次实际 GPU 长运行中得到部分工程证据：heartbeat 文件实际发布曾超过 0.5 秒，但采样 submit 没有同步等待，最大采样间隔仍为 0.376 秒，未触发 guard gap。这个结果表明异步 publisher 隔离了 heartbeat 文件写入的同步阻塞；它不能证明 R1 的 ext4 长尾根因，也不能证明下次 GPU 运行一定不会有 CSV、memory query、协调或调度长尾。

Guard 状态为 `stop_reason=child_wait_observed`、`continuous_monitoring=true`、无 monitoring gaps。Supervisor 记录 `termination_reason=timeout`，向 owned child 发送了 SIGTERM；raw wait returncode 为 0，但这不等同于正常完成，因为 supervisor 明确因 900 秒超时结束。没有遗留 R3 child、guard 或 supervisor 进程。完整机器摘要见 [monitor_summary.json](../evidence/g5_gear/g0_r3/monitor_summary.json) 和 [independent_runtime_review.json](../evidence/g5_gear/g0_r3/independent_runtime_review.json)。

## 物理和科学限制

GPU memory 使用和场景启动证明了一个真实 Isaac Sim GPU 进程曾运行，但没有可接受的 policy trace。非有限 robot/object states、reward、joint anomaly、reset discontinuity、官方 geometric success、contact force、penetration、`PHYSICAL_CONTACT_QUALITY` 和 `ACTUAL_GEAR_ENGAGEMENT` 均为 **NOT_MEASURED**。不能将本次 reference-action 结果解释为装配失败率或 PPO 性能。

PegInsert checkpoint 在运行后仍为 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`；注册源 hash、当前仓库和上游工作树完整性见 [source_and_checkpoint_integrity.json](../evidence/g5_gear/g0_r3/source_and_checkpoint_integrity.json)。

## 结论

R3 证明 R2 asynchronous HeartbeatPublisher 在一次真实 GPU 长运行中避免了 heartbeat 写入造成的采样 gap，但官方 GearMesh preflight 没有越过初始 reset/初始化阶段，且 supervisor 超时。故状态为 `G5_GEAR_G0_R3_PARTIAL`，不满足 R3 PASS。当前不能授权 G1 PPO 训练；需要用户审阅这次 900 秒初始化/reset 超时及 monitor 证据后，再单独决定是否进行新的工程诊断或 GPU 运行。本文到此停止，不自动重试。
