# G5-GEAR-G0-R0：监护间隙诊断与验收器加固

**G5_GEAR_G0_R0_READY_FOR_REVIEW。** 本阶段只做 CPU 检查、模拟监护测试、代码和文档修改；没有运行 Isaac Sim、GPU 仿真、PPO 训练或评估。此前 G0 的 GPU PhysX、完整 episode 和 reset 仍未验证。本文建议待用户审阅后再单独授权 G0-R1，不自动启动。

## 基线与保护范围

起点和远端 `main` 均为 `f0850e2a10ca29f2e827afb63b1d2264313464bf`，本地工作树干净，无后续远端提交。上游 Isaac Lab v2.1.1 为 `90b79bb2d44feb8d833f260f2bf37da3487180ba`，执行前后源码 hash 一致、工作树干净。[基线与取证](../evidence/g5_gear/g0_r0/baseline_and_forensics.json) 保存了修改前文件 hash；[源码与完整性检查](../evidence/g5_gear/g0_r0/source_and_integrity.json) 保存修改后本项目代码、上游控制、配置、articulation 和数学源码的 SHA256。G0 原始证据目录工作树未修改，基线 Git tree ID 亦保留于证据中。

可访问的 PegInsert epoch 200 checkpoint 在修改前后 SHA256 均为 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`，与既有清单一致；没有读入模型或重写该文件。没有修改官方任务、reward、观测、控制器、几何、物理、驱动或系统设置。

## 原始监护间隙：独立复核

从 [G0 原始 samples.csv](../evidence/g5_gear/g0/guard/samples.csv) 的单调时间相邻差独立计算：437 样本、436 个间隔；超过 200/300/400/500 ms 分别为 **15/7/3/1**，最大 **505.215516 ms**；最低空闲 **5480.5 MiB**，连续低显存最大计数为 **0**。原 guard 的 `monitoring_gap`、child 的 raw wait `-15` 与 launcher 124 可相互核对。原 `console.txt` 为空，无法确认 G0 的实际决策数或是否执行 GPU 物理。

原实现的时间戳在每次 `source.read()` 之后取得，相邻时间戳的差包含上一轮处理/协调、读取身份和状态文件、CSV flush、原子 heartbeat 写入、可能的绑定身份持久写入（含 fsync）、睡眠及唤醒、下一次 NVML 调用和中间调度延迟。原 CSV 只有样本完成时间，不能回溯各环节分别耗时。只读用户 journal 可见两个服务于 22:18:41 启动、22:19:26 以 124 结束；可访问的同窗口 kernel journal 无条目。历史逐时 CPU/IO 压力和调度轨迹没有记录；当前 `/proc/pressure` 不能替代历史遥测。

按证据强度排序：

1. **已确认**：间隔超过既有 0.5 s 上限，guard 执行了失效关闭；不是已观察到低显存三连击。
2. **尚不能区分的候选**：CPU 调度或系统负载、文件系统/元数据与 fsync、NVML 调用以及进程协调的耗时都可能构成间隔。原数据不足以给它们可靠排序，也不能归咎 GPU。
3. **低显存导致本次终止**与原采样和状态不符；采样仍不能排除未采到的瞬时显存变化，但不足以据此改写终止原因。

可用以下只读命令复算原始间隔（单位秒）：

```bash
python3 -c 'import csv; r=list(csv.DictReader(open("evidence/g5_gear/g0/guard/samples.csv"))); a=[float(x["monotonic_seconds"]) for x in r]; d=[y-x for x,y in zip(a,a[1:])]; print(len(r),len(d),*[sum(x>t for x in d) for t in (.2,.3,.4,.5)],max(d),min(float(x["free_MiB"]) for x in r))'
```

## 监护器计时加固

`gpu_guard.py` 使用单调时钟在内存中按类别累计次数、总时间和最大值：memory query、CSV 写入/flush、heartbeat 原子写入、前后处理与进程协调、请求/实际睡眠、睡眠超时及 deadline 迟到。发生 gap 时记录前一轮类别、前一次睡眠和本次 query 耗时；最终 status 一次性持久化摘要，不新增逐样本文件写入。每项是墙钟时间，也包含其间可能的抢占；oversleep 不是调度原因的证明，计时本身有少量开销，无法把系统调用耗时与调度严格分解。异常、挂起或 SIGKILL 仍可能使最终 status 缺失，应按原协议视为不完整。

仍使用 0.1 s 目标采样、严格 `gap > 0.5 s`、空闲低于 1024 MiB 连续三次停止，以及原有身份核验、锁、心跳、进程停止、超时和 fail-closed 机制。边界测试覆盖 0.499999、0.500000、0.500001 秒；模拟低显存、延迟 memory query、异常和进程丢失测试覆盖停止行为。未调整既有门槛。

第一轮 CPU 生命周期测试有 **1 项失败、56 项通过，17.20 s**：模拟 guard 在间隔 0.633306 s 后按规则中止 CPU 子进程，supervisor 随后记录 SIGTERM 和 broken pipe。新计时显示前一轮“CSV 前处理/协调”墙钟 **0.632739 s**、本次模拟 memory query 约 **4.4 µs**；该区段恰逢目标绑定并包含身份读取及 `attached_identity.json` 的持久原子写入，但也可能包含线程调度等待，不能再细分为 fsync 或调度的因果结论。这只是新的 CPU 模拟案例，不能作为历史 GPU 间隙的根因。相关 [失败记录](../evidence/g5_gear/g0_r0/cpu_failure_first/failure_summary.json)、监护状态及 CSV 已保留，没有放宽安全门槛。后续完整 CPU 回归通过。

## 独立验收器与几何验证

旧 `g5_gear_verify.py` 对空 JSONL 返回无错误和零退出码，且没有要求完整 episode、双 done、自动 reset 或证据完整性；仅用环境中间位置计算成功，无法发现中间位置与原始物体位姿不一致。现在默认只有完整 300 决策验收模式：

- 要求记录 task ID、1 环境、seed 0；恰好 300 条、连续决策号、策略循环计数每次加 8 并在第 300 次达到 2400。该计数不包含 reset 内部模拟步。
- 检查全部必需的 reward、19/43 维观测、关节、四个物体根状态、位置、四元数及 keypoint 距离的形状和有限性；损坏 JSON、重复 key、缺项、非有限值均拒绝。空、缺失、截断或多余记录不能通过。
- 要求仅第 299 条同时 `terminated=True` 和 `truncated=True`，捕获该步一次真实调用官方 `_reset_idx([0])` 并成功返回，reset 后 episode counter 为 0；第 300 条必须是下一个 episode 的第 1 步，其他步不得出现 reset。`ResetCapture` 只包装并原样调用官方方法，CPU 测试检查返回值、异常传播和恢复原方法。
- 从独立计算的每步成功重建 ever、terminal 和 first-success，与原运行中 `SuccessTracker` 的 episode summary 按值和类型逐项比较。官方成功在 reset 前捕获；不存在成功就提前结束 episode 的新规则。

采集器向原有逐步字段**追加** task/seed/env 数、环境原点、观测、实际 reset 调用记录和 episode summary；不改官方配置和行为。先前没有原始轨迹，因此不存在迁移旧轨迹的问题；缺少新字段的旧格式不能被误认作完整验收证据。在线 JSONL 每步 flush，但不保证主机故障时持久完整。

几何计算使用 Isaac Lab `root_state_w` 的世界 xyz 和 **wxyz** 根四元数、减去 `scene.env_origins` 后仍与世界轴平行；持有齿轮和固定底座各使用局部 `(0.02025,0,0)` m。独立 CPU 代码用四元数旋转矩阵第一列分别重建两参考点，并以逐坐标 **5e-6 m** 绝对容差与运行时中间值比较；四元数范数偏差须不超过 `1e-4`。容差用于 float32 位姿运算及原点相减的数值误差，不改变 2.5 mm / 1 mm 成功阈值：XY 和单侧 z 均严格 `<`，负 z 允许成功。边界附近如浮点计算路径不一致，验证器保守拒绝并需要复核，不静默扩大官方成功范围。旋转、非零环境原点、严格阈值和负 z 都有纯 CPU 控制测试。该验证不证明 PHYSICAL_CONTACT_QUALITY 或 ACTUAL_GEAR_ENGAGEMENT。

## 测试与下一步

最终命令（在仓库根目录；使用现有隔离 Python）为：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/envs/factory-g01/bin/python \
  -m pytest -v tests/test_g5_guard_timing.py tests/test_g5_gear_capture.py \
  tests/test_g5_gear_verify.py tests/test_success_metrics.py \
  tests/test_g3a_supervision.py tests/test_diagnostic_metrics.py
```

**92 项通过，pytest 报告 18.84 s，外部墙钟 18.995 s，返回 0**；测试逐项结果和精确 argv 在 [cpu_tests_final.json](../evidence/g5_gear/g0_r0/cpu_tests_final.json)。完整合成 299 步 episode + 第 300 步正例和空/截断/缺步/重复、物理计数、超时位置、双 done、reset 调用/计数、非有限奖励/状态/观测、格式/重复键、错误几何/谓词/统计、严格边界、负 z 等负控均通过；原成功统计及 supervisor 测试通过。期间还有先前 88 项、90 项的完整通过记录；首轮失败单独保留。

G0-R1 需事先重新注册唯一输出路径、版本/hash、seed、300 决策、900 秒超时、当前 GPU 准入与完整精确命令，并获得用户明确授权。运行后可用 `python3 scripts/g5_gear_verify.py <trace.jsonl>` 验证轨迹；其 0 返回码仅证明轨迹契约通过。仍须独立核对任务注册、运行时 GPU dynamics 与 CUDA PhysX view、实际过程及 supervisor 正常退出、guard 监控完整、资源采样和实际轨迹；检查未测的接触、穿透、关节限位与装配质量。任何 guard 间隙、缺状态或未达 episode 边界均不能当作完整通过。不得依靠本阶段 CPU 模拟或历史 32 步记录声称当前 GPU PhysX 成功；G1 训练需另行批准。
