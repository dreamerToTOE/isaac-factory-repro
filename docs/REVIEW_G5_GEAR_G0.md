# G5-GEAR-G0 源码审计、指标核验与有界预检

**工程状态：G5_GEAR_G0_PARTIAL。** 目标是审计官方 GearMesh 并验证完整 episode 与自动 reset；源码与 CPU 指标检查有结果，但本次唯一获授权的 GPU 预检被安全 guard 中止，未产生运行轨迹，完整 episode、运行时指标和本次 GPU PhysX 均未验证。没有启动 PPO，也不要求参考动作成功装配。本报告不推断未测结果。日期：2026-10-27（Asia/Seoul）。

## 来源、身份与资产

现有仓库起点/执行时远端 `main` 为 `38735e88bb3377cabccc45ef011bec28a255b1ca`；是用户指定提交，本地干净并 fast-forward。最终提交与远端 SHA 在发布后以 Git 核验结果为准：提交内容不能包含它自身的 SHA。锁定上游 Isaac Lab v2.1.1 Git `90b79bb2d44feb8d833f260f2bf37da3487180ba`，执行前后源文件 SHA256 一致，上游工作区干净。逐文件及启动脚本哈希、完整命令、seed 和预算见 [运行前注册](../evidence/g5_gear/g0/registered_plan.json)、[独立复核](../evidence/g5_gear/g0/independent_review.json)。主要源码：[环境](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py)、[环境配置](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py)、[任务和资产](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py)、[控制器](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py)、[PPO](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml)。所引用远程 USD 资产并未在本次启动中重新加载或独立几何测量。

任务使用 Franka Panda、夹持 `factory_gear_medium.usd`，装在 `factory_gear_base.usd` 的中间轴；两侧另有小、大齿轮 USD articulation。中齿轮配置直径 0.03 m（夹爪宽度用途）、高 0.03 m、质量 0.012 kg；底座高 0.02 m、base_height 0.005 m；中轴局部偏移 `(0.02025, 0, 0)` m，小/大轴 x 分别 0.05075/-0.03025 m。资产加载通过官方 Nucleus 路径；碰撞形状来自 USD，配置 contact offset 0.005 m、rest offset 0，接触传感器激活标志不代表本次采集了接触力。源码创建两侧 articulation，reset 在 `add_flanking_gears=True` 时直接放置两侧；虽然配置有 `add_flanking_gears_prob=1.0`，Factory 源码中没有用这个概率采样，不能描述为随机开关。持有件配置 disable_gravity=True，不应假设其完全由重力和握持维持。

## 观测和动作

构造时按顺序重算，配置中的 21/72 是占位数值，不是运行维度。源码与历史 GearMesh 运行记录均对应 actor 19、critic 43、action 6：

| 输入 | 分量与顺序 |
|---|---|
| actor 19 | 末端相对带噪固定目标位置 3；末端四元数 4；末端有限差分线速度 3、角速度 3；EMA 后动作缓存 6 |
| critic 43 | 末端位置 3；末端四元数 4；末端刚体线速度 3、角速度 3；7 关节位置；持有件位置 3、相对无噪目标位置 3、四元数 4；固定件位置 3、四元数 4；EMA 后动作缓存 6 |

位置先从世界位置减去环境原点，分量仍沿世界坐标轴；相对位置不是旋转到目标局部轴。四元数为 `(w,x,y,z)`；actor 的速度由位置和四元数差分，critic 使用刚体速度；actor 对固定观测目标加每次 reset 采样、整 episode 固定的独立高斯位置噪声（各轴标准差 1 mm）。默认不包含接触力。缓存为当前已滤波 `actions`，即用于下一次决策的历史动作，不是未滤波原始指令。

6 维对应 xyz 平移和轴角旋转增量；EMA `a_t=0.2 u_t+0.8 a_(t-1)`，分别乘 0.02 m、0.097 rad 阈值，目标相对于每个控制子步的当前末端生成。平移目标相对于带噪动作参考框逐轴裁剪到 ±0.05 m；roll 固定 π、pitch 固定 0，yaw 保留。`dt=1/120 s`、decimation 8，即控制物理子步 120 Hz、策略 15 Hz。任务空间 PD 默认比例 `[100,100,100,30,30,30]`、微分 `2 sqrt(kp)`，经雅可比转置映射并加入质量矩阵加权 nullspace 项（kp=10、kd=6.3246）；代码 torque 裁剪 ±100，模拟器 arm 关节 1–4 effort limit 87、5–7 为 12，手指 40；gripper 目标为 0，手指由 PhysX PD 驱动。没有改增益或其他官方控制配置。

## 奖励与成功定义

每个资产的局部 z 轴上以 0.15 m 缩放的四个均匀 keypoint 被变换到环境坐标；`d` 是四对点的欧氏距离平均值。定义 `f(d;a,b)=1/(exp(a*d)+b+exp(-a*d))`，默认 GearMesh 每步奖励为 `f(d;5,4)+f(d;50,2)+f(d;100,0)+I_engaged+I_success`。源码还计算动作及动作变化范数惩罚，但两个系数均为 0；没有接触力奖励。engaged 使用相同 2.5 mm 横向条件及 `z差 < 0.02×0.9=0.018 m`，不是机械接触检测。

令 `p_h = p_held + R_held*(0.02025,0,0)`，`p_t = p_fixed + R_fixed*(0.02025,0,0)`，位置已减环境原点，向量分量与世界轴平行。官方几何成功是 `norm((p_h-p_t).xy) < 0.0025 m` 且 `(p_h-p_t).z < 0.02×0.05 = 0.001 m`，均严格小于；z 是单侧上界，没有下界，也不检查旋转、接触力、保持时间、穿透或齿轮是否真正咬合。故 OFFICIAL_GEOMETRIC_SUCCESS、PHYSICAL_CONTACT_QUALITY 和 ACTUAL_GEAR_ENGAGEMENT 必须分开。`applied_wrench` 是控制器命令，不是传感器接触力。

成功不提前终止。`episode_length_s=20`、`dt=1/120`、decimation=8，最大长度 `ceil(20/(8/120))=300`；源码在自增 episode 计数 `>= max_episode_length-1` 时把 terminated 和 truncated **都**设真，所以预期第 299 个决策触发 reset；父环境先计算 reward/官方成功，再自动 reset，返回的 observation 已是新 episode。这是源码推导，不是本次运行验证。官方 `extras['successes']` 在任一 reset 时计算当步当前成功数/全部环境数，既不是 ever-success，也不一般等同仅对完成环境的 terminal-success；官方 first-success 的 `ep_success_times` 没有在 `_reset_buffers` 中清零，跨 episode 统计应谨慎。独立 evaluator 应在 reset 前捕获官方谓词，自行维护 episode ever、terminal 和 first-success，保留双重 done 标志。

## 重置与风险

固定件位置以配置默认值加各轴均匀 ±0.05 m；yaw 从 0 到 15° 均匀采样（代码不是 ±15°）。目标观测点在固定件局部 `(0.02025,0,0.025)` m；末端初始高度再加 0.035 m，并加位置噪声 ±(0.02,0.02,0.01) m 和 yaw 噪声 ±0.785 rad。Panda 先恢复指定关节姿态，关闭重力并用 DLS IK，位置和角度误差都需不大于 1e-3；失败时重抽且 `while True` 无重试上限，必须依赖外部超时，不静默兜底。然后把两侧齿轮放置到固定件姿态，将中齿轮按末端相对变换写入并施加夹内局部位置扰动（x ±0.003 m，y 0，z [-0.003,0] m），闭合夹爪约 0.25 s，再恢复重力。重置模拟步不计入父环境 `_sim_step_counter` 的策略循环步，不能将 `decisions×8` 解释为包含 IK 和抓取的总物理步。

## 历史证据与本次预检

历史 [32 步 GearMesh probe](../evidence/2026-10-23/gear_probe_1_pinned/result.json) 返回 0，记录 1 环境显式 reset、32 次有界参考动作、有限观测与状态、19/43 维，未出现 episode reset；全卡采样峰值 5194 MiB、测试进程 3502 MiB。它不是完整 episode 或成功啮合，更不能代替本次运行。

本次注册 task `Isaac-Factory-GearMesh-Direct-v0`、1 环境、headless `cuda:0`、seed 0、300 次决策上限、超时 900 s；输入 x 交替 -0.05/+0.05，其余为零，没有训练策略或 optimizer。精确 argv、所有源/配置哈希在 [registered_plan.json](../evidence/g5_gear/g0/registered_plan.json)；复现须先为输出路径和 systemd unit 指定未使用的新名称，并经单独授权。使用原 `run_sim.sh`、systemd guard/supervisor、独占 GPU 锁，未改上游物理或安全阈值。新采集器复用 `PreResetSuccessCapture` 和 `SuccessTracker`，设计为逐步保存官方谓词、预 reset 几何、机器人和物体状态及 done；但此采集器在本次没有进入可验证运行，不能声称其动态功能通过。

启动前 Ubuntu 22.04.5、Python 3.10.12、Sim VERSION `4.5.0-rc.36+release.19112.f59b3005.gl`、隔离环境 RL-Games 1.6.1；RTX 3070 8192 MiB、驱动 580.178.04，启动前全卡 约 2146 MiB 已用 / 5673 MiB 空闲（NVML reserved 导致总量不必与 used+free 完全一致），其中包含桌面和其他 GPU 程序。驱动可见、先前项目环境通过并不能单独证明本次 GPU PhysX。guard 在运行期间采得全卡峰值 2337.125 MiB、最低空闲 5480.5 MiB，共 437 样本；采样最大值不保证捕获瞬时峰值。独立 GPU 进程显存未单独采集。

guard 发现采样间隙 0.5052155 s 大于原有 0.5 s 上限，于启动约 44.7 s 发送 SIGTERM；未观察到连续低于 1024 MiB 的显存停止条件。supervisor 记录 child raw wait `-15` / SIGTERM，launcher 返回 124；guard 记录 `monitoring_gap`，supervisor 的 `child_signal` 与 guard 记录合并解释。见 [guard status](../evidence/g5_gear/g0/guard/status.json)、[采样](../evidence/g5_gear/g0/guard/samples.csv)、[supervisor status](../evidence/g5_gear/g0/supervisor/status.json)。console 为 0 字节，trace 不存在；实际决策数和实际物理步数未知，不能把它们写成零或 300。未能核验任务注册、显式 reset、GPU dynamics 和 CUDA PhysX view、finite 状态/奖励、done、自动 reset、运行期几何成功、接触异常或本次 GPU 仿真吞吐。没有第二次尝试，未放宽 guard。

## 独立数值检查与验收

CPU 回归：复用成功统计、supervision 和诊断测试 55 项通过；新增纯 Python 边界测试 1 项通过，覆盖严格 2.5 mm、严格 1 mm、二维范数与负 z 可成功的单侧条件，见 [测试记录](../evidence/g5_gear/g0/cpu_tests.txt)、[边界测试](../evidence/g5_gear/g0/metric_cpu_test.txt)。独立标量验证器已准备，但由于缺少 trace，无法对真实轨迹计算谓词、ever/terminal、物理步或非有限值；因此不能把 CPU 单元测试描述为本次动态指标通过。PegInsert epoch 200 原 checkpoint SHA256 在本次复核仍为 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`，历史 GearMesh 证据无改动。

| 指标 | 当前结论 |
|---|---|
| ENVIRONMENT_RUNTIME_PASS | 未验证；guard 提前中止 |
| METRIC_IMPLEMENTATION_PASS | CPU 边界与既有 bookkeeping 测试通过；运行轨迹交叉核验未完成 |
| EPISODE_RESET_PASS | 未验证 |
| PHYSICAL_DIAGNOSTICS_STATUS | 未测；接触质量、齿轮啮合和关节限位无结论 |
| PPO_TRAINING_NOT_STARTED | 是 |

建议先审阅 guard 的采样间隙和系统负载，并获得对一次新、唯一、有界 G0 预检的明确授权；保持安全门槛及官方行为，完成 GPU PhysX、完整 episode、reset 与独立轨迹复核后再决定是否进入 G1。候选资源和科学训练方案见 [G5_GEAR_TRAINING_PROTOCOL](G5_GEAR_TRAINING_PROTOCOL.md)，不是执行授权。技术学习和面试考察仍为 PENDING。
