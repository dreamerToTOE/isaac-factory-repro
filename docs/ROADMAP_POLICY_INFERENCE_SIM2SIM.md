# PegInsert 独立推理 → Sim2Sim → 其它 Factory 任务：后续执行路线

决策日期：2026-10-26（Asia/Seoul）

性质：**用户批准的未来工作顺序；不是实验结果或自动执行许可。**
目标仓库：`dreamerToTOE/isaac-factory-repro`

**Current status (2026-10-26): G3C-R5 has completed and been archived. G4-I0 is now the active next stage.**

## 2026-10-26 前瞻性更新：G4-I1 授权与学习安排

用户在 G4-I1 任务中已确认 **G4-I0 PASS，并明确授权实现和执行 G4-I1**。本次范围到 G4-I1 结束即停止；G4-I2、Sim2Sim、GearMesh、NutThread、seed 2 及其他训练均未获本次授权。上面的状态行及下文原路线保留为制定时的记录；当前执行范围以本更新及 G4-I1 任务要求为准。

用户同时改为**先完成工程项目，待 Sim2Sim 后在另一会话进行系统理论教学和知识评估**。从本更新起，工程交付不要求用户回答理论问题，学习门禁不阻塞已授权工程的实现、测试或验收；不得将学习状态误记为通过，当前学习状态仍为 **PENDING**。这只调整学习日程，不降低 checkpoint、数值 parity、真实 episode、错误处理、证据保存或独立工程验收要求，也不授权提前启动 Sim2Sim。

下文涉及“每阶段结束先问 2–3 题”或“用户独立解释后才能工程 PASS”的原安排，作为历史决策保留；对本更新后的工作，其学习提问和理解验收时间统一推迟到上述独立会话。工程状态与学习状态继续分别记录，详见 [`ENGINEERING_LEARNING_GATE.md`](ENGINEERING_LEARNING_GATE.md)。既有实验结果、报告及原始证据不追溯修改。

## 0. 当前证据和冻结边界

- 官方 Isaac Lab v2.1.1 固定于 `90b79bb2d44feb8d833f260f2bf37da3487180ba`；Isaac Sim 4.5.0，RL-Games PPO、双层 LSTM、Franka PegInsert；既有来源审计见 [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md)。
- G2：训练 seed 0，64 env × 200 epochs，1,638,400 transitions；独立 deterministic evaluation，ever/terminal = 280/288 = 97.22%。这是官方几何成功定义下的结果，不等于完整装配质量/真机可靠性。
- G3C-R4：另一个全新训练实验，seed 0，32 env × 400 epochs，同为 1,638,400 transitions；epoch 400 ever = 268/288 = 93.06%，terminal = 267/288 = 92.71%。环境数、每轮更新与随机流不同；不能从 G2 vs R4 的单 seed 比较推断 32/64 env 因果优劣。见 [`REVIEW_G3C_R4.md`](REVIEW_G3C_R4.md)。
- **G3C-R5（已完成并归档）**：32 env、training seed 1、400/400 epochs、1,638,400 transitions，child exit 0；独立确定性评估最终 ever/terminal success 均为 278/288（96.53%）。完整结果及评估期间 SSL 故障的独立恢复证据见 [`REVIEW_G3C_R5_FINAL.md`](REVIEW_G3C_R5_FINAL.md)。状态为工程 `COMPLETE`、科学 `ASSESSED_DESCRIPTIVELY`；未启动 training seed 2，两个训练 seed 不足以可靠估计种子总体方差。原路线制定时的“R5 运行中”已由本状态更新取代，历史原始证据不修改。
- `G3V` 可能存在本地未跟踪演示资产；未来须由运行该任务的窗口核实，不推断其已完成，也不删除或覆盖这些文件。

历史报告中的“下一步”均是其成文时的建议，不应修改为新决策。本文是新的前瞻性导航文件，不能追溯更改 G0–G3C 的科学记录。不能把训练 seed 和评估 seed 混为一谈。

## 1. 新顺序及停止条件

| 优先级 | 阶段 | 核心产物 | 门禁与下一步 |
| --- | --- | --- | --- |
| 0 | G3C-R5 归档（已完成） | 真实 child wait/status、checkpoint、独立评估与限制已归档 | `COMPLETE / ASSESSED_DESCRIPTIVELY`；不自动补跑、续训或启动 seed 2 |
| 1 | **G4-I：独立策略推理与控制回路测试** | 与 PPO 训练器解耦的推理入口、延迟/控制周期/限幅/失败轨迹基准、成功率对照 | **独立审阅 GO** 后才进入 G4-S；不等同真正 Sim2Sim |
| 2 | **G4-S：真正的跨引擎 Sim2Sim** | Isaac Lab/PhysX 训练的同一 policy 在第二种物理引擎中的无重训运行、接触与动力学差异审计 | 分层门禁；若资产或控制接口无法忠实映射可 PARTIAL/STOP，禁止包装为跨引擎成功 |
| 3 | GearMesh | 复用已审阅的训练/独立评估链，得到任务级结果和真实失败分析 | 需要用户在 G4-S 审阅后单独批准 GPU 长训练 |
| 4 | NutThread | 同上，额外审计旋合动作变换、接触/几何与成功标准 | 单独批准；不能因 GearMesh PASS 而自动启动 |

每一阶段独立提交、独立报告和验收；Codex 结束一个阶段后 **STOP**，不得自动越级开展下一个实验。建议每个 Codex 窗口只负责一个可验证的子任务，防止长上下文失控。

## 2. G4-I：独立推理工程验收（当前优先项）

这里的“独立”是指**策略运行与训练 loop/optimizer 解耦**，可复用原 Isaac Sim/PhysX 场景和官方低层控制器；不要求第一阶段立即拆为两个操作系统进程，也不要求 ROS2 或真机。现有官方 `rl_games/play.py` 及 D001/G2 evaluator 可作为**基准 oracle**，不能把原 play 脚本换名就宣称已经完成独立部署。

### G4-I0：只读接口审计与测试预注册（CPU 为主）

1. 冻结一个已确认的 G2 epoch200 checkpoint SHA 和一个 G3C-R4 epoch400 checkpoint SHA；优先用 G2 建立最小闭环，R4 只作后续适配对照。核对真实本地路径、版本、模型与输入 RMS。
2. 从实际配置/调用路径独立梳理：19 维 actor observation 的逐项定义、归一化、6 维 action 的单位与坐标系、动作映射/裁剪、低层控制器、物理 `dt`、子步与策略控制周期、episode reset、LSTM hidden/cell reset。**不可凭假设填写控制 Hz。**
3. 定义可复用的 `observation → normalization → recurrent policy → action postprocess → low-level controller` 接口；训练权重不变、不启动 optimizer、不改变 reward、physics 或 success predicate。保存接口契约、测试 seed、误差容差和采样/时钟口径。
4. 校验 checkpoint 来源可信性；按现有已审阅兼容流程加载自有 checkpoint，不为加载便利放宽未知文件反序列化安全性。
5. CPU mock 单测必须覆盖维度/归一化、LSTM 状态重置、动作裁剪、异常非有限动作、时序和错误处理。没有 GPU 或 checkpoint 时报告真实 blocker。

**I0 PASS**：接口契约、checkpoint/模型指纹、冻结对照协议及 CPU 测试齐全；**STOP** 等待审阅，不直接开展漫长 rollout。

### G4-I1：脱离训练循环的单环境推理

1. 创建新的独立 inference entry point，明确只使用 actor/normalization/RNN 推理，不初始化或执行 PPO 更新；先 headless、单环境短测，资源允许再进行 1–4 环境的 GUI 演示。
2. 同步输入/输出定义，与原独立评估器对**相同观测和 LSTM 初始状态**做逐步 action parity 测试。预注册数值容差，记录最大绝对差/不一致样本；不得通过调动作、阈值或随机流来凑成功率。
3. 使用确定性策略作为主要协议。首次启动和每次 episode reset 的 hidden/cell state、normalizer、动作处理必须一致。记录 post-reset 首个动作，防止跨 episode 记忆泄漏。
4. 独立推理闭环在原 Factory 场景真实 step，完成短 episode；记录完整 child exit、命令、resolved config、GPU 资源、trace 和官方 **pre-reset** 成功判据。原评估器的 ever-success、terminal-success、first-success timing 仍分开报告。

**I1 PASS**：独立运行、action parity（在预注册容差内）、至少一个完整 episode 和检查点身份链成立。若只跑通官方 `play.py`，只能判 `PARTIAL`。

### G4-I2：控制频率、延迟、限幅与失败轨迹

至少分开记录以下指标（优先在 headless 模式测试，渲染条件另报）：

| 项目 | 必须报告的定义/数据 |
| --- | --- |
| 控制周期与有效控制频率 | 物理 dt、每个策略动作对应的物理子步数、预期 policy Hz、真实闭环命令间隔 P50/P95/P99、deadline miss 次数；区分**仿真时间**与**墙钟时间** |
| 神经网络推理延迟 | 在 GPU 正确同步计时下的单步 policy forward P50/P95/P99、均值、样本数、warmup、CPU/GPU 型号；不要与整次 `env.step`、渲染或 FPS 混为一谈 |
| 端到端控制延迟 | 观测可用 → 预处理 → policy → 限幅 → 命令提交各段耗时和总耗时；记录真实控制频率与实时系数 |
| 动作约束 | 原始网络输出、归一化/物理单位下的动作、最终下发量、每维限幅/饱和率和非有限值；**不要把 6 维 policy action 直接叫电机 torque** |
| 失败轨迹 | 固定 seeds 与 episode ID，保存 pre-reset 前状态/观测、动作、末端与 peg/hole 几何、成功真假/首次成功、episode 终止原因；无法获取接触力时明确 `unsupported` |
| 策略有效性 | 固定预注册测试集（建议最终至少 288 episodes，3 evaluation seeds），与原评估器同 checkpoint、同成功定义/分母比较，记录不匹配根因及差异，不设靠改成功判据的“宽松通过” |

如果墙钟周期达不到期望值，报告 missed deadline 和限幅处理，而不是将仿真 FPS 冒充硬实时部署。异常输入/超时的 `no-command` 或预先定义的安全降级行为须有 mock 测试；此仿真安全逻辑不是经过真机安全认证的机制。RTX 3070 8GB 上的新 GPU 工作必须独立做资源预检，不自动继承 G3C-R4/R5 的 `user-authorized-no-gpu-guard` 例外；开始前确认没有其他正在运行的 GPU 任务。

**G4-I 最终 PASS** 需要：I0/I1/I2 证据、可重复命令、raw traces、时延分布、限幅统计、失败分类、成功率对照、退出状态，以及 [`ENGINEERING_LEARNING_GATE.md`](ENGINEERING_LEARNING_GATE.md) 中用户独立解释的学习验收。任何缺失项报告 `PARTIAL/FAIL`，不得凭可视化视频替代。

## 3. G4-S：真正的跨物理引擎 Sim2Sim

**术语约束**：在同一个 Isaac Sim/PhysX 中，把 RL-Games `play.py` 换成 standalone script、ROS2 node 或另一个 Python 进程，仍属于独立推理，**不能称为完成跨引擎 Sim2Sim**。G4-S 必须将同一训练后策略（先不微调）运行在**不同的物理引擎**。建议先评估 MuJoCo 可行性，再固定目标版本和资产；选择理由、替代方案或不可行结论都须归档。

### G4-S0：可行性与资产/接口映射（先 CPU，不启动大工程）

- 检查 Franka、夹具、peg/hole 的可复现模型来源、授权/许可、joint name/order、joint limits、初始姿态、坐标系、惯量、接触几何、摩擦及重力。
- 按已有 I0 契约列出 **逐字段**可观测量、单位、controller 与动作尺度；审计 PhysX vs 目标引擎的 contact solver、time step/substeps、驱动模式和耦合关系。
- 识别观测依赖于训练环境中的不可转移内部状态时，明确说明这些假设；禁止偷偷替换成目标引擎不可获取的数据后宣称完全相同的部署条件。
- 输出 mapping table 和 `GO / CONDITIONAL / NO-GO` 决策，附可复现的最小测试。若不能忠实建立 6D action → 末端/关节命令接口或 peg/hole 接触模型，**STOP**，不得用新物理引擎的不同任务代替原任务。

### G4-S1：目标仿真器一致性测试

- 无策略条件下先验证机器人关节顺序、正反运动学、坐标变换、单位、末端 pose、默认闭环 controller、静止保持、动作符号与饱和边界。
- 用冻结的 deterministic observations/actions 或简单已知参考命令比较两引擎响应；量化位姿误差、轨迹差、接触事件与 reset 行为。
- 建立等价的 **官方几何成功判据映射**，另外记录接触质量指标。不能把几何接近/碰撞直接当作完整装配成功。
- 记录 mismatch 与调参边界；物理参数对齐和策略微调严格分开，任何改变必须可追踪。

### G4-S2：零微调跨引擎推理评估

- 冻结 checkpoint 与 deterministic policy、LSTM/normalization、训练奖励不变；跨引擎仅使用经审阅的映射与控制器实现。
- 固定测试集，统计 ever/terminal、first-success、控制频率、P95/P99 延迟、动作饱和、碰撞/接触质量及失败轨迹；所有分母、seed、例外单独标记。
- 对照 G4-I 的 PhysX 结果描述 simulation gap；不能声称与不同随机流的 episode 逐位可比，不能用目标引擎失配证明 PPO 本身失效。
- 若性能下降，先分类动力学、摩擦/接触、控制器、动作尺度和观测映射问题；**本阶段不自动改 reward、重训、微调或启动真机**。

**G4-S PASS**：确实运行了第二物理引擎、接口映射与诊断证据完整、结果按预注册分母公开，包含零成功时的真实结论。**PASS 表示技术转移流程可信，不预设成功率必须达到 90%**。若跨引擎还没真实运行，只能 `PARTIAL/NO-GO`，严禁用同引擎演示冒充。

## 4. GearMesh / NutThread：放在 G4-S 审阅之后

完成 G4-S 后单独讨论是否继续 Factory 的两个官方 contact-rich task。拟定顺序为 **GearMesh → NutThread**；复用 G2/G3 的审计、训练监督、成功率独立评估、G4-I 的性能记录与失败轨迹框架。每项的首次长训练都须另行批准和预注册预算，记录从头训练 seed、完整 checkpoint、视频与任务特定成功定义。NutThread 额外审计官方单向 yaw 动作缓存变换、螺距阈值与接触/几何质量；不能假定 PegInsert 的动作语义和时间尺度直接通用。

## 5. 提交规范与审阅交付

- 建议各阶段文档/证据使用 `docs/REVIEW_G4_I{0,1,2}.md`、`docs/REVIEW_G4_S{0,1,2}.md` 和 `evidence/g4_i/`、`evidence/g4_s/` 的独立路径；每次新 run 唯一目录，先保存原始证据和 manifest 再做图表。
- 记录实际命令、入口/环境 SHA、checkpoint SHA、真实退出码、VRAM/时钟口径、固定 seed、成功分母、失败轨迹与改动清单。小文件提交 Git；大 checkpoint/视频/原始日志外置并保留 hash/清单，不覆盖历史。
- 不修改 Isaac Lab 原始 Factory physics、奖励、网络、成功判据或既有 checkpoint；允许非侵入式推理 instrumentation，标记 `[ENGINEERING]`，与科学基线分开。
- 如遇权限、GPU/图形环境或科学完整性阻塞，应明确 `PARTIAL/NO-GO` 并停止，不能绕过 GPU guard、权限或失败保护。
- **Codex 的代码/测试 PASS 与用户的理解 PASS 分开**；每次汇报后按 [`ENGINEERING_LEARNING_GATE.md`](ENGINEERING_LEARNING_GATE.md) 提出 2–3 个针对实测证据的追问。用户能自己解释输入、时序、控制器、约束和失败，才算达到学习目的。
