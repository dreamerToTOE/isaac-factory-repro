# Panda Sim2Sim：冻结 PPO 策略迁移后的控制振荡与脱抓

> **状态：INVESTIGATING — ROOT_CAUSE_PENDING / SOLUTION_PENDING / VALIDATION_PENDING**
>
> 首次记录：2026-10-26；仓库：`dreamerToTOE/isaac-factory-repro`。
>
> 本文是**正在推进的真实工程排障案例**，不是已解决的成功案例。下一阶段拟做 G4-S2A-R1 限定范围排障；**当前 G4-S2B NO-GO**，未运行后续完整评估，也未修复或微调 PPO。

## 1. 背景：为什么要做 Sim2Sim

使用 Isaac Lab v2.1.1 / Isaac Sim 4.5.0 (PhysX) 的 **Factory Panda PegInsert** 任务训练官方 PPO。冻结的 G2 epoch-200 策略在源 PhysX 环境的独立确定性评估中，原 Factory **几何判据** ever/terminal 成功率均为 **280/288 = 97.22%**。

研究目标是将**同一份 PPO checkpoint** 零微调部署到 **MuJoCo 3.5.0** 的 Panda PegInsert 中，判断策略是否仍能完成真实的接触插入。本项目采用明确标记的 **ADAPTED ZERO-SHOT SIM2SIM**：目标环境使用固定 MuJoCo Menagerie Panda 和独立建模的 peg/hole，不能声称 PhysX 与 MuJoCo 的资产、惯量、接触物理完全一致。

冻结 PPO checkpoint SHA-256：`f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`。MuJoCo 目标生成 XML SHA-256：`c60c33ee11eb4806294aee259f97d987308b23e83da62f2219cbc9950fb195c9`。

对应工程验收：[S1B 环境报告](../REVIEW_G4_S1B.md)（环境和接口 PASS，不是策略成功）、[S2A PPO 闭环报告](../REVIEW_G4_S2A.md)（PARTIAL）。

## 2. 问题：策略可运行，但早期物理失稳（OBSERVED）

S2A 使用 seed **1001** 完成首个 **149 次 PPO 决策 / 1192 次 MuJoCo 物理步**，原冻结策略、19D 观测、6D 动作、EMA、LSTM、RMS 与跨进程推理桥均通过审计；初始化抓握的末 10 个闭指步双侧接触有效。机器人进入策略控制后，很快发生脱抓和振荡。

| 观察事件 | 数值与时刻（仿真时间，排除 0.25s 预夹持） | 直接证据 |
| --- | --- | --- |
| 脱抓开始 | **第 3 物理步 / 0.025s**：peg-hand 滑移超过 1mm | [独立轨迹复核](../../evidence/g4_s/s2a/independent_trace_review.json) |
| 双侧夹持丢失 | **第 4 物理步 / 0.0333s**：右 pad 正接触力消失 | [S2A 报告](../REVIEW_G4_S2A.md) |
| 首次夹指越限 | **第 61 物理步 / 0.5083s** | [独立轨迹复核](../../evidence/g4_s/s2a/independent_trace_review.json) |
| 首次机械臂关节越限 | **第 76 物理步 / 0.6333s** | [独立轨迹复核](../../evidence/g4_s/s2a/independent_trace_review.json) |
| 手部与支撑体严重接触 | **第 685 物理步 / 约 5.708s**：最大瞬态全接触穿透约16.9174mm | [独立物理复核](../../evidence/g4_s/s2a/independent_physics_review.json) |

整回合最大机械臂关节越限 **0.321998 rad**；最大夹指越限 **0.873067mm**；最大关节速度 **63.6524 rad/s**；最大 peg-hand 平移滑移 **7.38673m**；arm motor 达到饱和的样本比例约 **45.9132%**。第一策略动作之后的 J7 请求力矩一度约 **+58.96 Nm**、**−107.76 Nm**，积分阶段电机力矩却限制在 **±12 Nm**。

没有发生 peg/hole 接触，也没有几何、物理插入或接触质量成功。唯一完成的回合存在明显关节越限和脱抓，**不计作通过技术门禁的有效回合**：技术有效回合数为 **0**；种子 1002、1003 因预设门禁被停止，不能从单次失败估计迁移成功概率。

保留 [S2A 机器验收门禁](../../evidence/g4_s/s2a/completion_gate.json)、[原始回合摘要](../../evidence/g4_s/s2a/run_20261026T140338/summary.json)、[归档清单](../../evidence/g4_s/s2a/run_20261026T140338/collection.json) 及 [失败起点图](../../evidence/g4_s/s2a/failure_onset.png)。

## 3. 为什么这是有价值的工程故障

- **运行成功不等于迁移成功**：PPO actor 正确加载并连续控制 149 步，不能代替机械臂稳定性与接触插入成功。
- **早于插孔的失效**：初始几毫秒内先出现动力学响应、振荡和脱抓，因此第一优先不是孔几何优化，而是控制、执行器和夹持接触。
- **执行器命令与实际响应不同**：`requested`、软件限幅、MuJoCo `actuator_force` 和广义力分别保存；源 PhysX 的实际施加力矩仍 **UNKNOWN**。
- **跨引擎不应要求逐位物理一致**，但观测含义、动作控制语义以及明确的任务有效性检查必须成立；物理差异需要可量化、可验证。

## 4. 竞争假说（HYPOTHESIS；均未获得因果验证）

| 假说 | 现有支持 | 尚需反证/实验 |
| --- | --- | --- |
| Menagerie Panda 与原 Factory 惯量/转动惯量差异 | 原 Factory 还有运行时各 body 加 `0.01I` 的行为，MuJoCo 目前未获等效认证 | 匹配同一初态、相同请求力矩下的加速度与质量矩阵，隔离 inertial-only 差异 |
| 关节执行器限幅与控制律互相作用 | 首几个物理步腕部请求大幅超过 12Nm，实际执行限幅；大量时间处于饱和 | 比较限幅命令与反馈、固定输入反事实；不能把 PhysX 配置限幅当实际施加力矩 |
| 任务空间 PD / nullspace / 每子步离散控制导致不稳定 | 高频速度反转，且会每个 1/120s 子步更新目标和 J/M | 固定目标、拆分 task/nullspace torque、控制变量短步响应 |
| 指尖 PD、接触软约束与重力/夹持适配失配 | 第4步失去正 pad 接触；S1B 明确 finger PD 只是 target-engine adaptation | 无机械臂动作时的夹持对照、固定参考动作下接触/滑移对照 |
| 初始 reset 分布、坐标/四元数或 FD 误差 | 当前只有一个随机初态；不同引擎物理响应不同 | 原始状态和记录中的 19D/6D 接口对照；不能把“测试通过”提升为全时序严格等价 |

[独立接口审阅](../../evidence/g4_s/s2a/independent_trace_review.json) 显示 RMS/推理请求响应匹配、EMA 原始 FP32 最大差约 `9.54e-8`，所有 1192 个 target 的最大差约 `2.42e-7`；这削弱部分纯映射错误的解释，但**不能证明**全量 Jacobian、质量矩阵、接触模型或闭环控制精确一致。

## 5. 当前排障协议（G4-S2A-R1：尚未执行）

拟进行**归档状态可视化**，优先展示初始夹持、物理步1–4、首次滑移、首次双侧接触丢失，再使用**严格受限的无策略对照**测试惯量、力矩限制、固定 target 与夹指/接触差异。每个变体最多 16 物理步，最多 12 个事先登记的变体；不重新调用 PPO、不额外跑完整策略回合。

GUI 使用真正打开的 MuJoCo 桌面 viewer；若没有图形会话，标记 `GUI_BLOCKED`，不伪称 GUI 已打开。可视化归档状态与重新积分生成的动态结果必须分别标注，画面和数值证据必须互相引用。

## 6. 根因与修复状态

**ROOT_CAUSE_PENDING**：控制器/惯量/执行器/夹持/观测多因素的贡献尚未通过隔离实验确定。

**SOLUTION_PENDING**：没有取得可作为正式配置的经验证修复。任何参数改动只可作为 `DIAGNOSTIC_ONLY` 隔离分支，不能事后抹去冻结基线，也不能以“更好看”的 GUI 画面选择最优配置。

**VALIDATION_PENDING**：没有修复后的完整 PPO 任务回合，因此也没有恢复成功率可报告。

## 7. 后续必须提供的修复前后对照

| 指标 | 冻结 S2A 基线（seed1001） | 修复后 | 证据要求 |
| --- | ---: | --- | --- |
| 首次 >1mm 滑移 | physics step 3 | PENDING | 相同可复现初态/物理步 |
| 首次失去双指接触 | physics step 4 | PENDING | 两侧 pad 力和接触分类 |
| 最大关节速度 | 63.6524rad/s | PENDING | 同一时间窗口、另报完整 episode |
| 最大 arm 越限 | 0.321998rad | PENDING | 关节范围与时序 |
| arm motor 饱和比例 | 45.9132% | PENDING | 同一执行器口径 |
| 官方几何/物理插入 | 均未发生 | PENDING | 任务统计分母、pre-reset 和独立物理判据 |

允许改善目标 MuJoCo 适配器或动力学，但**不得改变 PPO 权重**；每一次修订应记录是修 bug、模型适配还是诊断参数，并单独注册可复现评估协议。

## 8. 面试复盘（当前只可使用已验证事实）

**约30秒，当前进行时版本：**

> 我做过 Isaac Factory PegInsert 策略的跨仿真器迁移：训练阶段使用 PhysX 和 Panda，原 PPO 在源环境按几何成功条件完成了 280/288 回合。将冻结权重接到 MuJoCo Panda 后，19维观测、6维动作及 LSTM 推理接口可以工作，但第一次完整闭环实验出现了控制振荡、力矩饱和、脱抓和关节越限。我们把每一个物理步的执行器实际力矩、关节状态和夹指接触完整记录下来，定位到策略开始后仅0.025秒就发生明显滑移，随后双侧接触丢失；目前正通过无策略短步对照试验隔离惯量、执行器与接触动力学的影响，还未宣称完成成功迁移。

**推荐技术追问：** 为什么 `tau=JᵀF` 还需要研究动态一致 nullspace？为什么请求力矩和实际输出不同？为什么高成功率原 PhysX PPO 到 MuJoCo 会失稳？如何区分观测问题、动力学问题与抓握接触问题？为什么不直接微调 PPO？怎样证明修复在新 seed 上也有效？

待根因和修复真正验收后，再把此段改成已完成的“问题→定位→修复→对比验证”。

## 9. 源数据与更新纪律

- 源环境： [G2 训练结果](../RESULTS_G2.md)、[I2 独立推理](../REVIEW_G4_I2.md)。
- MuJoCo 构建：[S1A NO-GO（严格资产重建）](../REVIEW_G4_S1A.md)、[S1B 工程 PASS](../REVIEW_G4_S1B.md)。
- 首次失败：[S2A 报告](../REVIEW_G4_S2A.md)、[正式门禁](../../evidence/g4_s/s2a/completion_gate.json)。
- 原始轨迹校验：[collection.json](../../evidence/g4_s/s2a/run_20261026T140338/collection.json)；不覆盖失败数据。

每轮 G4-S2A-R1/R2 等实验在新证据目录记录：命令、环境、入场门禁、变体哈希、可视化来源、正负结果、时序统计、结论强度。后续若确实解决，可追加 VERIFIED_FIX 与 before/after 章节；禁止直接改写历史为“最初就成功”。
