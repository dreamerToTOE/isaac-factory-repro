# 工程理解与面试验收：不只“跑通项目”

设立日期：2026-10-26。服务于机械臂运动控制 + Robot Learning/RL 部署岗位。与 [`ROADMAP_POLICY_INFERENCE_SIM2SIM.md`](ROADMAP_POLICY_INFERENCE_SIM2SIM.md) 配套。

## 学习验收协议

**项目工程 PASS ≠ 用户学习 PASS。** Codex 可以实现代码、运行测试、整理日志，但不能代替项目负责人形成可解释的工程判断。后续每个子阶段完成后，ChatGPT 应先基于真实日志向用户提出 **2–3 个具体问题**，逐题听用户用自己的话回答，再从可核验源码/配置/数据中给出纠正和必要的知识补充，最后留下可供面试复习的简短记录。

提问规则：

1. 优先问“为什么这么设计、错了会发生什么、数据如何支持判断”，不只问术语翻译。每题尽量引用实际代码路径、配置值或实验测量；未知数据就请先查，不捏造。
2. 让用户先回答；不立即公布标准答案或直接让 Codex 代答。若回答部分正确，应追问一个反例或诊断实验，再解释关键差异。
3. 对新手难点给出直观例子和推导；对已经掌握的点尽量简洁，并升级到系统级取舍。
4. 每阶段记录“证据是否完整”和“用户是否理解”两个**独立状态**。理解状态可为 `PENDING / EXPLAINED / NEEDS_REVIEW`，不能因测试全部通过自动标为 `EXPLAINED`。
5. 用户此前已经强调工程优先、自己掌握关键原理；Codex 负责实现/测试/记录。先理解已有项目，不为了准备面试临时启动无关大型新项目。

## 六个贯穿后续工程的核心问题

这些问题在 G4-I 先提问，在 G4-S 和新任务中用真实证据再次追问；此处刻意不直接附答案，避免把背诵当成理解。

### Q1 — 策略输出到底控制了什么？

G2 的 PPO actor 输出六维动作：各维对应什么物理/控制意义、尺度、坐标系和更新周期？它通过哪些环节到达低层控制器，最终作用到仿真机器人的是位置、速度、力矩还是其它目标？请从 `source audit → resolved config → actual action path` 给出证据。

### Q2 — 网络速度能否等于控制频率？

推理延迟为若干毫秒时，是否就能断言控制频率为 `1/latency`？区分 physics timestep、substeps、policy decimation、`env.step`、GPU inference、wall-time command cadence。如何定义和测量 P95/P99 与 deadline miss？

### Q3 — 推理耗时超过控制周期怎么办？

若某次推理晚于约定的命令截止时间，保持上次动作、丢弃本次动作、缓冲异步动作或执行安全降级各有什么风险？为什么仿真快慢与真实机器人的实时性不是同一概念？应设计什么 mock/故障注入测试验证该决定？

### Q4 — 动作限幅为什么不只是 clip？

网络原始输出、归一化后/物理尺度动作、控制器收到的命令和真正的执行器限位有何区别？如何按维度统计饱和率？为什么长期饱和可能意味着 observation/action 尺度、奖励、控制器或者任务可行性的问题？

### Q5 — 相同 checkpoint 为什么换推理入口会失败？

针对 LSTM 策略，如何排查输入特征顺序、running mean/std、隐藏状态 reset、deterministic/stochastic mode、动作后处理和 episode reset 先后顺序？怎样用同一 observation/action trace 的 parity 测试定位问题？

### Q6 — 训练成功为何不能等于真实装配可靠？

为什么 PegInsert 官方几何成功判据不等价于力控安全、接触质量和持续保持？如果跨引擎成功率下降，如何区分策略问题与接触模型、摩擦、坐标系、控制器、执行频率或传感配置不一致？什么轨迹证据能支持而不只是猜测？

## 按阶段逐步升级的追问

| 节点 | 用户应独立解释的重点 | 需要引用的证据 |
| --- | --- | --- |
| G4-I0 | observation 19 维为何这样排列？归一化状态来自哪里？六维输出的真实含义是什么？ | 官方源码、resolved configs、checkpoint/normalizer |
| G4-I1 | 一个 episode reset 后 LSTM hidden/cell 如何处理？同样的 observation 为什么可能产生不同 action？ | 逐步 action parity、reset trace |
| G4-I2 | policy 频率、simulation FPS 和 wall-time 控制频率为何不同？饱和动作和 deadline misses 会怎样影响闭环？ | latency P50/P95/P99、周期分布、action saturation CSV、失败轨迹 |
| G4-S0 | 什么状态信息在训练器里可用却在目标仿真中不可得？接触/坐标映射哪些环节最容易出错？ | 逐字段 mapping 和资产/solver 对照 |
| G4-S1/S2 | 两个引擎出现位姿或成功率差异时，如何设计不依赖 PPO 的对照实验定位接触动力学与 controller？ | 无策略响应、固定动作轨迹、两引擎诊断统计 |
| GearMesh | 插孔与齿轮啮合的失败模式、奖励与几何判据何处不同？ | 两任务源码、成功分母与失败示例 |
| NutThread | 旋合任务的方向性、角度变换与接触质量如何影响控制器和成功率解释？ | 官方 NutThread 动作路径、旋合失败轨迹 |

## 每次验收记录模板（提交结果时填写，不得预填 PASS）

```text
Task / date / commit:
Engineering evidence gate: PASS | PARTIAL | FAIL
Checkpoint / model source SHA:
Actual measurement or source link:
Question 1 (with evidence):
User's explanation:
Review: EXPLAINED | NEEDS_REVIEW | PENDING
Question 2 (with evidence):
User's explanation:
Review: EXPLAINED | NEEDS_REVIEW | PENDING
Question 3 (optional):
User's explanation:
Review: EXPLAINED | NEEDS_REVIEW | PENDING
Outstanding conceptual gaps and next targeted practice:
```

完成项目的验收结论必须诚实记录；这里的学习门禁用于学习反馈，**不通过也不改写真实实验的科学 PASS/FAIL**。
