# G4-S1B 独立协议与源码复核

本次由与实现分离的 reviewer 只读读取用户完整 S1B 任务、冻结 Factory 环境/控制/数学/DirectRLEnv 源码、S0/S1A 审计以及固定 Panda 许可。没有载入 checkpoint、执行 actor、运行策略或自行推进物理。源码 SHA、精确检查列表和实现审阅结论见 `protocol_review.json`。

新任务明确授权 **ADAPTED ZERO-SHOT SIM2SIM**，解除的是“所有原资产和求解器必须精确复刻”的后续门禁；不会更改 S1A 对精确重建的 NO-GO。S1B 不应声称策略已迁移成功；目标是可运行且已验证的无策略工程接口。

关键独立核查点：

- 每个物理步从最新 TCP 重建位姿目标，八子步共享 EMA；原顺序 clip→EMA→scale。Euler helper 返回 modulo 2π，upright roll 常数为 3.14159。
- FD 保留立即前一物理状态及 1/120s；世界 dq 左乘并保留 sign(0)=0 分支。raw wxyz 不能作半球规范化。
- 原控制 J 是两夹指 body origin 的世界 Jacobian 平均，TCP pose 是单独末端帧；M 是完整广义矩阵的臂七轴主块。task torque 无 Lambda 前乘，nullspace 必须保留动态一致结构和 raw qvel。
- Reset action cache 不是全零：位置除以 .05；yaw 为原 unrot(-π)、包角和 270°映射；reset FD 为零与 LSTM reset 是不同状态。
- 原 success 是 FP32 严格 XY<2.5mm、有符号 z<1mm；不检查接触、深度下界或姿态。独立物理插入诊断必须另外报告。149步超时、pre-reset评估和 reset辅助时钟不得混淆。
- Panda 明确为固定 Menagerie Panda；Apache-2.0 来自模型目录许可和 README。原 Factory USD/衍生资产许可并未因此放行。

MuJoCo 官方 3.5 文档已浏览：

- [Jacobians / mass matrix API](https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html)：body origin 与 CoM/site Jacobian 分别存在，fullM 展开到 nv×nv。
- [MJCF reference](https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html)：gravcomp 与 actuatorgravcomp 的力路径不同；必须区分 passive 和 actuator 数量。

当前实现审阅尚待代码完成；本检查列表本身不是测试通过证明。最终报告须以真实执行日志、退出码和物理证据为准。
