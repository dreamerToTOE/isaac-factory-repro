# G5-GEAR-G1 训练提案（待用户批准，不是运行记录）

本文件只提出下一阶段。G0 不执行 PPO、不加载或更新 checkpoint。保留官方 GearMesh reward、控制器、PhysX 配置、RL-Games 网络、PPO 算法及超参数；在启动前重新审阅实际显存、source hash、资产和完整 G0 状态。不得将参考动作 smoke 的成功率解释为训练效果。

## 样本预算与配置

官方 pinned YAML：actor 与 central critic 均为两层 1024 LSTM 和 512/128/64 ELU MLP，horizon/sequence length 128，minibatch 512、mini-epochs 4，学习率 1e-4（adaptive，KL 0.008），gamma 0.995、GAE tau 0.95、clip 0.2、mixed precision；其余逐项以原 YAML 为准。默认 128 环境 × 128 × 200 epochs = 3,276,800 transitions。机器显存 8 GiB，建议 G1 先尝试 32 环境、seed 0、从头初始化，200 epochs = 819,200 transitions；同样本预算需 800 epochs。64 环境若后续经资源准入和受监护的训练证明安全，200 epochs = 1,638,400；同预算需 400 epochs。不同环境数改变 rollout batch、每 epoch 更新量与自适应调度，因此匹配 transitions 不等于严格重现官方训练轨迹。不要把 200 epochs 当成 GearMesh 已证明的收敛量；按验证曲线和预算事先决定扩展，而非事后挑最佳报告。

先采用 32 而非直接假设 64 可行，保留资源保护；准入不足则停止并保存证据，不更改驱动、系统或显存保护。每 25 epochs 保存 checkpoint 并记录 SHA256、文件长度、训练 epoch/frames；完成后还保存最终权重和优化器可用状态，使用现有 TensorBoard finalizer、guard、持久 supervisor 和独占锁。故障时保留 exit、日志、guard、checkpoint 原件；恢复前验证身份与框架 checkpoint 的已知状态缺失，不能宣称逐位续训。

## 独立评价

训练 seed 0；另预注册不重叠的评估 seeds，例如 1001/1002/1003 和每 seed 的完整 episode 数（建议每 seed 96）。在 25、50、100、150、200 epochs 以及预注册的最终 checkpoint 运行独立、确定性策略评价。捕获自动 reset 之前的官方几何谓词，分别报告 episode ever-success、terminal-success、first-success step、完成 episode 数、超时和双重 done 标志。成功率附分子/分母及置信区间；seed 分层，避免只展示最佳 seed 或将不完整 episode 计入分母。几何结果不等于机械啮合或接触质量。

分别记录非有限状态、关节上下限、速度异常、reset/IK 耗时和失败轨迹；若接触力、穿透及真实齿轮啮合数据不可获得，明确标为未测，不能把控制器输出 wrench 当作接触力。对于失败类别（未对中、姿态/高度异常、失夹、接触卡滞、超时、reset 失败），用可观察证据分类并允许未知。之后按资源决定多个独立训练 seed。

## 时间与准入

G0 一环境 reference-action 实测仅能建立带同步检查的仿真吞吐，无法推导 32/64 环境加 LSTM PPO 的可靠墙钟时间。G1 经批准后，用首个受监护训练窗口实际的 transitions/s、评估和保存耗时推算 `剩余 transitions / 实测总吞吐 + 评估与保存时间`，定期修正并记录条件；不依据单环境结果作多环境线性外推。训练中每 0.1 秒采样全卡显存，保留 1024 MiB 空闲下限、连续三次触发停止和 supervisor/guard 心跳保护，需区分全卡桌面占用和训练进程，采样峰值不保证捕获瞬时峰值。每次启动注册命令、源和配置 hash、seed、预算、超时与唯一输出路径。

G1 的最终启动命令必须在用户审阅后根据当前仓库、GPU 余量、训练目录和正式评估方案重新注册；不得直接把 G0 的脚本当成训练入口。继续使用 `scripts/g3a_launch_smoke.py`、`run_sim.sh` 和 pinned 官方 `scripts/reinforcement_learning/rl_games/train.py`。G0 后停止，技术学习和面试考察阶段仍为 PENDING。
