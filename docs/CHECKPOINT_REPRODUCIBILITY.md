# D001：Checkpoint 保存、恢复与可复现性

审阅日期：2026-10-23。对象为实际安装的 RL-Games **1.6.1**、Isaac Lab **v2.1.1**。未修改官方环境、PPO 或序列化实现。本页的“恢复”区分训练入口 `A2CAgent.restore` 与推理入口 `PpoPlayerContinuous.restore`。

## 1. 结论

**现有 checkpoint 可加载模型用于推理，也能继续执行 PPO；它不能复原一个中断前的完整训练过程。** 主要原因是独立 central critic Adam、调度所用的 `last_lr`、RNG、仿真状态和 LSTM 运行时 hidden/cell state 未完整保存恢复。权重可加载、参数继续变化，不等于可重复续训。

未来首次正式基线必须用固定 seed（建议 `0`）**从头初始化**：不传 `--checkpoint`，保留 `load_checkpoint=False`、`load_path=""`，不得把 G1/D001 smoke checkpoint 作为正式基线起点。固定 seed 是实验身份的一部分；不能由此承诺 PhysX/CUDA 的逐位确定性。

## 2. 源码身份与证据

- RL-Games 官方 `v1.6.1` 对应 Git SHA：`f5bd8f2a0022220a1109200a3da47d2e96cb9aa1`（`git ls-remote` 验证）。本地已安装源码与该 SHA 的逐文件 SHA256 比对见 [source identity](../evidence/d001/checkpoint_state_audit_source_identity.json)。
- 本地精确行号、内容摘录和文件 SHA256：[checkpoint_state_audit_source.txt](../evidence/d001/checkpoint_state_audit_source.txt)。以下链接固定到同一 RL-Games SHA。
- 官方训练保存入口：[a2c_continuous.py:66–72][agent]；实际 save/restore 字段：[a2c_common.py:600–668][full]。
- CPU 受控恢复实测：[checkpoint_state_audit.json](../evidence/d001/checkpoint_state_audit.json)。没有 GPU 训练或推理成功率含义。

## 3. 精确状态表

| 状态 | 官方文件保存内容 | 训练恢复行为 / 限制 |
| --- | --- | --- |
| Actor 网络（含 LSTM 学习权重） | `model = self.model.state_dict()` | `model.load_state_dict`；本地 checkpoint 的 28 个 tensor/buffer 完全相等恢复。包括模型中的 value head；实际中心价值估计另由 central critic 提供。 |
| Actor 输入 RMS | 嵌于 `model.running_mean_std.{running_mean,running_var,count}` | 随 model 恢复。不是依赖顶层额外 RMS key。 |
| Actor value RMS | 嵌于 `model.value_mean_std.*` | 随 model 恢复。启用独立 critic 时，训练代码的 `self.value_mean_std` 指向 central critic 的 value RMS；不能只检查 actor 的这组缓冲。 |
| Central critic 网络与输入/value RMS | `assymetric_vf_nets = central_value_net.state_dict()`，包括 `model.*` 子模块 | `central_value_net.load_state_dict`；24 个 tensor/buffer 完全相等恢复。[CentralValueTrain 是 nn.Module][central]；RMS 采用 [registered buffers][rms]。 |
| 主 Adam | `optimizer.state_dict()` | 恢复参数组与 Adam `step / exp_avg / exp_avg_sq`；实测 22 个有状态参数完整相等恢复。 |
| **Central critic 独立 Adam** | **没有保存** | `CentralValueTrain.optimizer` 是独立 Python optimizer，不属于 nn.Module 参数/缓冲。只加载 critic 的 module state 不会恢复它。新建 agent 的 critic Adam state 为空，丢失一阶/二阶动量和 step。 |
| AMP GradScaler | `scaler`（启用 mixed precision 时） | `scaler.load_state_dict`，包括 scale/growth tracker 等。CPU 测试中使用 CPU GradScaler 验证序列化标量相等；这不代表恢复了 CUDA 数值轨迹。 |
| Actor epoch / frame | 顶层 `epoch`, `frame` | `set_epoch=True` 时恢复；正常 `restore` 默认如此。 |
| Critic epoch / frame | 未保存（普通属性） | 新建为 0，恢复后仍为 0；不随顶层 actor counters 恢复。 |
| Actor optimizer LR | 存在主 Adam 的 `param_groups[*].lr` 中 | 加载 optimizer 后该 LR 恢复；**`self.last_lr` 没有恢复**，仍是配置初值 `1e-4`。[adaptive scheduler][schedule] 下一次按该变量和新 KL 更新，随后覆盖 optimizer LR。 |
| Scheduler / entropy coefficient | 无 scheduler state / `last_lr` / `entropy_coef` key | scheduler 根据配置重建；本 baseline entropy coef 为固定 0。不能把恢复 optimizer LR 称为恢复完整调度。 |
| Critic LR / scheduler | critic Adam 和 `self.lr` 未保存 | 官方 YAML 写 `adaptive`，但 [CentralValueTrain:42–52][central] 只特判 `linear`，其余实际使用 **IdentityScheduler**。在本版本这一配置下，critic 的 adaptive 配置不会启动自适应调度。保留原行为。 |
| Best reward / reward统计 | 保存 `last_mean_rewards` | restore 暂时恢复；进入 [ContinuousA2CBase.train:1301–1307][train] 后又设为 `-100500`。episode 计分队列、当前累计奖励/长度没有恢复。 |
| 环境状态 | 顶层 `env_state` key，但本 checkpoint 值为 `None` | [Isaac Lab RlGamesGpuEnv][wrapper] 未覆盖继承的 [IVecEnv get/set_env_state][ivecenv]，实际 getter 返回 None、setter 无操作；train 入口还会 reset。没有物体/机器人/接触/episode 状态快照。 |
| Python / NumPy / Torch CPU/CUDA RNG | 未保存 | 由新进程/配置 seed 重建；不是中断时随机流位置。 |
| LSTM hidden/cell、rollout与 minibatch顺序 | 未保存运行时状态 | 网络 LSTM 权重有保存，但 actor 的 `rnn_states` 在 [init_tensors:468–475][rnn] 重新初始化；critic 也构建默认 RNN state。没有 observation、dones、经验缓存或数据排列。 |

另外，`get_stats_weights(model_stats=False)` 默认生成的 `central_val_stats` 在实测文件中是空字典；不能据此说 normalization 未保存。RMS 本身已注册到 model state，随两套模型恢复。`set_stats_weights` 的独立顶层 `reward_mean_std` 分支条件和 critic 的空 setter 并不是本 checkpoint 恢复 RMS 所走的路径。[stats 的实际代码][full]、[model 内的 RMS 创建][models]。

## 4. CPU 独立验证

工具 [audit_checkpoint_state.py](../scripts/audit_checkpoint_state.py) 不启动 Sim，不执行优化。只接受显式给定且 hash 匹配的**本项目本地产生** checkpoint，使用 `weights_only=False` 加载。它按官方 YAML 构建原始网络（actor 19维、critic 43维、action 6维，2×1024 LSTM），通过实际 `A2CAgent.set_full_state_weights` 做 strict restore；另外调用实际 `get_full_state_weights` 验证 omission。

受检文件 SHA256：`6dd7f429bef5aede4c9d55fa1da54858f47569b182a044c5b634fb94952de442`，epoch **3**、frame **1536**。

- 10 项断言全部通过；actor/critic 模型和 RMS、主 Adam、AMP scaler、epoch/frame 相等恢复。
- 保存的主 Adam LR 是 **0.0003375**，恢复后 `self.last_lr` 仍是 **0.0001**。
- 新建 critic Adam 恢复后有 **0** 个 state，critic epoch/frame 仍为 **0/0**。
- 受控测试先给 critic Adam 放入非空 state、设置 `last_lr=2e-5`；实际 save 方法仍不包含它们，restore 后 critic Adam 仍为空，`last_lr` 保持配置初值。
- checkpoint 的 `env_state=null`，`central_val_stats={}`；actor/critic 的 RMS 缓冲确实存在。

在工作区根目录执行：

```bash
env OMP_NUM_THREADS=2 work/envs/factory-g01/bin/python \
  outputs/isaac-factory-repro/scripts/audit_checkpoint_state.py \
  work/runs/logs/rl_games/Factory/g01_peg4_pinned/nn/last_Factory_ep_3_rew__35.64205_.pth \
  --official-config work/upstream/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml \
  --expected-sha256 6dd7f429bef5aede4c9d55fa1da54858f47569b182a044c5b634fb94952de442 \
  --output outputs/isaac-factory-repro/evidence/d001/checkpoint_state_audit.json
```

Torch ≥2.6 的默认 `weights_only=True` 与 RL-Games 1.6.1 的旧式 `torch.load` 入口不兼容时，只对已核验 hash 的自有文件、单次运行使用 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。这是可信旧文件的加载兼容措施，不补齐训练状态，也不修改 PPO。大 checkpoint 保留在本地 `work/runs`，未加入 Git。

## 5. 实验协议

1. **推理兼容**：加载 actor 权重和对应输入 RMS，eval 模式，明确 deterministic/stochastic action、seed、episode 数；每 episode 清零对应 LSTM state。独立成功指标见 D001 evaluator 文档。
2. **可执行续训**：加载后仍能优化；报告所用 checkpoint hash 和上述缺失状态。不能标记为“与连续训练等价”。
3. **完全可复现续训**：当前官方保存路径不满足；后续若设计更完整状态快照，属于独立工程变更，需要另行审查，不能悄悄改本轮官方基线。
4. **首次正式 baseline**：全新 fixed-seed run，记录上游/依赖/配置/环境数/硬件身份。G1/D001 checkpoint 只用于 smoke、加载验证和评估工具验证。GPU 的 bitwise determinism 仍没有得到证明。

[agent]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/algos_torch/a2c_continuous.py#L66-L72
[full]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/common/a2c_common.py#L600-L668
[central]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/algos_torch/central_value.py#L12-L83
[rms]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/algos_torch/running_mean_std.py#L27-L31
[models]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/algos_torch/models.py#L37-L49
[schedule]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/common/a2c_common.py#L1213-L1227
[train]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/common/a2c_common.py#L1301-L1307
[wrapper]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_rl/isaaclab_rl/rl_games.py#L316-L355
[ivecenv]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/common/ivecenv.py#L28-L36
[rnn]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/common/a2c_common.py#L468-L475
