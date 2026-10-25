# G2 训练前检查

静态检查时间：2026-10-23 17:23:36（Asia/Seoul）。范围为首个 `Isaac-Factory-PegInsert-Direct-v0`、64 环境、seed=0、200 epochs 连续全新 PPO run。用户已在 G2 请求中授权此范围；G0/G1 和 D001 文档中的“等待审阅、不启动 200 epochs”是历史阶段限制，由本轮明确授权接续。

完成后补记（2026-10-25）：动态GPU/64环境/已知checkpoint检查全部通过，实际放行fresh200epochs，随后全部12次主要评估完成。最终CPU测试42项通过，上游SHA与干净状态再次验证。正式run的monitor中途消失，约123秒guard缺口、原训练退出码未知，故下方“全程guard生效”是启动前要求，不能当作事后全部满足的事实。实际完成、补充证据及限制以[RESULTS_G2](RESULTS_G2.md)为准。

本页静态部分仅执行只读检查，没有启动仿真或训练、安装包、修改上游源码。完整命令、原始输出、文件 SHA256 和断言见 [static_audit.json](../evidence/g2/preflight/static_audit.json)。GPU PhysX、任务注册和已知 checkpoint 完整 episode 的本轮动态检查须另看本目录中的独立运行结果；NVML 可读或包 metadata 一致本身不是 GPU 仿真通过证据。

## 1. 代码和安装身份

| 项目 | 本轮实测 |
| --- | --- |
| 复现仓库 HEAD | `0efe7fc5f5ab96b15008e8195fa936755b182a84`，与用户指定 D001 SHA 一致 |
| 既有历史 | `0efe7fc` ← `a77d215` ← `c54da28`；未 reset、覆盖或重建仓库 |
| origin | fetch=`https://github.com/dreamerToTOE/isaac-factory-repro.git`，push=`git@github.com:dreamerToTOE/isaac-factory-repro.git` |
| 上游 Isaac Lab | `90b79bb2d44feb8d833f260f2bf37da3487180ba`，精确 tag=`v2.1.1`，工作树干净 |
| Sim 安装 | `/home/ubuntu2004/isaacsim-4.5.0`；VERSION=`4.5.0-rc.36+release.19112.f59b3005.gl` |
| 独立 Python | `work/envs/factory-g01/bin/python`，3.10.12 |
| 已安装关键包 metadata | torch 2.7.0+cu128、torchvision 0.22.0+cu128、numpy 1.26.4、gymnasium 1.2.0、RL-Games 1.6.1 |
| Lab editable metadata | isaaclab 0.41.3、isaaclab_tasks 0.10.36、isaaclab_rl 0.1.8；项目发布身份以 Git SHA 为准 |
| RL-Games 源码完整性 | D001 审计的 9 个已安装文件 SHA256 全部仍等于官方 `f5bd8f2a0022220a1109200a3da47d2e96cb9aa1` 对应摘要 |

复现仓库工作树在本轮检查时不是干净状态：G2 agents 正在编写 evaluator、统计与训练观察脚本及其测试、协议文档。具体列表保存在 JSON；这些是本轮预期工程改动，未混入上游。此次检查记录了上游训练入口、PPO YAML、Factory 环境/控制/配置及父类文件 SHA256，结束时应再次验证上游干净。

保持现有 `run_sim.sh` / `python_entry.py` 的启动顺序与包断言。训练 stdout 必须出现正确的 `PINNED_PACKAGES` 与 `PACKAGES_AFTER_APP_LAUNCH`；只有前置 metadata 相同不足以排除 Kit archive 遮蔽。没有重装或替换工作环境。既有最小依赖缺口（pin-pink、dex-retargeting、moviepy）维持 D001 记录的范围；本轮 headless 训练不依赖这些未使用路径。

## 2. 运行资源快照

| 项目 | 17:23:36 静态快照 |
| --- | --- |
| OS / kernel | Ubuntu 22.04.5 LTS / `6.8.0-138-generic` |
| GPU / driver | NVIDIA GeForce RTX 3070，8192 MiB / 580.178.04 |
| 全卡显存 | used 1836 MiB；free 5983 MiB（NVML 驱动保留内存另计） |
| 温度 / 功耗 / utilization | 52 °C / 38.35 W / 14%，仅瞬时快照 |
| compute processes | 仅 `/usr/libexec/gnome-remote-desktop-daemon`，157 MiB；未见其它 GPU 训练 workload |
| RAM | 50,443,857,920 bytes，总量约 47.0 GiB；available 36,868,886,528 bytes，约 34.3 GiB |
| 磁盘可用 | 408,751,038,464 bytes，约 380.7 GiB |
| swap | 2 GiB，未使用 |

本轮沿用 [run_bounded.py](../scripts/run_bounded.py) 的只读 NVML **0.1 秒采样**与 **1024 MiB free 连续 3 次**停止线。该 guard 终止自己的训练子进程组，不改用户其它进程或设备权限。时间上限必须留出初始化、保存和正常性能变化；D001 的约 2 小时是估计，不是健康 run 的硬截止条件。

## 3. 200 epochs 与全新初始化的源码确认

官方训练入口将 CLI `--num_envs 64` 赋给 `env_cfg.scene.num_envs`；把 `--seed 0` 赋给 `agent_cfg.params.seed`，再赋给环境 seed。`--max_iterations 200` 的实际含义是 **`agent_cfg.params.config.max_epochs=200`**，不是 200 个物理 step 或 200 次 minibatch 更新。[train.py:90–120][cli]

官方 YAML 初始值 `load_checkpoint=False`、`load_path=""`；本轮不传 `--checkpoint`、不传 `--sigma`。官方训练入口只在 CLI checkpoint 非空时传恢复路径给 runner；RL-Games `_restore` 只有收到非空 `checkpoint` 参数才执行 restore。因而创建新 actor/critic 和新 optimizer，从头开始。[YAML:44–45][yaml]、[train.py:203–206][run]、[torch_runner.py:18–20,111–116][runner]

一个需记录的 seed 细节：RL-Games `Runner.load_config` 用 `if self.seed:`，seed=0 不执行该分支重新播种；Isaac Lab 在 runner 创建之前已执行 `DirectRLEnv.__init__` 的 `seed is not None` 分支，因此 0 仍传给官方 Sim `torch_utils.set_seed`，后者播种 Python、NumPy、Torch CPU/CUDA 和 Warp。这一原样顺序不需要工程补丁。源码及 Sim 函数摘录和 hash 已存 JSON；默认 `torch_deterministic=False`，因此不承诺 CUDA/PhysX 逐位可复现。[父环境初始化:92–94][envseed]、[seed 方法:395–412][seed]、[Runner seed 分支:57–97][runnerseed]

官方 rollout horizon=128、seq_length=128、actor/critic minibatch=512、mini_epochs=4、网络两层 1024 LSTM + MLP[512,128,64] 及其它 PPO 参数全部维持。64×128=**8192 transitions/epoch**，可被 512 整除。200 epochs 目标为 **1,638,400 transitions**，相对官方 128 环境同 epoch 预算为 1/2。[官方完整 PPO YAML][yaml]

## 4. 里程碑保存行为与选择协议

本轮允许的唯一保存设置覆盖是 `agent.params.config.save_frequency=50`，属于 **[ENGINEERING] checkpoint 写盘频率**；保留 `save_best_after=10` 和其它优化设置。RL-Games 的周期保存位于 `game_rewards.current_size > 0` 分支：已有至少一个完整 episode 时，epoch 整除 save_frequency 即保存 `last_Factory_ep_<epoch>_rew_<...>.pth`。正常 PegInsert 在第 149 个 policy step timeout，所以训练到第 50 epoch 前应已有多个完整 episode；是否实际保存最终以 inventory、epoch 字段与文件 hash 确认。[训练保存循环:1347–1378][save]

第 200 epoch 的 max_epochs 分支独立执行最后一次保存，即使没有已完成 episode 也会保存并打印 warning（reward 为 `-inf`），随后打印 `MAX EPOCHS NUM!` 结束。第 200 epoch 可能因周期保存与最终保存的文件名格式不同而有两份内容等价的 checkpoint；应按 checkpoint 内 `epoch/frame` 和 hash 去重，不能当作两个独立训练结果。[训练终止保存:1386–1413][stop]

`Factory.pth` 是根据训练 mean reward 选出的 best alias，可能多次覆盖。本轮主要比较预指定的 **50/100/150/200 epoch**，不依据独立 evaluation 结果事后改选 checkpoint；如果某里程碑未保存，报告缺失而非用别的 checkpoint 冒充。最终比较使用 episode 数及 seed 固定的 evaluator 协议。

官方另有 `score_to_win=20000` 提前结束分支，保持不改。PegInsert 每步 reward 上界为 `1/6+1/4+1/2+1+1=35/12≈2.9167`，149 步 episode 总 reward 上界约 **434.58**，远低于 20000；这是由官方奖励函数和默认零 penalty 推导的数学上界，不是实测奖励。[reward 源码][reward]、[PegInsert 参数][task]。若其它原因提前中断，必须报告 incomplete，不能恢复拼接后宣称 uninterrupted。

## 5. 允许覆盖与动态放行清单

建议命令的训练部分如下，`G2_RUN` 为唯一目录名：

```bash
bash "$FACTORY_REPRO/scripts/run_sim.sh" \
  "$ISAACLAB_PATH/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs 64 --seed 0 --max_iterations 200 \
  agent.params.config.full_experiment_name="$G2_RUN" \
  agent.params.config.save_frequency=50
```

实际执行命令须由外部监测器原样记录。唯一名称、路径、存档频率和只观察日志的 instrumentation 标作 `[ENGINEERING]`；环境数量/seed/epoch 是用户指定的实验配置。官方 dump 在环境构造与 `num_actors` 运行时更新之前，故 `agent.yaml` 可能仍显示 `num_actors=128`、env placeholder 21/72；实际配置 manifest 还须记录构造后的 **64 actors、19/43 observations、6 actions**，避免把早期 dump 当作最终对象状态。[train.py:136–180][dump]

最终启动前应汇总以下独立证据：

- 本页静态身份与资源断言通过；上游源码不变。
- 当前 CUDA / GPU PhysX 仿真和三 Factory ID 注册通过。
- D001 成功指标单元测试通过。
- 已知本地产生 checkpoint 的 hash 匹配，独立 evaluator 加载成功并完成 episode。
- 没有其它 GPU 训练、磁盘足够；全程 1 GiB free guard 生效。
- 实际训练启动对象确认 fresh epoch/frame、空 optimizer state、原样网络与 PPO 配置。

这些静态条件未发现阻碍；本页不把尚未读取的动态结果预先标为通过。全部实际执行结果由 [RESULTS_G2](RESULTS_G2.md) 和 `evidence/g2/preflight/` 汇总。

[cli]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/scripts/reinforcement_learning/rl_games/train.py#L90-L120
[yaml]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml#L1-L118
[run]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/scripts/reinforcement_learning/rl_games/train.py#L203-L206
[runner]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/torch_runner.py#L18-L20
[runnerseed]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/torch_runner.py#L57-L97
[envseed]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/envs/direct_rl_env.py#L92-L94
[seed]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/envs/direct_rl_env.py#L395-L412
[save]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/common/a2c_common.py#L1347-L1378
[stop]: https://github.com/Denys88/rl_games/blob/f5bd8f2a0022220a1109200a3da47d2e96cb9aa1/rl_games/common/a2c_common.py#L1386-L1413
[reward]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L487-L525
[task]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L68-L132
[dump]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/scripts/reinforcement_learning/rl_games/train.py#L136-L180
