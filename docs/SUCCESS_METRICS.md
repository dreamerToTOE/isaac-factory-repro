# D001 R2 成功指标与 checkpoint 评估

本文件定义仅用于评估的指标，不改变官方 Factory 成功条件、奖励或 episode 终止。评估器在每个环境步的官方 `_get_curr_successes` 调用中采集成功 mask；调用发生在 `DirectRLEnv.step` 的 reset 之前。本轮已完成两个seed、共8个完整episode的GPU评估；结果仅用于评估器链路验证。

## 官方顺序和观测点

固定上游提交为 `90b79bb2d44feb8d833f260f2bf37da3487180ba`。

1. 父环境执行 decimation 个物理步并将 `episode_length_buf += 1`。
2. `FactoryEnv._get_dones` 只返回 timeout；成功不会改变 done。
3. `FactoryEnv._get_rewards` 调用 `_get_curr_successes(success_threshold=cfg_task.success_threshold, ...)`，随后计算 reward。
4. 父环境依据 done 调用 `_reset_idx`，其中会随机化和重置 episode counters。
5. reset 后才生成并返回 observation。

因此评估器用临时实例方法包装官方 `_get_curr_successes`，先调用原方法、记录其原始返回值和调用时的 `episode_length_buf`，最后将原返回对象原样交还官方reward代码；没有复制或放宽判据。只选择官方success threshold和任务原本的rotation flag，排除第二次engagement调用。若每步没有恰好一个成功mask、终止时捕获比例与官方 `extras["successes"]` 不一致、或返回的counter不是reset后的0，评估器报错退出。代码证据：[FactoryEnv `_get_dones`/success/reward](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L425-L484)、[DirectRLEnv step order](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/envs/direct_rl_env.py#L341-L392)。

### 三个官方统计量不能混用

| 官方变量 | v2.1.1 实际含义 | 本独立评估的处理 |
| --- | --- | --- |
| `extras["successes"]` | `_get_rewards` 检测到任一 reset 时，取**当前这一时刻** `curr_successes` 的真值数量除以总环境数；不是 episode 内曾成功比例。在当前同步 timeout baseline 下对应末步成功比例。非 reset 步的 extras 值可能沿用上次写入。 | 只在 done 步做一致性核对，独立重新累计 terminal 指标。 |
| `ep_succeeded` | 每步将当前成功环境置1，因而记住本 episode 的曾成功；在 reset 后的下一次 `_pre_physics_step` 根据 `reset_buf` 调 `_reset_buffers` 清0。它没有被用于 `extras["successes"]` 的分子。 | 独立 tracker 保存自己的 ever 状态，每个 completed episode 后清零。 |
| `ep_success_times` / `extras["success_times"]` | 初始化为0，首次成功时写 episode step。`_reset_buffers` **只清 ep_succeeded，未清 ep_success_times**。后续无成功的 episode 可保留旧首次成功时间；聚合又按所有非零 entry 计算，未筛掉旧 episode，不能无条件解释为本批 completed episode 的成功首达均值。 | 完全不读取这两个字段作为独立真值；首次成功由本 episode 的捕获序列计算，失败输出 null。 |

固定源码证据：[初始化 148–149](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L148-L149)、[reset buffer / pre-physics 292–300](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L292-L300)、[官方日志和首次成功 468–482](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L468-L482)。这些是对现有实现的解释；未修补或重定义官方统计。

## 指标定义

对每个 completed episode 输出：

- `first_success_step`：官方成功 mask 首次为真时的 1-based episode policy step；没有成功则 `null`。
- `ever_success`：episode 内任一步官方成功为真，包括随后又失败/离开成功状态的 transient success。
- `terminal_success`：官方成功 mask 在 done 的**同一 pre-reset step**为真。
- `terminal_kind`：`timeout` 或 `terminated`。官方 Factory 当前只由时间到期 done，因此实际通常为 `timeout`。
- `steps`：done 前的官方 episode policy step。自动 reset 的额外 IK/夹持物理步不计入该值。

聚合同时输出 `first_success_rate`、`ever_success_rate`、`terminal_success_rate` 和首次成功步数均值。`ever_success_rate` 和 `terminal_success_rate` 有意分开：一个回答“曾经达到过官方几何条件”，另一个回答“在 episode 最后一个 pre-reset 步仍达到条件”。这两个指标都不替换官方 `extras["successes"]`；后者是官方 reset 时刻日志所使用的当前成功 mask 统计。

两个成功率的分母均为**完整结束的 environment episodes**；一次4环境同步结束计4个episode。`first_success_rate` 与 ever-success rate 数值等价，用来核对记录完整性。首达步数均值仅对 ever-success episodes 取条件均值，失败不写0、不写episode长度；无成功样本时均值为null。未完成/截断的评估不产生完整请求的成功率。`first_success_time_seconds = first_success_step * step_dt` 是策略仿真时间，不包含 reset IK/夹持步骤和真实墙钟；逐episode JSON保留原始step和seed。

由于官方成功不会提前结束 episode，出现 `ever_success=true, terminal_success=false` 是合法的 transient success，不是评估器错误。评估器也不添加保持时间、接触力、扭矩、掉落或 NaN 条件。

## 可复现命令

使用项目的 pinned 运行入口；`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 只在评估本项目自己生成的、已核对 SHA256 的 checkpoint 时需要。评估器默认确定性动作（网络均值），加 `--stochastic` 才采样策略动作。**官方player默认stochastic**；本评估的deterministic是显式评估设置，未改训练配置，也不保证GPU物理逐位确定。

本次D001约定：4 env、seeds0/1、每seed每个环境完整1 episode，合计 **8 completed episodes**。这是指标/加载链路验证，不是统计可靠的装配性能估计。各seed以独立进程启动，独立输出seed、checkpoint SHA256、实际运行包路径、capture调用数、逐episode首达时刻、ever/terminal与终止时官方日志一致性；不把两个seed的frame连续拼接为一次episode。

```bash
cd /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro
export FACTORY_WORKSPACE=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been
export FACTORY_VENV="$FACTORY_WORKSPACE/work/envs/factory-g01"
export FACTORY_CHECKPOINT="$FACTORY_WORKSPACE/work/runs/logs/rl_games/Factory/g01_peg4_pinned/nn/last_Factory_ep_3_rew__35.64205_.pth"
mkdir -p evidence/d001
for FACTORY_EVAL_SEED in 0 1; do
  python3 scripts/run_bounded.py \
    --output "$FACTORY_WORKSPACE/work/logs/d001_eval_seed$FACTORY_EVAL_SEED" --timeout 600 --min-free-mib 1024 -- \
    env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 \
    bash scripts/run_sim.sh scripts/evaluate_checkpoint.py \
    --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
    --checkpoint "$FACTORY_CHECKPOINT" --num_envs 4 --episodes 1 --seed "$FACTORY_EVAL_SEED" \
    --max_steps 160 --output "evidence/d001/metrics_seed$FACTORY_EVAL_SEED.json"
done
```

`--episodes` 是每个环境的完整episode数，因此请求总数=`episodes*num_envs`。脚本退出码非零或未完成请求数时，不报告成功率。输出 JSON 包含运行参数、每个 episode 的明细、聚合指标以及 `official_success_capture="pre_reset_factory_get_curr_successes"`。评估器需要 GPU/Isaac Sim，代码准备和下列CPU测试没有启动GPU。输出路径已存在时拒绝覆盖，失败也保留外层console日志。

## 测试边界

`tests/test_success_metrics.py` 是不依赖 Isaac Sim 的10项纯合成测试，覆盖 transient success、terminal timeout success、episode 状态清除、多环境独立完成、terminated/timeout区分，以及带自动reset的受控假环境。后者验证pre-reset末步成功即使被返回的post-reset状态覆盖为false，仍正确记录terminal success；同时检查原predicate返回对象不变、engagement被排除、关键词参数兼容、捕获缺失时失败与方法恢复。

另有 `tests/test_evaluation_runtime.py` 的5项CPU测试，直接核对已安装 RL-Games `BasePlayer` 的 observation解包、batch和`states` API，并验证 nested observation（包含critic states）的finite检查、inference tensor的LSTM hidden/cell state在done后安全清零。评估器在 `torch.inference_mode()` 内清零Player.states，避免对inference tensor在上下文外原地赋值报错。它不改训练agent的`rnn_states`。

```bash
PYTHONPATH=. "$FACTORY_VENV/bin/python" -m pytest -q tests/test_success_metrics.py tests/test_evaluation_runtime.py
```

CPU测试结果见 [metrics_tests.txt](../evidence/d001/metrics_tests.txt)。另存已知真值的4环境序列在 `evidence/d001/metrics_synthetic.json`：ever=3/4、terminal=2/4、transient-only=1/4，first steps为2、3、null、1。它只测试指标会计和采集时序，不测试官方几何判据，不是机器人实验结果。准备期间一次pytest命令使用错误相对venv路径，未找到解释器；改用正确绝对路径后通过，未安装额外依赖。

## D001真实checkpoint链路结果

[seed0](../evidence/d001/metrics_seed0.json)及[seed1](../evidence/d001/metrics_seed1.json)均exit0，每seed4env×1完整episode，共8episodes。每个episode149步；ever-success=0/8、terminal-success=0/8，first-success steps及条件均值均null。两次都捕获到reset前counter149、返回counter0，并与官方extras末步日志一致。3-epoch smoke模型无成功样本，不作训练收敛或性能推断。15项CPU测试提供成功/瞬时成功的已知真值正例。
