# G2 离线分析协议

本协议与 `scripts/analyze_g2.py` 只读取证据，使用 CPU 生成统计、PNG 与 PDF。不启动 Isaac Sim、不加载 checkpoint、不修改训练或评估配置。分析图本身不构成训练完成或复现成功的证据，须与实际退出状态、checkpoint 清单和独立评估共同审阅。

## 输入与命令

```bash
WORKSPACE=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been
REPRO="$WORKSPACE/outputs/isaac-factory-repro"
VENV="$WORKSPACE/work/envs/factory-g01"

# 将下面三个输入路径替换为本次 manifest 记录的实际目录。
# --output 必须是新目录，防止覆盖已交付的分析。
python3 "$REPRO/scripts/analyze_g2.py" \
  --train-log /absolute/path/to/official/experiment \
  --monitor /absolute/path/to/run_bounded/output \
  --eval-root /absolute/path/to/milestone/evaluations \
  --output "$REPRO/evidence/g2/analysis" \
  --tensorboard-python "$VENV/bin/python" \
  --num-envs 64 --horizon-length 128 \
  --expected-epochs 50,100,150,200 \
  --eval-seeds 1001,1002,1003 \
  --eval-num-envs 32 --final-episodes-per-env 3
```

可选 `--audit /absolute/path/training_audit.json` 接受 D001 格式的 `epochs` 列表，或每行一个完整 epoch JSON 的 `.jsonl`。优先使用 `epoch`、`environment_transitions`、`official_total_seconds`、`env_step_seconds`、`rollout_seconds`、`optimization_seconds`、`total_fps`、`env_step_fps`、`rollout_fps` 字段。没有 epoch audit 时，脚本读取官方 TensorBoard 的 timing 与 `info/epochs`；只有显式提供环境数、horizon 后才推导 transition 数，并在输出标记派生来源。

`--manifest` 可复制预注册 manifest 并保存其 SHA256；脚本不假定 manifest 的 schema，不用未核验的嵌套键悄悄补齐训练参数。输入 event 文件、NVML CSV、选定评估 JSON、脚本及可选 audit/manifest 均保留路径、大小和 SHA256。大原始日志及 checkpoint 留在复现仓库之外。

现有环境中，系统 Python 提供 Matplotlib，隔离学习 Python 提供 TensorBoard。`--tensorboard-python` 仅用既有学习 Python 的 `EventAccumulator` 导出 scalar，绘图在系统 Python 完成；无需安装或替换学习依赖。统计和图表均不导入 Torch 或 Isaac Sim。

### 监测中断后的分段输入

若原始 monitor 消失、训练子进程持续运行、随后另行附着 guard，使用 `--monitor-manifest /absolute/path/monitor_segments.json` 替代 `--monitor`，并显式提供 `--audit`。不得拼接两个局部 elapsed 时间轴，也不得把新 guard 的退出码当作原训练子进程退出码。manifest 示例仅说明格式，不代表实际 G2 测量值：

```json
{
  "schema_version": 1,
  "wall_evidence": {
    "start_utc": "2026-10-23T12:00:00+00:00",
    "start_source": "original pipeline status.json started_utc; pipeline start, not exact child spawn",
    "start_uncertainty_seconds": 0.1,
    "end_utc": "2026-10-23T14:00:00+00:00",
    "end_source": "guard observed matching PID/start-ticks disappear",
    "end_uncertainty_seconds": 0.1,
    "returncode": null,
    "returncode_source": "original parent disappeared; child exit code unavailable"
  },
  "segments": [
    {
      "name": "original",
      "directory": "/absolute/path/train",
      "elapsed_origin_utc": "2026-10-23T12:00:00+00:00",
      "elapsed_origin_source": "recorded monitor origin, or explicitly identified approximation",
      "elapsed_origin_uncertainty_seconds": 0.1
    },
    {
      "name": "reattached",
      "directory": "/absolute/path/reattached_guard",
      "elapsed_origin_utc": "2026-10-23T12:03:00+00:00",
      "elapsed_origin_source": "reattached_guard/identity.json attached_utc",
      "elapsed_origin_uncertainty_seconds": 0.001
    }
  ],
  "notes": ["Training was not restarted; missing NVML samples cannot be recovered."]
}
```

目录也可相对于 manifest 所在目录；时间戳必须包含时区。原始 CSV/result 保持不变，每段分别保留输入哈希、采样间隔和极值。绘图将时间转换为相对 `wall_evidence.start_utc` 的 elapsed，并在不同采样段之间断开，不跨越空白区连线。段内超过 `max(0.5 秒, 3 × 实际中位采样间隔)` 的间隔也单独视为观测空白。`sampled_span_coverage_seconds/fraction` 是各采样跨度的并集，包含普通采样间隔，并非连续观测证明；重叠段不会重复计入覆盖时间。

`coverage_gaps` 记录启动到首个样本、段间/段内长空白和末样本到进程消失的区间与时间不确定性。全局最大 used、最小 free 都只能称为“已观测样本极值”，不能声称覆盖空白区或保证全程未触发阈值。`wall_duration_seconds` 独立由所供 start/end 时间求差，明确标注 pipeline-start → disappearance 的时间来源与误差；原 monitor 缺失的 wait/exit 证据不因此恢复。`end_utc` 可省略，此时 wall duration 与端到端 FPS 为 `null`。无法回收子进程退出码时必须保留 `returncode: null`，报告不得写正常退出码 0；即使有完整 epoch、最终 checkpoint 和 `MAX EPOCHS NUM!`，也应将训练预算完成证据与 OS 退出状态分别陈述。

## 训练与效率口径

- `training_reward`：直接绘制官方 `rewards/iter`、`shaped_rewards/iter`。其中 shaped_rewards 还含 timeout value bootstrap，并非原始环境奖励。它们是 RL-Games 最近完成 episode 的 reward meter（默认 `games_to_track=100`），不是“该 epoch 新完成 episodes 的独立均值”，各 epoch 点的样本可重叠。下图单独绘制官方训练 `successes/iter`；Factory 在 episode 结束时更新全环境当前成功比例，observer 按 epoch 记录最近的 scalar，它可能从最近一次结束时刻保留到下一个 epoch，不是 episodic ever-success，也不是每 epoch 完整 episodes 的汇总。
- `training_losses`：直接绘制 actor loss、value loss、entropy、KL、官方自适应 learning rate 和 bounds loss，不平滑、不填补缺失数据。保留原始 TensorBoard step；central-value 与 actor 的 step 偏移不被悄悄移动。
- 此单智能体、64-env、horizon 128 配置每 epoch 为 8192 transitions。actor loss/timing 的 step 是 epoch 开始前累计 frame（epoch 200 为 1630208），central-value 的 step 是加完该 epoch 后的 frame（epoch 200 为 1638400）；`/iter` reward 的 step 才是 PPO epoch。完成预算由 audit 的 200 × 8192 = 1638400 求得，不能把最后一个 actor scalar 的横坐标当作完成 transitions。本次 RL-Games 默认 `use_experimental_cv=True`，actor 内 `losses/c_loss` 是实际 value-head loss，另有独立 central critic 的 `losses/cval_loss`；两者分别报告。另设 `use_experimental_cv=False` 的其它运行才可能令 actor c_loss 为零占位，不可套用于本次。实际落盘两类 loss 均到 epoch199；epoch200 标量缺失，不能把理论横坐标当作已有数据。
- `throughput_memory`：分别标注环境 stepping、rollout/inference、PPO total 的分母。PPO total FPS 的合计口径是总 transitions 除以官方 rollout/inference 与 optimization 秒数之和，不平均各 epoch FPS。
- 稳态效率从 epoch 11 起计算，以去除最初若干 epoch 的启动影响。若不足 11 epoch，稳态统计为 `null`，不会把短测伪装成长时稳态结果。
- 进程 wall time 来自外部 monitor，包含初始化、checkpoint I/O 和退出；与官方 PPO total seconds 分开。端到端 FPS 使用此 wall time 为分母。
- NVML 统计使用所有有效原始 sample 的最大 used、最小 free 及实际采样间隔。绘图使用至多约 1500 个时间窗的 min/max envelope；这只压缩绘图，不改变统计峰值。全卡 used 包含桌面，reserved 另计；10 Hz 不能保证捕获短于 100 ms 的峰值。
- scalar 有限只说明已记录的标量有限，不能单独证明每个梯度或 AMP 尝试有限。梯度、overflow/skipped update 必须另看 observer 的原始证据。

## 独立评估口径

主要协议：确定性策略（policy mean）、seeds `1001/1002/1003`、每个 seed 32 个环境；epoch 50/100/150 每环境 1 个完整 episode（各 96），epoch 200 每环境 3 个完整 episode（288）。这些 seeds 是评估随机化 seeds，不能代替多个独立训练 seeds。

评估 JSON 应为 `evaluate_checkpoint.py` 的输出。checkpoint epoch 可由 `checkpoint_epoch`/`epoch` 显式字段给出，或由目录名 `epoch050` / checkpoint basename 的 `_ep_50_` 识别。建议使用每 checkpoint、每 seed 单独的子目录，例如 `epoch050/seed1001/metrics.json`，并把 G2.1 已知旧 checkpoint 的加载预检放在另一目录。

| 指标 | 定义 |
|---|---|
| ever-success | 完整 episode 中任一 pre-reset policy step 满足官方成功判据的 episode 数 / 实际完成 episode 数。 |
| terminal-success | 完整 episode 最后一个 pre-reset policy step 满足官方判据的 episode 数 / 实际完成 episode 数。 |
| transient-only | 曾成功而终点未成功的完整 episode 数。 |
| first-success step | 首次满足官方判据的 1-based policy step，只在 ever-success 的完整 episode 中计算。 |
| first-success seconds | 首次成功 step × policy dt；为模拟时间，不是训练计算 wall time。 |
| Wilson 95% interval | 对实际完成 episode 的成功/总数计算双侧 Wilson score interval。描述 episode 计数不确定性，不量化训练 seed 的方差；同 seed、向量环境共享 policy，不能假设其完全独立。 |

每 seed 与跨预注册 seeds 的 aggregate 分开；aggregate 由原始 episode 记录重算，不对三个 seed rate 做简单平均。额外未注册 seed 保留单独行并发出提示，不进入主要 aggregate。首次成功时间提供成功 episode 的均值、中位数、IQR、范围和实际条件样本数；没有成功时为 `null`，不写 0 或 episode timeout。

最终 JSON 与 `.partial.json` progress snapshot 常含相同 episodes。脚本按 checkpoint epoch、seed 选择完整输出优先，其次已完成 episode 较多的快照，绝不相加。发现冲突 episode、同 epoch/seed 不同 checkpoint hash、重复 episode ID、episode 计数与 aggregate 不一致时报告 error。无完成 episode 的初始化 partial 可以读取，其 denominator 为 0、rate/CI 为 `null`。初始化/出错/超时评估保留实际完成 denominator，并且不通过完整协议检查。

缺失 checkpoint/seed 标为 `missing`，不记作失败 episode。`checkpoint_success` 仅绘制实际存在的 aggregate，以 `n` 标注 denominator，并对缺失里程碑标 `missing`；`first_success_conditional` 的 aggregate median/IQR 和每 seed median 都仅用实际成功 episode。

## 输出与验证

输出目录包含：

- `summary.json`：训练/监测/评估摘要、输入哈希、可见错误和限制。
- `tensorboard_scalars.json`：全部 scalar 与原始 step/wall_time，非有限值显式标注。
- `evaluation_summary.json`、`evaluation_per_seed.csv`：原始 episode 重算的 per-seed/aggregate 指标、实际分母和置信区间。
- 五组 PNG/PDF：`training_reward`、`training_losses`、`throughput_memory`、`checkpoint_success`、`first_success_conditional`。

`python3 -m pytest -q tests/test_analyze_g2.py` 覆盖瞬时/终点成功区分、首次成功条件分布、零成功与缺失评估的不同语义、partial 分母、重复快照不重复计数、未注册 seed 排除、冲突与损坏证据拒绝、JSONL 截断和重复 epoch。D001 的真实 64-env scalar/NVML 与 8 个完成评估 episode 已用于兼容性检查；测试产物仅保存在 `work/`，不作为 G2 实验结果。

分段监测回归另外覆盖：不同 elapsed 原点、未观测区间、段内长间隔、重叠覆盖不重复计数、新 guard 成功退出不覆盖未知子进程退出码、缺少 end 时不造 wall duration、PPO 与端到端 FPS 使用不同分母，以及不带时区/倒序/负不确定性等无效时间证据拒绝。
