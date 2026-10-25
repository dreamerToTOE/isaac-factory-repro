# G2 实际命令、运行身份与复算说明

本页记录 2026-10-23 已执行的 G2，供审计与复现；整理时间为 2026-10-25。训练只执行一个产出数据的全新 200-epoch run，随后评估预注册的四个 checkpoint。结果见 [RESULTS_G2.md](RESULTS_G2.md)，证据索引见 [evidence/g2/README.md](../evidence/g2/README.md)。下面的历史训练/评估命令包含已存在目录，不应直接重跑。离线复算需指定一个新的输出目录。

## 路径与环境

```bash
G2_WORKSPACE=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been
G2_REPRO="$G2_WORKSPACE/outputs/isaac-factory-repro"
G2_LAB="$G2_WORKSPACE/work/upstream/IsaacLab"
G2_VENV="$G2_WORKSPACE/work/envs/factory-g01"
G2_NAME=g2_peginsert_64env_seed0_baseline_rerun
G2_ROOT="$G2_WORKSPACE/work/logs/g2/$G2_NAME"
G2_RUN="$G2_WORKSPACE/work/runs/logs/rl_games/Factory/$G2_NAME"
```

训练与评估进程的工作目录均为 `$G2_WORKSPACE/work/runs`；现有启动器默认使用 `/home/ubuntu2004/isaacsim-4.5.0`。训练环境设置 `FACTORY_VENV="$G2_VENV"`、`OMP_NUM_THREADS=4`，通过 `FACTORY_G2_AUDIT_OUTPUT` 启用只读训练 observer。主评估去掉两个训练 audit 环境变量，另设 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 加载本机产生的可信 checkpoint；此变量不表示允许加载未知来源模型。评估续接器还显式设置同一 `ISAACSIM_PATH`。没有安装或替换环境。

固定 Isaac Lab `v2.1.1` / `90b79bb2d44feb8d833f260f2bf37da3487180ba`，RL-Games 1.6.1 的审计源码对应 `f5bd8f2a0022220a1109200a3da47d2e96cb9aa1`。最终逐文件复核见 [final_source_verification.json](../evidence/g2/final_source_verification.json)。

## 动态预检的实际子命令

四项预检均由 `run_bounded.py` 启动并成功退出。下列代码块从各自 `result.json.command` 逐参数转写，绝对路径保留原样；对应 result 同时保留 cwd、退出码、NVML 和时长。外层 preflight timeout 未在 result 中记录，本页不补造该值。首次已知 checkpoint 预检发生在 evaluator 元数据修复之前，其输出和源码快照仍保留。

### gpu

[原始 result](../evidence/g2/preflight/gpu/result.json)，returncode=`0`。

```bash
bash /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/scripts/run_sim.sh /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/scripts/gpu_probe.py
```

### factory64

[原始 result](../evidence/g2/preflight/factory64/result.json)，returncode=`0`。

```bash
bash /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/scripts/run_sim.sh /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/scripts/factory_probe.py --headless --device cuda:0 --num_envs 64 --steps 16
```

### observer64

[原始 result](../evidence/g2/preflight/observer64/result.json)，returncode=`0`。

```bash
bash /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/scripts/run_sim.sh /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/upstream/IsaacLab/scripts/reinforcement_learning/rl_games/train.py --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 --num_envs 64 --seed 0 --max_iterations 1 agent.params.config.full_experiment_name=g2_preflight_observer64 agent.params.config.save_frequency=50
```

### known_checkpoint

[原始 result](../evidence/g2/preflight/known_checkpoint/result.json)，returncode=`0`。

```bash
bash /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/scripts/run_sim.sh /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro/scripts/evaluate_checkpoint.py --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 --checkpoint /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/runs/logs/rl_games/Factory/g01_peg4_pinned/nn/last_Factory_ep_3_rew__35.64205_.pth --num_envs 4 --episodes 1 --seed 1001 --max_steps 200 --snapshot-limit 12 --progress-interval 10 --output /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g2_preflight/known_checkpoint/metrics.json
```

## 全新训练的实际命令

原 pipeline 和 monitor 的 argv 保存在 [orphaned_pipeline_status.json](../evidence/g2/orphaned_pipeline_status.json)。下面只将其已记录绝对路径替换为上面的等值变量：

```bash
cd "$G2_WORKSPACE/work/runs"
FACTORY_VENV="$G2_VENV" OMP_NUM_THREADS=4 \
FACTORY_G2_AUDIT_OUTPUT="$G2_ROOT/train/training_audit.json" \
/usr/bin/python3 "$G2_REPRO/scripts/run_bounded.py" \
  --output "$G2_ROOT/train" --timeout 43200 --min-free-mib 1024 -- \
  bash "$G2_REPRO/scripts/run_sim.sh" \
  "$G2_LAB/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs 64 --seed 0 --max_iterations 200 \
  agent.params.config.full_experiment_name=g2_peginsert_64env_seed0_baseline_rerun \
  agent.params.config.save_frequency=50
```

没有 `--checkpoint`，没有恢复训练；`--max_iterations 200` 由官方入口映射到 `max_epochs=200`。每个 epoch 为 64 × 128 = 8192 transitions，总量 1,638,400。`run_bounded.py` 默认 0.1 秒采样，连续 3 个样本 free < 1024 MiB 才终止自己的进程组。

`[ENGINEERING]` 覆盖包括唯一输出名、保存频率 50、官方入口生成的 train_dir，以及外部 observer/monitor。官方 PPO YAML 与 resolved agent dump 逐键比较只有 `full_experiment_name`、`save_frequency`、`train_dir` 三处差异；seed=0 和 max_epochs=200 本来就是该固定 YAML 的值。环境数在官方构造后成为 64，训练 observer 实测 obs/state/action 为 19/43/6。dump 在 runtime 更新前生成，保留的 `num_actors=128` 不能替代运行时证据。resolved env.yaml 与 D001 的 64-env env.yaml 字节一致。

## 启动与监督事故的保留方式

第一次名称为 `g2_peginsert_64env_seed0_baseline` 的 detached 启动，没有创建 train 输出、没有观察到 epoch 或 checkpoint，监督进程消失原因未知。其 manifest 在改名时被覆盖，只保留 [initial_launch_status.json](../evidence/g2/initial_launch_status.json) 中的命令和 hash；不能声称原 manifest 完整保存。

随后实际训练 run 于 UTC 12:12:02.991694 启动。原 pipeline/monitor 消失，但训练 PID 230594、birth ticks 4135139 持续运行。UTC 12:14:59.838041，独立 `g2-reattached-guard` systemd 用户服务附着同一存活进程；`g2-posttraining` 服务运行 `complete_g2_after_orphan.py`，等待这个 PID/birth 结束、验证 200 个连续 epoch、最终 checkpoint 与 `MAX EPOCHS NUM!`，然后顺序执行评估。

这两个服务没有启动或恢复训练。约 122.75 秒的 GPU 样本和显存保护缺口、原训练子进程退出码无法补回；退出码保留为 `null`。[launch_monitor_incident.json](../evidence/g2/launch_monitor_incident.json)、[monitor_segments.json](../evidence/g2/monitor_segments.json)、[continuation_status.json](../evidence/g2/continuation_status.json) 分别记录事故、时间轴、完成验证。服务的完整 `systemd-run` shell 包装 argv 没有归档，本页不将推测命令称为原始执行证据。

[reattach_guard_incident.py.txt](../evidence/g2/source_snapshots/reattach_guard_incident.py.txt) 是此次事故的定点脚本原样快照，含历史 PID 和绝对路径，仅作为非可执行事故档案保存；不得照抄重执行。`complete_g2_after_orphan.py` 也用于既存进程续接监督，不是启动新实验的入口。

## 评估修订与实际命令

[原 manifest](../evidence/g2/g2_manifest.json) 冻结了初始 evaluator SHA。首次 G2 主评估之前，发现 partial 快照缺少 checkpoint hash、pre-reset capture 身份和推理元数据，严格分析器会拒绝这些已完成 episode。修复只补元数据，未改策略、成功判据、seed、episode 预算或环境参数；原脚本保存在 [evaluator_before_partial_metadata_fix.py](../evidence/g2/source_snapshots/evaluator_before_partial_metadata_fix.py)。

[evaluation_amendment.json](../evidence/g2/evaluation_amendment.json) 于 UTC 12:19:50 冻结修复后的 evaluator SHA `415dff9b59ece8a21e6bbe0a6c9fff08e2ea594cb8dc2ef8fe73e299c94af707`；所有主评估随后才开始。续接器每次评估前核对修订 hash。原 manifest 未篡改；使用当前 evaluator 时必须同时读取 amendment。

主评估固定 policy mean、RMS eval 模式、`torch.inference_mode`，每个 done 重置相应 LSTM hidden state；成功值在 auto-reset 前抓取。三个评估 seeds 为 1001/1002/1003，每个 seed 32 env；50/100/150 epoch 各 1 episode/env，200 epoch 各 3 episode/env。这些是评估 seeds，训练仍只有 seed 0。未完成的 episode 不纳入分母；完整与 partial 快照去重。详见 [SUCCESS_METRICS.md](SUCCESS_METRICS.md) 和 [G2_ANALYSIS_PROTOCOL.md](G2_ANALYSIS_PROTOCOL.md)。

下面循环是 12 条已执行命令的等值展开，不用于再次写入既有目录。每条原始 argv 和退出状态都在 `evidence/g2/evaluation/epoch*/seed*/result.json`；全部 12 条 returncode=0。

```bash
cd "$G2_WORKSPACE/work/runs"
unset FACTORY_AUDIT_OUTPUT FACTORY_G2_AUDIT_OUTPUT
export FACTORY_VENV="$G2_VENV" OMP_NUM_THREADS=4
export ISAACSIM_PATH=/home/ubuntu2004/isaacsim-4.5.0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
for epoch in 50 100 150 200; do
  case "$epoch" in
    50) checkpoint=last_Factory_ep_50_rew_301.46692.pth; episodes=1 ;;
    100) checkpoint=last_Factory_ep_100_rew_346.65335.pth; episodes=1 ;;
    150) checkpoint=last_Factory_ep_150_rew_350.71515.pth; episodes=1 ;;
    200) checkpoint=last_Factory_ep_200_rew__350.4961_.pth; episodes=3 ;;
  esac
  for seed in 1001 1002 1003; do
    epoch_dir=$(printf 'epoch%03d' "$epoch")
    eval_dir="$G2_ROOT/evaluation/$epoch_dir/seed$seed"
    /usr/bin/python3 "$G2_REPRO/scripts/run_bounded.py" \
      --output "$eval_dir" --timeout 1800 --min-free-mib 1024 -- \
      bash "$G2_REPRO/scripts/run_sim.sh" "$G2_REPRO/scripts/evaluate_checkpoint.py" \
      --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
      --checkpoint "$G2_RUN/nn/$checkpoint" --num_envs 32 --episodes "$episodes" \
      --seed "$seed" --max_steps "$((160 * episodes))" \
      --snapshot-limit 12 --progress-interval 10 --output "$eval_dir/metrics.json"
  done
done
```

checkpoint 由预注册 epoch 选择，不依据独立评估挑选。200 epoch 使用无条件 final-save 文件；同 epoch 的 periodic-save 也保留，但文件 hash 不同，未宣称字节相同。`Factory.pth` 是训练 reward 的 best alias，未用作本轮主要结果。所有 6 个文件与 SHA 见 [checkpoint_inventory.json](../evidence/g2/checkpoint_inventory.json)，内部 epoch/frame、Adam 和参数差异验证见 [selected_checkpoints.json](../evidence/g2/selected_checkpoints.json)。

## 已执行的离线分析与安全复算

分析使用系统 Python 的 Matplotlib 和既有隔离 Python 的 TensorBoard，只读输入，不启动仿真或加载模型。交付图表来自以下参数；本次 `--output` 为 `$G2_REPRO/evidence/g2/analysis`，复算时必须换成新目录。

```bash
cd "$G2_REPRO"
python3 scripts/analyze_g2.py \
  --train-log "$G2_RUN" \
  --monitor-manifest evidence/g2/monitor_segments.json \
  --audit evidence/g2/training/training_audit.json \
  --eval-root evidence/g2/evaluation \
  --output evidence/g2/analysis \
  --manifest evidence/g2/g2_manifest.json \
  --tensorboard-python "$G2_VENV/bin/python" \
  --num-envs 64 --horizon-length 128 \
  --expected-epochs 50,100,150,200 --eval-seeds 1001,1002,1003 \
  --eval-num-envs 32 --final-episodes-per-env 3
```

`--monitor-manifest` 保留两个 elapsed 原点，缺口断线，不拼造缺失样本；wall time 来自 pipeline start → 原进程消失，原退出码仍未知。已记录 TensorBoard 点与 200-epoch audit 分开：TensorBoard 缺最后 epoch 的部分 scalar，图表不补值。`summary.json` 记录实际输入/脚本 hash 与警告，PNG/PDF 为派生产物。

主执行者另报告 CPU 回归测试结果 **42 passed in 14.87s**；本页整理者没有重复执行测试。初始预检记录的 34 tests 和 amendment 时的 35 tests 是各自当时的历史结果，不能替换为最终数量。

## 下一次独立复现的条件

下一次实验应先复查 GPU/磁盘、固定版本和 preflight，创建独立 run 名、空输出目录和新的 manifest，冻结当前训练脚本与修订 evaluator hash。原 manifest 的 evaluator hash 指向修复前快照，原 `run_g2_pipeline.py` 在评估阶段会按原 hash 检查；不能仅改 run_name 后复用旧 manifest。新的监督实现需在启动前确认其生命周期与 fail-closed 行为，保留 child wait/exit 和整段 telemetry，再重新冻结 hash。

新 run 若中断，记录 incomplete；恢复的训练不能计作全新 uninterrupted 200 epochs。此处没有启动新训练、延长预算或改为 128 env。完整原始日志/模型留在仓库外，路径、文件大小和流式 SHA256 清单见 [raw_artifact_inventory.json](../evidence/g2/raw_artifact_inventory.json)；Git 提交只包含小型证据、代码、配置和图表。
