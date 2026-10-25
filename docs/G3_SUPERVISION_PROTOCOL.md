# G3A：训练监督与日志可靠性协议

本协议对应仓库的 schema 2 实现，只规定启动、等待、监控、状态持久化和日志收尾。G2 原事件、manifest、GPU 序列与 checkpoint 只读保留。G3A 使用新的 run 名和新证据目录，不启动 G3B，不启动 seed-1/seed-2，也不恢复或拼接 G2 训练。

## 1. 实现与启动顺序

| 文件 | 职责 |
| --- | --- |
| `scripts/g3a_launch_smoke.py` | 保存完整 launch 计划，依次向用户 systemd 提交 guard 与 supervisor 两个独立 unit |
| `scripts/durable_supervisor.py` | 持有训练 child 的 `wait()`，保存真实 exit/signal，检查 guard liveness，维护全局 GPU lock |
| `scripts/gpu_guard.py` | 训练前即开始真实 NVML 采样，绑定 child 身份，独立检查 supervisor，执行显存停止规则 |
| `scripts/supervised_exec.py` | 等待 pipe 中的授权字节；未收到授权或 EOF 时拒绝执行训练 |
| `scripts/supervision_common.py` | boot/PID/birth 身份、单调时间、原子 JSON、受限进程组终止 |
| `scripts/tensorboard_finalizer.py` / `scripts/tensorboard_tail.py` | 显式 writer flush/close，以及独立的末 epoch scalar 检查 |

launcher 为 unit 设置 `Restart=no`、`KillMode=control-group`、`TimeoutStopSec=10`、明确的工作目录和有限超时。systemd 持有服务，提交它的终端或 Codex 会话消失不结束服务。launcher 的 `--wait` 只等待监督结果，不是 G3A 的完整科学/日志验收。

guard 先于 supervisor 提交并采样。supervisor 必须看到真实、有限、fresh 的 NVML 样本和存活 guard 身份后，才创建一个被 pipe 阻塞的 child。child 使用 `start_new_session=True`，并获得固定 `(boot_id, pid, birth_ticks)`；guard 读取 supervisor 创建的不可变 `child_identity.json` 后，将相同身份写入 heartbeat。supervisor 只有确认 guard 已绑定这个 child，才写 `exec_release.json` 并发送 `G` 字节让 child `execvp()` 训练命令。`exec` 保留 PID/birth，关闭 pipe 或 supervisor 在授权前消失会让 child 拒绝启动。

正式训练命令仍是 pinned Isaac Lab 入口的原 argv，监督层不修改它，也不自动添加 checkpoint 参数。`run_sim.sh` 最后以 `exec` 转为隔离 Python，标准训练的 child leader 身份保持至退出。训练不会自动 restart/restore；失败后必须保留旧目录，用新的 run id 明确启动另一短测。

## 2. 身份、独占与状态文件

Linux 身份包括 `/proc/sys/kernel/random/boot_id`、PID、`/proc/<pid>/stat` field 22 的 birth tick、PGID 和观测状态。liveness 比较以 boot/PID/birth 为准，并拒绝 zombie；不能仅比较 PID 或进程名称。停止前及 SIGTERM→SIGKILL 升级前再次比较身份和 PGID，只有已捕获 child leader 所拥有的进程组可被终止。systemd 的 control-group 是本次新建 unit 的进程清理边界，不能把无关用户服务纳入测试。

默认 `/run/user/<uid>/isaac-factory-gpu0.lock` 使用非阻塞 `flock`。锁 FD 通过 gate 和 exec 传给 child，supervisor 单独死亡时仍由 child 持有，因此第二个训练启动会被拒绝。测试必须使用临时 lock 文件，不得占用或终止已有训练。相同输出目录通过 `mkdir(exist_ok=False)` 拒绝复用；不可变 JSON 通过同目录临时完整文件和 hard-link 的排他创建拒绝覆盖。

```text
<fresh-run>/
  launch.json                     # argv、cwd、安全环境、unit命令、源hash；只创建一次
  guard_submission.json           # systemd提交结果；不代表训练退出码
  supervisor_submission.json
  guard/
    metadata.json                 # guard身份、固定安全策略
    attached_identity.json        # 已绑定的supervisor/child身份
    samples.csv                   # UTC/unix/monotonic/elapsed、used/free/reserved、低值计数
    heartbeat.json                # 当前采样和guard进程健康
    status.json                   # 原子、fsync的最终状态
  supervisor/
    command.json                  # 完整训练argv、cwd、安全环境、监督策略
    identity.json                 # supervisor身份
    child_identity.json           # immutable child身份和spawn时间
    exec_release.json             # guard绑定后放行时间
    child_exit.json               # wait得到的原始returncode和时间
    heartbeat.json
    events.jsonl                  # guard-ready/gated/released等事件
    console.txt                   # stdout/stderr（外部保留）
    status.json                   # 最终child退出码、signal、reason、错误、起止时间
```

`stamp()` 同时记录 UTC、Unix 秒和 monotonic 秒。stale/gap/wall duration 使用同一 boot 的 monotonic 值，UTC 供人阅读，避免墙钟调整影响 liveness。每个 heartbeat 还必须指向实际存活的 guard/supervisor 身份；guard heartbeat 必须具有真实 sample count，不能用空 heartbeat 冒充监测。

supervisor 最终状态分开保存 `raw_wait_returncode`、`child_exit_code`、`child_signal` 和 `termination_reason`。正常 exit0、普通非零 exit、被信号结束分别为 `0`、实际非零值、负信号 wait 值；信号结束时 `child_exit_code=null` 且 `child_signal` 明确。guard 的退出码、unit 的退出码或最终 checkpoint 都不能替代 child 的 `wait()` 证据。supervisor 因 SIGKILL 无法保存 final status 时，guard 记录 `supervisor_lost_or_stale`；此时不推断未知 child exit code。

## 3. GPU 安全策略与覆盖

固定 free-memory floor 为 **1024 MiB**，只有 **3 个连续有效样本** free `<1024 MiB` 才触发低显存终止。等于 1024 不算低值；任一回到阈值以上的样本重置计数。不会故意耗尽真实 GPU 来测试此规则。CPU 测试使用 mock 采样器和牺牲 child；`--mock-samples` 不能用于正式运行证据。

默认采样间隔 **0.1 秒**，独立 gap 上限 **0.5 秒**，guard/supervisor stale 上限 **2 秒**。0.5 秒是 `[ENGINEERING]` 调度/监测故障阈值，不改变 1024 MiB/连续3次的显存策略。样本间隔超过 0.5 秒会记录实际两端时间、duration 并 fail closed，不能被计为“持续正常”。NVML 异常、非有限/负内存值、identity 不匹配、guard/supervisor 丢失或 stale 也会终止自己的 child 并写入停止决定。

guard 从 spawn 之前开始采样，保持到 supervisor 发布 `child_exit.json` 的真实 wait 证据之后。成功覆盖必须交叉验证：`first_sample < child_spawn < exec_release <= child_wait <= last_sample`、所有样本间隔均未超过固定 gap 上限、错误为空、guard 最终 `stop_reason=child_wait_observed`。`continuous_monitoring=true` 的含义是该采样协议完整覆盖，不是连续物理测量；0.1 秒采样仍可能漏掉更短的显存峰值。

每行 GPU CSV 保存 UTC、Unix/monotonic/elapsed、全卡 used/free/total/reserved MiB、target PID、low-consecutive。最终 guard 状态保留 sample count、首尾样本、min free、peak used、最大连续低值、gap 数组、停止原因、错误和已发送信号。全卡内存包含桌面和其它应用；观测极值不能称作精确峰值。

## 4. 持久性与已知边界

运行时 CSV 每次采样 `flush`，heartbeats 采用同目录临时文件 + `os.replace` 的快速原子可见更新，不在每个 0.1 秒样本后 `fsync`，以免磁盘同步把采样周期拖长。元数据、事件和最终 `status.json` 使用 `fsync`；原子 helper 还同步文件所在目录。最终 CSV 在正常关闭时同步。失败替换应保留原完整 JSON，读者不会看到半个 JSON。

这些措施不能保证在主机断电、kernel/磁盘故障、文件系统不可写、服务被 SIGKILL 或用户 service manager 被整体关闭时总有最终文件。SIGKILL 无法运行 Python `finally`；两个服务同时消失也没有第三个进程替它们回收 child wait。此类情形通过缺失 final status、stale heartbeat、systemd 状态与已保留的时间戳标为不完整，不能报告正常完成。没有为登录会话退出或重启自动配置 linger；是否跨注销存活取决于当前 user-manager 生命周期，G3A 的直接测试范围是提交客户端/终端消失。

通用外部命令若自行 daemonize、改变 PGID、关闭继承 lock FD 或绕过 launcher，超出标准 `run_sim.sh` 的身份和独占保证。不要用不同 lock 路径并行运行同一 GPU 训练。任何新的长训练前都要重新确认没有已有任务、磁盘可写、guard readiness 和源身份。

## 5. TensorBoard `[ENGINEERING]` 收尾与检查

`python_entry.py` 仅在 `FACTORY_G3A_FINALIZE_TENSORBOARD=1` 时安装 wrapper。wrapper 保留官方 `Runner.run_train` 的调用与返回，临时捕获新 agent，最后对其 writer 先 `flush()` 再 `close()`，保留异常和操作结果；不改 PPO 更新、reward、环境或超参数。`FACTORY_G3A_FINALIZER_OUTPUT` 指定新 run 的操作结果 JSON。G2 未设置此 flag，原事件不修改、不补值。

直接源码事实是 pinned RL-Games 正常 `run_train` 路径没有显式 writer flush/close，tensorboardX 默认异步 flush 间隔为120秒。这解释了为何显式 finalizer 能改善收尾，但不能证明 G2 缺尾的历史根因。G2 supervisor 消失和未知 exit 路径没有被新 smoke 恢复。

新 2-epoch smoke 的严格尾部检查使用固定 required tags、每类 tag 的计数和原生 step。actor/performance tags 第2轮 step=8192；central-value 第2轮 step=16384；`info/epochs` 必须是 `[1,2]`。Factory episode149步、horizon128，reward首次在epoch2出现，因此只有1个 `rewards/iter` 是正常计数。缺tag、缺尾、非有限值或step不符返回 incomplete，不能合成补齐。

## 6. 精确复现命令

下列变量仅缩写已经存在的路径。运行时必须更换 `G3_RUN`、`G3_NAME` 和 `G3_UNIT` 为**未存在**的新名称，禁止复用历史目录。smoke 使用现有64环境、seed0、官方配置，只有2epochs；`save_frequency=1` 为短测checkpoint持久化覆盖，不改变优化。不得添加 `--checkpoint`。

```bash
G3_WORKSPACE=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been
G3_REPRO="$G3_WORKSPACE/outputs/isaac-factory-repro"
G3_VENV="$G3_WORKSPACE/work/envs/factory-g01"
G3_UPSTREAM="$G3_WORKSPACE/work/upstream/IsaacLab"
G3_RUN="$G3_WORKSPACE/work/logs/g3a/smoke_NEW_UNIQUE_NAME"
G3_NAME=g3a_smoke_64env_seed0_NEW_UNIQUE_NAME
G3_UNIT=g3a-NEW-UNIQUE-NAME
```

先运行 CPU-only 测试。禁用环境中无关的 ROS pytest entrypoint 自动加载，沿用已安装依赖：

```bash
cd "$G3_REPRO"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$G3_VENV/bin/python" -m pytest -q \
  tests/test_g3a_supervision.py tests/test_tensorboard_finalizer.py \
  tests/test_tensorboard_runner.py tests/test_tensorboard_smoke_tail.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$G3_VENV/bin/python" -m pytest -q tests
```

CPU 矩阵通过后，提交两个持久 unit。`--cwd` 指向仓库外已有 `work/runs`，官方 train_dir 由此生成；输出和大 checkpoint 留在该目录外置保留：

```bash
python3 "$G3_REPRO/scripts/g3a_launch_smoke.py" \
  --output "$G3_RUN" --unit-prefix "$G3_UNIT" \
  --cwd "$G3_WORKSPACE/work/runs" --timeout 1200 --wait \
  --setenv "FACTORY_VENV=$G3_VENV" \
  --setenv OMP_NUM_THREADS=4 \
  --setenv FACTORY_G3A_FINALIZE_TENSORBOARD=1 \
  --setenv "FACTORY_G3A_FINALIZER_OUTPUT=$G3_RUN/supervisor/tensorboard_finalizer.json" \
  --setenv "FACTORY_G2_AUDIT_OUTPUT=$G3_RUN/supervisor/training_audit.json" \
  -- bash "$G3_REPRO/scripts/run_sim.sh" \
  "$G3_UPSTREAM/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs 64 --seed 0 --max_iterations 2 \
  "agent.params.config.full_experiment_name=$G3_NAME" \
  agent.params.config.save_frequency=1
```

正常返回后另行检查事件尾，返回0才表示固定末轮scalar协议完整：

```bash
"$G3_VENV/bin/python" "$G3_REPRO/scripts/tensorboard_tail.py" \
  "$G3_WORKSPACE/work/runs/logs/rl_games/Factory/$G3_NAME/summaries" \
  --expected-epochs 2 --transitions-per-epoch 8192 --reward-first-epoch 2
```

`launch.json` 保存真实展开的 systemd argv、safe environment 和监督脚本 hash。读取 `supervisor/status.json`、`guard/status.json`、TensorBoard finalizer/tail、checkpoint 内epoch/frame及配置diff后才能判断G3A gate；launcher exit0本身不是全部验收证据。

## 7. 验收与停止

CPU sacrificial 测试覆盖普通/非零/signal退出、missing/crashed guard、supervisor crash、提交launcher SIGKILL、gated首末样本、连续/非连续低显存、采样错误/gap、PID birth误配、supervisor死亡后child继承锁、证据非覆盖、原子JSON并发可见和rename失败，以及TensorBoard缺尾/非有限值检测。测试不得终止已有训练或无关用户unit。

G3A 只有在新短smoke证明完整采样覆盖、真实child exit0、checkpoint、最终scalar及配置不变时才可PASS。证据不足时报告PARTIAL。G2的监督事故和缺失数据永久保留；G3A结束后停止，不自动执行G3B、额外seed或200epoch训练。
