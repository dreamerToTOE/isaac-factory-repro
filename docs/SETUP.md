# G0/G1 安装与执行步骤

本阶段只建立隔离环境、审计官方实现、执行有时间和迭代上限的 smoke test。上游 Factory 环境、成功条件与官方 PPO 源文件保持不变。

## 1. 目录与版本

本次工作区结构：

```text
$WORKSPACE/
├── outputs/isaac-factory-repro/   # 现有 GitHub 仓库的 clone
├── work/upstream/IsaacLab/        # 独立的官方上游 checkout
├── work/envs/factory-g01/         # 项目专用 Python 环境
├── work/logs/                    # 每次执行独立的原始证据目录
└── logs/rl_games/Factory/         # 官方训练入口生成的日志与 checkpoint
```

复现实验时先设置实际路径；`WORKSPACE` 必须是绝对路径：

```bash
export WORKSPACE="/absolute/path/to/isaac-factory-workspace"
export REPRO="$WORKSPACE/outputs/isaac-factory-repro"
export ISAACLAB_PATH="$WORKSPACE/work/upstream/IsaacLab"
export FACTORY_VENV="$WORKSPACE/work/envs/factory-g01"
export ISAACSIM_PATH="/home/ubuntu2004/isaacsim-4.5.0"
mkdir -p "$WORKSPACE/outputs" "$WORKSPACE/work/upstream" "$WORKSPACE/work/envs" "$WORKSPACE/work/logs"
cd "$WORKSPACE"
```

本机 `WORKSPACE` 为 `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been`。不把 Sim 安装目录复制进项目，不使用系统 pip，也不改动现有 conda 环境。

### 已有复现仓库

使用已存在的公共仓库：

```bash
git clone https://github.com/dreamerToTOE/isaac-factory-repro.git "$REPRO"
git -C "$REPRO" log -1 --format='%H %s'
git -C "$REPRO" status --short
```

`git clone` 仅在目标目录不存在时执行。已有 checkout 直接检查并复用，不能重新 `git init`、覆盖目录或强制 reset。本次起始提交为 `c54da28f61569ec6cc8249b46ffcea498ff27600`（`Initial commit`），保留 README、.gitignore、LICENSE 和原始历史。

### 官方 Isaac Lab

```bash
git clone --branch v2.1.1 --single-branch https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
git -C "$ISAACLAB_PATH" describe --tags --exact-match
git -C "$ISAACLAB_PATH" rev-parse HEAD
git -C "$ISAACLAB_PATH" status --short
```

预期 tag 为 `v2.1.1`，SHA 为 `90b79bb2d44feb8d833f260f2bf37da3487180ba`。若目标目录已存在，先检查版本与工作区状态；不能丢弃用户修改以取得该 tag。本次使用新的独立 checkout，未复用版本不匹配或来源不明的用户项目。

### 现有 Sim

```bash
cat "$ISAACSIM_PATH/VERSION"
"$ISAACSIM_PATH/python.sh" --version
nvidia-smi
```

实测构建标识为 `4.5.0-rc.36+release.19112.f59b3005.gl`，内置 Python 3.10.15。版本目录名称不能替代这些读取结果。

## 2. 独立学习环境

本次使用 `uv` 创建 Python 3.10.12 虚拟环境，未激活或修改已有 `isaaclab`、`isaac_env`、`unitree_rl_mjlab` conda 环境。

```bash
uv venv --python /usr/bin/python3 "$FACTORY_VENV"
"$FACTORY_VENV/bin/python" --version
```

只在环境目录不存在时创建；预期本机输出 `Python 3.10.12`。在其他机器上须显式选用 Python 3.10，不能用已有 Python 3.11 环境替代。

### 安装顺序

官方 v2.1.1 安装说明推荐 Torch 2.7.0 / torchvision 0.22.0 的 cu128 组合。先在隔离环境安装该组合，再安装冻结的必要依赖：

```bash
uv pip install --python "$FACTORY_VENV/bin/python" \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install --python "$FACTORY_VENV/bin/python" \
  setuptools==80.10.2 wheel

uv pip install --python "$FACTORY_VENV/bin/python" \
  --no-deps --no-build-isolation \
  -r "$REPRO/configs/requirements-g01.lock"
```

`configs/requirements-g01.lock` 记录本次第三方依赖版本，四个本地 Isaac Lab editable 包单独安装，不在锁文件内固定本机绝对路径。Torch 包的实际版本应为 `2.7.0+cu128`、torchvision 为 `0.22.0+cu128`，其余关键版本为 numpy 1.26.4、gymnasium 1.2.0、RL-Games 1.6.1。

锁文件是本次实际已安装包的精确版本清单；`--no-deps` 保持该清单，避免安装时自动升级目标组合。它不声称覆盖全部 Isaac Lab 可选依赖。源码扩展通过下面四条命令另行绑定本机的独立上游 checkout。

随后仅安装 Factory 与 RL-Games 路径所需的四个官方扩展：

```bash
uv pip install --python "$FACTORY_VENV/bin/python" --no-deps --no-build-isolation --editable "$ISAACLAB_PATH/source/isaaclab"
uv pip install --python "$FACTORY_VENV/bin/python" --no-deps --no-build-isolation --editable "$ISAACLAB_PATH/source/isaaclab_assets"
uv pip install --python "$FACTORY_VENV/bin/python" --no-deps --no-build-isolation --editable "$ISAACLAB_PATH/source/isaaclab_tasks"
uv pip install --python "$FACTORY_VENV/bin/python" --no-deps --no-build-isolation --editable "$ISAACLAB_PATH/source/isaaclab_rl"
```

这里选择显式最小依赖安装，没有运行会安装所有算法框架的 `./isaaclab.sh --install all`。本次没有安装其他 RL 算法或改动官方包源码。

### 已遇到的安装失败

- 最初 Factory 导入失败：缺少 `flatdict`。补齐必要支持依赖后重新测试。
- `flatdict==4.0.1` 的隔离构建失败：其 setup 使用 `pkg_resources`，但构建依赖未声明该模块。先固定 `setuptools==80.10.2`，再使用 `--no-build-isolation` 构建解决；没有修改第三方源码。上面的安装顺序保留该修复。
- 先前一次依赖安装等待同一虚拟环境的 uv 锁超时；后续顺序执行安装完成。安装同一环境时不能并发执行多个 uv 进程。

检查依赖声明：

```bash
uv pip check --python "$FACTORY_VENV/bin/python"
```

本 G0/G1 最小环境的声明依赖检查尚不完整：`pin-pink`、`dex-retargeting`、`moviepy` 未安装。它们分别用于未运行的运动学/重定向/视频路径；未把这次部分安装宣称为所有 Isaac Lab 功能可用。Factory headless 与 PPO 的可用性以实际运行证据单独验证。

## 3. 启动时保持实际包版本

仅在 `PYTHONPATH` 前面放置虚拟环境目录不足以固定运行时包。此前 Kit 启动时把 Sim 的 archive 路径插到前面，实际加载的是内置 Torch 2.5.1+cu118、numpy 1.26.0、gymnasium 0.29.1；这些运行不能当作目标 cu128 学习环境的验证。

项目启动器采用两步处理：

1. `scripts/run_sim.sh` 清除调用进程继承的 `PYTHONPATH` / `LD_LIBRARY_PATH`，设置 `PYTHONNOUSERSITE=1`，在该子进程加载现有 Sim 的 `setup_conda_env.sh`，再调用隔离环境的 Python。
2. `scripts/python_entry.py` 在 AppLauncher 启动前预先导入并核对 Torch、torchvision、numpy、gymnasium；AppLauncher 启动后再次核对模块身份、版本与路径，输出 `PINNED_PACKAGES` 和 `PACKAGES_AFTER_APP_LAUNCH`。

这些是启动与证据采集改动，不改变 PPO 数学实现、任务奖励、成功条件或物理配置。读取日志时必须同时看到启动前后正确的版本；安装 metadata 或 `pip freeze` 不能代替运行时版本证据。

修复后的 [4 环境训练日志](../evidence/2026-10-23/train_peg4_pinned/console.txt) 与 [16 环境训练日志](../evidence/2026-10-23/train_peg16_pinned/console.txt) 已确认 AppLauncher 启动后仍使用 Torch 2.7.0+cu128、torchvision 0.22.0+cu128、numpy 1.26.4、gymnasium 1.2.0。此前被内置 archive 遮蔽的结果仅作为排错记录。

```bash
bash "$REPRO/scripts/run_sim.sh" -c \
  'import torch, numpy, gymnasium; print(torch.__version__, numpy.__version__, gymnasium.__version__)'
```

此命令只做包检查，不启动 AppLauncher，故只输出启动前标记；完整 Factory / PPO 测试必须出现启动后标记。

## 4. 证据采集器

每次执行使用一个不存在的新目录。例如：

```bash
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
```

`scripts/run_logged.py` 保存命令、工作目录、开始 UTC 时间、返回码、超时状态、墙钟时间、console 输出，并约每秒采集：

- `gpu.csv`：整卡显存和 GPU 利用率，包含桌面及其他应用。
- `process_gpu.csv`：NVML compute-apps 中的进程显存，标注是否属于本次测试进程组。
- `result.json`：整卡采样最大值，以及可获得时的本次测试进程组采样最大值。

采样间隔还包含 `nvidia-smi` 执行时间；这些数值不是精确瞬时峰值。早期 gate 日志只采集整卡数值，不补造不存在的进程显存数据。采集器拒绝复用已存在的输出目录，防止覆盖证据；超时只终止自己创建的测试进程组。

本次原始运行输出保存在工作区 `work/logs/`，交付快照统一位于仓库 [evidence/2026-10-23](../evidence/2026-10-23/)。链接中的 `*_pinned` 是固定目标运行时后的最终结果。

## 5. 第一关：内置 Sim GPU 初始化

学习环境测试前先用现有 Sim 自带 Python 做无远程资产的 GPU 门槛：

```bash
python3 "$REPRO/scripts/run_logged.py" \
  --output "$WORKSPACE/work/logs/gpu_gate_$RUN_TAG" --timeout 600 -- \
  env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  "$ISAACSIM_PATH/python.sh" "$REPRO/scripts/gpu_probe.py"
```

通过标准是：日志识别 Vulkan 活动 RTX GPU，GPU dynamics 启用，物理视图张量为 CUDA，240 次步进后立方体在地面合理高度且状态有限，输出 `GPU_PROBE_RESULT` 的 `passed=true`，进程返回码 0。该关实际使用内置 Torch 2.5.1+cu118，必须与后续隔离学习环境区分。

## 6. 第二关：Factory 注册、reset 与 step

先运行 1 个环境，160 次 agent step 覆盖 PegInsert 的一次超时自动 reset：

```bash
python3 "$REPRO/scripts/run_logged.py" \
  --output "$WORKSPACE/work/logs/peg_probe1_$RUN_TAG" --timeout 600 -- \
  env FACTORY_VENV="$FACTORY_VENV" ISAACSIM_PATH="$ISAACSIM_PATH" OMP_NUM_THREADS=4 \
  bash "$REPRO/scripts/run_sim.sh" "$REPRO/scripts/factory_probe.py" \
  --headless --device cuda:0 --num_envs 1 --steps 160
```

日志必须列出以下三个已注册 ID，并输出 `FACTORY_PROBE_RESULT`：

- `Isaac-Factory-PegInsert-Direct-v0`
- `Isaac-Factory-GearMesh-Direct-v0`
- `Isaac-Factory-NutThread-Direct-v0`

注册三个 ID 不等于三个任务均已做完整动态测试。该命令实际初始化 PegInsert，验证 reset、CUDA observations、有限值 reward / observation / joint position 和短时间控制步进。探针有同步检查开销，其 `environment_steps_per_second` 不能称作 PPO 训练 FPS。

仅在小环境实测显存仍有余量后，把上述命令 `--num_envs 1` 改为 `16`，同时使用全新的证据目录。不能按 RTX 3070 的标称 8 GiB 推断 256/512 个环境可行。

## 7. 第三关：4 环境有界 PPO smoke

使用官方 v2.1.1 RL-Games 入口。4 个环境乘官方 horizon 128，得到 512 样本，与官方 actor / central critic minibatch 512 相符；避免为了单环境测试修改官方 PPO 批量参数。

```bash
python3 "$REPRO/scripts/run_logged.py" \
  --output "$WORKSPACE/work/logs/train_peg4_$RUN_TAG" --timeout 600 -- \
  env FACTORY_VENV="$FACTORY_VENV" ISAACSIM_PATH="$ISAACSIM_PATH" OMP_NUM_THREADS=4 \
  bash "$REPRO/scripts/run_sim.sh" "$ISAACLAB_PATH/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs 4 --seed 0 --max_iterations 3 \
  "agent.params.config.full_experiment_name=g01_peg4_$RUN_TAG" \
  agent.params.config.save_frequency=1 \
  agent.params.config.save_best_after=0
```

本 smoke 的命令行覆盖只有环境数、迭代上限、实验名与保存频率/起点；其他 PPO 及任务配置使用官方值。日志与 checkpoint 位于当前 `$WORKSPACE/logs/rl_games/Factory/g01_peg4_$RUN_TAG/`。训练 FPS 使用 RL-Games 本身输出；需要 checkpoint 中非零 Adam 更新状态及参数变化证据才能认定发生真实优化。

本次最终目标环境的 4 环境 / 3 epoch smoke 已通过，整卡 / 测试进程组采样最大值为 **5955 / 4238 MiB**，RL-Games 报告 total FPS 18–19。[执行结果](../evidence/2026-10-23/train_peg4_pinned/result.json)、[真实优化证据](../evidence/2026-10-23/checkpoint_peg4_pinned.json)。随后 16 环境 / 1 epoch 也通过，整卡 / 进程组为 **5659 / 4042 MiB**，total FPS 71；只覆盖一次短优化。[执行结果](../evidence/2026-10-23/train_peg16_pinned/result.json)、[checkpoint](../evidence/2026-10-23/checkpoint_peg16_pinned.json)。16 环境重放时使用新证据目录，把 `--num_envs 4` 改为 `16`、`--max_iterations 3` 改为 `1`，并更换实验名。

检查自己刚生成的 checkpoint，可在 CPU 上使用项目工具；`CHECKPOINT` 与 `EARLIER_CHECKPOINT` 均设置为本次生成文件的绝对路径：

```bash
"$FACTORY_VENV/bin/python" "$REPRO/scripts/inspect_checkpoint.py" \
  "$CHECKPOINT" --compare "$EARLIER_CHECKPOINT"
```

### 保存后加载并继续一个迭代

目标 Torch 2.7 中首次恢复失败：PyTorch 从 2.6 起将 `torch.load` 的 `weights_only` 默认值改为 `True`，RL-Games 1.6.1 未显式传入该参数，checkpoint 中的 NumPy 对象因此被拒绝。[失败原始日志](../evidence/2026-10-23/resume_peg4_pinned/console.txt)。

修复采用仅限恢复子进程的 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。适用对象仅为本项目自己生成、保存时已记录 SHA-256 的本地 checkpoint；未把该设置写入全局环境或修改 RL-Games 源码。本次原始 epoch 3 文件的 SHA-256 为 `6dd7f429bef5aede4c9d55fa1da54858f47569b182a044c5b634fb94952de442`，记录见 [checkpoint 检查](../evidence/2026-10-23/checkpoint_peg4_pinned.json)。

设置 `CHECKPOINT` 为上一步第 3 epoch 的实际保存文件，`EXPECTED_CHECKPOINT_SHA256` 为保存该文件时记录的 SHA-256。重放生成的新文件有自己的 SHA，不能直接套用原始运行的摘要。先核对，再使用另一独立实验名恢复：

```bash
printf '%s  %s\n' "$EXPECTED_CHECKPOINT_SHA256" "$CHECKPOINT" | sha256sum --check
```

```bash
python3 "$REPRO/scripts/run_logged.py" \
  --output "$WORKSPACE/work/logs/resume_peg4_$RUN_TAG" --timeout 600 -- \
  env FACTORY_VENV="$FACTORY_VENV" ISAACSIM_PATH="$ISAACSIM_PATH" OMP_NUM_THREADS=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  bash "$REPRO/scripts/run_sim.sh" "$ISAACLAB_PATH/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs 4 --seed 0 --max_iterations 4 --checkpoint "$CHECKPOINT" \
  "agent.params.config.full_experiment_name=g01_peg4_resume_$RUN_TAG" \
  agent.params.config.save_frequency=1 \
  agent.params.config.save_best_after=0
```

`--max_iterations 4` 是恢复后的总 epoch 上限，应从第 3 epoch 的 checkpoint 继续到第 4 epoch。再对恢复后的 checkpoint 与原文件执行 `inspect_checkpoint.py --compare`，检查有限值、优化器 step 递增和参数变化。

本次加入显式加载设置后的恢复已通过：epoch 3 → 4，frame 1536 → 2048，optimizer step 11 → 15，actor 与 critic 参数继续更新，全部 tensor 有限。[执行结果](../evidence/2026-10-23/resume_peg4_pinned_compat/result.json)、[恢复后检查](../evidence/2026-10-23/checkpoint_resume_peg4_pinned.json)。

## 8. 阶段结束

G0/G1 只形成环境、源码、有限步进、短 PPO 更新和保存/加载证据。完成后停止，不从 smoke 自动延伸至完整训练。正式训练的环境数与时长须经过本阶段结果审阅；尤其不能以短时吞吐或 GPU 门槛成功推断最终装配成功率。
