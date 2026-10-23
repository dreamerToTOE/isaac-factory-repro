# G0/G1 实验协议与后续训练建议

日期：2026-10-23（Asia/Seoul）。本阶段只验证安装、GPU 仿真、Factory 环境、有限次 PPO 优化与 checkpoint 恢复。完成这些检查后停止，等待审阅；本文件中的后续训练建议不构成启动长训练的授权。

## 1. 上游基准和工程覆盖

上游锁定为官方 Isaac Lab `v2.1.1` / `90b79bb2d44feb8d833f260f2bf37da3487180ba`，独立保存在复现仓库之外。官方三任务的行为和全部 PPO 参数详见 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)。原代码、奖励、成功标准、控制器及 PPO 算法不修改。

### 已采用的四环境 smoke 覆盖

| 参数 | 上游原值 | G0/G1 smoke |
| --- | --- | --- |
| 环境数 | 128 | 4；训练脚本同步更新 agent num_actors |
| max_epochs | 200 | 3 |
| save_frequency | 100 | 1 |
| save_best_after | 10 | 0 |
| full_experiment_name | test | 唯一的 smoke 运行名称 |

`seed=0` 与上游一致。恢复测试从 epoch 3 checkpoint 继续到总 epoch 4（`--max_iterations 4` 是总上限，非再训练4轮）。`--checkpoint` 和独立运行名称是恢复操作本身；不是替换初始模型或评估协议。显存验证另做16 env、1 epoch有限PPO测试；除环境数、epoch上限及运行名之外，沿用上表覆盖及其余官方参数。

以下保持官方值：actor 和 central-value `minibatch_size=512`、`horizon_length=128`、`seq_length=128`、`mini_epochs=4`、双层1024 LSTM、MLP `[512,128,64]`、PPO clipping、学习率/调度、奖励、成功判据、物理步长、decimation 和 PhysX 接触缓冲。4 env 每轮产生 `4×128=512` transitions，刚好匹配 minibatch512；没有为让小规模测试运行而缩小 minibatch 或替换 RNN。

`headless`、CUDA device、`OMP_NUM_THREADS=4`、隔离 Python 启动器、日志/显存采样器属于运行工程设置，须与配置差异一同记录。恢复时为兼容新版PyTorch使用的命令级环境变量另见第3节；它不改变PPO计算。仅从环境数减少不能推出8GB显存一定足够。

## 2. 每次运行必须保留的证据

每个测试使用唯一目录；不可覆盖失败日志，也不可只保留最后一次成功结果。

| 文件/字段 | 用途 |
| --- | --- |
| `result.json` | 完整 argv、cwd、UTC开始时间、返回码、是否超时、用时及采样峰值 |
| `console.txt` | GPU/Vulkan、环境初始化、配置解析、FPS、PPO epoch、错误与警告 |
| `gpu.csv` | 每秒左右采样的整卡使用显存/总显存/利用率；含桌面和其他进程 |
| `process_gpu.csv` | compute 进程显存、PID、是否属于测试进程组 |
| `params/env.yaml`、`agent.yaml` | 官方脚本构造前dump，核对覆盖范围；21/72维与num_actors=128会在后续构造/wrapper时覆盖 |
| checkpoint 检查 JSON | 路径、大小、SHA256、epoch/frame、张量有限性、Adam状态及参数差异 |
| AppLauncher **之后**的版本/路径 | `torch/numpy/gymnasium` 的 `__version__` 与 `__file__`；不能只记录 pip metadata |

采样最大值是**观测到的峰值**，实际采样间隔还包括 `nvidia-smi` 调用耗时；不声称捕获所有瞬时峰值。整卡峰值与本测试compute进程组峰值的定义不同，不能直接相减得到精确桌面峰值。启动/加载、reset、rollout、反向传播、checkpoint 写入都要包含在采样窗口内。

### 安装版本不等于运行版本

已发现 Kit/AppLauncher 会调整 `sys.path`，使捆绑 archive 遮蔽隔离 venv：早期4 env训练/恢复与16 env环境探针实际使用 Torch **2.5.1+cu118**、NumPy **1.26.0**、Gymnasium **0.29.1**，即使 venv metadata 显示 Torch2.7.0+cu128 / NumPy1.26.4 / Gymnasium1.2.0。

已通过独立入口 `scripts/python_entry.py` 在 SimulationApp 前预载锁定基础包，并在 AppLauncher 后验证版本、文件路径与模块对象一致。规范4 env训练、兼容恢复及16 env单epoch训练均记录 `PACKAGES_AFTER_APP_LAUNCH`，实际为 **Torch2.7.0+cu128、torchvision0.22.0+cu128、NumPy1.26.4、Gymnasium1.2.0**；来源均为项目venv，已消除本次观测到的遮蔽问题。入口同时增加AppLauncher证据打印包装，官方训练源码和PPO不变。[4 env启动后版本证据](../evidence/2026-10-23/train_peg4_pinned/console.txt)。

修复前数据保留为真实历史记录，与规范锁定环境复测分开，不混用FPS或显存结果。以后每次运行仍应检查实际模块，不能只以 `pip list`、启动前 `find_spec` 或启动器注释认定成功。具体设置见 [SETUP.md](SETUP.md) 与 [OPEN_ISSUES.md](OPEN_ISSUES.md)。

## 3. 分阶段放行条件

### A. 静态与版本检查

1. 核对上游标签、完整SHA和工作树状态。
2. 记录OS/内核、GPU/驱动、空闲显存、RAM、存储及Python；保留已有Sim与其他用户环境。
3. 使用项目隔离环境；核对实际 import 路径。记录依赖检查的失败或有意省略项。
4. 检查三个Factory ID在真实 `isaaclab_tasks` 导入后均存在于Gym registry。静态源码中的注册语句不能替代运行时结果。

### B. GPU门槛

必须同时确认Vulkan选中RTX3070、CUDA物理设备可用、GPU PhysX / tensor view工作，并完成有限步仿真。`nvidia-smi`或 `torch.cuda.is_available()` 单独成功不足以放行。GPU门槛失败则保留日志，停止Factory训练；不绕过权限、换CPU结果冒充GPU结果或修改系统设备权限。

### C. 小规模环境初始化、reset与step

1. 从1 env开始，运行有限的160策略步，检查CUDA观测、reward、joint state均为有限值。
2. 验证实际观测为 `policy:[N,19]`、`critic:[N,43]`，动作6维。
3. 覆盖至少一次episode超时自动reset；PegInsert nominal150步，源码判断路径推导第149步触发timeout，不能把reset过程中的额外物理步算作policy rollout。
4. 随后进行4 env的有限PPO测试。只有显存有余量才可进行16 env环境探针；16 env能reset/step不证明其PPO反向传播显存足够。

### D. 真正PPO优化与保存

四环境只跑3个epoch；16 env显存/优化验证只跑1个epoch，不延长为正式训练。最低验收条件：

- 日志出现完整epoch进度，运行返回0且没有超时。
- 四环境训练与恢复须证明actor及central-value参数均相较前一checkpoint发生变化；16 env单epoch没有前一训练checkpoint，则检查非零Adam动量、正step计数及写盘模型，按较窄的单轮优化证据报告。
- checkpoint内模型、optimizer相关tensor均有限；Adam有非零动量，step计数递增。
- checkpoint真实写入磁盘，记录其SHA256。只有“Started to train”或FPS日志不能证明优化已发生。
- console没有NaN/Inf状态、PhysX崩溃、非法内存访问或未处理仿真异常。上游非致命警告单列，不把它们等同训练失败。

### E. 恢复

从本机生成的epoch3 checkpoint启动独立短run，总上限epoch4。核对加载日志、epoch/frame/Adam step续接、actor与central-value参数继续变化，并保存新checkpoint。这证明actor/central-value参数与actor Adam恢复路径工作。RL-Games1.6.1未保存独立central-value Adam状态；也不证明物理场景、随机数、RNN rollout状态被逐位重放。框架源码摘录见[checkpoint源码证据](../evidence/2026-10-23/rlgames_checkpoint_source.txt)。

规范PyTorch2.7下第一次原样恢复失败：RL-Games1.6.1调用 `torch.load` 未指定 `weights_only`，PyTorch2.6起默认改变；本机checkpoint包含 `numpy.core.multiarray.scalar`，默认weights-only读取拒绝加载。失败日志保留在 [resume_peg4_pinned](../evidence/2026-10-23/resume_peg4_pinned/console.txt)。

兼容恢复仅对**本项目刚生成并校验SHA256的自有checkpoint**，在该恢复命令使用 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，使未显式指定的读取回到旧版完整checkpoint语义。没有改上游RL-Games/PPO源码，也不把该变量设为系统默认。所用epoch3文件SHA256为 `6dd7f429bef5aede4c9d55fa1da54858f47569b182a044c5b634fb94952de442`；兼容恢复已通过。[原checkpoint检查](../evidence/2026-10-23/checkpoint_peg4_pinned.json)、[恢复检查](../evidence/2026-10-23/checkpoint_resume_peg4_pinned.json)。

恢复只追加一个128步rollout，短于一个新的PegInsert episode，故可出现 `Max epochs reached before any env terminated at least once`，checkpoint文件名也可能包含 `rew_-inf`。这个标量/文件名不能直接判为网络出现NaN；需要结合有限性和checkpoint检查。反之，`all_tensors_finite`不能替代检查该标量及运行日志。

TensorBoard若 `scalar_tags=0`，结论为**无标量可供判断**，不是“所有loss有限”。本次兼容恢复与16 env单epoch运行属于此情形；相应优化证据来自checkpoint/Adam/参数变化及console。规范4 env三epoch运行有50个scalar tags，已记录标量均有限；不能把空记录检查返回的布尔值当作loss有限性通过。

## 4. 可复用命令

以下在**复现仓库根目录**执行；路径变量按实际隔离环境设置。`ISAACLAB_PATH` 指独立上游checkout；`FACTORY_LOG_ROOT` 指用户选择的日志目录。所有阶段串行执行，先检查上一阶段结果，并核验启动后的实际包版本。

```bash
export ISAACSIM_PATH=/home/ubuntu2004/isaacsim-4.5.0
export ISAACLAB_PATH=/absolute/path/to/IsaacLab-v2.1.1
export FACTORY_VENV=/absolute/path/to/factory-g01
export FACTORY_LOG_ROOT=/absolute/path/to/g01-logs
export OMP_NUM_THREADS=4
```

GPU门槛（10分钟上限）：

```bash
python3 scripts/run_logged.py --output "$FACTORY_LOG_ROOT/gpu_gate" --timeout 600 -- \
  bash scripts/run_sim.sh scripts/gpu_probe.py
```

单环境reset/step（10分钟上限）：

```bash
python3 scripts/run_logged.py --output "$FACTORY_LOG_ROOT/peg_probe_1" --timeout 600 -- \
  bash scripts/run_sim.sh scripts/factory_probe.py \
  --headless --device cuda:0 --num_envs 1 --steps 160
```

四环境3 epoch smoke（10分钟上限，示例run名需保持唯一）：

```bash
python3 scripts/run_logged.py --output "$FACTORY_LOG_ROOT/train_peg4_pinned" --timeout 600 -- \
  bash scripts/run_sim.sh "$ISAACLAB_PATH/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs 4 --seed 0 --max_iterations 3 \
  agent.params.config.full_experiment_name=g01_peg4_pinned \
  agent.params.config.save_frequency=1 agent.params.config.save_best_after=0
```

训练输出基于执行cwd写入 `logs/rl_games/Factory/<runname>/`。由真实输出确定checkpoint路径，不猜文件名或reward：

```bash
export FACTORY_CHECKPOINT=/absolute/path/to/epoch3-checkpoint.pth
export FACTORY_PREVIOUS_CHECKPOINT=/absolute/path/to/epoch2-checkpoint.pth
"$FACTORY_VENV/bin/python" scripts/inspect_checkpoint.py \
  "$FACTORY_CHECKPOINT" --compare "$FACTORY_PREVIOUS_CHECKPOINT"

python3 scripts/run_logged.py --output "$FACTORY_LOG_ROOT/resume_peg4_pinned_compat" --timeout 600 -- \
  env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  bash scripts/run_sim.sh "$ISAACLAB_PATH/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs 4 --seed 0 --max_iterations 4 --checkpoint "$FACTORY_CHECKPOINT" \
  agent.params.config.full_experiment_name=g01_peg4_pinned_resume_compat \
  agent.params.config.save_frequency=1 agent.params.config.save_best_after=0
```

对恢复生成的epoch4 checkpoint再执行一次 `inspect_checkpoint.py --compare "$FACTORY_CHECKPOINT"`。检查器只在CPU读取本项目自己生成的文件，不下载外部checkpoint。

16环境**环境探针**仅在前述步骤与显存余量允许时使用：

```bash
python3 scripts/run_logged.py --output "$FACTORY_LOG_ROOT/peg_probe_16" --timeout 600 -- \
  bash scripts/run_sim.sh scripts/factory_probe.py \
  --headless --device cuda:0 --num_envs 16 --steps 160
```

已执行的16 env、1 epoch有限PPO命令如下。保持其余PPO参数不变；不得把环境探针显存当作PPO峰值：

```bash
python3 scripts/run_logged.py --output "$FACTORY_LOG_ROOT/train_peg16_pinned" --timeout 600 -- \
  bash scripts/run_sim.sh "$ISAACLAB_PATH/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs 16 --seed 0 --max_iterations 1 \
  agent.params.config.full_experiment_name=g01_peg16_pinned \
  agent.params.config.save_frequency=1 agent.params.config.save_best_after=0
```

## 5. 规范运行结果与历史数据

### 规范锁定包环境

下表实际运行包版本由各console中的 `PACKAGES_AFTER_APP_LAUNCH` 验证，均使用第2节规范版本。FPS为console中的 `fps total`；用时包括启动，不用于反算同一个FPS指标。

| 测试 | 结果/优化证据 | 性能与显存（采样值） |
| --- | --- | --- |
| [4 env，3 epoch](../evidence/2026-10-23/train_peg4_pinned/result.json) | returncode0，无超时；frame1536；22个Adam状态，step11；相较epoch2，actor22个、central-value18个参数tensor变化；checkpoint tensor全部有限 | FPS19/18/18；103.529s；整卡5955MiB、测试compute进程组4238MiB |
| [首次规范恢复失败](../evidence/2026-10-23/resume_peg4_pinned/result.json) | returncode1；PyTorch weights-only兼容问题，未完成恢复优化 | 保留失败，不把运行时间/显存当作训练性能 |
| [兼容恢复至epoch4](../evidence/2026-10-23/resume_peg4_pinned_compat/result.json) | returncode0，无超时；frame2048；Adam step15；actor22个、central-value18个参数tensor继续变化；checkpoint tensor全部有限；未完成新episode | FPS19；42.718s；整卡5626MiB、测试compute进程组3842MiB |
| [16 env，1 epoch](../evidence/2026-10-23/train_peg16_pinned/result.json) | returncode0，无超时；frame2048；22个Adam状态，step15，非零动量；checkpoint tensor全部有限；未完成episode | FPS71；46.832s；整卡5659MiB、测试compute进程组4042MiB |

检查摘要：[4 env checkpoint与参数差异](../evidence/2026-10-23/checkpoint_peg4_pinned.json)、[恢复checkpoint与参数差异](../evidence/2026-10-23/checkpoint_resume_peg4_pinned.json)、[16 env checkpoint](../evidence/2026-10-23/checkpoint_peg16_pinned.json)、[四环境50项TensorBoard标量](../evidence/2026-10-23/events_peg4_pinned.json)。兼容恢复与16 env的TensorBoard无标量数据，不能据此声称loss检查通过。

16 env观测峰值低于4 env三epoch峰值，可能受运行长度、缓存、reset及桌面占用变化影响，不能外推为“环境越多占用越低”。16 env单epoch尚未覆盖episode reset；独立16 env、160步探针随后通过，16个环境均在第149步reset，整卡/进程组采样最大5353/3470MiB；见[探针结果](../evidence/2026-10-23/peg_probe_16_pinned/result.json)。上述测试仅证明短时间运行与优化可行，不证明长期物理稳定或装配收敛。

### 启动优先级修复前的历史运行

下表来自启动包优先级修复前的真实日志，使用Torch2.5.1+cu118 / NumPy1.26.0 / Gymnasium0.29.1。它们证明该运行组合完成了有限优化与恢复；不能冒充锁定包优先级修复后的性能。

| 测试 | 结果/优化证据 | 性能与显存（采样值） |
| --- | --- | --- |
| 4 env，3 epoch | returncode0，无超时；frame1536；Adam step11；与epoch2相比actor22个参数tensor、central-value18个参数tensor变化；所有tensor有限 | 日志 `fps total` 16–17；完整进程用时115.052s；整卡6208MiB、测试compute进程组4234MiB |
| epoch3恢复至epoch4 | returncode0，无超时；frame2048；Adam step15；actor22个、central-value18个参数tensor继续变化；所有tensor有限 | 完整进程用时48.849s；整卡5699MiB、测试compute进程组4034MiB；该轮未完成新episode |

历史证据：[train_peg4](../evidence/2026-10-23/train_peg4/result.json)、[resume_peg4](../evidence/2026-10-23/resume_peg4/result.json)。交付保存console、CSV、result与checkpoint摘要，checkpoint大文件保留本地并由SHA256追踪。进行中的进程不能记为通过。

## 6. 科学评估注意事项

- 原始 `extras["successes"]` 是episode结束时的**即时成功比例**；“episode内至少成功一次”的比例是另一指标。若后续记录ever-success，须另起字段，同时保留官方即时指标，不能替换官方标准。
- 成功没有保持时间判据，也不要求接触力、扭矩或真实机械锁合验证。Peg/Gear要求XY<2.5mm、z差<1mm；Nut阈值基于螺距且附末端yaw判据。不得添加成功提前终止、扩大阈值或把engagement称为接触检测。
- `success_times`上游缓存未在episode reset清零，可能携带旧值；且以步数记录。后续需先明确指标定义再使用，不直接当平均秒数或收敛时间。
- 官方player `deterministic=False`。若以后使用确定性评估，需要显式记录为评估设置差异；策略选择方式、seed、episode数与termination规则应一致。
- 不用短smoke的reward证明成功率；不拿不同环境数的同epoch曲线当相同样本预算；FPS区分环境探针step吞吐、训练step FPS和包括优化的 `fps total`。
- reset中的IK/夹紧会额外推进物理仿真，它们影响wall time，但不计作policy transitions。checkpoint恢复也没有保证环境状态逐位连续。

以上源码路径与固定SHA行链接集中在 [SOURCE_AUDIT.md 第4–9节](SOURCE_AUDIT.md)。

## 7. 审阅后的首个训练建议（当前不执行）

根据规范16 env单epoch已完成真正PPO优化与checkpoint保存，观测整卡峰值5659/8192MiB，建议审阅后的首轮选择 **16 env、seed0**。仍须监测reset和长期训练峰值，若桌面占用变化或余量缩小则退回已验证的4 env。保留官方网络/控制/奖励/PhysX、horizon128、seq128、两个minibatch512与其余PPO参数；不从此结果推断256/512 env能运行。

审阅后的第一轮可取官方 **200 epochs** 作为受限机器上的初步学习曲线；这仍是减少环境数的配置，不与官方训练样本预算等价，也不自动获得收敛保证。

| 配置 | 每epoch transitions | 200 epochs总transitions | 相对官方预算 | 匹配官方总预算所需epochs |
| --- | ---: | ---: | ---: | ---: |
| 官方128 env | 16,384 | 3,276,800 | 1 | 200 |
| 16 env | 2,048 | 409,600 | 1/8 | 1,600 |
| 4 env | 512 | 102,400 | 1/32 | 6,400 |

按 `num_envs×horizon_length×epochs` 计算。为匹配3,276,800 transitions而延长到1600或6400 epochs是更长实验，须以后单独审阅。即使总transition相同，rollout batch、每epoch minibatch数量及自适应学习率轨迹仍有差别，需标记“资源受限的环境数/预算覆盖”，不能声称与128 env训练完全相同。

本阶段结束条件是交付审计、有限实验日志、明确的失败与问题记录及Git提交；完成后等待用户审阅。**不执行200/1600/6400 epoch训练，不修改官方PPO基线。**
