# 环境要求与实机审计（G0/G1）

审计日期：2026-10-23（Asia/Seoul）。本文件区分静态环境审计与随后通过的 GPU 仿真探针；结论以对应命令和日志为依据。

## 锁定版本

- Isaac Lab：官方 `v2.1.1`，提交 `90b79bb2d44feb8d833f260f2bf37da3487180ba`；独立上游 checkout 初次审计为干净工作区。
- Isaac Sim：保留 `/home/ubuntu2004/isaacsim-4.5.0` 现有二进制安装；`VERSION` 原文为 `4.5.0-rc.36+release.19112.f59b3005.gl`，不能省略此构建标识。
- 强化学习：RL-Games PPO；v2.1.1 `source/isaaclab_rl/setup.py:44` 固定 `rl-games==1.6.1`，要求 `torch>=2.5.1`、`numpy<2`（同文件 22–23 行）。仅安装所需 RL-Games extra，不安装所有算法框架。
- 官方 v2.1.1 的二进制安装文档在 `docs/source/setup/installation/binaries_installation.rst:31` 记录 Sim 4.5 / Ubuntu 22.04 / NVIDIA 535.129 的测试组合；Python 安装文档 `pip_installation.rst:42` 要求 Python 3.10。当前驱动 580.178.04 比测试驱动新，不能据此直接宣称完整仿真兼容。

## 实机清单

| 项目 | 实测 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS，x86_64 |
| 内核 | `6.8.0-138-generic` |
| CPU | Intel Core i7-10700F @ 2.90 GHz，8 核 / 16 线程 |
| RAM | 46 GiB 总量，初始约 37 GiB available；2.0 GiB swap，未使用 |
| 根文件系统 | ext4，`df -h` 724 GiB 总量 / 398 GiB available，inode 6% 使用 |
| GPU | NVIDIA GeForce RTX 3070，8192 MiB，计算能力 8.6 |
| 驱动 | 580.178.04；`nvidia-smi` 的 CUDA Version 13.0 是驱动支持能力，不等于本项目 Torch 编译 CUDA 版本 |
| 初始 GPU 内存 | 1818 MiB 使用 / 6000 MiB 可用；桌面、远程桌面、浏览器占用，随时变化 |
| Sim 占用 | 约 16 GiB（`du -sh`） |
| Sim Python | 内置 Python 3.10.15 |
| Sim 预装包（metadata） | torch 2.5.1+cu118、numpy 1.26.0、gymnasium 0.29.1 |

## 终端设备与图形链路

- `/dev/nvidia0`、`nvidiactl`、`nvidia-uvm`、`nvidia-uvm-tools`、`nvidia-modeset` 存在且其他用户具有读写权限。
- `/dev/dri/card1` 和 `/dev/dri/renderD128` 存在；用户虽未加入 video/render 组，ACL 明确给予 `ubuntu2004:rw-`，无需修改权限。
- NVIDIA Vulkan ICD 位于 `/usr/share/vulkan/icd.d/nvidia_icd.json`，引用 `libGLX_nvidia.so.0`，声明 API `1.4.312`。`libvulkan.so.1`、`libcuda.so.1`、`libnvidia-ml.so.1` 可由动态链接器发现。
- `vulkaninfo` / `vulkaninfoSDK` 未安装或不在 PATH；并未为此安装系统包。
- NVML 查询成功、Torch 在已有环境里报告 `cuda.is_available() == True`、设备数 1，只是必要证据，尚不足以证明 Vulkan/RTX 初始化或 PhysX GPU 真正启用。正式放行以 Sim/PhysX GPU 探针为准。

## 已有环境与隔离要求

- 系统 Python 是 3.10.12，pip 22.0.2。
- 现有 conda `isaaclab`：Python 3.11.15、torch 2.7.0+cu128、isaaclab 0.54.3、isaaclab_tasks 0.11.16、isaaclab_rl 0.5.1、RL-Games 1.6.1。其 editable 安装指向已不存在的 `/home/ubuntu2004/isaacsim-5.1.0/.../IsaacLab`；不符合本项目锁定版本。
- 现有 conda `isaac_env`：Python 3.10.20、pip Isaac Sim 4.5.0.0、torch 2.11.0+cu130、gymnasium 1.3.0。不能直接当作 v2.1.1 基准环境。
- 还存在 `unitree_rl_mjlab` 环境；未改动。
- 用户 site 中存在 numpy 1.24.4，Sim 预装 archive 中为 numpy 1.26.0；内置启动器的 metadata 当前先解析到 1.26.0。避免激活上述不匹配 conda 环境，避免把系统 pip 当作 Sim pip。
- `/home/ubuntu2004/lmy/IsaacLab-main` 存在，但不是 Git checkout（`git rev-parse` 失败），不可用作准确 tag 来源。用户的 4.2 资产目录及其他项目已保留。

## 显存门槛

8 GiB 物理显存中初始约 1.8 GiB 已被桌面使用。本次按顺序完成 GPU gate、1 环境 reset/step、4 环境短 PPO，然后验证 16 环境 reset/step 与 1 个 PPO epoch。4 环境 3 epoch 的整卡采样最大值为 5955 MiB，16 环境 1 epoch 为 5659 MiB。两次运行时长和桌面占用不同，不能据此拟合环境数与显存的缩放关系；未测量 256/512 个环境，也未验证长时运行容量。

## 官方来源与检索限制

- [Isaac Lab v2.1.1 二进制安装说明](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/binaries_installation.html)
- [Isaac Lab v2.1.1 Python 安装说明](https://isaac-sim.github.io/IsaacLab/v2.1.1/source/setup/installation/pip_installation.html)
- [Isaac Lab v2.1.1 RL 依赖源代码](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_rl/setup.py#L19-L45)
- [NVIDIA Omniverse Linux 故障排查](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html)：NVML 的 GPU 枚举不代替 Vulkan 枚举，Vulkan 设备仍需实测。
- NVIDIA 的 `/4.5.0/installation/requirements.html` 和 `/4.5.0/installation/index.html` 在本次浏览时返回 HTTP 404。没有把当前最新版或旧 4.2 文档的最低要求冒充成 4.5 版本要求。

静态审计原始日志见 [environment_audit.txt](../evidence/2026-10-23/environment_audit.txt)；GPU 探针见 [console](../evidence/2026-10-23/gpu_gate_retry/console.txt)、[显存采样](../evidence/2026-10-23/gpu_gate_retry/gpu.csv) 与 [结果](../evidence/2026-10-23/gpu_gate_retry/result.json)。


## GPU 仿真门槛：实测通过

2026-10-23 11:07:11（Asia/Seoul；日志 UTC 02:07:11）启动独立 Sim 探针，正常退出，返回码 0，未超时，总墙钟 151.479 秒。日志 GPU 表明确显示 `Graphics API: Vulkan`，活动 GPU 为 NVIDIA GeForce RTX 3070，驱动 580.178.04。

探针启用 GPU dynamics 与 GPU broadphase，创建无远程资产依赖的 0.1 m 动态立方体，使用 PhysX torch rigid-body view，从 z=1 m 下落并步进 240 次。输出：

```json
{"cuda_available": true, "torch_version": "2.5.1+cu118", "gpu": "NVIDIA GeForce RTX 3070", "tensor_device": "cuda:0", "gpu_dynamics": true, "start_z": 1.0, "end_z": 0.049999989569187164, "finite": true, "steps": 240, "passed": true}
```

因此本终端实际通过 Vulkan 图形设备初始化、GPU dynamics 配置、CUDA 物理张量读取与落体接触步进。它是后续 Factory smoke 的环境门槛，不能据此推导 PPO 更新已经通过或大规模训练容量。

`nvidia-smi` 约每秒采样的整卡显存使用最大值为 **3803 MiB**。该值包含桌面和其他现有应用，是采样最大值，不能当作进程独占显存或精确瞬时峰值。

### 两个执行环境必须分别记录

| 用途 | Python | Torch | CUDA 编译版本 / 框架 |
| --- | --- | --- | --- |
| 独立 Sim GPU 门槛 | Sim 内置 3.10.15 | 2.5.1+cu118 | cu118；未加载训练框架 |
| 隔离学习环境 `work/envs/factory-g01` | 3.10.12 | 2.7.0+cu128 | cu128；RL-Games 1.6.1，numpy 1.26.4，gymnasium 1.2.0 |

学习环境的 Isaac Lab editable 包版本元数据为 0.41.3，但所对应的项目发布 tag 是 **v2.1.1**，以 Git SHA `90b79bb2d44feb8d833f260f2bf37da3487180ba` 为准。包内部版本号与项目发布 tag 不应混淆。

曾经仅前置 `PYTHONPATH` 时，Kit 启动后仍优先加载 Sim archive 中的 Torch 2.5.1 / numpy 1.26.0 / gymnasium 0.29.1。修复后的 `scripts/python_entry.py` 在 AppLauncher 前预加载目标包，并在启动后核对相同模块身份、版本与路径。最终采用的 Factory 与 PPO 日志均含 `PINNED_PACKAGES` 和 `PACKAGES_AFTER_APP_LAUNCH`，确认实际使用 Torch **2.7.0+cu128**、torchvision **0.22.0+cu128**、numpy **1.26.4**、gymnasium **1.2.0**。见 [4 环境训练运行时证据](../evidence/2026-10-23/train_peg4_pinned/console.txt)。此前被 archive 遮蔽的运行保留为排错证据，不计入目标学习环境的最终通过结果。

## 隔离学习环境：最终实测结果

下表均为正常返回码 0、未超时的运行；内置 Sim gate 与隔离学习环境明确分开。显存列为约 1 Hz 采样最大值。

| 测试 | 内容 | 整卡 / 测试进程组 MiB | 证据 |
| --- | --- | --- | --- |
| 内置 Sim gate | 240 个物理 step | 3803 / 未采集 | [结果](../evidence/2026-10-23/gpu_gate_retry/result.json) |
| pinned PegInsert 1 环境 | 160 个 agent step，1 次自动 reset | 5131 / 3470 | [结果](../evidence/2026-10-23/peg_probe_1_pinned/result.json) |
| pinned PegInsert 16 环境 | 160 个 agent step，16 次环境 reset | 5353 / 3470 | [结果](../evidence/2026-10-23/peg_probe_16_pinned/result.json) |
| pinned PPO 4 环境 | 3 epoch，1536 个样本 frame | 5955 / 4238 | [结果](../evidence/2026-10-23/train_peg4_pinned/result.json) |
| pinned PPO 16 环境 | 1 epoch，2048 个样本 frame | 5659 / 4042 | [结果](../evidence/2026-10-23/train_peg16_pinned/result.json) |
| pinned checkpoint 恢复 | epoch 3 → 4，frame 1536 → 2048 | 5626 / 3842 | [结果](../evidence/2026-10-23/resume_peg4_pinned_compat/result.json) |

1 与 16 环境探针的 actor / critic 维度分别为 19 / 43，观测在 CUDA 上；有限值检查通过，并在 agent step 索引 149 触发官方超时 reset。[1 环境日志](../evidence/2026-10-23/peg_probe_1_pinned/console.txt)、[16 环境日志](../evidence/2026-10-23/peg_probe_16_pinned/console.txt)。

4 环境训练 checkpoint 的 22 个 Adam state 均有非零一阶矩，epoch 2 → 3 时 optimizer step 7 → 11，actor 22 个参数张量与 critic 18 个参数张量发生变化；全部 tensor 有限。[checkpoint 对比](../evidence/2026-10-23/checkpoint_peg4_pinned.json)。16 环境 1 epoch 的 22 个 Adam state 也有非零更新，step=15，tensor 有限。[checkpoint 检查](../evidence/2026-10-23/checkpoint_peg16_pinned.json)。这些是实际优化证据，不代表已学会装配。

### 已解决的 checkpoint 加载兼容问题

目标 Torch 2.7 环境第一次恢复失败：PyTorch 从 2.6 起默认 `torch.load(weights_only=True)`，RL-Games 1.6.1 的加载函数没有传入该参数，而本地 checkpoint 含 NumPy 对象。[失败日志](../evidence/2026-10-23/resume_peg4_pinned/console.txt)。

仅对本项目自己生成、已记录 SHA-256 的本地 checkpoint，在恢复子进程设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，成功恢复到 epoch 4；optimizer step 11 → 15，参数继续变化，tensor 有限。[恢复日志](../evidence/2026-10-23/resume_peg4_pinned_compat/console.txt)、[恢复后检查](../evidence/2026-10-23/checkpoint_resume_peg4_pinned.json)。这是一项显式加载兼容设置，未修改 RL-Games 或 PPO 源码；不用于来源不明的 checkpoint，也不写入全局 shell 配置。
