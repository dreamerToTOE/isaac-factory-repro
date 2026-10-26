# G3C-R5：32 环境独立训练种子 1 完整报告

工程完成状态：**INCOMPLETE**；科学评估状态：**NOT_ASSESSED_AS_COMPLETED_400_EPOCH_RUN**。
本次从头训练使用 training seed **1**；独立评估使用 seeds **1001、1002、1003**。评估种子控制评估初始条件，不能当作额外的训练重复。

经连续 epoch 审计验证完成 **400/400 epochs**、**1,638,400/1,638,400 transitions**。最终 checkpoint 的 ever-success 为 **未评估**，terminal-success 为 **未评估**。

## 配置冻结与历史保留

起始提交：`04af151c7b78c0a3fe4adaf7a646ff151feecc86`。以完整 G3C-R4 的 32-env training seed 0 作为唯一直接对照。
科学配置固定为原始 Factory PegInsert、32 个环境、400 epochs、horizon 128、官方 PPO 与 LSTM。训练命令仅把 seed 0 改为 seed 1，并更换实验名称和输出目录。奖励、物理、成功判据、观测/动作空间、网络、PPO 超参数、checkpoint 周期与评估协议保持相同。
运行名称：`g3cr5_peginsert_32env_seed1_400ep_20261026t015040z`。checkpoint 每 25 epochs 保存；本次未载入 seed-0 checkpoint，不续训、不拼接旧训练。
R4 源文件、报告、checkpoint 和失败记录保留；历史 64-env G2、G3B 诊断 episode 及不完整训练均未进入本报告的 32-env 统计。保留校验与冻结 SHA 记录见同目录的 manifest、preflight 与 preservation evidence。

## 监督与退出证据

按照用户在 G3C-R5 中继续给予的授权，本次沿用显式 `user-authorized-no-gpu-guard` 模式。独立 GPU guard 不运行；缺失 GPU telemetry 或采样间隙不触发训练停止。CUDA 正常异常处理保持不变，未更换驱动、未终止无关用户进程。GPU 连续利用率/显存覆盖未测量，因此没有连续覆盖或真实显存峰值声明。
durable supervisor 保持 GPU0 独占锁、console、实际 child wait 与退出码记录。systemd `Restart=no`；TensorBoard 在官方训练退出路径执行 flush/close。控制器仅在完整 400-epoch 训练通过验收后，依次评估 4 个 checkpoint；不自动启动、恢复、终止或重试训练。
实际 child wait returncode：`0`；child exit code：`0`；child signal：`None`；退出原因：`child_exit`。
supervisor error：`None`；stop signals：`[]`；supervisor 收到的 signals：`[]`。

## 训练耗时与吞吐

| 指标 | training seed 0（R4） | training seed 1（R5） |
|---|---:|---:|
| Supervisor 训练用时（s） | 14,579.912 | 14,248.196 |
| 官方 PPO 累计时间（s） | 14,460.679 | 14,047.519 |
| 官方 PPO aggregate transitions/s | 113.300 | 116.633 |
| Epochs 11 起稳定段 transitions/s | 113.206 | 116.379 |
| 完成 transitions / 全程 wall s | 112.374 | 114.990 |

官方 PPO 吞吐的分子和分母来自同一连续完成 epoch 前缀。Supervisor 用时包含进程启动、训练与实际 child wait/finalization；它与官方 PPO rollout/optimization 累计时间不同。若训练不完整，未完成 rollout 的交互数量未知，未计入完成 transition 数。

## Loss、AMP 与优化器

观测 epoch loss 数组全部有限：`True`；记录的 loss/optimization scalars 全部有限：`True`。
实际 actor optimizer attempts：**12800**；实际 actor Adam steps：**12790**；实际 central-value Adam steps：**12800**。
AMP skip 事件：**10**；attempts−actor steps：`10`；与事件数一致：`True`。
完成 epoch 的 AMP scale：首次 `{'epoch': 1, 'scale': 65536.0}`；末次 `{'epoch': 400, 'scale': 2048.0}`；统计 `{'max': 65536.0, 'mean': 20142.08, 'median': 16384.0, 'min': 2048.0, 'n': 400, 'q25': 8192.0, 'q75': 32768.0}`。
计数来自只读训练 observer 的实际事件，未用预期 minibatch 数补齐。顶层优化器计数可能包含不完整 epoch 中已记录的工作；完成 epoch 的累计值单独保留在机器证据。未直接测量逐梯度有限性。有限 loss/参数只支持数值稳定性描述，不能证明收敛、任务成功或所有梯度有限。

| TensorBoard 标量 | n | 最小值 | 中位数 | 最大值 |
|---|---:|---:|---:|---:|
| `info/cval_lr` | 400 | 0.000100 | 0.000100 | 0.000100 |
| `info/kl` | 400 | 0.002249 | 0.009574 | 0.090515 |
| `info/last_lr` | 400 | 0.000002 | 0.000013 | 0.000100 |
| `losses/a_loss` | 400 | -0.015348 | -0.009038 | 0.000630 |
| `losses/bounds_loss` | 400 | 0.000000 | 0.128739 | 0.265419 |
| `losses/c_loss` | 400 | 0.001079 | 0.010990 | 1.493087 |
| `losses/cval_loss` | 400 | 0.000174 | 0.002854 | 1.486301 |
| `losses/entropy` | 400 | -2.305771 | 2.611228 | 8.496508 |

## Checkpoint 完整性

| Epoch | Frame | 大小（bytes） | CPU 数值有限 | SHA-256 |
|---|---:|---:|---|---|
| 100 | 409600 | 212889115 | `True` | `d152a4fadf28b06807d9396d8979602984f4c736670122453fe2d801224a2fb6` |
| 200 | 819200 | 212889115 | `True` | `b90b4667c3f205c67d79718a4f5556e7cc0a95ae0086246c7e9ece75e97fd1f3` |
| 300 | 1228800 | 212889115 | `True` | `2292bbac234384ebd48246e85797b0a08a7887db93e25c3cbbffe6e3e036826d` |
| 400 | 1638400 | 212889335 | `True` | `2b266c9b64574dc075634e10e3c6b9856899870cbabe54b157a70b7a8c9de54e` |

只对新产生且受信任的本地 checkpoint 在 CPU 上解包，核验 epoch/frame、actor/central-value 模型、Adam 状态与数值有限性；SHA 用于绑定独立评估所使用的同一策略。Epoch 400 按审计保存顺序选择官方最终 checkpoint，周期保存和最终保存均保留。

## 独立评估及 training seed 0 对照

评估采用确定性策略均值，恢复 checkpoint observation RMS，推理模式运行，LSTM 在 episode 边界重置。官方成功判据在自动 reset 前捕获。每个 checkpoint 使用相同 evaluation seeds 1001/1002/1003：epochs 100/200/300 每 seed 32 个完成 episodes、总分母 96；epoch 400 每 seed 96 个完成 episodes、总分母 288。未完成 episode 不补零，也不并入分母。

| Epoch / transitions | seed 0 ever-success | seed 1 ever-success | seed 0 terminal-success | seed 1 terminal-success | seed 1−0 ever / terminal（百分点） |
|---|---|---|---|---|---|
| 100 / 409,600 | 74/96（77.08%） | 未评估 | 74/96（77.08%） | 未评估 | 未记录 / 未记录 |
| 200 / 819,200 | 83/96（86.46%） | 未评估 | 83/96（86.46%） | 未评估 | 未记录 / 未记录 |
| 300 / 1,228,800 | 92/96（95.83%） | 未评估 | 92/96（95.83%） | 未评估 | 未记录 / 未记录 |
| 400 / 1,638,400 | 268/288（93.06%） | 未评估 | 267/288（92.71%） | 未评估 | 未记录 / 未记录 |

首次成功时间只针对 ever-successful 的已完成 episodes：

| Epoch | training seed 0（模拟时间） | training seed 1（模拟时间） |
|---|---|---|
| 100 | n=74；均值 2.651 s；中位数 2.067 s | n=0；均值 未记录 s；中位数 未记录 s |
| 200 | n=83；均值 2.545 s；中位数 2.200 s | n=0；均值 未记录 s；中位数 未记录 s |
| 300 | n=92；均值 2.585 s；中位数 2.067 s | n=0；均值 未记录 s；中位数 未记录 s |
| 400 | n=268；均值 2.411 s；中位数 2.267 s | n=0；均值 未记录 s；中位数 未记录 s |

失败 episodes 不被当作 0 秒或 timeout 秒数；因此首次成功均值/中位数会受成功样本组成影响。

## 每个 evaluation seed 的表现

| Checkpoint epoch | Evaluation seed | Training seed 0 ever / terminal | Training seed 1 ever / terminal | R5 首次成功（条件） |
|---|---:|---|---|---|
| 100 | 1001 | 25/32（78.12%） / 25/32（78.12%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 100 | 1002 | 29/32（90.62%） / 29/32（90.62%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 100 | 1003 | 20/32（62.50%） / 20/32（62.50%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 200 | 1001 | 28/32（87.50%） / 28/32（87.50%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 200 | 1002 | 31/32（96.88%） / 31/32（96.88%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 200 | 1003 | 24/32（75.00%） / 24/32（75.00%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 300 | 1001 | 32/32（100.00%） / 32/32（100.00%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 300 | 1002 | 32/32（100.00%） / 32/32（100.00%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 300 | 1003 | 28/32（87.50%） / 28/32（87.50%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 400 | 1001 | 91/96（94.79%） / 90/96（93.75%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 400 | 1002 | 91/96（94.79%） / 91/96（94.79%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |
| 400 | 1003 | 86/96（89.58%） / 86/96（89.58%） | 未评估 / 未评估 | n=0；均值 未记录 s；中位数 未记录 s |

## 解释、局限及 seed 2 建议

本次比较在相同 32-env 科学配置下，仅改变 training seed。相同 transition 预算和相同评估 seeds 提供直接描述性对照；它们不保证随机轨迹相同。两个训练策略仍不足以可靠估计训练种子总体均值、方差或失败概率。每个策略的 episode 共享策略和部分环境机制，episode 层面的 Wilson 区间仅作描述，不能解释为训练种子间的不确定性。
训练种子与评估种子分开报告；没有把历史 64-env baseline 混入 32-env 成功率，也没有把两个训练种子的 episodes 合并为单个性能分母。观察到的差异不触发任何超参数适配。
本次未达到完整训练与评估验收条件。建议先记录并解决已观察到的工程原因，保留本次全部证据；待另行授权后再用完全相同科学配置从头运行 training seed 2。不因本次表现更改超参数，不自动重试、续训或启动 seed 2。

## 验收、机器证据与可追溯性

机器证据包括 `analysis/report.json`、TensorBoard scalars/CSV/图、完整 console/audit/finalizer/supervisor status、checkpoint SHA/CPU inspection、12 次独立评估的逐 episode metrics、source freeze、配置差异和保留审计。CPU regression 与 fast-forward push 验证随最终交付记录保存。
验收失败项：`['completion controller has not confirmed training and all evaluations complete', 'evaluation epoch100/seed1001: metrics or supervisor status unavailable', 'evaluation epoch100/seed1002: metrics or supervisor status unavailable', 'evaluation epoch100/seed1003: metrics or supervisor status unavailable', 'evaluation epoch200/seed1001: metrics or supervisor status unavailable', 'evaluation epoch200/seed1002: metrics or supervisor status unavailable', 'evaluation epoch200/seed1003: metrics or supervisor status unavailable', 'evaluation epoch300/seed1001: metrics or supervisor status unavailable', 'evaluation epoch300/seed1002: metrics or supervisor status unavailable', 'evaluation epoch300/seed1003: metrics or supervisor status unavailable', 'evaluation epoch400/seed1001: metrics or supervisor status unavailable', 'evaluation epoch400/seed1002: metrics or supervisor status unavailable', 'evaluation epoch400/seed1003: metrics or supervisor status unavailable', 'one or more preregistered evaluation denominators are incomplete']`。
分析错误：`[]`；分析警告：`[]`。

| 输入 | SHA-256 |
|---|---|
| `analysis_report` | `5f049bc75df87719c2361026de5a48be5ac082b782dac78ac474ca50dd02498b` |
| `seed0_analysis_report` | `8c2797e056f34e6ec42abd80a91e73df51fd81e335dd12224ffc13fb44bad9bf` |
| `manifest` | `f3163299d669e355c401cb825661bb99705a3b10b79b1f16fed4b090bfb77103` |
| `report_generator` | `3fa77821eed0e67655e0d8ad3ea1cca91012cfb4fdbd17f5c4e3e72bf2684296` |
