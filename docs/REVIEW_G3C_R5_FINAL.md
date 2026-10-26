# G3C-R5：32 环境独立训练种子 1 完整报告

工程完成状态：**COMPLETE**；科学评估状态：**ASSESSED_DESCRIPTIVELY**。
本次从头训练使用 training seed **1**；独立评估使用 seeds **1001、1002、1003**。评估种子控制评估初始条件，不能当作额外的训练重复。

经连续 epoch 审计验证完成 **400/400 epochs**、**1,638,400/1,638,400 transitions**。最终 checkpoint 的 ever-success 为 **278/288（96.53%）**，terminal-success 为 **278/288（96.53%）**。

## 评估基础设施故障与独立恢复的来源

这份最终报告保留并区分最初的评估控制器与随后人工选择的仅评估恢复。原始训练没有重新运行、恢复或拼接；完成的 training seed 1 checkpoint 及其 SHA 保持不变。
首个评估进程虽然 child wait 为 0，但没有产生 metrics，不能认定为科学评估完成。原始控制器因此记录为 INCOMPLETE。Kit/OmniClient 记录在解析官方 S3 的 `default_environment.usd` 时出现 libcurl SSL connect error 35；该故障不属于训练失败或 CUDA OOM。
随后在新的 completion/evaluation 输出目录中，显式执行仅评估恢复。训练 checkpoint、环境、task/reward/success、确定性推理、评估种子与 episode 分母沿用冻结协议。最初的失败 console、controller status、部分分析和 result 仍保留，不被最终结果覆盖；不存在选择表现更好的训练重跑。
最终工程/科学状态描述本次训练和登记的评估恢复所获得的完整证据；不回填或改写原始失败控制器状态。具体恢复 manifest、原始来源与身份由下列机器证据保留：

恢复身份、4 个 checkpoint 的完整身份、原始失败状态 SHA 和新输出路径见 [evaluation_recovery_provenance.json](../evidence/g3c_r5/evaluation_recovery_provenance.json)；最终评估使用 [manifest_evaluation_recovery.json](../evidence/g3c_r5/manifest_evaluation_recovery.json)。机器证据完整保留，以下正文集中报告实验结果。

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

耗时与吞吐是在同一主机上本次运行的观测值；没有对系统负载进行受控实验，不能把吞吐差异归因为训练 seed。

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
| 100 / 409,600 | 74/96（77.08%） | 79/96（82.29%） | 74/96（77.08%） | 79/96（82.29%） | 5.208 / 5.208 |
| 200 / 819,200 | 83/96（86.46%） | 86/96（89.58%） | 83/96（86.46%） | 86/96（89.58%） | 3.125 / 3.125 |
| 300 / 1,228,800 | 92/96（95.83%） | 91/96（94.79%） | 92/96（95.83%） | 91/96（94.79%） | -1.042 / -1.042 |
| 400 / 1,638,400 | 268/288（93.06%） | 278/288（96.53%） | 267/288（92.71%） | 278/288（96.53%） | 3.472 / 3.819 |

首次成功时间只针对 ever-successful 的已完成 episodes：

| Epoch | training seed 0（模拟时间） | training seed 1（模拟时间） |
|---|---|---|
| 100 | n=74；均值 2.651 s；中位数 2.067 s | n=79；均值 2.666 s；中位数 2.133 s |
| 200 | n=83；均值 2.545 s；中位数 2.200 s | n=86；均值 2.605 s；中位数 2.067 s |
| 300 | n=92；均值 2.585 s；中位数 2.067 s | n=91；均值 2.525 s；中位数 2.133 s |
| 400 | n=268；均值 2.411 s；中位数 2.267 s | n=278；均值 2.401 s；中位数 2.067 s |

失败 episodes 不被当作 0 秒或 timeout 秒数；因此首次成功均值/中位数会受成功样本组成影响。

## 每个 evaluation seed 的表现

| Checkpoint epoch | Evaluation seed | Training seed 0 ever / terminal | Training seed 1 ever / terminal | R5 首次成功（条件） |
|---|---:|---|---|---|
| 100 | 1001 | 25/32（78.12%） / 25/32（78.12%） | 24/32（75.00%） / 24/32（75.00%） | n=24；均值 2.833 s；中位数 2.533 s |
| 100 | 1002 | 29/32（90.62%） / 29/32（90.62%） | 30/32（93.75%） / 30/32（93.75%） | n=30；均值 2.453 s；中位数 1.933 s |
| 100 | 1003 | 20/32（62.50%） / 20/32（62.50%） | 25/32（78.12%） / 25/32（78.12%） | n=25；均值 2.760 s；中位数 1.933 s |
| 200 | 1001 | 28/32（87.50%） / 28/32（87.50%） | 27/32（84.38%） / 27/32（84.38%） | n=27；均值 2.758 s；中位数 2.067 s |
| 200 | 1002 | 31/32（96.88%） / 31/32（96.88%） | 30/32（93.75%） / 30/32（93.75%） | n=30；均值 2.300 s；中位数 2.000 s |
| 200 | 1003 | 24/32（75.00%） / 24/32（75.00%） | 29/32（90.62%） / 29/32（90.62%） | n=29；均值 2.777 s；中位数 2.267 s |
| 300 | 1001 | 32/32（100.00%） / 32/32（100.00%） | 31/32（96.88%） / 31/32（96.88%） | n=31；均值 2.748 s；中位数 2.267 s |
| 300 | 1002 | 32/32（100.00%） / 32/32（100.00%） | 31/32（96.88%） / 31/32（96.88%） | n=31；均值 2.157 s；中位数 2.067 s |
| 300 | 1003 | 28/32（87.50%） / 28/32（87.50%） | 29/32（90.62%） / 29/32（90.62%） | n=29；均值 2.680 s；中位数 2.133 s |
| 400 | 1001 | 91/96（94.79%） / 90/96（93.75%） | 93/96（96.88%） / 93/96（96.88%） | n=93；均值 2.315 s；中位数 2.133 s |
| 400 | 1002 | 91/96（94.79%） / 91/96（94.79%） | 95/96（98.96%） / 95/96（98.96%） | n=95；均值 2.302 s；中位数 2.000 s |
| 400 | 1003 | 86/96（89.58%） / 86/96（89.58%） | 90/96（93.75%） / 90/96（93.75%） | n=90；均值 2.594 s；中位数 2.133 s |

## 解释、局限及 seed 2 建议

本次比较在相同 32-env 科学配置下，仅改变 training seed。相同 transition 预算和相同评估 seeds 提供直接描述性对照；它们不保证随机轨迹相同。两个训练策略仍不足以可靠估计训练种子总体均值、方差或失败概率。每个策略的 episode 共享策略和部分环境机制，episode 层面的 Wilson 区间仅作描述，不能解释为训练种子间的不确定性。
训练种子与评估种子分开报告；没有把历史 64-env baseline 混入 32-env 成功率，也没有把两个训练种子的 episodes 合并为单个性能分母。观察到的差异不触发任何超参数适配。
本次最终 ever-success 相对 seed 0 的观察差值为 +3.47 个百分点。建议在独立授权后继续一个完全相同配置的全新 training seed 2，以增加训练随机性证据；保持 32 env、400 epochs、horizon 128、同一评估协议和全部超参数。两个训练种子只能给出描述性重现性证据，尚不足以稳定估计训练种子总体分布。本任务到 seed 1 为止，不自动启动 seed 2。

## 验收、机器证据与可追溯性

最终机器结果见 [result_evaluated.json](../evidence/g3c_r5/result_evaluated.json)，分析、TensorBoard scalars/CSV/图见 [analysis_evaluated/report.json](../evidence/g3c_r5/analysis_evaluated/report.json)，完整 console/audit/finalizer/supervisor status、checkpoint SHA/CPU inspection 及 12 次独立评估的逐 episode metrics 位于 `evidence/g3c_r5/raw_collect_evaluated/`。大型 checkpoint 与原始 TensorBoard event 文件继续保留在 manifest 指定的本地 run 目录，Git 中保存身份、大小和 SHA，不把这些大文件打包进代码仓库。

最终 CPU 回归 **220 passed（14.15 s）**，实际命令、stdout/stderr、退出码见 [artifact_execution_evaluated/status.json](../evidence/g3c_r5/artifact_execution_evaluated/status.json)，JUnit 见 [final_cpu_regression_evaluated.xml](../evidence/g3c_r5/final_cpu_regression_evaluated.xml)。独立科学审计 **139/139 通过**，见 [final_scientific_audit.md](../evidence/g3c_r5/final_scientific_audit.md)。三张科学图均完成人工可读性检查。

运行结束、整合远端新文档之前的保留审计全部通过：起始提交的 tracked files **938/938**、R4 evidence **285/285**、R4 raw artifacts **277/277**、更早历史 artifacts **347/347**、预先存在的 G3V 文件 **5/5**、冻结源文件 **30/30**；分组存在交集，不能相加解释为独立文件总数。见 [postrun_preservation_audit.json](../evidence/g3c_r5/raw_collect_evaluated/postrun_preservation_audit.json)。原始起始提交保留在 Git 历史中；最终仅整合远端工程文档更新，未改变冻结科学源文件。最终提交及 fast-forward push 的远端 SHA 验证随交付 receipt 保存。
验收失败项：`[]`。
分析错误：`[]`；分析警告：`[]`。

| 输入 | SHA-256 |
|---|---|
| `analysis_report` | `72ae3575a493deb43d25a621fd699b78a3a13f2225fbd10fb71f8a99bf496733` |
| `seed0_analysis_report` | `8c2797e056f34e6ec42abd80a91e73df51fd81e335dd12224ffc13fb44bad9bf` |
| `manifest` | `826fc269feab1f7eb1ed9a481a265bd98d0a4017637ce0756ba2db4d6bd13eb1` |
| `report_generator` | `3cbd91563904266f282ae4143a4dc954636f4ec0f470763bc0767bc99ca2caf2` |
| `evaluation_recovery_provenance` | `b393ae5b38202724cba93acd99c0b7e120161fbf62a29f19f937b12dfd10ea71` |
