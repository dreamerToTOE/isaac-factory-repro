# G3C-R4：32 环境简化训练结果

本报告对应起点 `3bdbadd6b6289c778ccf805d8cd8009e3a9dc7b1`，结果提交见仓库历史。训练使用全新实验名 `g3cr4_peginsert_32env_seed0_400ep_20261025t140016z`，没有恢复、重试或拼接旧训练。

## 结果

- 训练协议：PegInsert，32 environments，training seed 0，400 epochs，horizon 128。
- 完成预算：400/400 epochs，1,638,400/1,638,400 transitions。
- 持续时间：监督器 wall time 14,579.912 s（4 h 2 min 59.912 s）；官方 PPO epoch time 合计 14,460.679 s。
- 退出：child wait `0`，`termination_reason=child_exit`，无 signal、无 supervisor stop signal、无错误。
- 速度：官方 PPO aggregate 113.300 transitions/s；按完整 supervisor wall time 计算 112.374 transitions/s。两者分母不同，前者排除启动/收尾与 I/O，后者包含整个监督区间。
- loss 稳定性：400 个 epoch 的观测 loss arrays 和 TensorBoard loss/optimization scalars 均有限。实际 Adam steps 为 actor 12,790、central value 12,800；记录 10 次 AMP skip。没有逐梯度有限性测量，因此这不是“所有梯度有限”或收敛证明。
- TensorBoard finalizer 成功 flush/close；严格 400-epoch tail 检查通过。

四个里程碑 checkpoint 均通过 CPU 检查，actor/central model、optimizer state 和数值有限性均通过：

| epoch | frame | SHA256 |
|---:|---:|---|
| 100 | 409,600 | `9ba32c0ae42b1d732beed25bac08083864a86c7ae8d5e2916928bc485b8f898a` |
| 200 | 819,200 | `2166165f8b9f75b84ed0db53a56bfe8c704904d2816dd68237d44ec2c7d43236` |
| 300 | 1,228,800 | `c0bb3dbadb406b4bae8a9e7d5727cfb2f868265fc10588710611776f01354272` |
| 400 | 1,638,400 | `da72a148ebe2b1e2f11f79c4316c24f515510780a3975e2304a41077436b3c36` |

## 独立评估

评估严格按预注册协议顺序执行，使用 32 个环境、evaluation seeds 1001/1002/1003、恢复 checkpoint normalization state，并在 episode 边界重置 recurrent state。epoch 100/200/300 各为 96 episodes，epoch 400 为 288 episodes；缺失 episode 不会被填成失败。

| checkpoint | episodes | ever-success | terminal-success | first-success timing（成功 episode 条件统计） |
|---:|---:|---:|---:|---|
| 100 | 96 | 74/96 = 77.08% | 74/96 = 77.08% | n=74，mean 2.651 s，median 2.067 s |
| 200 | 96 | 83/96 = 86.46% | 83/96 = 86.46% | n=83，mean 2.545 s，median 2.200 s |
| 300 | 96 | 92/96 = 95.83% | 92/96 = 95.83% | n=92，mean 2.585 s，median 2.067 s |
| 400 | 288 | 268/288 = 93.06% | 267/288 = 92.71% | n=268，mean 2.411 s，median 2.267 s |

epoch 400 有 1 个 transient-only episode：ever-success 为真但 terminal-success 为假。首次成功时间只在 ever-success episodes 上计算；失败 episodes 没有被赋予零秒或 timeout 时间。

## 与历史 64-env G2 baseline 的比较

相同 transition budget 下，R4 epoch 100/200/300/400 分别对应历史 G2 epoch 50/100/150/200。历史 G2 的 success 结果为 69/96（71.88%）、81/96（84.38%）、92/96（95.83%）和 280/288（97.22%）。因此描述性 ever-success 差异为 +5.21、+2.08、0.00、−4.17 percentage points；R4 epoch 400 terminal-success 92.71%，G2 对应 97.22%。

这不是环境数量的因果估计。32-env rollout 每轮 4,096 transitions，G2 的 64-env rollout 每轮 8,192；在 minibatch 512 和四个 mini-epochs 不变时，R4 每轮名义 actor update 数为 32，G2 为 64。总 transitions 相同但 on-policy grouping、trajectory collection、RNG/reset stream 和 update cadence 不同；每个协议也只有一个 training seed。evaluation seeds 是独立初始条件，不是额外 training seeds。

## 监督策略与保留

本轮按用户明确授权启用 `user-authorized-no-gpu-guard`：没有独立 GPU guard，GPU telemetry 缺失或 gap 不会停止 child；正常 CUDA error handling 保持不变。进程独占锁、systemd `Restart=no`、真实 child wait、console、TensorBoard finalizer 和 periodic checkpoint 保存均保留。没有修改 driver、无关进程、官方 PPO、reward、physics、success predicate、observation/action space 或 network architecture。

运行前 CPU 回归为 164 passed；完整 post-run 复核和 12 次评估 gate 均通过。645 个起始 tracked files、28 个冻结源指纹、40 个历史 checkpoint、302 个历史 raw evidence 文件以及 G3V 未跟踪文件均保留并通过 SHA256 审计。R4 新证据独立存放，未覆盖或混入历史 R2/G3B 分母。GPU monitoring coverage 没有被声称测量；该授权模式只保证缺少遥测不会成为终止条件。

完整机器可读数据位于 [`evidence/g3c_r4/analysis/report.json`](../evidence/g3c_r4/analysis/report.json)，摘要位于 [`evidence/g3c_r4/result.json`](../evidence/g3c_r4/result.json)。
