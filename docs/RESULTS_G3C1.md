# G3C-1 PegInsert 独立训练 seed 1 结果

日期：2026-10-25
状态：**工程 INCOMPLETE；科学结果 NOT ASSESSED**
起点：G3B commit `1a605d6abec4752d17948c96ac18f103bf159ffa`
预注册 commit：`308a976556b5b9deac87c0bb00d92233b9c4df54`

## 执行结果

G3C-1 按冻结 manifest 启动了一个全新 PPO 训练：Isaac-Factory-PegInsert-Direct-v0、64 environments、training seed 1、200 epochs、horizon 128，计划 1,638,400 transitions；没有 checkpoint restore。runtime audit 已确认 64 actors、horizon 128、observation/state/action 为 19/43/6，mixed precision 开启，且 seed=1。

训练在第一个 epoch 完成前被 G3A 独立 guard 按预注册规则停止。NVML free memory 连续 3 个有效样本低于 1024 MiB，最低 917.9375 MiB，峰值 used 6899.6875 MiB；采样 441 次、没有 monitoring gap。guard `stop_reason=gpu_headroom_floor`，随后 supervisor 终止 child。监督区间约 46.256 s，child 的原始 wait return code 为 0，但 supervisor termination reason 是 `guard_missing_or_stale`，并记录了 SIGTERM；因此 return code 不能把这次运行改称为正常完整训练。

没有完成 epoch，没有 optimizer step，没有 AMP skip，没有 PPO loss/reward epoch 记录，没有 checkpoint，TensorBoard event 文件为 0 bytes，finalizer 结果文件不存在。训练审计没有 instrumentation error，但“数值稳定”只能报告初始化阶段，不能外推到 PPO 更新或收敛。

## 评估与科学结论

epoch 50、100、150、200 均没有可用 checkpoint，因此按照预注册协议没有启动任何评估，也没有成功率分母、first-success timing 或 seed-0/seed-1 科学性能比较。G2 seed 0 的 280/288（97.222%）保持为历史主结果，不能与这个 0-transition incomplete run 合并或比较为训练种子结果。

因此 G3C-1 不判定“学会”或“未学会”，也不报告伪造的低成功率。当前只证明 seed 1 的训练资源安全闸门生效，并保留了失败原因和全部小型回执。

## 证据与复现身份

- [`manifest_frozen.json`](../evidence/g3c1/manifest_frozen.json)：训练前冻结的 task、seed、配置、source hashes、命令和 guard policy。
- [`CONFIG_COMPARISON.md`](../evidence/g3c1/CONFIG_COMPARISON.md)：与 G2 的唯一科学差异为 training seed 0→1。
- [`preflight.json`](../evidence/g3c1/preflight.json)：版本、GPU、磁盘、lock、CPU tests 和 fresh paths 检查。
- [`run_receipt.json`](../evidence/g3c1/run_receipt.json)：child/guard/审计/无 checkpoint 状态和外部原始文件 SHA256。
- [`gpu_guard_headroom.png`](../evidence/g3c1/plots/gpu_guard_headroom.png)：触发 guard 的 NVML free-memory 曲线。

完整 guard CSV、supervisor console 和 training audit 留在仓库外的 `work/logs/g3c1/g3c_peginsert_64env_seed1_baseline/`，没有把原始大日志或模型提交到 Git。预存 G3V 未跟踪文件没有进入本提交。

## G2 对照与 G3C-2 建议

G2 seed 0 完成 1,638,400 transitions，最终 280/288；G3C-1 seed 1 完成 0 transitions。两者不能构成独立训练种子科学比较。G2 的历史 exit-code/显存监控限制仍单独保留，G3C-1 的 guard coverage 和停止决定则完整可审计。

G3C-2 应先处理 RTX 3070 当前共享显存余量与 guard 触发条件，重新做资源 preflight，并以新的 run name/manifest 从 fresh seed 1 开始；不得从本目录恢复或拼接。只有 seed 1 完成后，才应决定是否开展 seed 2。当前不启动 seed 2、contact-sensor、GearMesh 或 NutThread。
