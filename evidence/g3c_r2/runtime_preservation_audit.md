# G3C-R2 独立只读审计

审计时间：2026-10-25T13:39:21.134472+00:00；起点 `32f784b43011470615dec251984aca26716a4b86`。

- 起点的 583 个 tracked 文件逐一按 Git blob 比较，全部字节一致；包括 G2、G3A、G3B、G3C-1、G3C-R1 报告与证据。
- 原有 5 个 G3V untracked 文件大小及 SHA256 均与 G3B 首次审计一致。
- G2 原有 6 个 checkpoint 大小与 SHA256 均匹配既有 inventory，未加载或修改 checkpoint。
- G3C-1 / G3C-R1 已登记的 16 个仓库外日志与监督记录大小及 SHA256 全匹配。
- 26 个官方/框架/监督脚本源码 SHA256 与 G3C-1 冻结记录一致。Isaac Lab HEAD `90b79bb2d44feb8d833f260f2bf37da3487180ba`，工作树干净。

## 配置与科学比较

- env.yaml 相对 G2 唯一差异为 `scene.num_envs: 64 → 32`。
- agent.yaml 相对 G2 仅 `max_epochs: 200 → 400` 与独立实验名变化；seed 仍为 0，restore=false。
- YAML 中 `num_actors=128` 与 G2 相同，因为官方脚本在 dump 后按实际环境数覆盖；运行 observer 证实 32 actors、horizon128、minibatch512、mini_epochs4、obs/state/action=19/43/6，mixed precision 开启，actor/central-value LSTM 与 G2 相同。
- 32×128×400=1,638,400 是计划量，不能称为 G2 的 seed-only replication。按相同累计 transition 分数登记的评估里程碑为 epoch100/200/300/400。

## 当前终止状态

- 10 个完成 epoch，已核实完成 epoch 的 transitions 为 40,960；终止前可能存在未记账的部分 epoch，实际全部环境交互总量未知。
- Supervisor wall=368.886014856 秒；raw wait returncode=0，同时 supervisor 明确因 `guard_missing_or_stale` 发送 SIGTERM，故工程状态必须为 INCOMPLETE。
- Guard stop=`monitoring_gap`：0.512318793 秒超过既定 0.5 秒上限；low-memory stop=false，min free=1408.6875 MiB，peak used=6408.9375 MiB。
- 不能将这次结果写成 32-env 显存容量失败；监控缺口具体成因未知。
- Actor 实际更新319/尝试320，central value320；epoch1有1次 AMP skip。10个完成 epoch 的 loss arrays 均有限，但不能据此声称所有原始梯度均有限。
- 终止时无 TensorBoard finalizer receipt；计划评估里程碑未到达，不应将 epoch10 best checkpoint 作为完成 baseline 评估。
- 仅能给出科学 NOT ASSESSED；不自动重启、恢复、拼接，也不启动 seed1/2。

完整逐文件字节数、SHA256、配置键差异、运行结构与终止字段见同目录 `g3cr2_runtime_preservation_audit.json`。
