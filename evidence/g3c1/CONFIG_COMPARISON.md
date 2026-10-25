# G3C-1 配置差异（冻结于训练前）

G3C-1 复现 G2 的 PegInsert PPO 训练，唯一科学配置差异是训练 seed：`0 -> 1`。

| 项目 | G2 | G3C-1 | 分类 |
|---|---|---|---|
| Task | Isaac-Factory-PegInsert-Direct-v0 | 相同 | 科学不变 |
| Isaac Lab | v2.1.1 / `90b79bb2d44feb8d833f260f2bf37da3487180ba` | 相同 | 科学不变 |
| PPO / architecture / horizon | 官方 RL-Games PPO，horizon 128 | 相同 | 科学不变 |
| Environments | 64 | 64 | 科学不变 |
| Training seed | 0 | **1** | 唯一科学差异 |
| Max epochs | 200 | 200 | 科学不变 |
| Planned transitions | 1,638,400 | 1,638,400 | 科学不变 |
| Checkpoint restore | false / no CLI checkpoint | false / no CLI checkpoint | 科学不变 |
| `save_frequency` | 50 | 50 | 允许的工程显式化 |
| Experiment name/path | `g2_peginsert_64env_seed0_baseline_rerun` | `g3c_peginsert_64env_seed1_baseline` | 允许的工程隔离 |
| Supervision/logging | G2 observer/monitor | G3A guard, supervisor, observer, finalizer | 允许的工程增强 |

禁止改变 PPO、奖励、观测、动作、控制器、物理、随机化、成功判据或通过 checkpoint 选择结果。若训练提前停止，保留目录并标记 incomplete；不恢复、不重启、不拼接。
