# G0/G1 原始证据（2026-10-23）

每个运行目录保留完整 argv/cwd/退出状态（result.json）、console.txt 与 GPU CSV。`*_pinned` 表示修复 Kit 包遮蔽后的运行；恢复失败及兼容重试分别保留。早期未 pinned 的 Factory/PPO 实测使用 Sim 自带 Torch2.5.1，不与最终 pinned2.7 性能混合。GPU gate 本来就使用 Sim 自带解释器。

| 运行 | 状态 | 用途 |
| --- | --- | --- |
| [gear_probe_1_pinned](gear_probe_1_pinned/result.json) | 通过 | 最终验收 |
| [gpu_gate](gpu_gate/result.json) | 超时 | 诊断历史 / 失败记录 |
| [gpu_gate_retry](gpu_gate_retry/result.json) | 通过 | 最终验收 |
| [nut_probe_1_pinned](nut_probe_1_pinned/result.json) | 通过 | 最终验收 |
| [peg_probe_1](peg_probe_1/result.json) | 失败 | 诊断历史 / 失败记录 |
| [peg_probe_16](peg_probe_16/result.json) | 通过 | 诊断历史 / 失败记录 |
| [peg_probe_16_pinned](peg_probe_16_pinned/result.json) | 通过 | 最终验收 |
| [peg_probe_1_pinned](peg_probe_1_pinned/result.json) | 通过 | 最终验收 |
| [peg_probe_1_retry](peg_probe_1_retry/result.json) | 失败 | 诊断历史 / 失败记录 |
| [peg_probe_1_support_fixed](peg_probe_1_support_fixed/result.json) | 通过 | 诊断历史 / 失败记录 |
| [resume_peg4](resume_peg4/result.json) | 通过 | 诊断历史 / 失败记录 |
| [resume_peg4_pinned](resume_peg4_pinned/result.json) | 失败 | 诊断历史 / 失败记录 |
| [resume_peg4_pinned_compat](resume_peg4_pinned_compat/result.json) | 通过 | 最终验收 |
| [train_peg16_pinned](train_peg16_pinned/result.json) | 通过 | 最终验收 |
| [train_peg4](train_peg4/result.json) | 通过 | 诊断历史 / 失败记录 |
| [train_peg4_pinned](train_peg4_pinned/result.json) | 通过 | 最终验收 |

安装原始日志保留成功、flatdict 构建失败和 uv 锁告警。`environment_audit.txt` 是只读硬件/环境采集；`git_access.txt` 是不含凭证的协议诊断。

`checkpoint_*.json` 保存校验和、优化器状态、有限性及相邻模型差异；`checkpoint_locations.json` 给出当前本地大文件位置。生成时原路径为工作区根目录 `logs/`，完成实验后整目录移动到 `work/runs/logs/`，原始日志中的执行路径没有改写。大文件不提交 Git。

`events_peg4_pinned.json` 有50个有限标量标签；恢复与16环境单epoch事件文件未提供标量，相关检查返回非零，属于无数据，不能宣称 loss 有限检查通过。checkpoint张量检查独立进行。

`manifest.json` 记录官方 SHA、源树干净状态、脚本/框架源文件 SHA256。解析后的官方训练配置见 ../../configs/resolved/；env.yaml 在环境构造前保存，其中21/72维占位值会在构造时改成19/43，agent.yaml的num_actors=128也在wrapper之后才改成实际4/16；以源码执行顺序、CLI和运行形状为准。
