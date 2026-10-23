# G0/G1 结果报告

日期：2026-10-23（Asia/Seoul）。本阶段已完成；停止于 G0/G1，等待审阅。没有启动正式长训练，也没有修改官方 PPO、奖励、成功阈值、控制器或 PhysX 配置。GitHub 认证按用户要求作为独立非阻塞问题处理，本次交付本地提交。

## 1. 环境与兼容性

Ubuntu22.04.5 / kernel6.8.0-138、i7-10700F（8核16线程）、RAM约46GiB（初始可用37GiB）、磁盘初始可用398GiB。RTX3070为8192MiB，驱动580.178.04，初始桌面等占约1.8GiB。具体采集见 [REQUIREMENTS](REQUIREMENTS.md) 与 [原始审计](../evidence/2026-10-23/environment_audit.txt)。

现有 Sim `/home/ubuntu2004/isaacsim-4.5.0` 的原始 VERSION 为 `4.5.0-rc.36+release.19112.f59b3005.gl`。GPU gate 日志明确 Vulkan 活动RTX3070；GPU dynamics=true、rigid-body状态张量cuda:0，0.1m方块经过240物理步从z=1m落稳到约0.05m。不存在以CUDA枚举代替GPU仿真的情况；未更改设备权限或绕过限制。

独立官方 Isaac Lab `v2.1.1` / `90b79bb2d44feb8d833f260f2bf37da3487180ba` 位于 `work/upstream/IsaacLab`，结束时源码树干净。已有v1.2.0工作区及其他用户项目、Sim安装均保留。隔离学习环境在 `work/envs/factory-g01`：Python3.10.12、Torch2.7.0+cu128、torchvision0.22.0+cu128、NumPy1.26.4、Gymnasium1.2.0、RL-Games1.6.1；启动后实际模块路径已核对。

## 2. 动态测试与显存

以下最终验收运行全部返回0。全卡与测试进程组显存为约1Hz采样最大值，包含启动/初始化/运行/保存窗口；采样可能漏掉瞬时峰值。全卡列包含桌面等其他进程。不同运行长度/桌面占用不同，16环境峰值较低不表示其长期占用一定更小。

| 测试（链接含精确命令） | 状态 | 墙钟秒 | 全卡MiB | 测试进程MiB | PPO总FPS |
| --- | --- | ---: | ---: | ---: | --- |
| [gear_probe_1_pinned](../evidence/2026-10-23/gear_probe_1_pinned/result.json) | 通过 | 37.452 | 5194 | 3502 | — |
| [gpu_gate_retry](../evidence/2026-10-23/gpu_gate_retry/result.json) | 通过 | 151.479 | 3803 | 未采集 | — |
| [nut_probe_1_pinned](../evidence/2026-10-23/nut_probe_1_pinned/result.json) | 通过 | 38.641 | 5220 | 3476 | — |
| [peg_probe_16_pinned](../evidence/2026-10-23/peg_probe_16_pinned/result.json) | 通过 | 52.962 | 5353 | 3470 | — |
| [peg_probe_1_pinned](../evidence/2026-10-23/peg_probe_1_pinned/result.json) | 通过 | 47.863 | 5131 | 3470 | — |
| [resume_peg4_pinned_compat](../evidence/2026-10-23/resume_peg4_pinned_compat/result.json) | 通过 | 42.718 | 5626 | 3842 | 19 |
| [train_peg16_pinned](../evidence/2026-10-23/train_peg16_pinned/result.json) | 通过 | 46.832 | 5659 | 4042 | 71 |
| [train_peg4_pinned](../evidence/2026-10-23/train_peg4_pinned/result.json) | 通过 | 103.529 | 5955 | 4238 | 19, 18, 18 |

- 三个Factory任务ID均运行时注册。PegInsert在1与16环境各跑160策略步，均在第149步自动reset；GearMesh/NutThread各初始化、显式reset并步进32次。后两者未覆盖完整episode自动reset，也未做训练。
- 观测形状实际为policy `[N,19]`、critic `[N,43]`，action6维。探针逐步检查观测、奖励、所有scene articulation根状态和关节速度有限。Peg探针最大末端零件线速度约0.026/0.034m/s，最大关节速度约1.41rad/s。短测试没有检出非有限物理状态或未处理仿真异常；不据此声称长期物理稳定。
- 4环境3epoch共1536 transitions，epoch2→3的actor22个参数张量、central-value18个参数张量变化；22个Adam状态非零，step7→11。checkpoint全部张量有限。50个TensorBoard scalar tags全部有限。
- 从epoch3恢复到epoch4后frame1536→2048、actor Adam step11→15，actor22/critic18个参数张量继续变化。16环境1epoch产生2048 transitions，22个Adam状态step15、动量非零、checkpoint张量有限。混合精度下实际step计数按文件记录，不假定等于计划mini-epoch次数。
- 恢复和16环境单epoch未走完新episode，RL-Games以`rew_-inf`命名checkpoint并打印未完成episode警告。这是缺少episode奖励统计，不是已发现网络NaN；两次TensorBoard没有scalar标签，loss日志检查无数据，不能写作通过。

完整命令、日志和CSV见 [证据索引](../evidence/2026-10-23/README.md)；可复用操作见 [SETUP](SETUP.md) 和 [EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md)。这里的PPO FPS来自RL-Games `fps total`，探针吞吐含逐步同步检查，不与训练FPS混用。

## 3. checkpoint 与恢复边界

| 文件用途 | SHA256 |
| --- | --- |
| pinned4env epoch3 | `6dd7f429bef5aede4c9d55fa1da54858f47569b182a044c5b634fb94952de442` |
| pinned恢复 epoch4 | `5952151b497ed38c5775703b967ee743a4aa293dcc06cb7f9e16ee4b75e77208` |
| pinned16env epoch1 | `6e672dad70bbe1442e993e1541e5af3f3b723d0624f397abc8de39ed62f3be05` |

这些文件约213MB/个，保留在本地 `work/runs/logs/rl_games/Factory/`，没有塞入Git；具体绝对路径见 [checkpoint_locations.json](../evidence/2026-10-23/checkpoint_locations.json)。官方训练脚本输出的配置YAML已提交在 `configs/resolved/`，checkpoint检查JSON和50项scalar数据已提交证据目录。该脚本在环境构造、wrapper创建之前保存YAML，因此env.yaml的21/72维以及agent.yaml的num_actors=128仍是构造前值；运行时按任务/实际环境数更新到19/43维和4或16 actors，以console、CLI与固定源码执行顺序为准。

恢复证明actor/central-value权重和actor Adam可加载且继续更新。RL-Games1.6.1的`get_full_state_weights`保存actor optimizer与central-value的nn.Module.state_dict；未保存独立central-value Adam状态。故不声称完整优化器、RNG、RNN rollout或物理状态逐位续接。精确安装源码行摘录及SHA256见 [框架checkpoint源码证据](../evidence/2026-10-23/rlgames_checkpoint_source.txt)。

## 4. 科学源码审计摘要

所有结论对应固定v2.1.1源码路径和行号，详见 [SOURCE_AUDIT](SOURCE_AUDIT.md)。

- 实际维度为19/43/6，而配置21/72是构造前占位。策略15Hz，物理/任务空间PD控制120Hz；6维动作经过EMA、目标增量及边界处理，末端roll/pitch受约束。
- 三尺度奖励是在同一个平均关键点距离上计算三条`1/(exp(a*x)+b+exp(-a*x))`曲线，再加engaged和success奖励。不是三个分辨率、也不是力奖励。
- Peg/Gear成功要求XY<2.5mm、z差<1mm；Nut z阈值0.75mm并检查末端yaw。没有保持时长或力阈值。engagement是几何测试。成功不提前终止，Factory只按时间返回done。
- 默认观测没有接触力；`applied_wrench`是控制命令，不是测量外部接触力。默认随机化主要是重置位姿和每episode固定的1mm位置观测误差，不包含默认质量/摩擦/PD随机化。
- 官方actor与central critic均为2层1024 LSTM，MLP512/128/64；horizon=seq128、两个minibatch512、mini_epochs4、lr1e-4、gamma0.995、tau0.95、PPO clip0.2、mixed_precision=true。没有以小MLP替换官方网络。
- Nut动作切片写回、成功时间缓存未清零、IK重试无上限、timeout同时置terminated/truncated等上游细节均记录而未静默修正。

用户提供的项目描述没有更具体的既有科学断言；上述澄清用于避免把常见假设误写为官方实现。

## 5. 失败、修复与剩余问题

1. 首次Sim冷启动240s超时：保留失败，900s上限重试在151.479s正常退出；不是GPU权限失败。
2. 缺少flatdict与其隔离构建`pkg_resources`失败：隔离环境中固定setuptools并用`--no-build-isolation`完成安装。一次uv锁等待告警后顺序重跑。
3. Kit遮蔽venv包：仅设置PYTHONPATH无效，早期实验实际运行捆绑2.5.1组合。通过启动前预载固定包、启动后断言身份/版本解决；固定2.7组合重新做了环境、PPO、恢复验证，历史数据另存。
4. 固定Torch2.7下原样恢复失败：RL-Games1.6.1未显式设置weights_only，触发PyTorch2.6以后默认行为。仅对本次本地已知SHA的checkpoint在该命令设置`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，重试通过；未改PPO源码、未全局修改加载行为。
5. `uv pip check`仍报告pin-pink、dex-retargeting、moviepy未安装。这是未使用的Pink/OpenXR/视频路径的声明依赖缺口，未宣称所有Isaac Lab功能完整可用。
6. NVIDIA旧4.5 requirements网页本次404；官方Lab v2.1.1本地文档与真实动态测试作为版本/兼容性证据。没有vulkaninfo，但Kit的Vulkan设备表和GPU物理探针已通过。
7. GitHub HTTPS origin缺少写入认证；SSH早期失败但最终认证与推送预检通过。按用户要求仍只做本地提交。完整诊断和推送步骤见 [OPEN_ISSUES](OPEN_ISSUES.md)。这不构成仿真/训练阻塞。

长时显存、稳定性、装配成功率、学习曲线与跨seed复现均尚未验证；本阶段不宣称已复现论文性能。

## 6. 下一次审阅后的首训建议

建议16环境、headless、seed0、官方网络与全部奖励/控制/PhysX及PPO参数；horizon=seq128、两个minibatch512。首轮可用官方200epochs观察初步曲线并持续监测reset和显存，显存余量不足退回4环境。本次没有启动该训练。

16环境×128×200=409600 transitions，只有官方128环境×128×200=3276800的1/8；如目标是匹配官方样本预算，16环境需要1600epochs，须在后续实验单独审阅。4环境200epochs仅102400（1/32），匹配预算需6400epochs。环境数变化也改变batch/调度轨迹，不等同完全相同训练。

## 7. Git交付

复现仓库为现有`dreamerToTOE/isaac-factory-repro`，在原始提交`c54da28f61569ec6cc8249b46ffcea498ff27600`之后增加提交，保留README原文、原.gitignore全文与LICENSE。没有重建Git历史、强推或将上游仓库嵌套进复现仓库。

最终本地提交SHA在提交完成后由`git rev-parse HEAD`返回并报告给用户；本报告所属提交本身无法预先包含自己的SHA。推送尚未执行，远程main仍为原始提交；具体状态以最终Git核验为准。
