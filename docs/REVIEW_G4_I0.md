# G4-I0：冻结策略推理接口审计

日期：2026-10-26（Asia/Seoul）。工程结论：**G4_I0_PASS（仅 I0 审计范围）**。用户理解门禁：**PENDING**。

G2 epoch-200 主参考、R4 epoch-400 次参考和 R5 epoch-400 附加证据已经核验；19D actor、43D critic 边界、归一化、LSTM、6D 动作、控制器和 reset 时序均已从实际源码及 checkpoint 冻结。本阶段没有执行 actor forward、Isaac Sim、GPU 工作、训练或后续阶段。I0 PASS 表示接口审计与 CPU 验证完成；独立推理运行和仿真 parity 尚须 G4-I1 验收。

## 起点和历史保存

修改前通过 fetch、祖先关系检查和 `pull --ff-only origin main` 将此前 R5 交付 `851d1b844109020e0ae050b265925391d6011461` 更新到 **`9c2ce260589a32036ec346b1573e776b8f36c432`**。该 main 提交明确将 G3C-R5 归档、G4-I0 设为当前阶段。已核阅 README、路线、理解门禁、G2 报告/原始清单、R4 和 R5 最终报告/证据；不重新解释或改写旧科学结果。

[historical_preservation.json](../evidence/g4_i/i0/historical_preservation.json) 验证起点的 1,359 个 tracked 文件内容和执行权限一致，987 个唯一历史文件（16,207,906,067 bytes）按原清单重新计算 SHA，无缺失/不一致；覆盖 G2、R4、R5、早期 checkpoint、日志和失败记录。G2 原先的 manifest、退出码和监控证据缺口仍按原报告保留。

原先未跟踪的 `docs/DEMO_G3V.md`、`scripts/g3v_playback.sh` 和 `evidence/g3v/` 五个文件均保留，未提交到 I0；记录的是本次检查时哈希，没有虚构 I0 开始时的对照。I0 新增内容只在本报告、I1 计划、`evidence/g4_i/`、两个 CPU 审计辅助模块和一个测试模块中。

## 检查点身份和依赖

下表路径相对于工作区根目录 `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been`；绝对路径、文件大小、完整模型张量形状及三套 agent/env 配置路径和 SHA 均在 [checkpoint_and_source_identity.json](../evidence/g4_i/i0/checkpoint_and_source_identity.json) 与 [checkpoint_audit.json](../evidence/g4_i/i0/checkpoint_audit.json)。G2 使用明确 epoch-200 final-save 文件，不能以 best alias 或 R4/R5 替代。

| 角色 | 精确路径 | SHA-256 | 训练配置 |
|---|---|---|---|
| G2 primary | `work/runs/logs/rl_games/Factory/g2_peginsert_64env_seed0_baseline_rerun/nn/last_Factory_ep_200_rew__350.4961_.pth` | `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f` | seed0，64 env，200 ep，horizon128 |
| R4 secondary | `work/runs/logs/rl_games/Factory/g3cr4_peginsert_32env_seed0_400ep_20261025t140016z/nn/last_Factory_ep_400_rew__345.99203_.pth` | `da72a148ebe2b1e2f11f79c4316c24f515510780a3975e2304a41077436b3c36` | seed0，32 env，400 ep，horizon128 |
| R5 additional | `work/runs/logs/rl_games/Factory/g3cr5_peginsert_32env_seed1_400ep_20261026t015040z/nn/last_Factory_ep_400_rew__346.54318_.pth` | `2b266c9b64574dc075634e10e3c6b9856899870cbabe54b157a70b7a8c9de54e` | seed1，32 env，400 ep，horizon128 |

三者 frame 都为 1,638,400，但训练 env 数/epoch/seed 不同，归一化和权重各自独立，不能合并。agent.yaml 的 `num_actors=128` 是 wrapper 更新前的 dump；实际 env 数由 env.yaml 和既有运行证据核实，不能将该 dump 值当作真实训练环境数。

CPU 检查首先验证 allowlist SHA，随后只对这三份可信本地文件执行 `torch.load(map_location="cpu", weights_only=False)`，核对 epoch/frame、模型 key 全集、形状、dtype、有限值、RMS 和配置。没有重训、改权重或写入 checkpoint。checkpoint 含 optimizer/critic/AMP scaler，但推理不恢复 optimizer；三者 `env_state=null`，没有可恢复的 episode hidden/cell。官方 Player 的模型恢复见 RL-Games `algos_torch/players.py:72–80`，使用 strict `load_state_dict`；本项目三份 RMS 已嵌入 `model`，不存在需优先覆盖的 top-level RMS。

依赖：Python **3.10.12**、Torch **2.7.0+cu128**、RL-Games **1.6.1**、Gymnasium **1.2.0**、NumPy **1.26.4**、PyYAML **6.0.3**；Isaac Lab **v2.1.1 / `90b79bb2d44feb8d833f260f2bf37da3487180ba`**，本机 Isaac Sim **4.5.0**（VERSION build=`4.5.0-rc.36+release.19112.f59b3005.gl`，见 `environment_verification.json`）。安装版 RL-Games 的实际执行文件 SHA 也已冻结，不能只凭包版本宣称实现完全相同。

## 源码索引

以下 LAB 表示上述工作区的 `work/upstream/IsaacLab/`，RLG 表示 `work/envs/factory-g01/lib/python3.10/site-packages/rl_games/`，REPO 表示本仓库。14 个执行源码文件的绝对路径及 SHA 见 `checkpoint_audit.json.source_files`。

| 代码 | 与本审计有关的精确行 |
|---|---|
| LAB `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py` | 17–41 维度；45–67 噪声/控制；72–118 时序/顺序；159–179 执行器；187–191 PegInsert |
| LAB `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py` | 26–31 runtime维度；115 fingertip body；191–225 pose/FD；254–290 obs；296–304 EMA；342–423 target/controller；425–456 done/success；527–537 reset；691–710 frame/noise；845–884 reset cache |
| LAB `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py` | 19–91 PD/Jacobian/nullspace/torque；94–135 shortest quaternion pose error |
| LAB `source/isaaclab/isaaclab/envs/direct_rl_env.py` | 239–260 dt/max length；329–392 step/auto-reset顺序 |
| LAB `source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py` | 60–75 finite-horizon配置 |
| LAB `source/isaaclab_rl/isaaclab_rl/rl_games.py` | 238–264 action/done/device；273–311 observation/state包装 |
| RLG `algos_torch/running_mean_std.py` | 9–31 epsilon/buffers；45–80 eval更新条件/FP32转换/clip |
| RLG `algos_torch/network_builder.py` | 227–246 LSTM结构；397–448 实际共享网络分支；460–476 初态 |
| RLG `algos_torch/models.py` | 34–52 RMS；239–299 logstd/sample/mu |
| RLG `common/layers/recurrent.py` | 24–56 done masking；74–77 torch.nn.LSTM |
| RLG `algos_torch/players.py` | 11–15 action rescale；45–80 deterministic和restore |
| RLG `common/player.py` | 26 clip_actions；267–272 init_rnn；196–206 obs_to_torch；335–344 done状态管理，文件SHA已保存 |
| REPO `scripts/evaluate_checkpoint.py` | 103–158 配置和wrapper；244–274 inference、pre-reset capture、reset RNN |
| REPO `scripts/evaluation_runtime.py` | 18–27 在inference_mode内清零h/c |

## 19D actor 与 43D critic

`factory_env.py:286–289` 实际拼接顺序如下。输入为 FP32 `[B,19]`，同 actor device；单样本 helper 支持 `[19]`，真正模型边界必须显式 batch。各特征拼接前没有逐项比例缩放，随后统一进入冻结 RMS。

| 索引 | 字段/维度 | 物理意义、单位、坐标 | 更新来源 |
|---|---|---|---|
| 0:3 | fingertip_pos_rel_fixed /3 | `p_tip-(fixed_pos_obs_frame+noise)`，m，世界轴 | robot body position减env_origins；reference在reset冻结；每次返回obs |
| 3:7 | fingertip_quat /4 | `panda_fingertip_centered` 的 body_quat_w，wxyz，无量纲 | 世界朝向原始符号，不额外规范四元数正负；每次返回obs |
| 7:10 | ee_linvel /3 | 相邻物理步位置差 / physics_dt，m/s，世界轴 | physics dt=1/120 的FD，每policy边界返回最新值 |
| 10:13 | ee_angvel /3 | `q_now*conj(q_prev)`，按w符号取短路径、axis-angle/dt，rad/s，世界轴 | 同上，不用整段8dt代替差分dt |
| 13:19 | prev_actions /6 | `self.actions`：EMA动作缓存，无量纲 | 每policy步更新一次；reset由初始位姿计算 |

critic 的43D按序为 fingertip pos3、quat4、simulator linvel3/angvel3、joint pos7、held pos3、held-relative pos3、held quat4、fixed pos3、fixed quat4、cache6。它的速度源和 actor 不同；不把 critic states 送入 actor。配置 placeholder 21/72 在构造时被覆盖为19/43。

**隐含 privileged 依赖已确认**：actor 虽没有独立 fixed-pose字段，其 relative position 的 fixed frame 由 simulator fixed_pos/fixed_quat 加几何tip offset变换而来（699–710），再加 reset 时采样的每轴 sigma=0.001m 噪声（691–695）。它目前不是外部视觉/定位测量。fingertip pose也来自 simulator body truth。独立同引擎入口可读取同样的量，但真机或其他引擎的观测获取尚无证据；不能宣称19D天然完全可部署。controller 还额外依赖 joint pos/vel、平均左右手指 Jacobian、arm mass matrix，这些不由19D本身提供。

## Normalization、网络与 recurrent state

三份完整19维 mean/variance及scalar count保存在 `checkpoint_audit.json.checkpoints.<role>.rms`，G2 count=1,638,401；不同checkpoint必须使用各自buffer。RMS以float64保存，**运算前转float32**，实际表达式是：

```python
x = clamp((obs_fp32 - mean.float()) / sqrt(var.float() + 1e-5), -5.0, 5.0)
```

这与 wrapper 的 raw observation clip不同：这里wrapper clip_observations默认为inf。必须 `model.eval()`，仅 no_grad/inference_mode 不会阻止 training-mode RMS更新。CPU审计对三份实际RMS分别调用安装版官方 `RunningMeanStd.eval()`，与合同计算逐元素完全一致（max_abs=0），并验证buffer未变化。该验证不是actor forward parity。

真实actor：19 → **torch.nn.LSTM（2层，每层1024）→ LayerNorm1024 → MLP512/ELU→128/ELU→64/ELU → Linear mu6**。mu无bounded激活；另有state-dependent Linear logstd6，sigma=exp(logstd)，以及value1。`separate=false`是该模型共享网络分支；43D central critic独立。`concat_input=true`在before_mlp=true分支不执行，不把19D额外拼到1024D后。checkpoint的第一层LSTM输入权重[4096,19]、第二层[4096,1024]、首MLP[512,1024]及mu[6,64]与此一致。

h/c分别为float32 `[2,B,1024]`、同policy device，第一次obs前置零；B为当前推理batch，与原训练64/32个env无绑定。每次forward延续状态；done对应列在下一episode首次action前清零。action cache与h/c是两种不同状态，reset不能都置零。

独立评估使用policy mean。训练rollout采样Normal(mu,sigma)，训练优化分支使用prev_actions计算logprob/entropy；保存配置player deterministic=false已由原独立evaluator覆盖。**官方模型即使最终Player选择mu，也会先产生一个被丢弃的sample，消耗RNG**。I1若改为只算mu，必须处理这一RNG差异，不能仅凭同seed声称后续reset随机流相同。原训练AMP不等于推理必须启用AMP；I1先固定FP32。

## 6D动作到控制器

六维原始mu均无内在范围限制。Player先clip[-1,1]再按Box[-1,1]作恒等rescale；wrapper另做clip[-1,1]。Factory每policy步执行 `cache=.2*incoming+.8*previous_cache`，**EMA后不再clip**。

| 维度 | 物理解释与尺度 | 最终处理 |
|---|---|---|
| 0/1/2 | 世界轴x/y/z末端目标位置增量，每cache单位0.02m | 当前位姿加增量；target相对fixed_action_frame逐轴限制±0.05m |
| 3/4/5 | 世界轴轴角旋转向量x/y/z，每cache单位0.097rad | 转增量quat并左乘当前quat；转Euler后roll固定字面值3.14159、pitch=0，保留所得yaw，再转quat |

因此旋转三维不是三个独立自由Euler目标，亦不能直接称为六个torque、velocity或绝对position。零/极小旋转角<=1e-6使用identity quaternion。PegInsert `unidirectional_rot=false`；配置rot_action_bounds=[1,1,1]不在此动作路径实际使用。每个physics子步都以**最新**末端位姿重新生成target，不能在15Hz只算一次pose target后保持8步。

最终位姿进入task-space PD：Kp=[100,100,100,30,30,30]，Kd=2sqrt(Kp)；wrench经J转置和mass-matrix相关nullspace姿态项（kp_null=10,kd_null=6.3246）变成7个arm effort targets。软件clamp±100N·m后写joint effort；PhysX另有joint1–4限87、joint5–7限12N·m，不能把±100当作最终执行能力。arm position PD为零；fingers通过position target=0闭合，stiffness7500、damping173、effort_limit40。controller不是policy直接给出torque。

reset先建立 `fixed_action_frame=fixed_obs_frame+episode_noise`；位置cache=(current_pos-fixed_action_frame)/0.05，**不是/0.02**；旋转x/y为0，yaw按源码862–877的unrot180、包角和270度映射初始化。完整公式在机器spec，cache不保证落在[-1,1]。reset还清零FD速度buffer。CPU测试专门验证cache=2经EMA后1.6不会被错误再clip。

## 时序和终止

physics dt=1/120s（120Hz），decimation=8，policy/env step=1/15s（15Hz）。同一个smoothed cache在8个physics子步保持，controller/target每个子步重算。

名义episode_length_s=10，max_episode_length=ceil(10/(8/120))=150；counter先加1再以 `>=150-1` 判断，实际是**149 policy steps = 9.9333333333模拟秒**超时，G2既有轨迹也为149步。成功不提前终止；timeout时Factory同时返回terminated/truncated为true，wrapper合并done并提供time_outs。reset/IK/grasp另外推进physics，不算在这149个policy步内。

step顺序是pre_physics_step→8个apply/physics→counter+1→done/reward/pre-reset成功抓取→auto-reset→新episode obs。evaluator拿到done和新obs后清零h/c，下一次action才用新obs/新cache。不能用返回的post-reset obs判断上一episode末态成功。

15Hz仅是仿真时间下的策略节奏。I0没有测量墙钟推理延迟、真实命令频率或deadline miss；I1完成首个episode，I2再做GPU同步的时延分布/控制频率验证。

## CPU验证和复现

机器契约：[interface_spec.json](../evidence/g4_i/i0/interface_spec.json)，包含逐字段source、dtype/device、RMS、reset、动作处理、控制频率、伪代码和异常门禁。`scripts/g4i0_interface.py`仅为CPU合同helper，`scripts/audit_g4i0.py`只读SHA/配置/张量/RMS，不含独立policy runner或forward。

新增针对性测试 **52 passed**（由最终完整回归JUnit按该模块统计），覆盖19D切片与顺序、非法shape/rank/dtype/batch、RMS负方差/非有限/缺项/count/固定epsilon及clip、h/c初态与选择性reset、mock policy输入输出和确定性重放、raw clip/EMA/物理scale/position saturation、no-reclip、仿真时钟/8子步target重算、异常时mock controller零提交、NaN/Inf、checkpoint身份和key/shape/dtype/epoch/frame不兼容。mock不是训练后policy正确性的证据。

最终完整回归 **272 passed，0 failed，exit 0**（14.24秒pytest / 14.72秒子进程墙钟）结果见 [cpu_validation.json](../evidence/g4_i/i0/cpu_validation.json) 和 `g4i0_acceptance_cpu.{txt,xml,json}`，使用指定venv、`CUDA_VISIBLE_DEVICES=''`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`及唯一`/dev/shm`临时目录，**不排除既有测试**。初次完整回归232passed/2failed及排除G3A的诊断结果都保留，不能把诊断当完整PASS。两项G3A失败在原样16项G3A重跑中均未复现；存储/时序敏感性是推断，原guard临时证据消失，不能确定原始根因，也没有弱化测试或改监督实现。第一次审计collector误按Assign读取实际AnnAssign导致StopIteration，已修复并保留原失败console；最终三checkpoint审计成功。

从仓库根目录复现（必须给新output，不覆盖证据）：

```bash
CUDA_VISIBLE_DEVICES='' ../../work/envs/factory-g01/bin/python -m scripts.audit_g4i0 --output /tmp/g4i0_new_audit.json
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../work/envs/factory-g01/bin/python -m pytest -q tests --basetemp=/dev/shm/g4i0_new_tests
```

命令只有CPU审计和回归，不会启动历史训练脚本。

## 风险、接受计划和结论

G4-I0必需接口组件均已源码/metadata核验；没有把I1尚未运行的runtime测试伪称为通过。未测项目：真实actor输出parity、GPU dtype/device、upright quaternion逐步数值parity、reset首动作/随机流、完整episode、接触质量、墙钟性能与异常时物理安全行为。外部fixed-frame感知和controller状态获取仍是部署依赖；官方几何成功不等于力控安全或持续保持。

[G4_I1_ACCEPTANCE_PLAN.md](G4_I1_ACCEPTANCE_PLAN.md) 预注册FP32 action/h/c分阶段容差、同一观测trace、RNG处理、至少一个完整episode、身份链、no-command mock、真实退出码和资源前检。I1尚未启动；先等待用户审阅。此前R4/R5的no-GPU-guard授权不自动扩展到未来GPU工作。

结论：**G4_I0_PASS**。I1接口准备完成，I1运行验收仍为NOT_RUN；用户工程理解为PENDING。停止在I0，没有启动I1/I2、Sim2Sim、GearMesh、NutThread、seed2或任何新训练。
