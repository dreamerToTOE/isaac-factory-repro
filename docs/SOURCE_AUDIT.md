# Isaac Factory v2.1.1 科学源码审计

审计日期：2026-10-23（Asia/Seoul）。范围仅 G0/G1。本文依据本地实际代码逐行审阅；所有链接固定到同一提交，不跟随 `main`。本文件的静态结论不能替代 GPU 仿真、任务注册运行时导入或 PPO 更新测试；执行结果见项目环境审计与实验记录。

## 1. 上游身份和审计范围

| 项目 | 固定值 / 证据 |
| --- | --- |
| 官方仓库 | `https://github.com/isaac-sim/IsaacLab` |
| 标签 | `v2.1.1`（本地 `git describe --tags --exact-match`） |
| 提交 | `90b79bb2d44feb8d833f260f2bf37da3487180ba`（本地 `git rev-parse HEAD`） |
| 源码工作树 | 审阅时 `git status --short` 无输出；本审计没有修改上游 |
| Isaac Sim 版本关系 | 标签内 [README.md:41–54][R41] 明确列 `v2.1.1 → Isaac Sim 4.5`；同 README 徽标及 RL 包分类器指向 4.5.0 |
| Python / RL 依赖 | [source/isaaclab_rl/setup.py:20–48,66–74][P20]：Python `>=3.10`、分类器 3.10；RL-Games extra 固定 `rl-games==1.6.1`，附带 `gym`；不需要为此任务安装所有 RL 后端 |

核心路径均相对于这个上游仓库：

- `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py`：运行逻辑。
- `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py`：环境、观测、控制与 PhysX 配置。
- `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py`：三项任务、资产与奖励参数。
- `source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py`：任务空间控制与 IK。
- `source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml`：三个任务共同使用的 PPO。
- `source/isaaclab/isaaclab/envs/direct_rl_env.py`、`direct_rl_env_cfg.py`：父环境步进和终止语义。
- `source/isaaclab_rl/isaaclab_rl/rl_games.py` 与 `scripts/reinforcement_learning/rl_games/train.py`：训练包装、超时信号与命令行覆盖。

### 三个任务 ID

[factory/__init__.py:16–44][I16] 注册：

| Gym ID | 环境配置 | agent 配置 |
| --- | --- | --- |
| `Isaac-Factory-PegInsert-Direct-v0` | `FactoryTaskPegInsertCfg` | `rl_games_ppo_cfg.yaml` |
| `Isaac-Factory-GearMesh-Direct-v0` | `FactoryTaskGearMeshCfg` | 同上 |
| `Isaac-Factory-NutThread-Direct-v0` | `FactoryTaskNutThreadCfg` | 同上 |

这证明源码包含注册语句。运行时注册是否成功仍必须由安装后的真实导入结果确认。

## 2. 观测、特权状态和动作

### 2.1 实际维度：actor 19、critic 43、action 6

不能直接抄配置类中的 `observation_space=21` 和 `state_space=72`。[factory_env_cfg.py:17–41,71–89][C17] 定义维度表与排列；[factory_env.py:26–31][E26] 在构造时按排列求和，再各追加 6 维动作。以原样默认配置计算得到：

| actor 输入顺序 | 维数 | 含义 |
| --- | ---: | --- |
| `fingertip_pos_rel_fixed` | 3 | 末端位置减固定件观测框架位置及该 episode 的观测偏移；并非把位置旋转到固定件局部坐标系 |
| `fingertip_quat` | 4 | 末端四元数，标量在前 `wxyz` |
| `ee_linvel` | 3 | 相邻物理步末端位置有限差分 |
| `ee_angvel` | 3 | 相邻末端四元数差转轴角后除物理步长 |
| `prev_actions` | 6 | `self.actions.clone()`，即内部处理后的动作缓存，不保证等于网络原始输出 |
| **合计** | **19** | 所有三项任务共用 |

critic 按 `fingertip_pos(3), fingertip_quat(4), ee_linvel(3), ee_angvel(3), joint_pos(7), held_pos(3), held_pos_rel_fixed(3), held_quat(4), fixed_pos(3), fixed_quat(4), prev_actions(6)` 拼接，总计 **43**。critic 的线速度/角速度使用仿真原始 body velocity，actor 使用有限差分速度；critic 的固定件位置不加观测噪声。配置维度表虽列 `task_prop_gains`、`ema_factor`、阈值等候选字段，默认 `state_order` 没有它们，不能称其为实际默认输入。实现见 [factory_env.py:194–229,254–290][E194]。

### 2.2 动作处理与频率

- 网络动作是 6 维连续量：前三维位置增量，后三维轴角增量。RL-Games wrapper 先裁剪到 `[-1,1]`；环境执行 `a_t = 0.2 * a_raw + 0.8 * a_previous` EMA。[rl_games.py:238–250][W238]、[factory_env.py:296–304][E296]、[factory_env_cfg.py:49–67][C49]。
- 每个**物理子步**把位置目标设为当前末端位置加 `a_xyz * [0.02,0.02,0.02]` 米，并相对固定的 `fixed_pos_action_frame` 裁剪至每轴 ±0.05 米。该 action frame 在 reset 时设为固定件观测框架加观测噪声，之后不逐步跟随资产更新。[factory_env.py:342–368][E342]、[factory_env.py:849–877][E849]。
- 轴角乘 `[0.097,0.097,0.097]` 弧度，再形成目标四元数。但目标 Euler roll 强制 `3.14159`、pitch 强制 `0`，所以不能解释为完全自由的六自由度位姿控制；抓爪目标始终闭合 `0.0`。[factory_env.py:356–391][E342]。
- NutThread 开启 `unidirectional_rot=True`，yaw 分量执行 `-(a_z+1)/2`。须注意该赋值使用 `self.actions` 的切片视图，因此会写回缓存；父环境在每次策略步执行 8 次 `_apply_action`，不能把实际实现简化成只在策略步开始做一次变换。[factory_tasks_cfg.py:377–378][T364]、[factory_env.py:357–360][E342]、[direct_rl_env.py:329–356][D305]。保留此上游行为，G0/G1 不修算法。
- `dt=1/120 s`、`decimation=8`：物理/低层控制 **120 Hz**，策略步 **15 Hz**。默认环境数 **128**；本机能容纳的环境数不能由源码推出。[factory_env_cfg.py:71–118][C71]、[direct_rl_env.py:239–259][D239]。

## 3. 低层机器人控制及物理设置

训练时不是直接输出关节力矩的 policy，也不是纯关节位置 PD：

1. 取左右指 Jacobian 的均值作为末端 Jacobian，取机械臂 7×7 广义质量矩阵。[factory_env.py:205–210][E194]。
2. 计算末端位置/姿态误差；以任务空间 PD 得到命令 wrench：`w = Kp * pose_error - Kd * ee_velocity`，旋转误差采用最短路径四元数的轴角。[factory_control.py:44–66,94–135,178–196][F19]。
3. 机械臂 torque 是 `Jᵀw` 加使用质量矩阵与动态一致投影的 null-space posture torque，结果裁至 ±100。训练默认任务空间 `Kp=[100,100,100,30,30,30]`，`Kd=2*sqrt(Kp)`；nullspace `kp=10`、`kd=6.3246`。[factory_control.py:64–91][F19]、[factory_env.py:393–423][E393]、[factory_env_cfg.py:59–67][C49]。
4. 机械臂 PhysX implicit actuator stiffness/damping 均 0，由 effort target 控制；仿真关节 effort limit 前四关节 87、后三关节 12。手指使用 PhysX PD：stiffness 7500、damping 173、effort limit 40；手指 torque 缓存清零，关节位置目标闭合。[factory_env_cfg.py:158–182][C158]、[factory_env.py:418–423][E393]。
5. reset 过程使用 DLS IK（`lambda=0.1`）移动手臂，按位置/姿态误差各 `1e-3` 检查并重新采样直到成功；没有最大重试上限。随后用 reset 增益 `[300,300,300,20,20,20]`、旋转导数除以 10，闭合手指约 0.25 s，再恢复训练增益。[factory_control.py:156–161][F138]、[factory_env.py:568–603][E568]、[factory_env.py:712–764][E712]、[factory_env.py:827–886][E827]。

上游还给机器人各 body 惯量对角加 `0.01`；这属于原实现，而非本复现引入的修补。[factory_env.py:41–47][E41]。PhysX 使用 solver_type=1、位置迭代上限192、速度迭代上限1、`gpu_max_rigid_contact_count=2**23`、`gpu_max_rigid_patch_count=2**23`、`gpu_max_num_partitions=1`。这些大容量接触缓冲可能影响 8GB GPU 的初始化内存，不能因少量 env 就保证能启动；G0/G1 不默认修改它们。[factory_env_cfg.py:97–115][C71]。

## 4. 多尺度关键点奖励

### 4.1 几何与公式

每件物体沿自身 z 轴放置 4 个等距关键点：`(0,0,linspace(0,1,4)-0.5) * keypoint_scale`，分别经 held base pose 和 target held base pose 变换。距离是对应点三维欧氏距离的均值：`x = mean_i ||p_held_i - p_target_i||₂`。参考 [factory_env.py:123–156][E123]、[factory_env.py:231–251][E231]。

三个尺度对**同一个 x**计算不同曲率的函数，既非三个不同采样分辨率，也非三个 keypoint 长度：

`s(x;a,b) = 1 / (exp(a*x) + b + exp(-a*x))`

`reward = s_baseline + s_coarse + s_fine + 1_engaged + 1_success - λ_action ||actions||₂ - λ_grad ||actions-prev_actions||₂`

所有任务默认 `λ_action=λ_grad=0`。实现里的 `action_penalty` 没有 `dim` 参数，实际是整个动作张量的 norm；默认零系数时不影响奖励。奖励每步计算，不是只在 episode 末结算。[factory_env.py:487–525][E487]、[factory_tasks_cfg.py:68–84][T68]。

| 任务 | 关键点线段总长度 | baseline `(a,b)` | coarse `(a,b)` | fine `(a,b)` |
| --- | ---: | --- | --- | --- |
| PegInsert | 0.15 m | (5,4) | (50,2) | (100,0) |
| GearMesh | 0.15 m | (5,4) | (50,2) | (100,0) |
| NutThread | 0.05 m | (100,2) | (500,2) | (1500,0) |

来源：[factory_tasks_cfg.py:77–81,126–132][T68]、[factory_tasks_cfg.py:286–291][T267]、[factory_tasks_cfg.py:389–397][T364]。Peg/Gear 关键点 z 偏移为 `[-0.075,-0.025,0.025,0.075] m`；Nut 为 `[-0.025,-0.008333…,0.008333…,0.025] m`（由源码公式计算）。

### 4.2 成功和 engagement 的真实定义

共用函数首先要求 `xy_dist < 0.0025 m`，再要求 `z_disp = held_base_z - target_held_base_z < height_threshold`。是严格 `<`，z 没有对称绝对值、下界、力阈值或持续时间判据；因此不能把此二值指标直接称为实物机械质量检验。[factory_env.py:431–456][E431]。

| 任务 | 目标 pose 的定义 | success 的 z 条件 | engaged 的 z 条件 | 额外条件 |
| --- | --- | --- | --- | --- |
| PegInsert | held base 是 peg 的资产根；target 是固定孔资产根 | `<0.025*0.04=0.001 m` | `<0.025*0.9=0.0225 m` | 都要求 XY<2.5mm；无旋转测试 |
| GearMesh | 默认 medium gear；held base 与 fixed target 都按各自 pose 变换 medium gear offset `[0.02025,0,0]` | `<0.02*0.05=0.001 m` | `<0.02*0.9=0.018 m` | 都要求 XY<2.5mm；无旋转测试 |
| NutThread | held base 在 nut 局部 z=`0.01 m`；target 在 bolt 局部 z=`0.01+0.025-0.002*1.5=0.032 m` | `<0.002*0.375=0.00075 m` | `<0.002*0.5=0.001 m` | success 还要求映射后的末端 yaw `<0`；engaged 不检查 yaw |

目标变换与几何参数来源：[factory_env.py:91–144][E91]、[factory_env.py:231–251][E231]、[factory_env.py:539–550][E539]、[factory_tasks_cfg.py:88–132][T88]、[factory_tasks_cfg.py:186–210,286–291][T186]、[factory_tasks_cfg.py:345–397][T345]。

Nut 的高度阈值实际乘的是 **thread_pitch=2mm**，不是注释泛称的固定件高度。yaw 是末端世界 Euler yaw；实现把大于 235° 的值减 2π，再与 `ee_success_yaw=0` 比较，不能解释成螺纹转数、相对 bolt yaw 误差或达到指定拧紧扭矩。[factory_env.py:344–346][E342]、[factory_env.py:441–454][E431]、[factory_tasks_cfg.py:389–397][T364]。

三任务 engaged/success 都作为每步 +1 的奖励。engagement 来自几何位置测试，并未读取真实接触。`extras["successes"]` 只在 reset 时记录**该时刻**成功比例；不是“episode 内曾成功”比例。另有 `ep_succeeded` 和首次成功步数日志，应分别解释。[factory_env.py:458–484][E458]。

## 5. Episode 终止与 reset 的实际语义

`_get_dones` 唯一判据是 `episode_length_buf >= max_episode_length - 1`，并返回 `(time_out,time_out)`，即 terminated 与 truncated 同时置位。**成功不会提前终止；没有掉落、过大接触力、越界或 NaN 自动终止条件。** [factory_env.py:425–429][E425]。

父类给出 `max_episode_length = ceil(episode_length_s / (dt*decimation))`，且先 `episode_length_buf += 1` 再检查 done。由这条代码路径推导，在计数由 0 开始时三任务分别于第149、299、449个策略步触发 time out；配置名义长度为150、300、450步（10、20、30秒）。reset 中的额外仿真步也不计入这个策略步计数。此为静态推导，仍需运行时日志核验。[factory_env_cfg.py:187–205][C187]、[direct_rl_env.py:252–259,358–392][D239]、[direct_rl_env.py:592–593][D572]。

`DirectRLEnvCfg.is_finite_horizon=False`；RL-Games wrapper 合并两种 done，同时把 truncated 写入 `extras["time_outs"]`，配合 PPO 的 `value_bootstrap=True`。评估和重放必须沿用这套行为，不应自行把成功改为 terminal。[direct_rl_env_cfg.py:60–75][DC60]、[rl_games.py:238–264][W238]。

`_reset_idx` 注释明确假定所有环境同步 reset；随机化函数含整批张量写入。默认只有同长度 timeout 符合此假定，后续增加独立 failure reset 会影响该假定，属于需单独设计与验证的工程修改。[factory_env.py:527–537][E527]、[factory_env.py:691–710,804–825][E658]。

## 6. Domain randomization：实现了什么、没有实现什么

随机化在 reset 的 `randomize_initial_state` 中执行，初始化暂时把全局 gravity 设 0，夹紧后恢复配置重力。不是训练中持续随机外力。[factory_env.py:658–695][E658]、[factory_env.py:827–886][E827]。

| 参数 | PegInsert | GearMesh | NutThread | 分布/频率 |
| --- | --- | --- | --- | --- |
| 固定件位置 | 每轴±0.05m | 同左 | 同左 | reset 均匀分布 |
| 固定件 yaw | [0°,360°) | [0°,15°) | [120°,150°) | `initial + range*U(0,1)`，不是中心±range |
| 手的位置偏移（相对固定件 tip） | z=0.047m | z=0.035m | z=0.015m | 每轴再加均匀±[0.02,0.02,0.01]m |
| 手初始 Euler | [3.1416,0,0] | [3.1416,0,0] | [3.1416,0,1.83] | 再加均匀±[0,0,0.785]rad；Nut 改为±[0,0,0.26]rad |
| 手中零件位置扰动 | ±[0.003,0,0.003]m | x±0.003m，y=0，z∈[-0.003,0]m | ±[0,0.003,0.003]m | reset，在 held asset 局部变换中施加 |
| 固定件观测位置误差 | σ=[0.001,0.001,0.001]m | 同左 | 同左 | reset 高斯采样，该 episode 内固定；不是每步重新采样 |

采样实现：[factory_env.py:664–710][E658]、[factory_env.py:712–744][E712]、[factory_env.py:784–825][E784]；任务参数：[factory_tasks_cfg.py:104–132][T104]、[factory_tasks_cfg.py:267–291][T267]、[factory_tasks_cfg.py:364–397][T364]；观测误差配置：[factory_env_cfg.py:44–46][C44]。Nut 固定的手中旋转初始化 -90° 实际仅在 nut 分支使用；不可据 Gear 的同名配置字段推断执行了相同旋转。[factory_env.py:605–633][E605]。

摩擦、质量、PD gains、动作阈值、EMA 默认都是确定常量；默认没有配置 `events`/额外 action noise/observation noise model。源码将 held/fixed/robot 摩擦固定为配置数值（通常0.75，Nut held为0.01），不是每个 episode 随机化。Nut friction 行的解释性注释与当前数值不一致，实际执行 `0.01`；不采用注释中的历史 `-0.25` 推算值。[factory_env.py:49–73][E49]、[factory_tasks_cfg.py:14–37,345–351][T14]、[direct_rl_env_cfg.py:102,172,209][DC102]。

Gear 默认添加两侧齿轮，`add_flanking_gears=True`；虽有 `add_flanking_gears_prob=1.0` 字段，当前 reset 分支只检查布尔开关，没有采样该概率。[factory_tasks_cfg.py:267–269][T267]、[factory_env.py:768–782][E768]。

## 7. 接触传感与力观测

机器人及 held/fixed 资产 spawn 配置包含 `activate_contact_sensors=True`，但 `_setup_scene` 只创建 articulation、桌面、地面和光源；Factory 没有实例化 `ContactSensor`、读取 `net_forces_w` 或把接触力加入 actor/critic。观测字典和奖励也没有力/力矩输入。[factory_env_cfg.py:120–142][C120]、[factory_tasks_cfg.py:134–177][T134]、[factory_env.py:158–189][E158]、[factory_env.py:254–290][E254]、[factory_env.py:487–525][E487]。

`self.applied_wrench` 是 `compute_dof_torque` 返回的**任务空间 PD 命令 wrench**，不是 PhysX 测得的外部接触 wrench；也没有加入默认观测。因此不能把这项官方基线描述为“以接触力观测驱动的力反馈 PPO”。真正的接触物理由 PhysX 处理，不能从“contact-rich”名称推导出显式力感知策略。[factory_env.py:399–423][E393]、[factory_control.py:40–66,90–91][F19]。

## 8. 官方 RL-Games PPO 完整配置

三个任务共同使用同一 YAML。以下都是固定提交的配置值；RL-Games 的类名 `a2c_continuous` 不等于关闭 PPO，配置明确 `ppo: True`。[rl_games_ppo_cfg.yaml:1–118][Y1]。

| 类别 | 官方值 |
| --- | --- |
| seed / 算法 / model | `seed=0`；`algo.name=a2c_continuous`；`model.name=continuous_a2c_logstd` |
| action clipping | `env.clip_actions=1.0`；未显式配置 clip_observations，训练脚本默认 `math.inf` |
| actor network | `name=actor_critic`；`separate=False`；MLP `[512,128,64]`、ELU、`d2rl=False`；default initializer、无 regularizer |
| continuous head | `mu_activation=None`、`sigma_activation=None`；mu default initializer；sigma const initializer `val=0`；`fixed_sigma=False`（logstd 模型） |
| actor RNN | LSTM，units=1024，layers=2，`before_mlp=True`、`concat_input=True`、`layer_norm=True` |
| checkpoint 初始值 | `load_checkpoint=False`、`load_path=""` |
| 运行 | `name=Factory`、`device=cuda:0`、`full_experiment_name=test`、`env_name=rlgpu`、`multi_gpu=False`、`num_actors=128` |
| PPO / 数值 | `ppo=True`、`mixed_precision=True`、`normalize_input=True`、`normalize_value=True`、`normalize_advantage=True`、`value_bootstrap=True` |
| reward shaper | `scale_value=1.0` |
| discount / GAE | `gamma=0.995`、`tau=0.95` |
| 学习率 | `learning_rate=1.0e-4`、`lr_schedule=adaptive`、`schedule_type=standard`、`kl_threshold=0.008` |
| 训练/保存 | `score_to_win=20000`、`max_epochs=200`、`save_best_after=10`、`save_frequency=100`、`print_stats=True` |
| loss / 梯度 | `grad_norm=1.0`、`truncate_grads=True`、`entropy_coef=0.0`、`e_clip=0.2`、`critic_coef=2`、`clip_value=True`、`bounds_loss_coef=0.0001` |
| rollout / minibatch | `horizon_length=128`、`minibatch_size=512`、`mini_epochs=4`、`seq_length=128` |
| central value optimizer | `minibatch_size=512`、`mini_epochs=4`、`learning_rate=1e-4`、`lr_schedule=adaptive`、`kl_threshold=0.008`、`clip_value=True`、`normalize_input=True`、`truncate_grads=True` |
| central value network | `name=actor_critic`、`central_value=True`；MLP `[512,128,64]`、ELU、d2rl=False、default initializer、无 regularizer；LSTM units=1024、layers=2、before_mlp=True、concat_input=True、layer_norm=True |
| player | `deterministic=False` |

这是有记忆的 asymmetric actor/critic 基线，不应替换为无 RNN 的简单 MLP 并称为原样复现。默认 rollout batch=`128*128=16384` transitions，4 mini epochs、512 minibatch；只改 env 数时必须检查 batch/RNN sequence/minibatch 的整除关系。训练脚本会将 agent `num_actors` 更新为实际环境数，`--max_iterations` 实际覆盖 `max_epochs`，并保存最终 env/agent yaml/pkl；`--checkpoint` 设置 load_checkpoint/load_path 并把 checkpoint 传给 runner。[train.py:87–105,136–145,176–206][TR87]。

### 工程 smoke 覆盖的界限

1–4 env、短 `max_epochs`、更小 minibatch（actor 和 central value 同时调整）及更频繁 checkpoint 保存可以作为单列的 smoke 覆盖；需要保存解析后的配置和每项差异。保留原 `horizon_length=seq_length=128`、RNN/MLP、奖励、成功判据、控制器与 PhysX 设置，有助于减少差异。以 1 env 为例 batch=128，不能原样套用 minibatch=512；4 env batch=512，原 minibatch 在维度上可以整除。此处只是源码推导，不代表本机已跑通或有足够 VRAM。

正式首训的 env 数需要在实际 GPU 初始化、PPO 更新和 VRAM 记录通过后再决定。默认128、256或512都不构成 RTX3070 8GB 可运行的证据。成功阈值、PPO 算法或网络的改动必须另行审批并标记，不属于 G0/G1 baseline。

## 9. 需保留并跟踪的上游细节

- `ep_success_times` 初始化为零，却未在 `_reset_buffers`（只清 `ep_succeeded`）或 reset 主体中清零；新 episode 尚未成功时，`success_times` 日志可能携带先前 episode 的数值。不要将其直接用于严谨收敛评估。[factory_env.py:148–149,292–295][E148]、[factory_env.py:472–482][E458]。
- Nut 的动作切片写回与 decimation 交互、timeout 双 done、149/299/449步推导等均已记录。它们是此提交的行为，未静默“修正”。
- reset 的 IK 重试无上限；若启动卡在 reset，应保留日志并定位 IK/资产/物理原因，不能把训练未开始宣称为 PPO 成功。
- USD 资源来自 `${NUCLEUS_ASSET_ROOT_DIR}/Isaac/IsaacLab/Factory`，桌面还依赖 `${...}/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd`；Factory Python 源码克隆成功不代表这些资产已经在机器上缓存可用。[factory_tasks_cfg.py:9–11][T9]、[utils/assets.py:24–34][A24]、[factory_env.py:160–170][E158]。

## 10. 本次审计命令与证据边界

实际执行了 `git rev-parse HEAD`、`git describe --tags --exact-match`、`git status --short`、`rg --files`、`rg -n` 与 `nl -ba … | sed -n …`，逐行阅读以上源文件。一次以裸 `python` 发起的静态 AST 维度计算失败（`python: 未找到命令`），随后用 `python3` 重跑成功，输出：

```json
{"policy_dim": 19, "critic_dim": 43, "action_dim": 6,
 "note": "Static AST calculation only; not simulation initialization"}
```

AST 检查只从维度表、obs_order/state_order/action_space 读取 Python literal 并按构造器规则求和，没有导入 Isaac Sim、启动仿真或执行 PPO。本文没有填造 FPS、显存峰值、成功率、checkpoint 成功或优化次数。所有仿真及训练结论必须以独立运行日志为准。

用户提供的原始要求没有声称具体观测维度、力传感策略或成功保持时间；上面的澄清用于防止后续项目描述引入这些错误假设。已明确纠正的事实包括：官方默认128环境、实际19/43维、没有默认力观测、三尺度是同一距离的三条函数、Nut高度阈值基于螺距、成功不提前终止，以及默认随机化范围有限。

## 固定源码链接

下列引用的 URL 均固定同一上游 SHA；显示标签给出准确文件及行号。

[R41]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/README.md#L41-L54
[P20]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_rl/setup.py#L20-L74
[I16]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/__init__.py#L16-L44
[C17]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py#L17-L89
[C44]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py#L44-L46
[C49]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py#L49-L67
[C71]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py#L71-L118
[C120]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py#L120-L142
[C158]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py#L158-L182
[C187]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py#L187-L205
[E26]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L26-L31
[E41]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L41-L47
[E49]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L49-L73
[E91]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L91-L144
[E123]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L123-L156
[E148]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L148-L295
[E158]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L158-L189
[E194]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L194-L290
[E231]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L231-L251
[E254]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L254-L290
[E296]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L296-L304
[E342]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L342-L391
[E393]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L393-L423
[E425]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L425-L429
[E431]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L431-L456
[E458]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L458-L484
[E487]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L487-L525
[E527]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L527-L537
[E539]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L539-L550
[E568]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L568-L603
[E605]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L605-L633
[E658]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L658-L825
[E712]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L712-L764
[E768]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L768-L782
[E784]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L784-L825
[E827]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L827-L886
[E849]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L849-L877
[F19]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py#L19-L196
[F138]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py#L138-L175
[T9]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L9-L11
[T14]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L14-L351
[T68]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L68-L132
[T88]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L88-L132
[T104]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L104-L132
[T134]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L134-L177
[T186]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L186-L291
[T267]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L267-L291
[T345]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L345-L397
[T364]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L364-L397
[D239]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/envs/direct_rl_env.py#L239-L392
[D305]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/envs/direct_rl_env.py#L305-L356
[D572]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/envs/direct_rl_env.py#L572-L593
[DC60]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py#L60-L75
[DC102]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py#L102-L209
[W238]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_rl/isaaclab_rl/rl_games.py#L238-L264
[Y1]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml#L1-L118
[TR87]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/scripts/reinforcement_learning/rl_games/train.py#L87-L206
[A24]: https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab/isaaclab/utils/assets.py#L24-L34
