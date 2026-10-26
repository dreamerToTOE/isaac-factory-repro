# G4-S0 原动作、控制器与物理参数审计

本附录是只读源代码、历史已解析配置和已有 G4-I2 原始记录的审计，未导入 Isaac Sim、未执行环境、未进行策略 rollout、未安装依赖或修改 checkpoint。`controller_audit.json` 保存 19 份源文件/配置/结果的完整 SHA-256、行号和 14 行物理兼容性记录。原 Isaac Lab checkout 为 `90b79bb2d44feb8d833f260f2bf37da3487180ba` / `v2.1.1`，工作区无改动；Isaac Sim 为历史 4.5.0，已安装 tensor 扩展为 106.5.7。

**结论：CONDITIONAL。** 原动作和控制器的数学接口有可信的重建路径；S0 没有实现目标引擎控制器，也没有验证两引擎动力学相同或策略可迁移。本文的 VERIFIED 表示直接审阅冻结源码/历史数据；PROPOSED 表示后续实现方案；UNKNOWN 表示证据不足。

## 源证据与配置一致性

冻结源码分别是 `factory_env.py`、`factory_env_cfg.py`、`factory_tasks_cfg.py`、`factory_control.py`、`direct_rl_env.py` 和 Isaac Lab 的 rl_games wrapper。它们当前字节 SHA 均与 G4-I0 `checkpoint_and_source_identity.json` 的登记 SHA 相同。逐项读取 G2 原训练 `params/env.yaml` 与 G4-I2 R1 A 的 `env.yaml`，使用 `yaml.BaseLoader` 仅读取数据，不实例化 Python 标签；`sim`、`decimation`、`episode_length_s`、`obs_order`、`ctrl`、`task_name`、`task` 和 `robot` 全部相同。环境数分别为 64 和 1，不能把环境数差异隐去。源码与配置校验未重新执行任何历史实验。

下文 `E`、`C`、`EC`、`TC` 分别指固定版本 [factory_env.py](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py)、[factory_control.py](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py)、[factory_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py)、[factory_tasks_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py)。完整本地文件路径和额外底层源码均见 JSON `sources`。

## 6D 动作的实际顺序（VERIFIED）

题目列出的“scaling → EMA”是待核查流程，源码实际为 **clipping → EMA → scaling**，不能据题目预期调整实现顺序。

令策略确定性均值为 `mu_t ∈ R^6`，原缓存为 `a_(t-1)`：

```text
b_t = clip(mu_t, -1, 1)               # Player/standalone
b_t = clip(b_t, -1, 1)               # rl_games wrapper 再裁剪
 a_t = 0.2 b_t + 0.8 a_(t-1)         # Factory，只在每次 policy step 做一次
                                   # 之后不再裁剪缓存
```

原 Factory 的 `_pre_physics_step` 本身不裁剪动作，依赖上述 wrapper / Player 边界。参考 E:296–304、rl_games wrapper:238–244、`scripts/g4i1_runtime.py`:95–107。输入单位无量纲。位置/旋转物理缩放发生在每次 `_apply_action`，即每个 physics step。PegInsert 的 `unidirectional_rot=False`（TC:57、104–124），没有 NutThread 使用的单向 yaw 重映射。

每个 1/120 秒物理步读取当前 fingertip pose，重新计算：

```text
dp = a_t[0:3] * [0.02, 0.02, 0.02] m
r  = a_t[3:6] * [0.097, 0.097, 0.097] rad
p_star = fixed_pos_action_frame
       + clip(p_current + dp - fixed_pos_action_frame, -0.05, +0.05)
```

位置增量沿世界坐标轴。观测/动作坐标中的环境原点只减去平移，没有转为 hole 的旋转局部坐标。源码 `torch.clip` 下界取 `-pos_action_bounds[0]`、上界取 `pos_action_bounds[1]`，作为三轴公共标量；冻结向量均为 0.05，因此本配置恰好三轴相同。不要把一般向量规则误称原实现。`rot_action_bounds=[1,1,1]` 虽有配置，并未在本控制链用于额外旋转裁剪（E:353–368）。

旋转处理为：

```text
angle = norm(r)
delta_q = quat_from_angle_axis(angle, r / angle)
if angle <= 1e-6: delta_q = [1, 0, 0, 0]
q_pre = delta_q ⊗ q_current          # WORLD LEFT MULTIPLY，wxyz
(roll, pitch, yaw) = get_euler_xyz(q_pre)
roll = 3.14159                       # 按源码常数，不能替换为数学 pi
pitch = 0
q_star = quat_from_euler_xyz(roll, pitch, yaw)
gripper_position_target = 0
```

参见 E:370–391。这是“axis-angle world 增量 → Euler upright 投影”，不是直接把后三个动作当 Euler 增量，也不是将 yaw 锁零。即使改变 x/y 旋转动作，约束后的 yaw 仍可能受原四元数/Euler 组合影响，不能直接丢弃两维。

**时序含义：** 同一个 `a_t` 在八个物理步保持，但位置/姿态目标、FD 速度、Jacobian、M 和 torque 每物理步重算。目标是“当前 pose + 固定动作增量”，不是只在 15 Hz 更新一次并保持八步的 Cartesian target，不是每秒 0.02m 的速度，也不是对旧目标做独立累加。原 `DirectRLEnv.step`:334–356 明确循环八次 `_apply_action → write_data_to_sim → sim.step → scene.update`。

## EMA / 速度 / LSTM 的不同 reset（VERIFIED）

E:845–884 在 reset 抓握之后保存当前 fingertip pose；`fixed_pos_action_frame = fixed_pos_obs_frame + init_fixed_pos_obs_noise`。缓存初始化为：

```text
a_xyz = (p_fingertip - fixed_pos_action_frame) / [0.05, 0.05, 0.05]
a_rot_x = a_rot_y = 0
q_rel = quat_euler_xyz(-pi, 0, 0) ⊗ q_fingertip
y = get_euler_xyz(q_rel).yaw
if y > pi/2: y -= 2*pi
if y < -pi:  y += 2*pi
a_rot_z = 2*(y + pi)/(3*pi/2) - 1
```

位置分母是 **0.05，不是 0.02**。缓存没有 [-1,1] 保证，也不应重新裁剪：G4-I2 R1 `a_events/trace.jsonl` 首帧 z 缓存实际为 `1.0918008089065552`。两个 task FD 速度缓冲清零；raw joint velocity 并未因此整体清零，已有首帧 raw arm joint velocity 仍可非零。done 后 actor 的 h/c 各自按环境列清零，不能把“LSTM 清零”等同于“EMA 缓存清零”。

任务线速度使用 `(p_now-p_prev)/dt`；角速度为 `axis_angle(q_now ⊗ conjugate(q_prev)) / dt`，先把差四元数乘 `sign(w)`（E:214–225）。控制使用同一世界轴的 FD task 速度；nullspace 项使用 simulator 提供的 **raw joint velocity**，而不是 `joint_vel_fd`（E:399–414）。调用更新时间戳门限及 `_get_dones` 的最终刷新也须保留，不能改为每 15 Hz 才差分一次。不可直接用 MuJoCo 的任意 body spatial-velocity 数组替换这一差分计算。

## 原低层 torque 公式（VERIFIED）

关节输入形状 `[B,9]`，前七轴为 arm；task pose error `[B,6]`，`J[B,6,7]`，`M[B,7,7]`；函数产生 `[B,9]` torque，finger 两轴在随后被设为零 effort、位置目标零。

位姿误差（C:94–135）：

```text
e_p = p_star - p
若 dot(q_star,q)<0，先令 q_star=-q_star
q_err = q_star ⊗ inverse(q)
inverse(q) = conjugate(q)/||q||²
e_R = axis_angle_from_quat(q_err)
e = concat(e_p,e_R)
```

任务 PD（C:178–196、EC:62、E:393–397）：

```text
Kp = [100,100,100,30,30,30]
Kd = 2 sqrt(Kp) = [20,20,20,10.954451...,10.954451...,10.954451...]
F  = Kp * e - Kd * concat(v_FD_world, omega_FD_world)
tau_task = J.T @ F
```

目标 task velocity 明确为零；没有目标差分前馈。`F` 是计算出的 PD wrench，并非测得接触力。此 task torque 项 **没有再乘 Lambda**，不能因称其为 OSC 而改成另一个质量整形控制律。

Nullspace（C:71–87、EC:65–67）：

```text
q_default = [-1.3003,-0.4015,1.1791,-2.1493,0.4001,1.9425,0.4754]
dq = ((q_default - q_arm + pi) mod (2*pi)) - pi
Lambda = inverse(J @ inverse(M) @ J.T)
J_hash = Lambda @ J @ inverse(M)
u = M @ (10*dq - 6.3246*qdot_arm_raw)
tau_null = (I_7 - J.T @ J_hash) @ u
tau_arm_requested = tau_task + tau_null
tau_software = clip(tau_requested, -100, +100)
```

源码没有 nullspace 伪逆阻尼、矩阵 inverse 正则化、Coriolis/centrifugal 前馈或重力补偿项，也没有对 arm position command 做额外 joint-limit clip（C:19–91）。出现奇异矩阵时，未来适配层应报告或拒绝无效状态；若悄悄加入阻尼/新 gain，就已不是原控制器。可以在 S1 选择数值求解表达式，但必须与固定公式、精度和误差界核对。

## Jacobian、质量矩阵和关节映射（VERIFIED 原接口 / PROPOSED 目标接口）

原 pose 来自 `panda_fingertip_centered` 的世界 body frame（减去 env origin 平移），而 J 并不是这个 fingertip frame 的直接 J：原始 PhysX link J 的 `panda_leftfinger` 和 `panda_rightfinger` 两行组分别取 `body_index-1, :6, :7`，然后求平均（E:112–115、200–212）。`-1` 是固定底座 Jacobian 缺省根 link 行的索引约定。

已安装的 Isaac Sim `omni.physics.tensors/impl/api.py`:1762–1793 直接声明 Jacobian 将 DOF velocity 映射到 world-space link velocity；原 Factory 没有根坐标旋转变换。行序 `[linear xyz, angular xyz]`；质量矩阵取完整 generalized mass matrix 左上 `7×7`。不要误用 Isaac Lab 其他 root-frame OSC helper 的转换规则来改写本任务。

MuJoCo 的可信方案为（PROPOSED）：

1. 按原名字 `panda_joint1..7` 取得 `jnt_qposadr` 和 `jnt_dofadr`，分别映射 q 与 qdot。MJCF 全系统可能含 peg free joint，不能假设 qpos/qvel 下标一一相同。
2. 从原 finger link origin 的世界几何 Jacobian 取得 `jacp`、`jacr`，按前线后角拼成 6×nv，再按七个 arm DOF 取列并求左右均值。原 pose 仍取独立 fingertip frame。仅在独立证明几何点等价后才可简化为一个 site J。
3. 从目标版本的 `mj_fullM` API 展开 full M，然后按七个 arm DOF 取 principal submatrix。保留 finger/hand body 对 arm 的惯性贡献。
4. 依据 qvel 取 raw arm joint velocity，task damping 则忠实保留 physics-rate pose FD。
5. 使用七个直接 torque motor、可审计的单位 transmission/gear；不改为 arm position actuator。实际 joint force/actuator-force 路径和 clipping 必须另行验证。

这些能力由 [MuJoCo 官方 Jacobian / inertia API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac) 支持，S0 未调用目标 J/M，也没有声明跨引擎相等。API 签名应以将来锁定的 MuJoCo release 为准，不能把当前 `stable` 文档签名强加给另一个版本。

## Actuator、重力和惯性的关键差异

| 项目 | 原 Isaac/PhysX，VERIFIED | MuJoCo 路径，PROPOSED / 仍未知 |
|---|---|---|
| arm software torque clip | 全九维 ±100，随后 finger effort 清零（C:90、E:420） | 保留同一 software clip；不能用 actor action clip 冒充 torque clip |
| arm actuator | implicit；Kp=Kd=0，friction=0，armature=0；joint1–4 配置 87Nm，5–7 配置 12Nm（EC:159–174） | 直接 torque motor；单独保存 configured limits 和实际执行 force。不能未经证据把 ±100 直接改成 87/12 软件 clip |
| hand | 两个原 finger coordinate、mimic；target=0；implicit PD 7500 / 173、40N、friction .1（EC:175–182） | 匹配 mimic 与两个坐标，finger 可用专门 PD，但要验证其 implicit integration / saturation 差异；不能用通用 Panda 单 tendon 假定等价 |
| applied torque | G4-I2 实际 applied effort **UNKNOWN**，0 条测量记录 | S0 无新动力学测量，仍 UNKNOWN |
| 场景重力 | `[0,0,-9.81]m/s²`；randomize 时全场暂设零，最后恢复 | 保留场景值以及 reset 时间线 |
| per-body 重力 | robot **disable_gravity=True**；held peg **True**；fixed hole False（EC:125–126、TC:139–140、164–165） | 对原禁用重力 body 单独补偿；须核对 `gravcomp` 计入 passive/actuator force 的路径。不能默认对所有 body 加重力，也不能把整个 qfrc_bias 加入原 torque |
| robot 惯性 | E:41–47 对 API 返回的每个机器人 body 惯性矩阵三个对角元各加 **0.01** | 将实际 body inertia 加 `0.01 I`，按正确惯性 frame 转换；**不是** joint `armature=.01`。源 joint armature=0 |
| peg / hole mass | 0.019kg / 0.05kg（TC:88–100、151、176） | 转换真实质心/张量；不能仅凭 nominal mass 声称惯性一致 |

`ImplicitActuator.compute` 的源码（`actuator_pd.py`:115–140）计算近似 `computed_effort` 和 `applied_effort` 后，**原样返回 control_action**。`articulation.py`:1512–1544 再把该 action 的 effort 写入模拟目标；203–210 将 effort 和 implicit position/velocity target 交给 PhysX。近似 diagnostic `applied_effort` 不是实际求解器 joint force 传感读数。配置 DOF max force 对直接 feed-forward effort 的精确作用尚未独立隔离；这也是后续要核对的引擎接口，而非 S0 已证明的 clipping 行为。

MuJoCo [MJCF 官方参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html) 提供 body gravcomp/inertial 与 motor/gear/force-range 字段；存在表达能力不代表原 PhysX actuator 或惯性数据已经匹配。

## 物理与接触矩阵

| 参数 | 原冻结值 / 语义（VERIFIED） | 目标表示与风险 |
|---|---|---|
| physics / policy | dt=1/120s；decimation8；120Hz / 15Hz | 可设置同 timestep 和八步 schedule。无目标实测吞吐 |
| wall clock | G4-I2 complete command=**3.4699399575795047Hz**；mean interval288.1894246658842ms；1278/1278 超过66.667ms；RTF .2313575393 | 不与15Hz混同；S0 没有新的 realtime 结果 |
| solver | TGS；actor192 position /1 velocity；场景最大192/1；CCD/stabilization/enhanced determinism=False | MuJoCo solver/integrator/iterations/tolerance 需登记；迭代数不等于 substep，也没有一一对应 |
| material friction | 场景默认 static/dynamic1；运行时 robot/peg/hole 的两项均改为 **.75**；default average combine / patch friction | target sliding .75 仅为起始映射；不能凭同系数推断相同摩擦。torsional/rolling、cone/condim、pair mixing 另行核对 |
| contact generation | contact_offset .005m、rest_offset0；friction offset .01m、correlation distance .00625m | 不能直接认定 MuJoCo margin/gap 等价 |
| compliance | 默认 compliant stiffness0、damping0，即该 PhysX compliant 模式未启用；restitution0；bounce threshold .2m/s | solref/solimp 无唯一换算；不能照抄“零 stiffness”使目标接触失效 |
| limits / impulse | max depenetration5m/s；max contact impulse1e32 | solver-specific；无 exact contact parity |
| body damping / gyro | robot/peg/hole linear/angular damping0，gyroscopic=True | 不凭习惯加新 joint/body damping |
| body velocity cap | max linear1000m/s；max angular**3666deg/s** | schema 明确角单位 deg/s，不能当 rad/s（schemas_cfg.py:86–93） |
| self collision | robot enabled_self_collisions=False | 保留原 filter；不把另一个 Panda 默认的 self collision graph 当作冻结任务 |
| peg / bore | peg diameter .007986m，hole .0081m；nominal diametral clearance .000114m、radial .000057m | 这只是配置名义差值；真实 authored/cooked geometry、孔道凹形/convex decomposition 必须保留。单个 hole convex hull 会封闭孔，不可用 |
| joint limits | 来源为 USD；环境配置没有另写数值 joint range；控制没有 q 位置 clip | 必须提取原 joint axis/range/mimic，不能引用别的 Panda 值冒充；soft-limit 求解仍会不同 |
| mass/inertia | peg .019kg、hole .05kg、robot authored inertias + .01I/body | 不从 visual/collision 简化外形重新估算后宣称等价 |

[MuJoCo 官方 solver 参数说明](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters) 支持显式设置约束参数；contact/friction 参数名称相似不构成引擎等效证明。没有为使策略成功而提出任何奖励、predicate、gain、collision 缩放或容差放宽。

## Reset 与 episode 约束（VERIFIED / PROPOSED 重建）

原 E:527–537、635–886 为真实带辅助 physics 的 reset：默认 arm pose；暂时关闭场景重力；fixed root 初始位置 `(0.6,0,0.05)m` 加三轴 uniform ±.05m，yaw uniform `[0,2pi)`；观测 fixed frame 是 hole 顶部，另叠加每 episode Gaussian sigma=.001m 的三轴噪声。hand 初始化在 fixed tip 上 .047m，加位置 uniform ±[.02,.02,.01]m，roll3.1416、pitch0、yaw0 加 uniform ±.785rad。IK 位置/角误差阈值各1e-3，失败环境重新采样，不可把随机初态强行改为一个方便 pose。

peg 在手内的相对变换来自 E:605–633，其 z 为 `.05−.017608=.032392m`，再组合 fingertip 的翻转 frame、逆变换及 hand-local uniform ±[.003,0,.003]m 噪声（E:784–825）。finger 随后闭合至少 .25s 模拟时间；reset Kp `[300,300,300,20,20,20]`，旋转 Kd 额外除10，结束恢复 episode gain。peg 是接触抓握中的独立物体，不能在目标中暗中 weld 到手上代替。

reset 辅助推进不计入主 episode 步数；相同随机种子在两引擎/随机库里不等于相同采样或初态。未来 S1 应分别冻结随机变量和实际 post-grasp 初态，并记录原随机流可能被 Player 丢弃的 Normal sample 消耗影响。task 默认假定所有环境同时 reset；批次隔离和 done actor h/c 要另外保留。

episode 为 `ceil(10/(8/120))=150` 上限，counter 在 step 后加一，原 timeout 条件 `episode_length_buf >=149`；所以完整 episode 为149 policy steps、9.933333...s主计数模拟时间。terminated 与 truncated 在这个任务都返回同一 timeout；success 不提前终止。奖励与成功先在旧 episode 计算，再 reset，最后返回新 observation（DirectRLEnv:358–392）。本审计不改原 success，也不把 reset 后 pose 用来评估 terminal success。

## G4-S1 进入条件（建议，不启动）

先锁定合法可用的原 robot/assembly USD 及依赖，提取 joint/mimic、坐标、mass/CoM/inertia 与实际 collision decomposition；取得目标引擎版本和授权安装条件后，才实施结构等价 MJCF。先做 CPU 解析与静态 pose/J/M parity，再做原公式 controller 的固定合成输入 parity；必须记录 float32 源计算与目标通常 double 的误差、quaternion 分支和 singularity 行为。验证 original gravity-disable、body +.01I、finger implicit PD 与 direct effort limit 路径，再开展经单独批准的短物理检查。接触参数差异、实际 applied torque 未知和 nominal57µm clearance 都是明确技术风险。

S0 只支持可信的接口映射与后续实现范围，不能宣称模型已重建、controller 已部署、MuJoCo 上策略成功或 Sim2Sim PASS。

后续资产审计补充：原USD关节限位、mimic及robot惯性元数据已在 `assets_audit.json` 提取；运行时质量覆盖后的惯量、cooked碰撞与目标模型等价性仍未验证。此补充不改变原控制器公式或历史applied torque UNKNOWN。
