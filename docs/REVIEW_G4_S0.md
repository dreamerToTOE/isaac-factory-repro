# G4-S0：跨物理引擎 Sim2Sim 可行性与接口映射

日期：2026-10-26（Asia/Seoul）。仓库 `dreamerToTOE/isaac-factory-repro`，分支 `main`。

**工程状态：COMPLETE；可行性决策：CONDITIONAL；Sim2Sim：NOT RUN / NOT ASSESSED。**

本任务只做 S0。已核对当前 README、路线、G4-I0/I1/I2 报告与冻结源码/配置/checkpoint，
完成资产与许可调查、19D/6D/控制与接触映射，以及 CPU 验证。没有实现目标装配环境、
运行物理步、策略 rollout、训练、微调或修改权重。MuJoCo 可作为真实第二物理引擎；
**已有模型能解析不等于原 Factory 已被忠实重建，更不表示策略在 MuJoCo 有效。**

首要未决项为原 Factory 修改资产的转换/公共再分发许可、孔的 SDF/凹形碰撞迁移、
运行时质量覆盖与惯量、夹持与 actuator-force 语义、原始 quaternion sign 和 matched
pose/J/M。没有用 FR3、通用 Panda position actuator 或更容易的插孔任务替代原问题。

## 1. 起点、冻结身份与证据口径

读取当前仓库后，从 `85b49ef547e07df063d2efe58457488c99935a70` fast-forward 到
**`280634679633bf127e13349741379b3d38a27a1f`**；中间 `772f091`、`2806346` 仅更新
README/路线导航。它们明确 G4-I0/I1/I2 已验收、S0 尚待授权；本次用户任务提供 S0
授权，历史“尚未执行”文字保持原样，不追溯改写历史结果。学习考核仍 PENDING，按用户
安排推迟至 Sim2Sim 后；本阶段不设置学习测验或理解门禁。

证据标签统一为：**VERIFIED**=本次实际测试或直接审阅冻结源码；**PROPOSED**=可审查
的实现路线，尚未实施；**UNKNOWN**=证据不足；**BLOCKED**=在本次授权范围内不能完成。
“原语义 VERIFIED / 目标实现 PROPOSED”不能合并成跨引擎等价 VERIFIED。

| 冻结项 | 核查结果（VERIFIED） | 直接依据 |
|---|---|---|
| Isaac Lab | v2.1.1 / `90b79bb2d44feb8d833f260f2bf37da3487180ba` | 本地 upstream checkout + 源码 SHA 与 I0 一致 |
| Isaac Sim / PhysX | Isaac Sim4.5.0，历史VERSION build4.5.0-rc.36；tensor extension106.5.7 | 既有环境证据与已装 API 源码；未启动引擎 |
| Task / robot | `Isaac-Factory-PegInsert-Direct-v0`；Factory Panda `franka_mimic.usd` | Factory env/task cfg 与缓存 USD metadata；不是 FR3 |
| Primary checkpoint | G2 final epoch200，seed0、64env、frame1,638,400 | 212,889,225 bytes；读取前后 SHA 与冻结清单一致 |
| SHA-256 | `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f` | 真实 checkpoint 字节哈希，非 best alias |
| Actor | 19→LSTM(2×1024)→LayerNorm1024→512/ELU→128/ELU→64/ELU→mu6 | 配置、源码分支与 checkpoint 权重形状；`concat_input=true` 在 before_mlp 分支无效 |
| RMS | mean[19]/var[19]/count FP64保存，计算转FP32；count1,638,401 | CPU元数据探针逐项 torch.equal，与 I0 完全相同 |
| Normalization | `(obs_fp32-mean.float())/sqrt(var.float()+1e-5)`，clip[-5,5] | 原 RunningMeanStd；wrapper raw observation clip=inf；不更新统计 |
| LSTM | h/c各[2,B,1024] FP32；首步全零、done列在下一回合首动作前清零 | 冻结 standalone；EMA cache reset不是h/c reset |
| Timing | physics dt1/120s，decimation8，policy dt1/15s | G2与I2 resolved configs / DirectRLEnv；CPU时钟算术复核 |
| Timeout | 实际149 policy steps=9.9333333333模拟秒；success不提前终止 | counter加1后以`>=150-1`超时；terminated/truncated均为timeout |

Primary真实路径、全部RMS、模型形状、源码行号与 SHA 见
[observation_success_audit.json](../evidence/g4_s/s0/observation_success_audit.json)。
G2 与 G4-I2 R1 A 的 sim/ctrl/task/robot 等配置通过 inert YAML BaseLoader 逐项比较，
科学配置完全一致；只保留原64env与I2单环境的数量差异，不把 cfg dump 的128actors
误当真实训练数量。R4/R5没有替代主模型；三份 checkpoint 均在本次回归前后重新哈希。

原 Player 即便选确定性mu，仍会消耗一次被丢弃的 Normal sample；已验收 FrozenPolicy
保留此行为。S1应复用该冻结模块，不能以同seed假定跨引擎 reset/RNG逐位相同。
CPU回归包含既有 actor 数学测试，但没有新建策略运行/环境 rollout 或 optimizer。

## 2. 第二引擎与资产可行性

### 2.1 已有环境与最小执行

系统 Python 和冻结 Factory venv 均未安装 MuJoCo；记录为真实缺项，没有安装包。
已有独立环境 `/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python` 可导入
**MuJoCo3.5.0**，实际库版本相同。现有 Menagerie 位于
`/home/ubuntu2004/mujoco_menagerie`，干净于
`8161bba264d7fa7c99ca301e91e7fb44737676ad`。

仅调用 `MjModel.from_xml_path` 解析已有 Panda/FR3：分别 nq/nv/nu=9/9/8 与7/7/7，
编译耗时约0.240s/0.245s，进程peak RSS316,224/415,608KiB。没有创建 MjData，
没有 forward、step、viewer。FR3仅用于排除错误候选，没有选择它代替 Panda。
这些测试只证明依赖和候选模型解析可用，不测试接触、FK、J、M或策略效果。两候选编译后的
默认dt均为0.002s，尚未改成冻结任务的1/120s；其默认驱动和时间步不是已批准的目标配置。

### 2.2 原资产来源、版本与完整性

下表链接仅表示来源，不包含在本仓库中重新分发的原资产。缓存使用不透明名称，
原 basename 搜索未命中被保留为调查过程，随后通过已有文件与官方 HEAD 找到关联。
本地 SHA-256 已计算；远端Content-Length与ETag/本地MD5相符是强来源关联证据，
**没有下载远端正文后独立计算其 SHA-256，不能将它表述成远端SHA核验**。

| 资产 | 冻结来源 | 本地 SHA-256 | 审计边界 |
| --- | --- | --- | --- |
| 机器人、手、夹指、末端参考体 | [Factory/franka_mimic.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/franka_mimic.usd) | `86d0017a093669b222a8c640bf581cc3e3715035aa7c596600f478da011f6c8e` | 关节、质量惯量、所有 mesh 的顶点/面计数、碰撞属性与包围盒已读 |
| Peg8mm | [Factory/factory_peg_8mm.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/factory_peg_8mm.usd) | `b5040c8c639f3dc0731adfb6022c0d63383b7550354e413dc24e7417b600869d` | 原 mesh 与 SDF 碰撞属性已读，未导出 mesh |
| Hole8mm | [Factory/factory_hole_8mm.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/factory_hole_8mm.usd) | `4bad48cde5a2e523454f9ed584023fae53d619a5ffb3d6ab08f29a6240014c28` | 同上；孔拓扑不能由单个凸包保留 |
| 桌面顶层 USD | [SeattleLabTable/table_instanceable.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd) | `479ed8fd374d8eafdedf3ca3b9d0d5981ffc6d5c5016929347f748be93d5df17` | 仅 Sdf 层元数据与依赖清单；完整桌面依赖、许可及几何仍 UNKNOWN |


Factory原 robot 为 metre/Z-up；Peg/hole层 metadata为0.01/Y-up，但原坐标与任务米制
尺寸相符。已有 USD0.22.11 的 CPU reference→metre/Z-up host 测试表明几何世界包围盒
原样保持，没有自动缩放/换轴。因此转换应复核原引用/局部根变换，不得按metadata
再缩小100倍或旋转。该测试没有运行 Isaac Sim。

Robot/peg/hole的 reference/payload/sublayer 依赖闭合；22个机器人 mesh、13个含固定
joint、13个显式质量体均读取。原 peg1154顶点/1152面，hole1897顶点/3806面，二者
碰撞 approximation均为 PhysX SDF。shader/MDL等视觉资源尚未闭合审计；table顶层
有外部`table.usd`引用，只读其层依赖，完整桌面仍 UNKNOWN。原peg/hole没有显式
完整质量/惯量组合，运行时 mass覆盖后实际 PhysX惯量仍须验证，不能由名义质量推断。

| 项目 | Menagerie Panda | Menagerie FR3 | 对冻结 Factory 的意义 |
| --- | --- | --- | --- |
| 来源 | [固定提交 Panda README](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/README.md)，来自 Franka ROS 的简化 MJCF | [固定提交 FR3 README](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_fr3/README.md)，来自 franka_description | Panda 可作为审查模板；FR3 不是本任务允许的替代品 |
| 关节命名/顺序 | `joint1..7,finger_joint1,finger_joint2` | `fr3_joint1..7` | 未来必须显式名映射，不能按数组位置猜测；FR3 没有原 Panda 夹手 |
| 7 臂关节范围，rad | ±2.8973；±1.7628；±2.8973；[-3.0718,-0.0698]；±2.8973；[-0.0175,3.7525]；±2.8973 | ±2.7437；±1.7837；±2.9007；[-3.0421,-0.1518]；±2.8065；[0.5445,4.5169]；±3.0159 | FR3 限位不同；原 USD 的精确 degree 数值见 JSON，转换要保留浮点差异 |
| 臂驱动 | 有位置伺服增益/bias；damping=1、armature=0.1 | 有位置驱动、摩擦、armature | 均不能原样接收冻结 torque controller；不得靠位置驱动使实验“跑通” |
| 质量/惯量 | 臂参数与原资产有共同来源；但手指每个 0.015kg | 臂质量、惯量不同 | 原 Factory 手指每个约 0.0297kg，额外 force_sensor 约0.01kg 与末端参考体；Panda 也未获整体动力学等价认证 |
| 碰撞与视觉 | link5 凸分解、人工设计 fingertip geoms、视觉 mesh 转换与碰撞过滤 | 独立 FR3 mesh | 原 Factory collision mesh、指垫、接触材料、self-collision 关闭及质量覆盖应独立迁移；视觉相似不证明接触等价 |


原 arm names `panda_joint1..7`，finger names `panda_finger_joint1/2`；候选 Panda 为
`joint1..7,finger_joint1/2`。关节按名字映射，再分别使用 qposadr/dofadr，不能根据
数组下标或字典排序猜测。原 revolute USD限位单位degree；转换后的7轴限位与候选
Panda最大差约2.765e-7rad，**差值虽小也不是字节相同**。所有原度值/候选弧度值、
局部轴、body mass/CoM/主惯量/主轴和文件哈希已逐项保存于
[assets_audit.json](../evidence/g4_s/s0/assets_audit.json)。Panda共67、FR3共36个直接
引用mesh存在且已哈希；没有复制36/57MB资产入仓库，也未下载它们。

原 peg直径0.007986m、长度0.05m；hole名义孔径0.0081m、高0.025m。名义径向间隙
57µm是配置差值，不是网格/SDF有效孔径的测量。不能改成理想圆柱、更大的孔、简化
夹手或weld payload来提高成功率。普通mesh凸包会填实孔；有误差界的凸分解或SDF
插件是 PROPOSED，尚未验证。源PhysX SDF cooked buffer没有可直接跨引擎复用证明。

### 2.3 许可与再分发

MuJoCo3.5.0与固定 Menagerie Panda/FR3自带许可证为Apache-2.0，使用/再分发应履行
相应许可、NOTICE/归属和改动标记要求。Isaac Lab代码为BSD-3-Clause；冻结源码树
存在一般Franka资产Apache-2.0许可，但不能据此推断覆盖全部Forge/Factory改动、peg、
hole和table。安装的Isaac assets许可要求逐机器人/prop查询hosted-folder许可。

常见Factory LICENSE/另三个许可名称 HEAD均404，负结果原样保留；这只表示没有在
这些URL发现条款，不表示资产无许可或法律上禁止使用。当前许可网页也不能追溯证明
原4.5资产的适用条款。**原Factory资产使用/转换/公开再分发范围 UNKNOWN；其公共
衍生资产分发门禁 BLOCKED。** 本次只提交来源和派生元数据，不复制或公开原USD/mesh。

许可与文本哈希、官方URL见资产JSON与
[资产审计附录](../evidence/g4_s/s0/assets_audit.md)。主要原始依据：
[MuJoCo3.5许可](https://github.com/google-deepmind/mujoco/blob/3.5.0/LICENSE)、
[固定Panda许可](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/LICENSE)、
[冻结Franka许可](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/docs/licenses/assets/franka-license.txt)。

## 3. 19D观测逐项映射

以下**原语义 VERIFIED，MuJoCo计算 PROPOSED**。p为TCP世界位置减env origin，q为
`panda_fingertip_centered`世界wxyz四元数；f0/R0为reset时固定物体根pose；
`b=f0+R0*[0,0,.025]+n`，n为每回合一次sigma=.001m的三轴Gaussian噪声，reset缓存。
不是孔局部旋转坐标；观测使用孔顶部参考，而success使用当前孔根。

|索引|字段 / 原 PhysX 来源|原坐标与单位|MuJoCo 对应计算（PROPOSED）|必需变换与精度|无 privileged state 可观测性|精确复现结论|
|---|---|---|---|---|---|---|
|0|relative position x；env.py194-201,262|世界轴，m|`(site_xpos[tcp]-origin-b)[0]`|保留 root→tip .025m、reset缓存和减噪声；FP32运算|否；孔 reset 真值来自模拟器|数学映射可行；TCP/孔根帧待核|
|1|relative position y；同上|世界轴，m|同式分量1|不乘固定物体逆旋转；FP32|否|同上|
|2|relative position z；同上|世界轴，m|同式分量2|观测参考是孔顶端，不是成功判据的孔根；FP32|否|同上|
|3|fingertip quaternion w；env.py201,263|世界朝向 wxyz，无量纲|exact TCP body `xquat[...,0]` 或 site矩阵转换|不作 w≥0 规范化；保留原 sign；FP32|编码器FK+准确标定原则可得；原值是模拟器真值|旋转可映射；原始符号一致性 UNKNOWN|
|4|fingertip quaternion x；同上|世界朝向，无量纲|同式分量1|q与−q不能互换，RMS对其不对称|同上|同上|
|5|fingertip quaternion y；同上|世界朝向，无量纲|同式分量2|site_xmat转换会丢失四元数符号信息|同上|同上|
|6|fingertip quaternion z；同上|世界朝向，无量纲|同式分量3|不能误用xyzw或局部朝向|同上|同上|
|7|ee linear velocity x；env.py214-216,264|世界轴，m/s|`(p_now-p_prev_physics)[0]/dt`|dt=1/120，非1/15；每新physics timestamp更新；FP32差分|可由标定FK历史估计；非原真值保证|相同pose样本下语义可复现|
|8|ee linear velocity y；同上|世界轴，m/s|同式分量1|不直接替换成qvel或body速度；reset为0|同上|同上|
|9|ee linear velocity z；同上|世界轴，m/s|同式分量2|必须保留8子步内的前一pose；FP32差分误差乘120|同上|同上|
|10|ee angular velocity x；env.py219-225,265|世界轴，rad/s|`axis_angle(sign(dq.w)*dq)[0]/dt`，`dq=q_now*conj(q_prev)`|左空间差；同源axis-angle eps1e-6；FP32|标定朝向历史原则可估计|数值语义可复现；符号/轨迹一致性未验证|
|11|ee angular velocity y；同上|世界轴，rad/s|同式分量1|不能反乘成body-local差或用原生qvel冒充|同上|同上|
|12|ee angular velocity z；同上|世界轴，rad/s|同式分量2|原`sign(0)=0`在恰180°差时会零化delta；不静默修正|同上|同上|
|13|prev_actions position x；env.py258,266,302-304|内部EMA、归一化动作单位|cache0=`.2*clip(mu0)+.8*cache0`|每policy步一次，reset=`(p-b)[0]/.05`，不是/.02|命令内部状态可知；reset继承孔真值依赖|可按原程序复现，target未实现|
|14|prev_actions position y；同上|内部EMA、归一化动作单位|同式分量1|reset=`(p-b)[1]/.05`，不再clip|同上|同上|
|15|prev_actions position z；同上|内部EMA、归一化动作单位|同式分量2|reset=`(p-b)[2]/.05`，cache可能越[-1,1]|同上|同上|
|16|prev_actions rotation x；同上|内部EMA、归一化动作单位|同式分量3|reset=0；不把它当raw mu或已缩放角速度|是，内部状态|同上|
|17|prev_actions rotation y；同上|内部EMA、归一化动作单位|同式分量4|reset=0|是，内部状态|同上|
|18|prev_actions rotation z；同上|内部EMA、归一化动作单位|同式分量5|reset为下述非零yaw映射；不额外clip|是，内部状态|同上|


MuJoCo body `xpos/xquat`或匹配site `site_xpos/site_xmat`可提供相应状态；COM字段
`xipos`不能代替body origin。精确TCP/site与模型frame必须先匹配。矩阵转四元数会
丢失符号信息；原 q 没有w≥0 canonicalization，q与−q虽然旋转相同，却是不同actor
特征（RMS qx均值约.98157）。S1必须检查raw sign，不能只比较旋转矩阵。

FD使用前一物理时间戳的pose及1/120s；不使用原生body velocity替代，也不以两次
policy观测之差/1/15替代。dq=`q_now*conj(q_prev)`后乘sign(dq.w)，再轴角/dt；
原恰180°且w=0时sign(0)=0会零化差分，不能未经登记“修正”。reset首观测FD显式为0。

**privileged依赖仍在。** 相对位置的固定装配体真值、reset噪声/参考缓存来自模拟器，
不是视觉观测；不能从机器人编码器直接得到孔位置。目标仿真可使用同类内部真值，
但不能把任意外部估计代入后宣称观测等价。TCP本身也依赖机器人标定/FK真值。

完整19维RMS不改变，运算前转FP32、epsilon1e-5、clip±5、eval冻结。h/c与动作缓存
单独reset；cache位置=(p-b)/.05，旋转x/y=0，yaw按原unrot(-pi)/包角/270°公式，
不是全零，可能超[-1,1]。完整公式与19行各八项属性见
[观测与成功审计](../evidence/g4_s/s0/observation_success_audit.md)。

## 4. 6D动作与低层控制器

实际顺序从源码核实为 **mu→clip±1→EMA→物理缩放**，并非题目预期顺序的scaling→EMA。

```text
b_t = clip(mu_t,-1,1)               # Player/standalone和wrapper边界
cache_t = .2*b_t + .8*cache_(t-1)  # 每policy步一次，之后不再clip
每physics步（120Hz）:
  p_pre = p_current + .02*cache_t[:3]                 # 世界轴，m
  p_target = fixed_action_frame + clip(p_pre-fixed_action_frame,-.05,.05)
  r = .097*cache_t[3:]                               # 世界轴轴角，rad
  dq = quat_from_rotvec(r)，norm(r)<=1e-6时identity
  q_pre = dq ⊗ q_current                            # 左乘
  euler = 原get_euler_xyz(q_pre)
  roll=3.14159，pitch=0，yaw保留，再转目标quat
```

六维不是torque、Euler独立增量或速度。8个物理步保持cache，但每步用最新pose重建
目标、FD/J/M和控制器，不能以15Hz算一个target保持8步，也不能对旧target累计位移。
`rot_action_bounds=[1,1,1]`在此路径未额外使用，PegInsert无单向yaw重映射。

源控制器使用task-space PD与动态一致nullspace。令e=[p_target-p,轴角(q_target*q^-1)]，
旋转误差先按target/current点积选择shortest quaternion：

```text
Kp=[100,100,100,30,30,30], Kd=2*sqrt(Kp)
F = Kp*e - Kd*[v_FD_world,omega_FD_world]
tau_task = J.T * F                      # 没有额外Lambda乘task wrench
Lambda = inverse(J*inverse(M)*J.T)
J_hash = Lambda*J*inverse(M)
e_q = wrap_to_[-pi,pi)(q_default-q_arm)
u_null = M*(10*e_q - 6.3246*qdot_arm_raw)
tau_requested = tau_task + (I - J.T*J_hash)*u_null
tau_software = clip(tau_requested,-100,100)
```

q_default=[-1.3003,-.4015,1.1791,-2.1493,.4001,1.9425,.4754]。
源函数q/qdot[B,9]、J[B,6,7]、M[B,7,7]、wrench[B,6]、torque[B,9]；finger effort随后
归零并设置position target0。没有Coriolis、全qfrc_bias或重力前馈，也没有inverse
阻尼正则化。奇异性必须显式拒绝/记录，悄悄加阻尼或改gain会改变控制律。

| 项目 | 原接口 VERIFIED | MuJoCo方案 PROPOSED / 未决 |
|---|---|---|
| Cartesian error | 世界pose；target-current；shortest quaternion差 | 同一TCP/frame、同源Euler/axis-angle分支，FP32对照 |
| 速度 | task damping用physics-rate FD；nullspace用raw joint_vel | qposadr/dofadr分别映射；不把MuJoCo原生body velocity替代FD |
| Jacobian | world [linear,angular]；左右finger body-origin J取前7列再平均 | 分别`mj_jacBody`或同点几何J，拼jacp/jacr；不能默认等于TCP site J |
| M | 完整广义M的arm7×7主子矩阵 | `mj_fullM`后按arm DOF取块，保留hand/finger惯性贡献 |
| 重力 | global-9.81；robot和heldpeg各disable_gravity=True | 按body匹配gravity-disable；不能整体加qfrc_bias冒充原控制器 |
| 惯性 | 每个robot body惯性对角另加.01 | 是body tensor+.01I，不是joint armature=.01；frame/导入后值待核 |
| arm actuation | implicit Kp/Kd0；friction/armature0；直接effort target | 单位gear直接torque motor；不得采用Panda默认position servo |
| torque limits | 软件±100；配置joint1–4为87Nm，5–7为12Nm | 分别登记，需测actuator force路径；不能静默改成87/12软件clip |
| fingers | 两原坐标/mimic，PD7500/173，40N，joint friction.1，夹持自由peg | 匹配implicit积分/力饱和与接触抓握；不是weld或简单一个tendon等价 |

原implicit actuator的`applied_effort`为近似诊断，并把原action原样交给PhysX；不能
把该字段或已知effort limit当作实际求解器施加扭矩。**G4-I2 actual applied joint
torque仍UNKNOWN，本次没有填补该历史缺口。** F也是PD计算wrench，不是接触力传感器。

MuJoCo3.5的[Jacobians/M API](https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html)、
[actuator/inertial/gravcomp表示](https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html)
提供相应建模能力；S0没有调用目标J/M或实施这些actuator，不能据API存在声称控制等价。
原源码/精确行号与额外依赖见
[controller_audit.json](../evidence/g4_s/s0/controller_audit.json)。

## 5. PhysX—MuJoCo物理/接触兼容性

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


目前只决定可保留的120Hz/8/15Hz**模拟时间**schedule，没有测得MuJoCo闭环吞吐。
原G4-I2 complete wall command rate **3.4699399576Hz**、完整间隔超66.6667ms比例
1278/1278=100%，继续作为历史限制保留。网络forward P99约1.116ms不能改写这一
完整闭环结论；CPU模型编译时间也不是控制频率。

原PhysX TGS/材料/offset/compliance与MuJoCo软约束接触、cone、solref/solimp、margin
没有唯一数值换算；相同friction=.75也不保证接触/摩擦等价。目标应登记选择、做无策略
接触一致性试验，不能改变孔径、reward、阈值或PPO来掩盖差异。
[MuJoCo3.5接触模型依据](https://mujoco.readthedocs.io/en/3.5.0/computation/index.html)。

## 6. 官方成功判据、reset与回合统计

原`factory_env.py:_get_curr_successes`直接检查，PegInsert check_rot=False；held-base与
fixed-target-base局部offset都是[0,0,0]，所以使用当前两者root，不是COM、孔顶部观测
参考或reset缓存。精确数学条件：

```text
s = (sqrt((p_fixed.x-p_held.x)^2 + (p_fixed.y-p_held.y)^2) < .0025)
    AND (p_held.z-p_fixed.z < .025*.04)
```

世界XY严格小于2.5mm，**有符号Z严格小于1mm**；等号失败，没有abs、深度下限、
orientation、接触、保持时间或force条件。几何成功不保证稳固/无穿透装配；大幅低于
孔根也可满足z子条件，不能私自把此现象修成新成功谓词。目标程序映射保留原FP32：

```python
held = torch.as_tensor(current_held_root, dtype=torch.float32)
fixed = torch.as_tensor(current_fixed_root, dtype=torch.float32)
xy = torch.linalg.vector_norm(fixed[:, :2] - held[:, :2], dim=1)
z = held[:, 2] - fixed[:, 2]
success = (xy < 0.0025) & (z < 0.025 * 0.04)
```

S0实际提取并执行了SHA核验后的**原函数AST**，以等号、FP32 nextafter邻点和负z
合成输入对照该映射；这不是目标引擎 rollout。helper中NumPy norm没有被宣称在所有
浮点状态与GPU求值逐位相同，S1应保留上面的原torch程序并验证near-boundary数值。

每policy步演化后、auto-reset前评估success；ever=该回合mask之OR，terminal=done步
的pre-reset mask，first_success=首次true的官方1-based counter乘8/120s。
成功不提前终止；timeout149步，返回post-reset obs不能用于判断前回合terminal。

原reset包含关闭全场重力、fixed root初始(.6,0,.05)加uniform±.05m和yaw[0,2pi)、
hand目标在孔顶上.047m加位置±[.02,.02,.01]及yaw±.785rad、IK失败重采样。peg在手内
z相对量.032392m并加hand-local±[.003,0,.003]m扰动，随后至少.25s模拟夹紧。
reset Kp[300,300,300,20,20,20]、旋转Kd另除10，结束恢复episode gains与重力；辅助
physics时间不计主episode149步。peg独立接触抓握，不能暗中weld。完整reset/yaw-cache
公式见两个审计附录。相同seed不证明两个引擎初态配对；未来应固定抽取的随机量与
post-grasp状态，单列原始与映射差异，不拼接不完整回合。

## 7. CPU验证、失败与保全

| 项目 | 执行结果 | 证据边界 |
|---|---|---|
| 全部相关CPU回归 | **559 passed，1 skipped，0 failed，exit0**，15.856s | 原534项通过均保留；新增S0 25项通过；缺matplotlib导致原唯一绘图skip |
| 系统Python绘图补测 | **1 passed，exit0**，1.317s | 仅重跑该skip，无安装；去重后共560个case通过 |
| MuJoCo import/候选compile | 两个model PASS，零physics步 | 版本3.5.0、模型结构/API和文件解析，不是物理一致性 |
| 原USD CPU读取/引用 | PASS，3.821s，peak123,660KiB | 原3层包围盒reference保持一致；没有Kit/physics |
| Primary checkpoint CPU元数据 | PASS | SHA前后一致、epoch/frame/架构/RMS精确匹配；未改权重 |
| 内容保全 | **1,616 tracked文件、3 checkpoint、5项原G3V未跟踪文件前后相同** | 全部SHA与执行位；14个冻结执行源码/配置亦核对 |

新增25项覆盖19D排序/FP32组装、q顺序与sign、世界变换、1/120FD及reset0、恰180°
分支、非交换左旋转、clip→EMA→scale/no-reclip、/.05 reset cache、每子步target重建、
显式joint映射、严格success边界/one-sided z、冻结RMS与LSTM列reset、原controller
输入输出维度/nullspace与奇异拒绝、非法配置及NaN/Inf。没有实现完整目标旋转投影、
MuJoCo控制器、碰撞世界或rollout；这些仍是PROPOSED。

CPU `g4s0_mapping.py` 的几何/动作代数使用float64合成夹具，观测拼接和success显式
FP32；这不是对所有NumPy/SciPy/PhysX/GPU舍入的逐位认证。controller测试执行原
`compute_dof_torque`，仅在identity姿态夹具把已知零角误差作为输入；验证9D输出、
6D wrench、J/M/nullspace代数维度，不验证真正applied torque。

保留的负证据包括：默认系统/Factory Python无MuJoCo、Factory常见许可URL四次404、
`assets_audit_initial_failed.json`的reference prim路径错误，以及
`assets_audit_serialization_failed.json`的USD authored inf严格JSON序列化错误。
后两项是审计器开发缺陷，修复后保留原失败与新成功记录；不伪称物理引擎或策略失败。

复现（输出必须新建，拒绝覆盖已有证据）：

```bash
# 系统Python：原资产读取/现有依赖CPU解析；可选HEAD仅取header，不下载资产正文
python3 scripts/g4s0_asset_audit.py --output /tmp/g4s0_assets_new.json --network-head
# 全CPU regression、绘图补测与前后保全（内部使用现有冻结Factory venv）
python3 -m scripts.audit_g4s0 --output evidence/g4_s/s0/cpu_validation_NEW
```

测试进程固定`CUDA_VISIBLE_DEVICES=''`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，临时目录
唯一且位于/dev/shm，不排除原有测试。完整命令、JUnit/console、依赖缺项及保全清单见
[cpu验证汇总](../evidence/g4_s/s0/cpu_validation_20261026T_g4s0/summary.json)。
只读资产probe的命令、Python路径/版本、returncode、原stdout/stderr、RSS与源hash见资产JSON；
PYTHONPATH/LD_LIBRARY_PATH设置位于其中hash固定的collector源码。
原失败不被成功日志覆盖。总证据索引为[summary.json](../evidence/g4_s/s0/summary.json)。

## 8. 验收门禁与G4-S1入口估计

| S0验收项 | 结论 | 理由 |
|---|---|---|
| 真正不同的引擎 | VERIFIED | MuJoCo3.5.0已导入/CPU解析；不是PhysX独立推理换名 |
| 同一robot/assembly可重建 | CONDITIONAL | 原缓存与结构数据存在；完整依赖、惯量和孔碰撞转换未完成 |
| 19D接口映射 | VERIFIED原定义 / PROPOSED目标 | 逐项可计算；privilege与raw quaternion sign必须显式保留 |
| 6D/controller路线 | VERIFIED原公式 / PROPOSED目标 | 直接torque、world J/M/FD可表达；actuator路径尚未验证 |
| 原success可保留 | VERIFIED定义与CPU边界 / PROPOSED目标调用 | 原AST边界复核；pre-reset/149步语义已明确 |
| 关键mismatch识别 | VERIFIED审计 | 指质量、mimic、SDF、重力、+.01I、contact、precision与随机流 |
| 资产许可约束满足 | **UNKNOWN / 分发BLOCKED** | 不足以覆盖全部Factory修改资产；故不得给GO |

因此选择**CONDITIONAL**而非GO；也没有证据说明忠实重建必然不可行，故不武断NO-GO。
G4-S1必须另获授权，并且先满足：

1. 确认逐资产使用、转换和必要分发范围；冻结准确来源、依赖、版本与hash。许可未决
   期间不公开衍生模型；若最终无法获得必需授权，转NO-GO而不是静默换任务。
2. 补齐桌面/必要collision依赖；按实际引用坐标、degree→rad、mass覆盖后的CoM/inertia
   和每body+.01I构建专用Panda模型，保留finger/force-sensor/TCP与夹持关系。
3. 为孔道保留拓扑并量化转换几何误差，误差应相对57µm名义间隙可审查；不能扩大孔、
   缩小peg、替换指垫或放宽成功阈值。
4. 固定MuJoCo版本/积分器/solver、.75摩擦映射及gravity-disable，先做无策略的同姿态
   FK/J/M、19D/raw sign/FD/RMS、原6D target/torque公式对照。
5. 单独核对直接effort、configured limits、实际actuator force、finger PD/mimic以及
   reset/随机量/主回合和辅助时钟，再审阅后决定短physics测试。不得提前rollout。

**PROPOSED工程工作量估计**（不是已授权计划或完成日期）：资产/依赖与惯量转换约
1–3工程日；frame/FK/J/M与观测对照约1–2日；原目标/torque/actuator映射约1–2日；
孔几何与接触/夹持、reset短检查约2–4日，合计约5–11工程日，存在重叠。此估计来自
本次已识别接口与差异，许可审批/缺失资产获取时间无法可靠估计；若孔SDF/惯量无法
忠实转换，工期与可行性需重新审阅。没有预设S2成功率。

## 9. Git交付与STOP

本次只提交新增G4-S0报告、CPU审计脚本/测试与`evidence/g4_s/s0/`，旧源码、历史
实验/失败、reward/成功谓词、physics、checkpoint和原工作树G3V均保留。提交前再fetch，
确认远端为本地祖先；只做fast-forward push，之后独立`ls-remote`核对SHA。最终
commit及实际远端核对在交付回复给出，避免在提交内嵌自身SHA形成自引用。

**停止在G4-S0。没有开始G4-S1、G4-S2、GearMesh、NutThread、seed2、训练、微调或
长物理实验；没有声称Sim2Sim PASS。**
