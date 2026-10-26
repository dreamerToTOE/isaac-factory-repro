# G4-S1B：适配型 MuJoCo Panda PegInsert zero-shot Sim2Sim 工程门

日期：2026-10-26（Asia/Seoul）。仓库 `dreamerToTOE/isaac-factory-repro`，分支 `main`。
起点：`6e282557e2fd0c689932b6a572b53d987ad4c8d3`（G4-S1A 已推送结果）。

**工程状态：`G4_S1B_PASS`。科学 zero-shot 策略效果：`NOT_ASSESSED`。**

这里的 PASS 只表示适配后的 MuJoCo 环境、观测/动作接口、控制器数学链和无策略物理
验证已经建立并通过验收。它不是原 Factory 的精确重建，也不是冻结 PPO 在 MuJoCo
中的成功率结果。G4-S1A 对精确 NVIDIA Factory USD/PhysX cooked SDF 重建的 NO-GO
保持有效；本阶段使用新的、明确批准的 **ADAPTED ZERO-SHOT SIM2SIM** 协议。

本次没有加载或执行 PPO actor，没有反序列化 checkpoint，没有运行 policy episode、
训练、微调、G4-S2、GearMesh 或 NutThread。S2 只能在本报告审阅后另行授权。

## 1. 适配协议和边界

冻结的源策略身份仍是 G2 epoch-200 checkpoint，SHA-256：
`f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`。本阶段只验证
其接口所需的目标环境能力，不读取该文件。

| 项目 | S1B 协议 |
|---|---|
| 目标引擎 | MuJoCo 3.5.0，实际安装环境版本核对为3.5.0 |
| Panda | Menagerie `franka_emika_panda/panda.xml`，commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`，Apache-2.0 |
| peg/hole | 本仓库独立原创圆柱 peg、256块保孔凸扇区、孔底和支撑体；不使用或转换 NVIDIA Factory mesh |
| 任务维度 | 7臂关节 + 2夹指关节；peg为独立 free joint，nq/nv=16/15 |
| 时间 | physics dt=1/120s；每次参考动作8个MuJoCo步；仿真策略频率15Hz |
| 主尺寸 | peg直径0.007986m；hole直径0.0081m；名义径向间隙57µm |
| 状态假设 | 使用 simulator ground-truth 的 TCP pose、hole参考和peg pose；这是 privileged-state 适配，不是感知/传感器部署 |
| 成功口径 | 原 Factory 几何 predicate 独立计算；接触、孔内、孔底和穿透为另一个物理诊断，不混为成功 |

原 raw Factory 几何在 S1A 中测得的直孔段间隙≥505.921µm、cooked SDF未知；本报告的
新模型不声称与该 raw 或 cooked几何相同。路线文档和 README 已以前瞻性方式记录本目标
变化，旧的 S0/S1A 文字未追溯改写。

## 2. 资产、许可和来源

完整来源和逐文件 SHA 在 [protocol_review.json](../evidence/g4_s/s1b/protocol_review.json)、
[environment_no_policy_final.json](../evidence/g4_s/s1b/environment_no_policy_final.json)
及 [models/g4_s1b/README.md](../models/g4_s1b/README.md)。固定 Panda 的 LICENSE、
Isaac Lab BSD-3-Clause source-port notice 和 README 均保存在 `models/g4_s1b/`。

- 读取了固定 Menagerie Panda 及其 78 个 XML/license/mesh 文件；collector 对其哈希进行
  复核，没有复制这些外部 mesh 到仓库。
- peg、孔壁、孔底、支撑体由本仓库 MIT 代码按解析尺寸生成；没有 Factory USD、顶点、
  三角面、PhysX cooked buffer 或其衍生物进入新模型。
- 原 Factory 资产的许可缺口仍由 S1A 保存；本适配环境没有以 Panda Apache-2.0 或本仓库
  MIT 许可证重新授权 NVIDIA 内容。

## 3. 环境实现和碰撞几何

实现位于 [g4s1b_env.py](../scripts/g4s1b_env.py)。`model_xml()` 只读取固定 Menagerie
Panda，检查 commit 和 Panda 子树无未提交修改，然后在内存中改写 timestep、重力、
碰撞过滤、TCP site 和 actuator，并添加原创装配体。

### Panda、夹指和 peg

- 继承固定 Panda 的7轴、2个slide finger、质量/惯量和机器人碰撞 mesh。
- 移除 Menagerie 默认7个位置伺服和split-tendon执行器，使用7个直接 torque motors；
  软件命令先限幅到±100Nm，MuJoCo actuator再按源前4轴87Nm、后3轴12Nm的`forcerange`
  限制。每步保存 requested torque、software-limited torque、`actuator_force`和
  `qfrc_actuator`。
- peg有独立 free joint、质量0.019kg、解析圆柱碰撞体和独立位姿；最终配置没有
  peg-to-hand weld。模型 equality 只有两夹指对称 joint 约束，`has_weld=false`。
- hand增加 hand-local z=.11207100003957748 的`factory_tcp` site。原 controller所需的
  Jacobian仍取左右 finger body-origin world geometric Jacobian 的平均值；TCP site
  Jacobian另行计算和差异报告，不能混用。

### 孔道

孔由256个凸梯形棱柱扇区组成，内半径采用0.00405m的名义配置值，外半径0.02m，
高度0.025m；底部是厚0.003m的独立实体。普通单凸包不会填死开口。模型编译后测得
内apothem范围0.004049694575–0.004049695506m；理想内弦侵入约0.305µm，理想剩余
径向间隙约56.695µm。该编译测量是浮点读数，不是严格的全局 rounding certificate。

所有装配 geom 显式使用 margin=0、gap=0；接触 solref/solimp、Newton solver、100
iterations、elliptic cone 记录在机器证据中。MuJoCo 接触模型不是 PhysX 等价物。

## 4. 19D observation 和 6D action

实现位于 [g4s1b_contract.py](../scripts/g4s1b_contract.py)，完整字段和源 SHA 在
[controller_contract.json](../evidence/g4_s/s1b/controller_contract.json)。

19D 顺序固定为：

1. TCP world position − noisy reset hole tip，3个 float32 m；
2. world TCP quaternion，wxyz，4个分量，不做符号规范化；
3. 每个1/120s物理步的 world linear finite difference，3个 m/s；
4. `q_now * conjugate(q_previous)` 的 shortest-sign axis-angle finite difference，3个 rad/s；
5. EMA action cache，6个 normalized action分量。

reset 时 FD为零，EMA translation cache按原规则不裁剪，yaw cache保留原 modulo/branch；
episode noise可由独立seed一次生成或显式提供。冻结RMS只接受外部不可变mean/variance，
按float32执行`(obs-mean)/sqrt(var+1e-5)`并clip到[-5,5]，不更新统计。LSTM h/c没有加载；
ReferenceRunner只记录未来 actor reset pending，不能冒充已执行策略。

动作/控制链保持：

`raw mu → clip[-1,1] → EMA(.2,.8) → 每个物理步从当前pose重建Cartesian target → world task-space PD → Jᵀ wrench → dynamically-consistent nullspace → software ±100Nm → MuJoCo actuator`。

位置增量为world坐标×0.02m，并围绕fixed action frame逐轴±0.05m限幅；旋转为axis-angle
×0.097，world-left quaternion multiplication，然后保留source EulerXYZ yaw并强制
roll=3.14159、pitch=0。任务 wrench为`Kp*pose_error-Kd*[FD linear,FD angular]`，
Kp=[100,100,100,30,30,30]，Kd为原critical-damping值。nullspace严格使用7×7 arm mass
block、source default q、wrap差、J/M inverse和`(I-JᵀJ#)u`，不加入gravity/Coriolis
feedforward、damped pseudoinverse或隐式调参。

控制器对非有限输入、非单位quaternion、非正定/奇异mass或rank-deficient J显式拒绝，
不以零力矩、伪逆或自动重试掩盖错误。24个独立源AST Torch float32 fixture 与 NumPy
float64 port 的最大误差为：target `1.8884e-7`、software torque `7.0201e-6Nm`、
wrench `6.6733e-6`、pose error `2.2253e-7`、dynamic-null residual `2.7303e-15`。
这些是固定fixture界，不是全球误差证书。

夹指采用一个明确登记的目标引擎离散隐式PD适配：保留7500/173和40N参数，用当前
finger质量矩阵的后向Euler自由运动预测计算每个物理步的force，再进入MuJoCo。它不是
PhysX implicit drive 的等价证明。作为独立负控，直接使用affine PD并在±40N饱和会让
finger qpos一步到−0.165185m，越限约0.165185m，已拒绝为最终配置。

## 5. 无策略 MuJoCo 物理验证

最终收据为 [environment_no_policy_final.json](../evidence/g4_s/s1b/environment_no_policy_final.json)，进程命令、退出码和输出 SHA 另见 [environment_process_receipt.json](../evidence/g4_s/s1b/environment_process_receipt.json)。
真实MuJoCo进程退出码为0，执行518个有界物理步，运行约0.961s，peak RSS约由JSON记录。

| 检查 | 结果 |
|---|---|
| 真实加载/结构 | MuJoCo3.5.0；nq/nv/nu=16/15/9；free peg joint存在；无weld |
| FK/Jacobian | TCP Jacobian FD最大误差1.561e-9；source finger-origin linear J FD最大误差1.339e-9；TCP与source J的最大结构差0.0517（按设计分别保留） |
| mass matrix | 对称误差0；最小特征值0.006292，7×7 arm block正定 |
| gravity/static hold | robot gravcomp=1；24步/0.2s最大关节漂移2.22e-15rad |
| torque direction | 7个关节各施加+0.001Nm，7个own qvel均为正；方向检查通过 |
| saturation | requested[200,-200,101,-101,20,-20,12.5]→software[100,-100,100,-100,20,-20,12.5]→actual[87,-87,87,-87,12,-12,12]，qfrc一致记录 |
| free fall | peg关闭gravcomp后12步，终速−0.981m/s、下落0.0531375m，与半隐式解析值一致 |
| open bore/wall/rim/floor | 轴心无接触；偏移到wall有真实peg-wall contacts；rim、floor和above-opening分类均保留 |
| physical grasp | 独立peg、无weld；90步中82步双指正法向接触，最后30步全为双指接触；hand-relative slip约23.57µm |
| direct insertion | 高速gravity-on free drop未成为物理插入候选，结果保留为失败负控 |
| gentle insertion | gravity-compensated peg施加显式−0.001N bounded reference force，最终floor contact、inside bore、final penetration约0.269µm，物理候选为true；全程峰值瞬时signed penetration约−96.49µm，故不能称无穿透路径证书 |
| reference bridge | 3个显式6D reference actions；每次8步、19D返回、targets每个物理步重建；不包含actor |

低速插入结果只证明当前适配几何存在一个有界、直接放置的插入终态；它不证明机器人
能把peg抓起并完成插入，也不证明接触求解器无穿透。高速自由落体失败和瞬时穿透被
保留在证据中，不能通过放宽阈值改写为PASS。

## 6. CPU回归和保全

最终回归收据：[cpu_validation_20261026T_g4s1b/summary.json](../evidence/g4_s/s1b/cpu_validation_20261026T_g4s1b/summary.json)。

- 新增controller与environment测试：**27 passed**，其中环境测试通过独立MuJoCo3.5.0子进程执行真实collector。
- 选定历史回归：562 collected，561 passed、1 skipped、0 failed；另系统Python绘图测试1 passed。
- 73个会执行actor/from_checkpoint的历史policy测试按S1B“不得执行actor”要求排除，原测试文件未修改；排除清单和历史数量在 `audit_g4s1b.py` 收据中。没有声称全套actor回归已运行。
- no-policy tripwire的独立dummy负控在函数体执行前退出97；正式两组pytest guard收据均 `active=true, forbidden_calls=[]`。
- 起点历史tracked文件、20份冻结源码/配置、3个checkpoint（只哈希）、4个原USD、5个G3V文件以及78个固定Panda文件的前后哈希一致。README和路线文档是本阶段获准的前瞻性更新。
- 真实退出码、命令、stderr、JUnit、runner源码和保全快照均在CPU evidence目录。没有安装包，没有改动checkpoint，没有GPU调用。

## 7. 与 PhysX 的差异和限制

1. Menagerie Panda 的质量、碰撞mesh、指垫、右指frame与Factory Panda不同；S1A的精确
   asset/geometry NO-GO仍适用。
2. 新模型采用名义57µm尺寸，原raw hole约4.5mm和cooked PhysX SDF未知；不能把两者
   作为同一碰撞几何。
3. MuJoCo Newton/implicitfast、soft contact、solref/solimp、friction和solver iterations
   不是PhysX TGS参数的一一对应；没有做轨迹或接触力逐位parity。
4. TCP quaternion使用MuJoCo hand xquat链的目标引擎分支；跨引擎raw quaternion sign
   parity未知。NumPy float64控制器与源Torch float32也有固定舍入差异。
5. 夹指step-implicit force是适配实现，不是PhysX implicit drive重建；实际grasp结果为
   独立静态close/hold诊断，不是机器人pickup、policy或官方成功率。
6. reference bridge仅使用显式人工6D actions，未加载checkpoint；因此本报告没有
   ever-success/terminal-success、first-success或zero-shot policy性能数字。
7. 低速插入终态有微小最终接触穿透、快速drop有更大的瞬态穿透；这限制了后续接触
   质量解释，不能被隐藏为理想无穿透。

## 8. 复现命令和S2入场条件

```bash
# 真实MuJoCo3.5.0无策略物理collector（拒绝覆盖已有文件）
/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python \
  scripts/g4s1b_env.py --output evidence/g4_s/s1b/environment_NEW.json

# controller + environment CPU tests
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/envs/factory-g01/bin/python \
  -m pytest -q tests/test_g4s1b_contract.py tests/test_g4s1b_env.py -p no:cacheprovider

# 选定无策略历史回归和保全收据（新目录必须不存在）
python3 scripts/audit_g4s1b.py --output evidence/g4_s/s1b/cpu_validation_NEW
```

S2尚未启动。可单独考虑S2的前置条件为：审阅本报告与全部JSON；接受“适配模型”而非
精确Factory模型的术语；冻结当前MuJoCo XML/geometry/controller hash；决定如何处理
低速插入的瞬态穿透；确认checkpoint读取安全流程、评估seed/分母、LSTM/RMS和pre-reset
记录协议；另外授权有限的policy episodes。S2不得把本阶段人工reference动作结果当作
zero-shot策略结果，也不得因当前PASS自动启动。

**STOP：G4-S1B结束；G4-S2、训练、GearMesh、NutThread均未开始。学习与理论考核仍为PENDING。**
