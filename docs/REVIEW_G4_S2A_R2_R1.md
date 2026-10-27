# G4-S2A-R2-R1：晚期抓持失败、手指惯量与接触动力学调查

日期：2026-10-27（Asia/Seoul）；仓库 `dreamerToTOE/isaac-factory-repro`，`main`。起点 `5abe4375c2bea828816a7d2f0ff04e9ee0673ff0`。

**工程门：`G4_S2A_R2_R1_PARTIAL`；候选判定：`CANDIDATE_PARTIAL`；R3 / S2B：NO-GO。** 本阶段是无策略有界诊断，不是任务级 zero-shot 成功率实验。八条轨迹完成并有一条部分 raw trace，但一条监督进程丢失使九格记账未达到 PASS。

## 1. 问题与冻结边界

S2A 在 MuJoCo 中出现早期振荡与脱抓。R1 发现 wrist J7 在达到执行器限幅前已反向放大；R2 隔离物理 plant 与臂 controller 的 M 后，适配物理惯量能够抑制三初态的最初振荡，而仅替换 controller M 不能。因此早期问题的局部效应有证据，但 R2 的 I3 在较晚时间仍出现指垫接触损失和严重穿透。这是第二个问题，不能把早期稳定误写成整个抓持过程或策略迁移已修复。

保持原 S1B 默认模型、所有历史证据和 checkpoint；本阶段不执行 PPO、训练或微调，不生成替换初态，不更改 peg/hole、finger 质量/CoM/碰撞/限位、PD/力限、接触求解器、步长和原 success predicate。P_FULL 与 P_ARM_HAND 是明确标注的目标引擎诊断，而非精确 Factory PhysX 重建；PhysX runtime 惯量和力仍未知。

历史依据：[S2A](REVIEW_G4_S2A.md)、[R1](REVIEW_G4_S2A_R1.md)、[R2](REVIEW_G4_S2A_R2.md)、[归档取证及原始哈希](../evidence/g4_s/s2a_r2_r1/forensics_v1/forensics.json)。

## 2. R2 归档失败重建：先找事件，不预设根因

独立读取 I3_B 的 150–177 步、I3_D 的 190–216 步，并与 I2_B/D 的对应窗口比较。I3_B 第174步四个常见 peg/finger 几何接触仍存在；第175步左 `left_finger_geom_4` 从记录的接触集合消失、左总正法向力仍约0.146N，第176步左正接触全失、右约0.0684N，第177步重新出现多个接触对，`peg_collision`/`left_finger_geom_7` 穿透 1000.772µm 并立即按原阈值早停。I3_D 对应第213步仍有四对，第214步 `left_finger_geom_4` 消失，左总力约0.144N，第215步左正接触全失、右约0.0670N，第216步相同 pair 穿透1047.671µm并早停。I2_B/D 同时间窗口仍有双侧正接触，不能将不同初态当成可交换样本。

在失去接触前一步，两例最大穿透分别为约6.863µm/5.814µm；失接触时约7.113µm/5.053µm。失接触后的下一步，finger qvel 左右分别约 −0.08269/−0.08070m/s 与 −0.08193/−0.08000m/s；相对转动分别跳至0.05884/0.06077rad，越过0.05rad通过门。I3_B 第176→177步左finger qpos约3.21579→2.52673mm；I3_D 第215→216步约3.18026→2.49747mm。再出现的左侧接触法向总力约74.56/89.30N；它是接触反力，不能等同 finger 执行器输出或40N cap。

归档前积分原始 PD 指令在末步约−24.03N/−23.74N，隐式预测后实际施加 finger force约−0.182N/−0.180N（左右分别记录）。原始 PD 由前状态 q/qvel 与冻结 gains 重算，数值和归档一致；必须区分 raw PD、隐式命令、积分执行器力与接触法向反力。完整每步 q/qvel、各指接触对与施力、完整物理/控制器 M 和 post-forward 汇总约束量见取证时间线。

这显示 **接触集合变化→左正力失去→下一步严重穿透** 的记录次序，而非证明接触 pair 重分类是唯一原因。所存 contact position 是接触中点；用静态 FK 投影可看到部分点接近 finger box 边界，但不能由中点证明真正碰撞特征属于边/角或判定几何缺陷。归档 qacc、contact force 和汇总 `qfrc_constraint` 是每步后 `mj_forward` 的重新计算结果，不等于之前积分使用的接触冲量；原归档没有分开的 equality/contact solver impulses、完整子步 manifold 或迭代过程，不能据此唯一解释 solver/等式约束。详见[几何投影](../evidence/g4_s/s2a_r2_r1/forensics_v1/geometry_projection.json)。

## 3. 初态准备质量与历史准入

历史 reset admission 检查仍为 **RESET_ADMISSION_PASS**，不追溯改变结果。另行标记 **RESET_PREPARATION_CONTACT_QUALITY_WARNING**：I3 原30步准备第2步 `peg_collision` / `right_finger_geom_7` 瞬态峰值873.514µm，步骤2–12超过后续50µm pad质量阈值，最后恢复至约6.524µm，最终左右正法向力约0.1692/0.1683N。I1 原归档准备第1步峰值316.328µm、步骤1–7超过50µm，末步约6.947µm；I2峰值22.444µm，未超过50µm，末步约6.816µm。I1 最终还在后续R2 B/D维持235步，因此瞬态警告是应调查的候选解释，不充分证明 I3 晚期失败由准备瞬态导致。

I3 最终双侧力、末10步相对姿态及无非pad干涉满足旧准入协议；并不能反推整个准备轨迹均满足后来制定的50µm质量门。原始准备和接触几何中点完整记录见 [I1](../evidence/g4_s/s2a_r2_r1/forensics_v1/I1_preparation.json)、[I2](../evidence/g4_s/s2a_r2_r1/forensics_v1/I2_preparation.json)、[I3](../evidence/g4_s/s2a_r2_r1/forensics_v1/I3_preparation.json)。

## 4. 惯量与 controller 隔离

| 物理配置 | 物理转动惯量改变 | 指惯量 |
|---|---|---|
| P_FULL | link0..7、hand、left/right finger 的11个映射 body 各加 `0.01 I kg·m²` | 增加；对应R2 P1 |
| P_ARM_HAND | link0..7 和 hand 各加 `0.01 I kg·m²` | 保留 Menagerie 原值 |

原 finger 主惯量约 `2.375e−6 / 2.375e−6 / 7.5e−7 kg·m²`，FULL 变为 `0.010002375 / 0.010002375 / 0.01000075`。仅改变转动惯量，不改指质量0.015kg、CoM、形状等。固定的 link0 按映射记录；源 sensor/TCP body无一一映射。物理 M 的变化为 `ΔM=Σ 0.01 JωᵀJω`，不是同值关节 armature。

C0 始终取原 S1B 模型在同一完整 q 的臂 controller M；C1 取当前所选物理配置在同一 q 的臂 controller M。原 finger step-implicit 控制仍读取真实 plant 完整 M 的逆矩阵 finger block，C0/C1 不替换这条支路；因此更改 finger inertia 可能改变该预测输出和接触载荷，但这是待检验假设。静态 MuJoCo 3.5.0 检查验证两个物理模型 × 两个控制器选择、对称正定、原 finger 与冻结属性、独立 Jacobian 动能 oracle 和无物理积分；完整矩阵及模型身份见[预注册](../evidence/g4_s/s2a_r2_r1/matrix_v2/preregistered_matrix.json)与[静态门禁](../evidence/g4_s/s2a_r2_r1/regression_prelaunch_v3/static_models_guard.json)。

## 5. 预注册、时间与质量门

使用R2已记录I1/I2/I3，不重新准备或筛选：I3_FULL_C0/C1、I3_ARM_HAND_C0/C1、I2_ARM_HAND_C0/C1、I1_ARM_HAND_C0/C1为8条主对照；I3_FULL_PASSIVE_ARM 是单独的零臂命令负控，不与主对照合并。所有轨迹共享R2已归档的两条动作：步1–8第一条、步9–16第二条；步17–235固定该初态 admitted TCP 参考。每条最多235步，最多9条、合计2115新增积分步，协调器含 GUI 与记录 wall budget 900s；失败保留、不调参重试。矩阵 v1 在0新增积分时保留，复审增加 GUI 同步之后及积分边界的外力检查后重新预注册为 v2，详见[修订记录](../evidence/g4_s/s2a_r2_r1/preregistration_amendment.json)。

质量门沿用 R2：每记录步左右 pad 力均 >1e−6N；平移滑移 <1mm、旋转滑移 ≤0.05rad、臂速度 ≤3rad/s、臂/指越限分别 ≤1e−6rad/m、pad穿透 ≤50µm、装配接触穿透 ≤10µm、全有限且无新增数值警告并保留完整力矩与接触证据。早停：全接触穿透 >1mm、平移 >5mm、旋转 >0.2rad、臂速度 >15rad/s、臂越限 >1e−5rad、指越限 >1e−4m、连续3步丢失正接触或其他数值危险。偶发一次失接触即不能通过持续抓持门，即使未达到三步早停。早停轨迹不是235步成功，进程exit0也不是物理质量通过。1mm滑移是持握门，不是57µm间隙插入精度。

## 6. 实际 GUI 和可复现性

实际执行使用 MuJoCo 3.5.0 原生 `mujoco.viewer.launch_passive`，桌面为 `DISPLAY=:1`、X11、X.Org 1.21.1.4；已完成子进程记录 `IsViewable` 窗口和 render-thread join。同步后与积分前均检查外力。交互回放命令：

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority /home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/g4s2a_r2_r1_viewer.py --matrix evidence/g4_s/s2a_r2_r1/matrix_v2 --run I3_FULL_C0
```

Space 播放/暂停，左右键单步，L 跳到接触损失前，S 慢速，V/Z 切换视角。保存状态回放只加载 qpos/qvel 并 forward，不重建历史 solver impulse；微米和毫米定量来自 raw trace。截图按 `LIVE_PHYSICS_INTEGRATION` 与 `SAVED_STATE_PLAYBACK` 分开标注，污染原图由 `visuals/private_capture_availability.json` 记录并保存在 Git 外。

## 7. 新实验与独立数值复核

新的8条轨迹完整结果与 I3_FULL_C1 的部分 raw trace 均保留在 `evidence/g4_s/s2a_r2_r1/matrix_v2/`：I3_FULL_C0 在177步复现R2的176/177接触损失/穿透；I3_FULL_C1 有211条physics结果，随后第212请求处监督进程丢失，退出码未知且不重跑；P_ARM_HAND 的 I1/I2/I3 六条对照在15、25、7、7、8、8步早停；I3_FULL_PASSIVE_ARM 在154步早停。独立 NumPy reader 对8条有完整 receipt/result 的轨迹重算401个可审阅 physics rows；durable trace 的原始事件记账为613个 integration requests，其中612个有对应 physics results，包含 I3_FULL_C1 的211行。`numerical_review_passed=false` 仅反映缺失 I3_FULL_C1 receipt/result。P_ARM_HAND 的负结果支持 finger inertia/contact branch 敏感性；被动臂失败削弱“臂力矩唯一原因”；唯一晚期根因仍 `INCONCLUSIVE`。

## 8. 候选判定与剩余未知

候选判定为 **`CANDIDATE_PARTIAL`**。P_FULL 保留为版本化目标引擎诊断，但 P_ARM_HAND 反例、I3_FULL_C0/PASSIVE_ARM 早停以及 I3_FULL_C1 记账中断，阻止 `CANDIDATE_SUPPORTED`。当前不推荐 R3；需先解释 I3 接触质量边界并在等价受控条件下完成有效记账。PhysX runtime M、实际 impulse、完整 contact manifold 和原始指驱动仍 UNKNOWN；不声明全局根因，不修改 S1B default。

## 9. CPU回归、保全与交付

[prelaunch_v3](../evidence/g4_s/s2a_r2_r1/regression_prelaunch_v3/summary.json)：721 passed、1 skipped，额外系统绘图1 passed；27项新测试，75项明确排除（73会执行actor的历史用例、1真实策略前向、1含518步无头物理采集）；5种禁止入口负控通过。静态检查和回归没有策略、checkpoint反序列化、积分或GUI。预启动 v1 因汇总读取旧字段名而误报 FAIL，静态子进程实际通过；原失败收据保留。v2 修复汇总通过，后因GUI外力检查加强而0积分重新预注册，v3通过。

保护2316个已有跟踪文件及3个 checkpoint、历史 USD 源、配置和 Menagerie 文件，唯一允许更新既有案例正文/索引。prelaunch v3 为721 passed、1 skipped，系统绘图1 passed，27项新增测试；静态MuJoCo四组合和五类入口负控通过，0 actor/physics/checkpoint。R2-R1 durable trace 记录613/2115个 integration requests，其中612个完成 physics result；另有第212请求未完成且监督丢失，没有自动重试。最终提交与远端 SHA 在交付提交后补入本节。
