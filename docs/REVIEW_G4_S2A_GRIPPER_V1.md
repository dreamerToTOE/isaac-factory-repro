# G4-S2A-GRIPPER-V1 — MuJoCo 原生 Panda 夹爪重建

**工程状态：`G4_S2A_GRIPPER_V1_PARTIAL`；候选：`NATIVE_GRIPPER_PARTIAL`；R3 / S2B：`NO-GO`。** 单一版本化原生夹爪已实现且静态力学映射通过。I2 唯一一次准备第一步即严重穿透早停；I3 实际积分前状态与登记初态不同，该次实验无效，不能评价 I3。两条操作诊断按门禁均未启动。独立数值复核总状态为 `INCOMPLETE`。这不是冻结 PPO 的任务成功，也不是 PhysX 精确复刻。

## 起点、隔离和保全

起点 `ce8815c2586e157a04e1dafb7a25d726c8dba264`；开始前本地与远端 main 一致。历史 S1B、自定义指控制器、全部原始模型、checkpoint 与既有失败证据未覆盖。新候选只在 `scripts/g4s2a_gripper_v1_model.py` 中构造；没有训练、PPO 推理或完整插入 episode。保留两根实体 finger joint、独立自由 peg、开放孔洞和原碰撞；没有焊接、禁碰或修改 success predicate。机器人 P_FULL 的源启发转动惯量改动仍是适配，不是已证明的 PhysX 惯量。

## 单一原生设计和静态门

在冻结 S1B XML 上应用既有 P_FULL，再仅移除双指 motor 和 finger joint equality，以单位系数固定 tendon `L=q1+q2` 和一个原生 position actuator 取代。原本每指 `Kp=7500 N/m, Kd=173 N·s/m`；对称时位置和速度误差在 tendon 坐标加倍，选 `kp=3750 N/m, kv=86.5 N·s/m`，每指目标 `q_target` 映射为 `L_target=2*q_target`。控制范围 0–0.08 m，执行器力限 ±40 N；编译后的两个 moment arm 均为 +1，因此该模型每指广义驱动力不超过 40 N。反力与执行器力不同。

Tendon 是传动，不是强制等位的约束；取消旧 equality 后，两个独立指允许在不对称载荷下产生差动运动。仍保持相同手指碰撞、摩擦、peg/hole 尺寸和质量、机器人惯量、7 臂驱动/限力、求解器及 implicitfast 的 1/120 s，策略名义频率 15 Hz。此方案不是先前失败的两个独立 affine PD motor，也不保证稳定。

独立静态审计用 MuJoCo 3.5.0 的编译模型和独立 MjData `mj_forward`，没有积分：逐元素验证非夹爪 XML 和已编译的机器人质量/惯量、几何、碰撞掩码、摩擦、arm actuator、时步和 solver；检查 free joint、无 weld、两指限位；以有限差分和不对称位置/速度验证 `dL/dq=[1,1]`，正负饱和和广义力映射均正确。故意改变力限、摩擦、臂力限、增益的负控均被拒绝。证据见 `static_v1/independent_static.json` 和静态测试；静态通过不等于动态抓持通过。

## 预注册和真实执行

`matrix_v1/preregistered_matrix.json` 在物理积分前冻结源文件、XML、seed 2101/2102 的 reset draw/IK/闭合前状态、动作序列、两个阶段、门槛、无重试、最大 530 步/900 s。重建的是归档 `admission.json.initial`，不是旧的闭合后成功状态；30 步准备要求最终 10 步正双侧接触与姿态稳定，并单独对全程及末窗指垫穿透实行 50 µm 质量门。I2、I3 各一次准备；只有两个准备均通过 reset admission 与 physical quality 才运行新准备状态上的两条最多 235 步操作诊断。对应历史 `episode_noise_m` 只为动作参考保留，未把旧抓持状态当成新准备。

| 条件 | 请求 / 积分结果 | 数据有效性 | 结果 |
|---|---:|---|---|
| PREP_I2 | 1 / 1 | **有效初态**，与归档 qpos/qvel 一致 | 第 1 步严重穿透早停，reset / quality 均 FAIL |
| PREP_I3 | 1 / 1 | **INVALID_INITIAL_STATE** | 实际第一步使用模型默认状态，不能据此判定 I3 夹爪质量 |
| DIAG_I2、DIAG_I3 | 0 / 0 | 未执行 | 准备门禁未满足 |

I2、I3 子进程均真实 exit 0，表示记录与受控退出，不表示物理成功。协调器报告总共 2 次积分、15.125 s，未重试，无策略调用。I3 identity 与注册状态一致，但记录的第一步 pre_state 已变为默认的零关节与默认 peg 位置；变化发生在登记后、首次积分前，现有证据不能确定具体机制或输入来源。I3 的 `ARM_LIMIT` 是错误初态的运行记录，不得解释为该登记条件的物理结果。也不能把无效试验事后修正成有效或重复运行。未来另行授权的实现必须在 viewer 同步/截图之后、每次积分请求之前检查完整 qpos、qvel、时间与注册/前步状态，发现变化即阻止积分，并证明可视窗口未被无关内容遮挡。

## I2 接触和求解阶段

I2 闭合前每指开度 0.00499125 m，`L=0.0099825 m`、`Ldot=0`、目标 0。第一步 `mj_step` 使用 tendon actuator 力 −37.434375 N，moment arm `[1,1]`，两指广义执行器力各 −37.434375 N，均未超 40 N。积分求解阶段无接触/无 EFC 行；步末两指约 0.001458372 m。随后原 wrapper 的 `mj_forward` 在新状态记录 10 个 peg/pad 接触，最大指垫穿透 **1.901796 mm**，左/右 pad 法向合力约 **553.439 / 453.937 N**。这些是重新 forward 的反力，不是已用于上一步积分的接触力，更不是冲量。原阈值 1 mm 的严重穿透触发早停；50 µm 质量门也未满足。

可观察到一个离散步从无接触跨入深度相交；这支持该初态、参数和步长组合不满足抓持准备门，但不能仅凭一步确认全局根因、接触模型缺陷或所有原生 tendon 架构均不可用。高反力不是更好的夹持证据。I2 的历史 P0 准备全程最大约 22.444 µm，而历史 I3 约 873.514 µm；历史 P_FULL I2 在另一历史闭合状态完成 235 步。它们与新 P_FULL+tendon 准备并非相同机械机制或匹配状态，不是无偏因果或成功率比较。

## 监督、GUI 和图像

R2-R2 supervisor 在每次真实操作前 fsync request，积分后保存 solver-stage result，再保存进程 wait status；两次各 1 request / 1 result、无未决请求，进程无重试。`physics.jsonl` 按步保存积分阶段和 post-forward 接触/EFC/Jacobian、执行器与广义力、位置/速度、孔/手相对状态、警告及终止原因。积分阶段 qpos/qvel 是步末量，而 solver/actuation 数据对应该步求解，不能混为一个重新求解的状态。

实际 Ubuntu Xorg 会话、`DISPLAY` 和 `XAUTHORITY` 在启动前检查；每次使用 `mujoco.viewer.launch_passive`，并记录 `IsViewable` 窗口及关闭回执。三张实时屏幕截图经逐张检查含无关桌面内容，已删除 PNG，保留各原始 SHA-256 与元数据；一张原生截图有遮挡，不作精细诊断。`playback_i2_v1/` 中两张经检查的干净截图来自独立 data 的**保存状态回放**，没有新积分或 PPO；不能用截图测量微米穿透或重建冲量。回放脚本提供空格暂停/播放、左右单步、S 慢放、L 跳故障、V/Z 和鼠标调视角，已有无积分自动测试及截图路径执行；各按键未逐项人工验收，不宣称其通过人工测试。`playback_i3_v1/` 使用实际积分前状态并标记初态无效；有一张出现来源不明的边缘而谨慎移除，保留哈希。

## 独立复核和工程决策

`scripts/analyze_g4s2a_gripper_v1.py` 只读原始数据，重算 tendon、限幅、广义力、EFC 映射/`Jᵀf`、接触聚合、限位、相对位姿、早停、准备门、持久账本和真实 wait 状态。`numerical_review_v2.json` 将 I2 标为有完整一步数据和准备 FAIL，I3 标为 `INVALID_INITIAL_STATE`，两个诊断标为 `UNSTARTED`，整体 `INCOMPLETE`。最初 v1 读者将状态不连续视为异常并漏计 I3 的一步；保留 v1，v2 修复只读记账后正确计数 2，未修改或重跑实验。

候选技术上可编译且映射成立，但 I2 不满足物理质量，I3 没有有效对照，故为 `NATIVE_GRIPPER_PARTIAL`，没有 supported。停止本参数配置，不搜索增益或提升力限。未来的研究需要用户另行决定，并应先解决状态一致性和截图隔离，再提出有物理依据的准备机制；当前不推荐 R3 或 S2B。即使将来获得两个 235 步通过，也不能直接声称 PPO 插入成功。

## 回归、可复现入口与限制

物理前 51 个静态/合成监督测试通过，零积分。最终受保护历史 CPU 回归 770 passed、5 skipped，另绘图 1 passed；75 个会实际调用冻结 actor 或额外运行未经注册物理积分的历史用例按范围排除。MuJoCo 环境下 56 个模型、静态审计、监督器与复核定向测试通过；独立回放测试另见 `viewer_final_tests.txt`。历史保护检查通过；本次新增源码与证据、案例文档以外不修改起点已追踪文件。测试通过不抹去上述 I3 缺陷。

在仓库根目录执行（独立静态审计必须写入不存在的文件）：

```bash
/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python -m scripts.audit_g4s2a_gripper_v1_static --output /tmp/gripper-v1-static-new.json
/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/envs/factory-g01/bin/python scripts/analyze_g4s2a_gripper_v1.py evidence/g4_s/gripper_v1/matrix_v1 /tmp/gripper-v1-review-new.json
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority /home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/g4s2a_gripper_v1_viewer.py --matrix evidence/g4_s/gripper_v1/matrix_v1 --run PREP_I2
```

GUI 环境变量仅为本机实测值，重现时须动态确认；回放为零积分。物理协调器具有一次性登记和输出保护，禁止对本次试验重试。实际 GUI 测试和原始轨迹不能代替完整 30 步准备或两个 235 步诊断。学习与技术面试考核继续 PENDING。本任务到此停止。
