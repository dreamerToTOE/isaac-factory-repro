# G4-S2A-GRIPPER-V1-R1 — 状态隔离与限速物理准备

**工程状态 `G4_S2A_GRIPPER_V1_R1_PASS`；候选状态 `NATIVE_GRIPPER_REJECTED`；R3 和 S2B 为 `NO-GO`。** 工程 PASS 指按预注册方案完成状态隔离、有效初态、两条独立准备、监督账本和独立复核，并如实记录失败；不是夹爪或 PPO 任务成功。对本次固定模型、增益、步长和准备协议，I2/I3 均在第 23 步严重违反手指关节限位，未达到物理抓持门；该否决不能推广到所有原生夹爪或所有参考方案。

## 状态问题的取证

起点 `d5290698fe20c52b1d8c539ce9e4e7ba8561466a`，任务开始前本地和远端 main 一致。V1 的 I3 登记初态与首次积分前的实际状态不一致，实际 qpos 看起来是模型默认值；历史记录只能将变化范围限定在登记之后与首次积分之前，无法确定具体操作或责任。本阶段在一次性数据对象上，禁止 `mj_step/mj_step1/mj_step2`，逐边界记录环境构造、`mj_resetData`、受授权初态赋值、`mj_setState`、`mj_forward`、旧共享数据 GUI 的启动、同步、摄像机、窗口激活和截图。本次共享路径复现没有观察到失配，不证明历史没有发生，也不证明 viewer 是或不是原因。见 `shared_forensic_v1/shared_viewer_forensics.json`；历史根因仍 `UNKNOWN`。

独立预检 `state_preflight_v1/state_preflight.json` 在不积分条件下检查 I2/I3 初态、独立模型/数据、单向复制与显示重新 forward、GUI 同步、镜头、截图、窗口激活、viewer-only 修改及 I2→I3 切换。各项无未授权物理状态变化；相关截图已逐张检查。显示复制针对的是**独立模型和独立 MjData**，其中 hole 模型位置与物理模型一致；不把显示数据写回物理。两份模型对质量、惯量、接触几何、传动、约束和求解器等选定的已编译物理数组计算指纹并一致；不声称覆盖所有可视属性或不透明 plugin 内存。

## 积分守卫与负控

预注册冻结由规范 reset、归档 I2/I3 qpos/qvel 和 hole 位姿、一次 `mj_forward` 构造出的完整初始 integration-state 快照及哈希；它不伪称历史档案包含不存在的初始 warmstart。首次真实积分前按位核对该快照和注册输入。每步使用上一个已完成 `mj_forward` 的权威快照，在显式授权当步 arm 和 tendon ctrl 变化后，再紧贴真正 `mj_step` 前逐位检查；不匹配即保留预期/实际及字段、停止且不请求积分。快照包括 qpos/qvel/time、ctrl、act、history、warmstart、mocap、外力、equality 激活、userdata/plugin 和完整 integration 向量。派生接触、qacc 不与另一求解阶段作错误的位级等同。

I2 与 I3 各自对 arm qpos、finger qpos、peg 自由位置、peg 四元数、qvel、time、外力和 warmstart 注入篡改，16 个负控全部在真实积分边界之前拒绝且没有 request；还测试 ctrl、模型差异、显示复制方向和 signed zero。预启动测试 55 passed、18 subtests passed，状态和 GUI 预检均零物理积分。46 个实际步骤的初态、控制授权及前后状态链经独立读者逐位复核一致。

## 已冻结的限速方案

沿用 V1 的 P_FULL、两个实体 finger、自由 peg、碰撞几何、摩擦、solver、`implicitfast`、1/120 s、原 7 臂控制、单位系数 sum tendon、`Kp=3750 N/m`、`Kv=86.5 N·s/m` 和 ±40 N 力限。指目标仍是每指开度，tendon 控制始终为 `2*q_target`。

注册开度每指 4.99125 mm，peg 直径 7.986 mm。对编译模型的 `mj_geomDistance` 静态检查，I2/I3 最近 pad 净距约 1.218282/1.237752 mm。预先固定每步目标最多向关闭方向变化 **10 µm**，即 1.2 mm/s，为 50 µm 指垫质量阈值的五分之一；以初始几何净距作尺度参考，不表示指实际位移、接触穿透或动力学稳定被限制在 10 µm。首个任一侧正 pad 法向力 >1e−6 N 时永久冻结目标；连续 10 次双侧正力进入 hold，额外 10 次双侧观测完成，失接触重新计数且不恢复闭合。每次最多 240 步（2 s 模拟时间），不是原 Factory 的 30 步过程；没有依结果改参或延长。

## 预注册实验和结果

矩阵在积分前绑定源文件、完整初态、模型 XML、静态几何、无积分预检、质量及早停阈值：I2 seed 2101 与 I3 seed 2102 各一次，准备独立；只有两次 reset 和全程接触质量均通过才运行同 R2 归档动作的两条 235 步操作诊断。无策略推理，最大 950 步、协调器 900 s、无重试。

| 实验 | 登记初态与守卫 | 积分 / 请求 | 早停及质量 | 接触和诊断 |
|---|---|---:|---|---|
| PREP_I2_RATE_LIMITED | 逐位一致；全部状态链有效 | 23 / 23 | 第 23 步 `FINGER_LIMIT`；reset FAIL / quality FAIL | 所记录步无正 pad 接触；未进入 acquisition / hold |
| PREP_I3_RATE_LIMITED | 逐位一致；全部状态链有效 | 23 / 23 | 第 23 步 `FINGER_LIMIT`；reset FAIL / quality FAIL | 所记录步无正 pad 接触；未进入 acquisition / hold |
| DIAG_I2_NATIVE / DIAG_I3_NATIVE | 准备门不满足 | 0 / 0 | 按预注册门禁未启动 | 无 PPO 成功结论 |

协调器耗时 67.287 s，46 request 与 46 result 均持久化；两子进程真实 exit 0 表示有监督的正常早停，不表示物理通过。没有重试或另一个架构。两次各 23 步只模拟约 0.192 s，未完成规定的接触取得和保持，故无“成功抓持状态”或成功诊断截图可供提供。

## 接触、力和历史对照

两组指轨迹几乎一致；第 1 步由 −0.075 N 起，第 20–23 步积分阶段 actuator 力依次约 +19.14、−26.19、+35.83、−40 N；对应每指开度约 5.412、3.974、5.918、−177.324 mm，第 23 步指速约 −21.989 m/s，超过原关节限位早停门。单步最大记录位移约 183.242 mm，尽管目标每步只改变 10 µm。不存在提高力限；积分力和每指广义力独立核对。记录的 post-forward 端点没有 pad 正力、pad/assembly 穿透最大为零，但高速步内未经连续碰撞验证，不能据此声称连续路径无碰撞或安全。没有利用图像估计穿透，也不把 post-forward 力称作上一积分的力或冲量。

历史 V1 I2 以骤闭目标第一步产生约 1.902 mm 的步后穿透；本次以固定速率参考从相同登记初态开始，但实际中间状态不同、未形成持续接触，在第 23 步发生增长振荡和越限。结果支持该固定参考和原生驱动组合在此步长下不适合准备；不足以唯一归因给某个求解器、几何或 PhysX 差异，也不是对两个方案的无偏成功率评估。保留先前已验证的 P_FULL 对初始臂振荡的局部效果，不将其改写为全局修复。

## GUI、复核和回归

真实 Ubuntu Xorg 的 `DISPLAY` 和 `XAUTHORITY` 在运行时验证，调用 `mujoco.viewer.launch_passive`；独立数据的单向复制让画面显示真实保存物理状态，GUI 同步不会作为新物理积分。两组 live 的初态及第 23 步截图，以及 I2 的保存状态回放第 5、15、22、23 步均经逐张隐私检查；图像哈希和阶段标识在 `image_qa.json` 和各元数据中。接触取得和 hold 未发生，无法提供相应真实画面。`scripts/g4s2a_gripper_v1_r1_viewer.py` 提供暂停、播放、慢放、单步、跳故障和镜头控制；状态机自动测试通过，但未逐项人工操作 GUI 按键，不能称为人工验收。保存回放只对独立数据 forward，不执行新积分。

独立 `numerical_review_v1.json` 对两组完整初态、全量位级连续性、每步限速/阶段、执行器限幅和两指广义力、接触、关节限位、早停和 wait/event 账本进行重算，`COMPLETE`、无分歧；正确记录两条诊断 `UNSTARTED`。受防护历史 CPU 回归 **803 passed、8 skipped**，另绘图测试 1 passed；75 项会执行 actor 或额外未注册物理的历史用例按任务限制排除。MuJoCo 定向测试 **62 passed、18 subtests passed**，结果见 `final_targeted.txt`。历史保护审计通过；与起点比较仅授权更新的两份 case-study 文档发生修改，其余既有受跟踪文件未改，原 PPO、S1B 和 V1 全部证据保持。

## 复现与下一步

仓库根目录下可以只读复核或打开已保存轨迹（输出路径必须不存在；回放不积分）：

```bash
/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/envs/factory-g01/bin/python scripts/analyze_g4s2a_gripper_v1_r1.py evidence/g4_s/gripper_v1_r1/matrix_v1 /tmp/gripper-v1-r1-review-new.json
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority /home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/g4s2a_gripper_v1_r1_viewer.py --matrix evidence/g4_s/gripper_v1_r1/matrix_v1 --run PREP_I2_RATE_LIMITED
```

`DISPLAY`/`XAUTHORITY` 为本机实测值，复现时须动态检查。用户授权的当前 native tendon 恢复尝试到此结束：不自动搜索增益/接触参数、不重跑、不执行 PPO。建议用户另行决定是否为物理合理的替代夹爪/执行器方案单独立项，或停止 MuJoCo Sim2Sim 分支；任何新方案须保留冻结策略接口和独立 peg，且先通过物理准备及诊断门。学习和技术面试考核保持 `PENDING`。
