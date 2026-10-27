# G4-S2A-R2-R2：夹爪驱动、接触求解器与晚期抓持失败调查

日期：2026-10-27（Asia/Seoul）；仓库 `dreamerToTOE/isaac-factory-repro`，分支 `main`；起点 `d1a2097b1bf43d0cc95b9bae7da231644b43503a`。

**工程状态：`G4_S2A_R2_R2_PASS`（审计门）；本阶段新修正候选：`CANDIDATE_BLOCKED`；历史 P_FULL 适配候选仍为 `CANDIDATE_PARTIAL`；R3 / S2B：`NO-GO`。** 本阶段严格是无策略、离线读取与 CPU 合成监督测试，没有新的 MuJoCo 积分、PPO 推理、episode、训练或微调。

## 1. 冻结边界与结论

本阶段读取 S1B、S2A、R1、R2、R2-R1 报告和对应机器证据，保留全部历史 checkpoint、XML、轨迹、失败结果和公开/私有截图。S1B 默认模型、P_FULL、peg/hole 几何、finger 质量和碰撞、solver 设置、阈值、原 success predicate 与 PPO 接口均未修改。

结论是：

- R2-R1 发现的晚期失败具有可重复的接触 manifold 变化时序；这支持“接触/约束切换先于 finger 速度和 peg 旋转突变”的局部机制。
- `AdaptedPegInsertEnv.physics_step()` 的手指自由运动后向欧拉公式与归档 actuator force 独立复算完全一致，未发现实现错误。
- `raw PD≈24–28 N` 到 `implicit force≈0.18–0.21 N` 是该公式在约 `0.01493 kg` 有效手指质量和 `dt=1/120 s` 下的预期衰减，不是应直接替换 controller 的证据。
- MuJoCo 3.5.0 的 `contact.efc_address`、`efc_type`、`efc_id` 关系已按实际 API 审计；post-forward 的 `qfrc_constraint`、`qacc` 与 `mj_contactForce` 不能冒充前一积分步的历史冲量。
- 被动臂不是机械锁定臂，但接触丢失时 arm 速度仍低于 `0.001 rad/s`，随后 finger 速度和 peg 旋转才突变；这削弱“arm torque 是唯一原因”，不能证明 solver 是唯一根因。
- 没有足够证据选择一个可信的单一 target-engine correction，因此不执行新的 6 条物理验证矩阵。

## 2. Stage A：历史接触取证

独立 reader 只读取 R2 的 I3_B/I3_D、稳定 I2_B/I2_D、R2-R1 的 I3_FULL_C0 和 I3_FULL_PASSIVE_ARM。没有加载 MuJoCo、调用 `mj_step`/`mj_forward`、打开 GUI 或加载策略。

I3 正常 arm 的 failure window 是：

| 轨迹 | 首个相关接触 pair 变化 | 左侧正 pad force 首次丢失 | 下一步结果 |
|---|---:|---:|---|
| R2 I3_B | step 175，`left_finger_geom_4` 消失 | step 176 | step 177，`peg-left_finger_geom_7` 约 1.000772 mm 穿透并早停 |
| R2 I3_D | step 214，`left_finger_geom_4` 消失 | step 215 | step 216，约 1.047671 mm 穿透并早停 |
| R2-R1 I3_FULL_C0 | step 175 | step 176 | step 177，约 1.000772 mm 穿透并早停 |

失效前的 pad 穿透仍约数微米；失去左侧正接触后，下一 physics result 的 finger `qvel` 约 `−0.08 m/s`，peg-hand rotation slip 越过 `0.05 rad`，post-forward manifold 由少量接触扩展到多个深穿透接触。接触中点只证明存储的接触位置，不能证明碰撞特征一定是边、角或某个 mesh feature。

I2_B/D 在相同时间窗保持双侧正接触，不能把不同初态混为同一个因果样本。完整机器结果见 [forensics.json](../evidence/g4_s/s2a_r2_r2/forensics_v1/forensics.json)。

## 3. 初始抓持准备异常

旧 reset admission 结果仍是 `RESET_ADMISSION_PASS`，没有追溯改写。另加质量警告：I3 准备阶段 step 2 曾出现 `peg_collision/right_finger_geom_7` 约 `873.514 µm` 瞬态穿透，之后恢复至约 `6.524 µm`；I1 峰值约 `316.328 µm`，I2 峰值约 `22.444 µm`。这使准备瞬态成为合理候选线索，但 I1 后续仍可保持 235 步，因而不足以证明 I3 晚期失败的唯一原因。

## 4. Stage B：手指 actuator 数学审计

源实现使用 `q`（m）、`qdot`（m/s）、`kp=7500 N/m`、`kd=173 N·s/m`、`dt=1/120 s`，并从完整 `M` 的逆矩阵抽取手指 DOF `[7,8]` 的 `A=(M⁻¹)ff`：

\[
q^+=q+dt(v+dtAf),\quad v^+=v+dtAf,
\]
\[
f=K_p(q_d-q^+)-K_dv^+,
\]
\[
[I+dt(K_d+dtK_p)A]f=K_p(q_d-q)-(K_d+dtK_p)v.
\]

I3_FULL_C0 step 1 的独立复算为：

- raw PD：`[-25.7982, -25.7991] N`；
- `A` 对角约 `66.9912 kg⁻¹`；有效质量约 `0.0149277 kg`；
- 隐式系统 `B` 对角约 `132.47`；
- applied force：`[-0.195670, -0.195678] N`；
- 公式与归档 actuator force 的最大残差：`0.0 N`。

因此 raw/implicit 差异是公式定义的自由运动衰减。该近似没有包含 bias、Coriolis、gravity、arm torque、接触反力或 equality reaction；它不是 PhysX implicit drive 的等价证明。完整计算、单位、条件数和多个 failure step 见 [math_api_audit.json](../evidence/g4_s/s2a_r2_r2/api_audit_v1/math_api_audit.json)。完整质量矩阵的条件数约 `2.2e7`，主要受 peg free-joint 小转动惯量影响；手指 `A` 和 `B` 的 2×2 条件数约 `1.01`，所以该小系统不是数值病态导致的输出衰减。

## 5. Stage C：接触与 equality API 审计

安装版本为 MuJoCo `3.5.0`。静态编译的 P_FULL XML 只用于模型元数据审计，没有调用 `mj_step` 或 `mj_forward`；结果见 [api_audit.json](../evidence/g4_s/s2a_r2_r2/api_audit_v1/api_audit.json)。

模型有一个 `mjEQ_JOINT` equality（`eq_type=2`，对应两个 finger joint），接触行使用 `mjCNSTR_CONTACT_ELLIPTIC=7`。在 condim=4 时，一个 contact 通过 `contact.efc_address` 映射到四个连续 EFC scalar rows；`efc_type`、`efc_id`、`efc_force`、`efc_pos`、`efc_vel` 是按 constraint row 排列，不能按 `mjData.contact` 数组逐项解释。

R2-R1 已保存两种时序：`integration_stage` 是原 `mj_step` 之后、wrapper `mj_forward` 之前的观察；`post_forward_solver` 是 advanced state 的重新 forward。I3 step 176 的 integration-stage manifold 仅余 `left_finger_geom_6` 与两条右侧接触，而 step 177 post-forward 出现多个左、右 finger 几何深穿透。两套 `qacc/qfrc_constraint` 不应被合并成一条“历史冲量”解释。

## 6. Stage D：被动臂负控

I3_FULL_PASSIVE_ARM 将 arm command 置零，但没有把 arm 机械锁定。右侧 `right_finger_geom_5` 在 step 142 消失，右 pad 正法向力在 step 143 丢失；step 144 finger `qvel` 约 `[-0.0825,-0.0847] m/s`，peg-hand rotation slip 约 `0.06117 rad`，而 arm 最大速度仍仅约 `0.000832 rad/s`。这说明接触/手指扰动出现在显著 arm 运动之前，削弱 arm torque-only 假设；它不能证明 contact solver 是唯一根因。

## 7. Stage E：监督进程修复与边界

旧 R2-R1 coordinator 的 child receipt 在 `wait()` 返回后才写入；因此监督进程在 I3_FULL_C1 第 212 request 处消失时，只能保留 613 个 durable requests、612 个 physics results 和未知 child exit status。该历史缺口保持 UNKNOWN，未补造 result 或 exit code。

新增的 [g4s2a_r2_r2_supervisor.py](../scripts/g4s2a_r2_r2_supervisor.py) 是 additive、无 MuJoCo 的 one-shot supervisor：

- request 写入 JSONL、flush、fsync 后才调用现有 operation；
- 只有 operation 成功返回才写 result；异常记录为 `integration_exception`，不伪造 result；
- 每个 child 使用独立 process group；timeout 只终止自有 group；
- 明确区分 `CHILD_EXIT`、普通 `CHILD_SIGNAL`、fatal `CHILD_CRASH`、`TIMEOUT`、`UNKNOWN_TERMINATION`；
- 保存 `last_event.json`、console、实际 wait return code 和最终状态；
- 输出目录一次性创建，禁止自动 retry。

合成子进程测试 12/12 通过，0 physics、0 MuJoCo import、0 policy call，结果见 [supervisor_audit.json](../evidence/g4_s/s2a_r2_r2/supervisor_v1/supervisor_audit.json)。这修复了未来诊断的记账边界，但没有篡改历史 R2-R1 收据。

## 8. 修正候选与实验矩阵决定

本阶段没有发现可验证的 controller 数学错误、错误的 equality 配置或错误的 MuJoCo API 使用。接触 manifold/constraint coupling 是支持中的机制，但缺少 solver iteration history、独立 equality/contact impulse 分解和 PhysX runtime 对照。

因此：

- **R2-R2 新 correction candidate：`CANDIDATE_BLOCKED`**；不执行新的 6 条 trajectory matrix，不超过本阶段零新 physics budget。
- **历史 P_FULL adaptation：仍 `CANDIDATE_PARTIAL`**，不能被本阶段改写为 supported 或 rejected。
- 不调整 finger PD、friction、solver、force limit、几何或 PPO 接口。
- R3 / S2B 继续 `NO-GO`，直到 contact-quality 和 sustained-grasp prerequisites 有可辩护的证据。

## 9. GUI、数值和独立复核

由于没有新的 physics integration，本阶段没有新增 live GUI 窗口或截图，也没有把保存状态 playback 当作 solver impulse 重建。上一阶段已经完成的真实 MuJoCo GUI 和逐张截图 QA 仍可从 [R2-R1 报告](REVIEW_G4_S2A_R2_R1.md) 和 [公开截图清单](../evidence/g4_s/s2a_r2_r1/visuals/public_manifest.json) 复现。

本阶段独立复核由两个互不依赖的 CPU reader 构成：

- Stage A/D reader：重建 pair chronology、pad force、finger q/qdot、peg slip、passive-arm timing、EFC metadata；
- Stage B/C reader：独立重算 implicit finger formula、mass conditioning、contact/equality mapping；
- supervisor synthetic reader：验证 request/result/exit/timeout/signal/unknown contract。

三者都没有执行 policy 或 physics。`PREPHYSICS_FORENSICS_COMPLETE` 收据为 PASS；数学审计的 correction recommendation 为 blocked，而不是把缺少证据误报为 candidate PASS。

## 10. 回归、保全和交付

历史 S1B、S2A、R1、R2、R2-R1 evidence 和 checkpoint 均保留。R2-R2 新增测试只做 CPU 合成 supervisor 检查；禁止会加载 checkpoint、调用 actor、执行新 MuJoCo integration 的测试均明确排除。最终回归结果写入 [regression_r2_r2_v1](../evidence/g4_s/s2a_r2_r2/regression_r2_r2_v1/summary.json)。

工程审计门为 `G4_S2A_R2_R2_PASS`，因为只读取证、API/数学审计、监督修复和独立 CPU 测试均完成，且没有未记账的新 physics。候选修正仍 `CANDIDATE_BLOCKED`；这不等于 Sim2Sim 或 PPO 成功。

科学内容提交 SHA：`fa455e76cbdecee2edcaaa30d443c5314afe6bb1`。推送前已验证起点远端 SHA `d1a2097b1bf43d0cc95b9bae7da231644b43503a` 是该提交祖先；推送后 `git ls-remote https://github.com/dreamerToTOE/isaac-factory-repro.git refs/heads/main` 返回同一 SHA。本报告不会改写历史候选结论。

## 11. 下一阶段建议

在没有新的证据前不启动 R3/S2B。若要继续，应先获得独立的 solver/equality impulse 记录或可验证的源引擎对应关系，再单独授权一个明确的 target-engine correction 和最多六条匹配诊断轨迹。不得通过放宽阈值、增加 actuator limit、weld peg、关闭 collision 或调参搜索制造通过结果。

学习和技术面试考核仍为 `PENDING`。
