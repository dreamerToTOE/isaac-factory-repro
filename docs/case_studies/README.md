# Engineering Case Studies｜机器人工程问题复盘

本目录记录已经**实际执行并保留证据**的工程故障、定位过程、修复尝试和结果。目标是让项目不仅可以展示最终性能，也能在技术面试中解释判断依据、失败边界和可复现的解决过程。

## 案例目录

| 案例 | 当前证据状态 | 研究问题 | 入口 |
| --- | --- | --- | --- |
| Panda Sim2Sim 控制振荡与脱抓 | **GRIPPER_V1_R1_PASS / NATIVE_GRIPPER_REJECTED / R3_NO-GO / ROOT_CAUSE_PENDING** | PhysX 中训练的冻结 PPO 在 MuJoCo 中出现快速脱抓与关节失稳；状态隔离验证通过；限速准备的 I2/I3 均因指关节越限早停 | [案例正文](SIM2SIM_PANDA_CONTROL_INSTABILITY.md) · [GRIPPER-V1-R1报告](../REVIEW_G4_S2A_GRIPPER_V1_R1.md) · [GRIPPER-V1报告](../REVIEW_G4_S2A_GRIPPER_V1.md) · [R2-R2报告](../REVIEW_G4_S2A_R2_R2.md) · [R2-R1报告](../REVIEW_G4_S2A_R2_R1.md) · [R2报告](../REVIEW_G4_S2A_R2.md) · [R1报告](../REVIEW_G4_S2A_R1.md) |

新增其他案例时，使用 [CASE_STUDY_TEMPLATE.md](CASE_STUDY_TEMPLATE.md) 作为结构模板。

## 证据与表述规范

1. **OBSERVED**：日志、代码审计或可复现测试直接支持的观察。写明阶段、实验 ID、分母和对应证据链接。
2. **HYPOTHESIS**：有待设计对照实验的解释；不能当成已证实根因。
3. **VERIFIED_FIX**：仅当独立的后续实验在预先设定的验收指标下完成，才可以把修复标记为已验证。
4. **LIMITATION**：记录资产、版本、环境、统计范围、误差和未开展的实验；不因后续成功而改写旧的失败证据。
5. **INTERVIEW_SUMMARY**：面试叙述须随证据状态更新；未解决时只能说“如何排障”，不得声称“已解决并恢复成功率”。

技术术语区分：
- **Physics-/simulator-level interface running**（跨引擎模型和控制接口能运行）≠ **successful task-level zero-shot transfer**（冻结策略完成真实任务）。
- **Factory original geometric success** ≠ **physical insertion/contact-quality validated success**。
- **Adapted Sim2Sim** ≠ **exact original Factory asset/physics reconstruction**。

证据优先指向本仓库不可覆盖的历史报告、机器摘要和保留的原始轨迹；大型视频与二进制文件应只提交校验清单和再现命令，避免把大量数据直接塞入 Git。

## 更新纪律

新阶段使用新的证据目录、来源版本及哈希；修复案例时追加“基线→变更→验证”对照，不得覆盖旧观测数据。每次更新记录审核范围、未经验证事项和可复现入口。
