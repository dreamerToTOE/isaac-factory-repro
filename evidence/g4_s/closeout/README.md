# G4 Sim2Sim closeout evidence

本目录只包含文档结项、历史数据整理和CPU完整性核验；**没有新物理积分、PPO推理/训练、checkpoint反序列化或夹爪实验**。起点 `85017ba626e123ed95c397c32c4b852e9118abe4`。

- [milestone_summary.json](milestone_summary.json)：从11个历史机器门禁直接读取原字段、源G2与I2分母分开、最终状态、局部发现、失败和UNKNOWN、引用SHA与授权边界。
- [preservation_before.json](preservation_before.json) / [preservation_after.json](preservation_after.json)：起点全Git跟踪文件、引用外部科学文件及work/runs全部76个`.pth`只读SHA；checkpoint不反序列化、不复制。
- [source_hashes.json](source_hashes.json)：起点G4报告、脚本/配置及模型文本的SHA索引；历史源身份仍以原清单为准，不扩张资产许可。
- [cpu_tests.txt](cpu_tests.txt)：6项stdlib测试真实输出；[cpu_test_receipt.json](cpu_test_receipt.json)记录命令、返回码和时间。
- [consistency_checks.json](consistency_checks.json)：本次文档链接、机器门禁、全历史保护与外部hash对比结果。没有调用会运行actor/physics的全仓测试。
- [chart_data.json](chart_data.json)：图表原数据、单位、原路径/SHA及图像SHA；[visual_review.json](visual_review.json)记录本次图表检查和历史截图QA引用。
- [publication.json](publication.json)：起点及科学交付提交的正常fast-forward推送/独立远端核验；最终包含该收据的提交SHA由最终答复提供，避免自引用。

## 只基于归档的图表

![同初态共同前5步惯量对照](inertia_common_prefix.png)

数据来自[R2 factorial_effects](../s2a_r2/numerical_review_v2/factorial_effects.json)，I1前5步统一窗口。A=P0/C0、B=P_FULL/C0、C=P0/C1、D=P_FULL/C1；对数轴标明单位。只说明该局部确定性窗口，不推断全局稳定或成功率。

![原生V1与限速参考的实际响应](gripper_target_vs_actual.png)

左图为同注册I2初态**第一步**的目标/实际位移，右图为R1 I2/I3各23步指位置，超出±8 mm使用symlog轴。骤闭总窗1步、R1总窗23步；不能比较不等全窗极值来估计性能。输入限速不能证明真实速度有界，保存端点不能证明连续无碰撞。

## 可复核性与输出约束

生成器为[scripts/g4_sim2sim_closeout.py](../../../scripts/g4_sim2sim_closeout.py)和[plot脚本](../../../scripts/plot_g4_sim2sim_closeout.py)，只读取JSON/JSONL、普通源文本和hash字节，禁止导入MuJoCo/torch/Isaac/rl_games。历史实验执行器未导入。新CPU测试可在仓库根目录运行：

```bash
python3 tests/test_g4_sim2sim_closeout.py
```

生成器使用独占创建拒绝覆盖已存在JSON收据；起点snapshot要求HEAD匹配。它们记录本次生成流程，未来不能借此覆盖提交证据或自动复跑实验。所有原始长轨迹继续在历史目录，不重复复制。报告：[中文结项](../../../docs/REVIEW_G4_SIM2SIM_CLOSEOUT.md)；[面试指南](../../../docs/case_studies/SIM2SIM_INTERVIEW_GUIDE.md)。

最终：PARTIAL / CLOSED_WITH_NEGATIVE_RESULT / NOT_DEMONSTRATED；R3、S2B NO_GO，唯一根因未建立。学习和理论考核PENDING；停止。
