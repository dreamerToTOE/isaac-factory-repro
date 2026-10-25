# G3B PegInsert 失败模式诊断审阅

日期：2026-10-25
状态：**PASS（诊断阶段）**
起点：G3A 提交 `668d31f412404542bc5d3ccd4822512577663115`
checkpoint：`last_Factory_ep_200_rew__350.4961_.pth`，SHA256 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`

## 结论

G3B 使用同一最终 checkpoint，对 1001、1002、1003 三个评估 seed 顺序运行 32 个环境、每环境 3 个完整 episode，得到 288/288 个完整 episode。官方 ever-success 与 terminal-success 都是 **280/288（97.222%）**，逐 seed 为 1001：95/96、1002：96/96、1003：89/96；这与原始 G2 分母和计数一致。G3B 诊断池与 G2 主结果分开统计，没有替换 G2 结果。

原始 G2 的 8 个失败标识在 G3B 中全部再次为 terminal failure：1001/env11/ep0，以及 1003/env4/ep0、env12/ep0、env25/ep0、env29/ep0、env5/ep1、env1/ep2、env5/ep2。这里是 seed/env/episode 标识及结果的再次出现，**不是 exact replay**：G2 没有保存这些失败的完整历史轨迹，不能声称逐位相同模拟。

轨迹记录在官方 success/engagement 调用、自动 reset 之前完成；累计 447 policy steps/seed，NPZ 形状为 447×32，每个 episode 149 步。分析重新计算了官方 predicate，trajectory 与 evaluator outcome 一致，官方 predicate disagreement 和 engagement disagreement 均为 0。记录只 clone 现有 GPU 张量，不改变 PPO、奖励、随机化、控制器、观测、动作、物理或 success criterion。

## 分类结果与边界

8 个新 terminal failures 的重叠分类为 A=3、B=8、C=0、D=0、E=0、F=0。A 是末端/全过程最小 XY 仍达到官方 2.5 mm 严格门槛的 XY alignment 条件；B 是末端轴向 gap 仍不满足官方单侧 `<1 mm` 条件；A/B 可以同时出现。所有 8 个失败的 peg/hole +Z 轴角最大值为 2.26–3.84°，低于预注册的 5°次级描述阈值，因此没有 C。没有 episode 曾满足官方 success 后又丢失，因此没有 E。D 不能由停滞、动作、EE速度或 controller command 推断：本版本没有可用的独立 measured contact sensor，applied/commanded wrench 没有当作接触力。

失败量化证据见 [`summary_compact.json`](../evidence/g3b/summaries/summary_compact.json) 和 [`episode_summary.csv`](../evidence/g3b/summaries/episode_summary.csv)。8 个失败的 XY 最小值为 0.026–2.819 mm、末值为 0.285–7.853 mm；轴向 gap 最小/末值为 2.543–25.004 / 2.543–25.199 mm；轴角最大值为 2.26–3.84°。EMA action RMS 为 0.457–0.518，末 20 步 RMS 为 0.459–0.561；EE 采样路径为 58.6–116.7 mm，末 20 步平均线速度为 0.00324–0.00803 m/s。这些是描述性轨迹量，不能证明动作导致失败或接触阻塞。

次级物理质量检查只在 280 个新官方 terminal successes 上统计，分母明确为 280：末端 world-Z 小于 0 的为 0，hole-axis projected gap 小于 −1 mm 的为 0，轴角大于 5°的为 0；sampled success dwell 中位数 8.067 s，范围 3.0–8.8 s。负 base-plane gap 只是几何观测，不是独立的穿透证明；这些次级量没有成为新的 success rate。

## 监督与资源证据

1001、1002、1003 的有效运行均由 G3A guard/supervisor 完成，child exit code 为 0、guard continuous monitoring 为 true、无 monitoring gap。显存峰值分别为 6308.0、6259.75、6309.8125 MiB，最小空闲分别为 1509.625、1557.875、1507.8125 MiB。1003 第一次提交在仿真启动前因 guard 发现 0.839 s 监控缺口而 fail-closed；该目录和 supervisor/guard 证据保留，未计入样本。随后使用新唯一目录恢复，完成有效 1003 诊断；没有覆盖、拼接或伪装第一次尝试。

## 验证

- CPU 回归：121 项通过（`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 env -u PYTHONPATH .../factory-g01/bin/python -m pytest -q tests`）。
- 轨迹序列化：NPZ 禁止 pickle、禁止 object/non-finite array、独立 metadata 和 SHA256；CPU 集成控制验证了 pre-EMA action、EMA action、world-origin、149-step reset 及关闭前落盘。
- 诊断分析：3×96 episode、447×32 pre-reset records/seed、每 episode 149 步；outcome 和官方 predicate 校验全通过。
- 图表：XY、轴向 gap、姿态轴角、policy/EMA action、success/failure 描述性 median/IQR、分类计数均在 [`evidence/g3b/plots/`](../evidence/g3b/plots/)；灰色成功曲线是不同起点的观察对照，不是 counterfactual。

复现命令和冻结参数见 [`protocol_frozen.json`](../evidence/g3b/protocol_frozen.json)，实际 child/guard 结果见 [`run_receipts.json`](../evidence/g3b/run_receipts.json)。原始 NPZ、metrics 和监督 console 保留在仓库外 `work/logs/g3b/`，避免把大原始日志提交到 Git。

## 未决问题与 G3C 建议

G3B 说明了这些可观测失败主要满足轴向未达标条件，其中 3 个同时保持 XY 超阈值；它没有建立接触阻塞的因果机制，也不能从一次三 seed 诊断推断更广泛的泛化率。若 G3C 要研究接触机制，应先单独预注册可复核的 read-only contact sensor/力学观测和对照实验，再决定是否扩大评估或训练 seed。G3B 结束后停止；不启动训练 seed1/seed2，不改变 PPO 基线。
