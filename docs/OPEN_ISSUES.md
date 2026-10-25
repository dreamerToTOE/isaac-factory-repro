# 未决问题、失败及解决状态

更新：2026-10-25（Asia/Seoul）。G2已完成64环境、seed0、200epochs及全部主要评估；下方G0/G1、D001的“未启动/等待审阅”等语句保留为历史状态。

## G3A 当前工程验收

**PASS（监督与日志可靠性短测）**：77项CPU测试通过；独立systemd guard在训练前采样，绑定boot/PID/birth后放行执行闸门。64环境2epoch smoke实际child exit0、16384transitions，829样本无>0.5秒gap，checkpoint及50tag末轮TensorBoard通过。见[REVIEW_G3A](REVIEW_G3A.md)、[协议](G3_SUPERVISION_PROTOCOL.md)、[证据](../evidence/g3a/README.md)。截至 G3A 工程验收时未启动 G3B 或长训练；G2科学PASS不变，当前 G3B 状态见文末。

已补正常writer显式flush/close、真实wait状态、双方liveness、身份/独占/非覆盖与原子最终记录。此前手工迟启guard的失败/部分完成smoke保留；G2约123秒缺口、原退出码、TensorBoard缺尾仍无法追回，根因未确证。

剩余限制：Linger=no，不保证跨完整注销/重启；两服务同时SIGKILL、断电或文件系统故障可丢失最后状态；0.1秒采样不保证瞬时峰值；通用daemonize/PGID逃逸不在标准run_sim保证内。新长训练须重新preflight、新manifest/目录。G3B bounded 诊断已完成；新长训练仍须重新 preflight、新 manifest/目录，并不会自动启动。

## G2 当前状态与未决项

**PASS（资源受限初步基线）**：1,638,400 transitions，最终独立确定性ever/terminal均280/288=97.222%，42项CPU测试通过；里程碑、回放、几何和日志见[RESULTS_G2](RESULTS_G2.md)及[证据索引](../evidence/g2/README.md)。上游保持pinned且干净，没有参数调优或追加训练。

| 问题 | 已确认事实及当前边界 | G3建议 |
| --- | --- | --- |
| 原父监督/monitor消失 | 原因未知；训练原PID/birth继续，未restore。约122.747秒无NVML/guard；训练OS退出码null。预算完成由epoch1…200、final checkpoint和MAX EPOCHS标记佐证 | 持久化监督、独立liveness/guard、退出状态落盘；短smoke故障验证 |
| 首次无数据launch manifest未完整保留 | 启动未产生epoch/checkpoint，原hash/status仍在，不能恢复原manifest全文 | 新实验manifest只读/唯一命名；不得覆盖历史 |
| TensorBoard缺末epoch | actor/cval各199点，reward198点；epoch200没有落盘scalar，与异步flush遗漏相容但原因未证实 | 显式flush/close及末epoch校验作为工程改动；不补造G2事件 |
| 长跑显存余量较紧 | 观测used峰值6909.6875MiB、free最低907.9375MiB；仅1次低于1024，未达连续3次停止条件；未知区间不保证安全余量 | 新GPU运行继续guard，先测资源再考虑128环境 |
| 8次AMP actor skip | 12800尝试、12792实际Adam；central value12800。保留官方GradScaler，没有逐梯度有限性证据 | 多训练seed复测，分开记录loss、skip和参数有限性 |
| 单训练seed/相关episodes | 3个评估seed不是3次训练重复；Wilson区间只描述episode计数；150/200差异无显著性结论 | 预注册训练seeds1、2，同64env/200epochs及固定评估协议 |
| final8次失败未获完整轨迹 | seed1001有1次，seed1003有7次；不能臆测卡滞等机制 | 对具体env/episode诊断几何与动作，额外诊断不改主要分母 |
| 官方成功语义有限 | XY<2.5mm且单侧z<1mm，没有保持时长/力/旋转/下界约束 | 如需更严格质量指标，未来独立定义并保留原官方指标 |

保留D001精确续训、依赖最小安装、其它Factory任务等限制。该段记录 G3A 审阅时的历史状态；G3B 后续状态见文末。G3A工程验收见上节。不会自动400epochs、128环境、调PPO/奖励或转Gear/Nut。

## D001状态（历史，2026-10-23）

- **推送已解决**：原提交a77d215通过既有SSH写入并独立ls-remote核验；首次push返回Broken pipe但服务端已接受。保留HTTPS fetch地址，origin push URL改为SSH，没有请求/读取令牌或修改SSH配置。原始HTTPS无credential helper，历史失败见[git_access.txt](../evidence/2026-10-23/git_access.txt)，本轮见[r1_push.json](../evidence/d001/r1_push.json)。
- **容量短测已扩展**：16/32/64各3epochs+加载后1epoch，10Hz NVML记录，未OOM，最小空闲均超过2GiB。128及长期内存增长仍未验证。
- **损失/梯度证据已补齐**：逐次观察actor和central critic。全部损失及实际应用梯度有限；fresh首次AMP overflow被skip，未改AMP。旧0-tag TensorBoard的历史事实保留，不再依赖它证明本轮loss。
- **指标独立验证已完成**：15项CPU测试；两个seed共8完整episode，ever/terminal均0/8。官方缓存未清零问题保留，独立评估不以该缓存为真值。
- **仍不支持完整可复现续训**：critic Adam、last_lr、RNG、环境与运行时RNN等缺失。正式首轮必须fresh seed0，见[状态表](CHECKPOINT_REPRODUCIBILITY.md)。

当前建议GO（64环境fresh run）；没有GPU权限或Git写入阻塞。未启动200epochs，长期收敛/显存及统计性能仍是下一阶段验证内容。详细边界见[REVIEW_D001](REVIEW_D001.md)。

## 已解决：保留失败证据

| 问题 | 证据 | 处理与状态 |
| --- | --- | --- |
| 首次冷启动240秒超时 | gpu_gate | 保留日志，gpu_gate_retry通过GPU物理240步 |
| Factory缺少flatdict | peg_probe_1 / peg_probe_1_retry | 在独立venv补依赖；未改系统或Sim包 |
| flatdict构建缺pkg_resources | install_support.txt | 固定setuptools80.10.2，no-build-isolation安装成功 |
| uv同时安装锁等待告警 | install_learning.txt | 顺序重跑完成 |
| Kit archive遮蔽学习包 | peg_probe_16运行时版本、早期train_peg4 | python_entry预载与启动后断言；最终pinned2.7环境重测通过 |
| Torch2.7默认weights_only与RL-Games1.6.1不兼容 | resume_peg4_pinned | 自有已知SHA checkpoint命令级兼容env；resume_peg4_pinned_compat通过 |

均见 [证据目录](../evidence/2026-10-23/README.md)，没有删去失败后只报告成功。

## 仍存在的验证边界

- 有意最小安装未包含pin-pink/dex-retargeting/moviepy，pip check声明检查不完整；Factory headless无视频路径已实测。
- 原G0/G1显存为约1Hz、16环境1epoch；D001已补10Hz及16/32/64短测，仍未验证长时增长或128环境容量。桌面占用随时间变化。
- 原G0/G1的16环境及恢复TensorBoard scalar tags为0（无数据）；D001用优化调用观察补齐loss/gradient，旧日志没有被改写为通过。
- checkpoint加载/继续更新通过，actor Adam状态递增；独立central-value Adam未被RL-Games1.6.1原样checkpoint保存，RNG/物理/RNN rollout逐位续接也未验证。后续精确续训应单列工程协议，当前不改基线。
- 源码`success_times`缓存、Nut动作切片、无上限IK重试、timeout双标记等详见[SOURCE_AUDIT](SOURCE_AUDIT.md)。保留上游行为，后续修补必须独立标记。
- Gear/Nut只做单环境reset+32步，不代表其完整episode或PPO验证；本阶段训练仅PegInsert。
- 未进行200epoch正式训练、收敛或统计可靠的多seed性能实验。D001的8episode评估只验证工具，不能当作性能结论。

已完成的最小测试没有剩余GPU权限阻塞。后续训练配置见[EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md)，等待审阅后再执行。


## G3B 状态（2026-10-25）

G3B bounded diagnostic 已完成并通过：保留 checkpoint SHA256 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`，三 seed 共 288/288 完整 episode，官方 ever/terminal 280/288。原始 8 个 seed/env/episode 标识全部复现为失败，但没有历史完整轨迹，因此不宣称 exact replay。A/B/C/D/E/F 重叠计数为 3/8/0/0/0/0；A/B 是观察到的几何条件，D 因无独立 measured contact 保持 unsupported。

完整诊断回执和边界在 [`REVIEW_G3B`](REVIEW_G3B.md) 与 [`evidence/g3b`](../evidence/g3b/README.md)。G3B 不改变 G2/G3A 证据、PPO、奖励、物理或 success criterion，也没有启动 seed1/seed2 训练。下一步建议停止并单独预注册 G3C 的接触传感器和对照协议。


## G3C-1 状态（2026-10-25）

G3C-1 的 seed 1 PPO 训练按冻结配置启动，但在第一个 epoch 前由独立 GPU guard 依据 1024 MiB free、连续 3 次低值规则停止：最低 free 917.9375 MiB，child 原始 wait code=0，supervisor 记录 guard 触发和 SIGTERM。运行标记 **工程 INCOMPLETE**；没有 epoch、optimizer step、AMP skip、checkpoint 或可评估分母，因此科学结果 **NOT ASSESSED**，不能报告 seed 1 成功率。

回执和资源曲线见 [`RESULTS_G3C1`](RESULTS_G3C1.md) 与 [`evidence/g3c1`](../evidence/g3c1/README.md)。不恢复、不重启、不拼接本次运行；不启动 seed 2。G3C-2 应先解决资源余量并重新预注册 fresh seed 1 运行。


## G3C-R1 状态（2026-10-25）

G3C-R1 对 RTX 3070 做了 60 秒只读 idle inventory 和一次新的 64-env、2-epoch supervised capacity smoke。idle minimum free 为 4780 MiB；smoke peak used 6798.75 MiB、minimum free 1018.875 MiB，按既定 1024 MiB×3 guard 停止。因此当前容量判定为 **CAPACITY_NO_GO**，不能安全支持另一次完整 64-env/200-epoch run。

建议 idle preflight minimum free 约 5500 MiB，且 2-epoch smoke 实际保留约 1.5 GiB free 后再考虑 fresh seed1。没有终止桌面或 remote desktop、没有降低 guard、没有启动长训练或 seed2。详细比较见 [`REVIEW_G3C_R1`](REVIEW_G3C_R1.md) 与 [`evidence/g3c_r1`](../evidence/g3c_r1/README.md)。
