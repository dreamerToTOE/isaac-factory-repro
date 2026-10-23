# 未决问题、失败及解决状态

日期：2026-10-23（Asia/Seoul）。GPU仿真、Factory初始化与有限PPO已通过；当前停止G0/G1。

## D001当前状态

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
