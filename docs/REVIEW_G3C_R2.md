# G3C-R2：受控容量恢复与 PegInsert 条件训练审阅

日期：2026-10-25（Asia/Seoul）。起点 `32f784b43011470615dec251984aca26716a4b86`。

**结论：64-env 容量 margin NO-GO；32-env smoke 工程 PASS；32-env / 400-epoch seed0 完整训练工程 INCOMPLETE，科学 NOT ASSESSED。** 本轮已遵照停止条件结束，没有恢复、重试、拼接或启动其他训练 seed。32-env 的终止原因是监控缺口，不能称为32-env显存不足或强化学习种子失败。

## 1. 目标、授权与分支

本轮先做60秒只读inventory，只有 idle minimum free≥5500MiB 才允许一次64-env2epoch smoke；该smoke需满足child正常退出、无guard触发/监控缺口、checkpoint/有限loss/TensorBoard完整，并以1536MiB作为约1.5GiB的冻结margin。64分支不满足时，获准切换为新科学协议“PegInsert 32-env / 400-epoch baseline”，先做一次32-env2epoch smoke，通过后从头训练seed0。所有分支均保留1024MiB连续3个样本的运行时显存保护，以及0.1s目标采样、0.5s gap、2s stale规则。

配置保持IsaacLab v2.1.1 / `90b79bb2d44feb8d833f260f2bf37da3487180ba`、IsaacSim4.5.0、RL-Games1.6.1、Torch2.7.0+cu128。驱动580.178.04的nvidia-smi显示最高CUDA13.0支持；Torch构建使用CUDA12.8，两者含义不同。没有改官方PPO、LSTM/MLP结构、奖励、观测、动作、控制器、随机化、物理或success predicate。

| 实验 | 环境 | seed | epochs | transitions | 定位 |
|---|---:|---:|---:|---:|---|
| G2历史正式训练 |64|0|200|1,638,400|保留的原始基线 |
| 本轮64-env smoke |64|1|2|16,384|容量检查，不作为训练种子复现 |
| 本轮32-env smoke |32|0|2|8,192|新协议容量检查 |
| 本轮32-env完整训练计划 |32|0|400|1,638,400|新的科学基线，运行提前结束 |
| 本轮32-env已核实完成部分 |32|0|10|40,960|计划预算2.5%，不作为完整baseline |

32-env每epoch rollout为4096 transitions，G2为8192；相同minibatch512和mini_epochs4意味着每epoch优化尝试数量不同。相同总transitions不能证明相同优化轨迹、调度或科学等价。新的计划评估epoch100/200/300/400与G2 epoch50/100/150/200在目标transitions上对应；最终目标分母仍288，其余各96，不能把旧G2/G3B样本并入新分母。

## 2. Phase A：只读资源预检通过

新inventory完成600样本、约60秒；全卡used峰值2088MiB、free最低5730MiB、reserved约375MiB，超过5500MiB门槛。唯一compute列表报告gnome-remote-desktop PID2882约157MiB；其余graphics/driver/未归因分配没有被声称可回收。实际nvidia-smi/进程/RAM/systemd快照保留在外部inventory并有hash。没有新Isaac/训练争用。

与R1的4780MiB相比，本次idle余量改善，但这不是本agent执行清理的效果，也没有建立具体外部占用变化的因果归因。本轮没有杀用户进程、停止桌面或remote desktop、改驱动/全局systemd、删缓存或降低guard。

## 3. 两次容量smoke与分支门槛

| 检查 |64-env smoke|32-env smoke|
|---|---|---|
| supervisor wall秒 |86.974680|81.504845|
| completed epochs / transitions |2 / 16,384|2 / 8,192|
| child raw wait / reason |0 / child_exit|0 / child_exit|
| guard samples / gaps |871 / 0|816 / 0|
| peak used MiB |6360.5|6213.1875|
| minimum free MiB |1457.125|1604.4375|
| final checkpoint epoch/frame |2 / 16,384|2 / 8,192|
| checkpoint tensor finiteness |通过|通过|
| recorded epoch loss finiteness |通过|通过|
| actor actual/attempts，central updates |128/128，128|63/64，64|
| AMP skips |0|1（首个actor尝试）|
| TensorBoard finalizer与严格末轮检查 |complete|complete|
| 1536MiB margin |未通过，少78.875MiB|通过|
| 分支决策 |NO-GO，不启动64-env长训练|PASS，允许新的32-env seed0 baseline|

64-env smoke的正常退出不抵消margin失败；没有放宽“约1.5GiB”以获得PASS，没有再次尝试64-env。smoke checkpoint只作工件完整性检查，没有作为主要独立评估模型。

## 4. 32-env完整训练：监控异常停止

唯一完整训练run：`g3cr2_peginsert_32env_seed0_baseline_400ep`；监督目录：`work/logs/g3c_r2/train_32env_seed0_400ep/`。从fresh policy/optimizer开始，没有checkpoint参数。runtime实测actors32、horizon/seq128、minibatch512、mini_epochs4、obs/state/action=19/43/6、mixed precision开启。与G2的resolved env.yaml只差scene.num_envs 64→32，agent.yaml只差max_epochs200→400和独立实验名。pre-run agent dump内num_actors128是官方runtime覆盖前快照；实际32由observer交叉验证。

监督区间为2026-10-25 22:29:34.871–22:35:43.757 KST，**368.886015秒（6分8.886秒）**；child spawn-to-wait和exec-release-to-wait分别是独立更短时间定义，见analysis，不混用。已记录连续epoch1…10，共40,960 completed-epoch transitions；被停止时未完成rollout的额外交互未独立计数，不能把40,960写作逐step精确全运行总量。已完成epoch的官方PPO时间合计341.653234秒、聚合119.887640 FPS。

最后两个guard样本monotonic分别32664.940079581与32665.452398374，间隔**0.512318793秒>0.5秒**。guard在22:35:41.966 KST发送SIGTERM，stop_reason=`monitoring_gap`。supervisor随后也记录SIGTERM，reason=`guard_missing_or_stale`；真实child wait=0、child_signal=null。请求发送过SIGTERM与OS最终wait返回0是不同证据；exit0不能证明自然完成400epochs。

观测3662个NVML样本，peak used6408.9375MiB、minimum free1408.6875MiB，最大low consecutive=0，未触发显存floor。末样本free1664.75MiB。guard覆盖不完整（最后样本早于child wait），极值仅适用于观测样本。0.512秒延迟的底层原因尚未确定；没有证据可归因到某个应用、磁盘或GPU故障，也没有将gap阈值调宽。32-env smoke PASS不能保证数小时无调度或监控故障。

## 5. 数值、日志与checkpoint边界

已记录10个epoch的loss arrays均有限、observer instrumentation_errors为空；actor实际Adam更新319/320尝试、central更新320，首个actor AMP skip使scale从65536降至32768。轻量observer没有逐梯度采样，不能声称所有原始/应用梯度都有限；AMP skip不等价于已应用非有限参数更新。

唯一完整run checkpoint为训练reward挑选的`Factory.pth`，CPU检查内部epoch10/frame40960、actorAdam step319、actor/critic/optimizer tensor全部有限，SHA256 `d24556a2189da37aaf0f07e1397167b711235e811f946872070ffa81937470c9`。这不是预注册里程碑或final checkpoint，不选它来制造一个新的主要成功率。

finalizer flag已启用，但因中断没有finalizer receipt。TensorBoard不是空：actor/central主loss及info/epochs各10点；bounds_loss仅9点，reward/shaped/episode_length各8点止于epoch9，缺失epoch10尾部。没有补造或用checkpoint的训练reward填补TensorBoard。按400epoch检查和按10epoch前缀检查均不完整。

## 6. 独立评估与科学结果

计划的100/200/300/400 checkpoint均不存在，没有启动任何GPU评估。新run的ever-success、terminal-success和conditional first-success time都为**未评估/null**；不是0%或0/288。“0个已评估episode”只表示没有分母。G2的280/288=97.222%仍是历史64-env主结果，G3B诊断仍单列；没有追加到本轮统计。

科学状态是**NOT ASSESSED**，既不能判定收敛，也不能判定seed学习失败。仅有早期训练reward/loss曲线，不能证明PegInsert独立成功或与G2等价。

## 7. CPU验证、保留与交付

本轮全量CPU回归**126 passed**（121既有+5新analysis控制），包括预算/分母分离、缺失评估不填0、统计时序和不完整日志处理。CPU checkpoint和TensorBoard检查使用既有工具，实际命令、returncode、hash与tag计数在`cpu_artifact_checks.json`。

只读审计确认起点583个tracked文件、26个已冻结源文件、6个原G2 checkpoint、16个G3C-1/R1原始文件记录及5个预存G3V未跟踪文件均匹配。历史证据不覆盖；本轮新增独立receipt、命令、summary和图表。guard/supervisor/训练科学入口未改动，分析代码不启动GPU。

参见[证据索引](../evidence/g3c_r2/README.md)、[当前结果](../evidence/g3c_r2/result.json)、[CPU工件检查](../evidence/g3c_r2/cpu_artifact_checks.json)、[逐文件保留审计](../evidence/g3c_r2/runtime_preservation_audit.json)。大checkpoint、完整console/guard CSV/TensorBoard保持在仓库外；文件大小/hash在`raw_artifact_inventory.json`，small direct state/launch records另有提交副本。

## 8. 最终建议与停止

**下一次长训练NO-GO，先审阅监控可靠性。** 当前结果不是新的32-env显存NO-GO：有完成的容量smoke且全运行观测free均高于1024MiB，但不能保证完整长期采样。后续应单独调查采样/心跳延迟原因并验证安全策略，保持0.5s gap和1024MiB floor不变；之后重新做资源preflight并经审阅决定是否用新run从头开始32-env seed0。

本轮已经停止：不自动恢复epoch10、拼接记录、重跑32-env、减到16/8环境、启动seed1/2、contact sensor或其它Factory任务，也不租用云资源。32-env baseline尚未完成，不能进入其seed1/2阶段。
