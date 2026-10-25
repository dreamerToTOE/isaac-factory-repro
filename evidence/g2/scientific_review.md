# G2 scientific audit — read only

审计日期：2026-10-25（Asia/Seoul）。审阅任务由主 agent 委派；本次未启动 GPU 作业、未改仓库文档、未推送、未发送 GitHub 消息。

仓库：`/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro`。
原始运行：同一旧 workspace 下 `work/logs/g2/g2_peginsert_64env_seed0_baseline_rerun`。

## 结论与 decision gate

在主 agent 完成代表性 playback 核验后，建议 **PASS（限本次 64-env、单训练 seed=0、200 epochs 资源受限初步基线）**。预算完成、独立成功和内部一致性已有直接证据；不能写为官方 128-env 等样本复现、跨训练 seeds 可靠性证明或完整无缺监测。

必须与 PASS 同时披露两个证据缺口：原监控父进程消失造成约 122.747 秒 NVML/安全 guard 空白，原训练 OS exit code 不可回收；TensorBoard 尾端 epoch200 scalar 未持久化。预算完整性由连续 epoch audit、最终 checkpoint 内 epoch/frame、optimizer step 及 `MAX EPOCHS NUM!` 交叉支持，而非伪造 exit0 或插值补齐日志。没有训练中断或 checkpoint restore 证据；同一 PID/start-ticks 持续训练，与监控中断应分开描述。

如 playback 尚未完成，最终交付尚有这一验收项；现有 144 条 pre-reset 几何记录不能冒充视频/截图。另须避免把 PASS 解读为所有工程缺口已解决。

## 配置、源码与 runtime 身份

- 实际 Isaac Lab HEAD 为 `90b79bb2d44feb8d833f260f2bf37da3487180ba`，`git status --short` 空，官方 Factory 环境、控制器、奖励和配置未编辑。
- 本次 frozen training scripts 的 SHA256 全部与 `g2_manifest.json` 一致；当前 evaluator 及相关依赖全部与 `evaluation_amendment.json` 一致。amendment 在首次 G2 evaluation 之前，只补 partial provenance/supervision，不改 seeds、episodes、deterministic policy 或成功定义。
- 独立逐键比较官方 PPO YAML 与本次 `params/agent.yaml`：仅 `save_frequency:100→50`、experiment name、train_dir 不同。官方 seed 已为0、max_epochs已为200。实际64 actor 的覆盖在 dump之后发生；**保存的 agent.yaml 仍写128，不是最终 runner runtime 数量错误**。官方 `train.py` 先 dump YAML（约143–146行），之后 `agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs`（约180行）。应同时交付 env.yaml 和 observer runtime config，明确前者是 pre-runner snapshot。runtime observer确证64 actors、horizon128、minibatch512、mini_epochs4、max_epochs200。
- runtime 为 actor19维、critic43维、action6维；两层1024 LSTM、MLP[512,128,64]、ELU 等一致。保持 gamma .995、tau .95、lr1e-4/adaptive、KL.008、clip.2、entropy0、gradient norm1、seq128、AMP、normalization、value bootstrap。`player.deterministic=False` 是官方训练配置；独立评估显式 deterministic=True，与训练参数变更无关。
- `load_checkpoint=False`、`load_path=""`、CLI无 checkpoint，observer所有 checkpoint事件为save、无restore；这是fresh训练。既有 orphan attempt 未产生epoch/checkpoint，不与此次拼接。
- stdout中 `PINNED_PACKAGES` 和 `PACKAGES_AFTER_APP_LAUNCH` 都是既有隔离环境Torch2.7.0+cu128、Torchvision.22.0+cu128、NumPy1.26.4、Gymnasium1.2.0，无 Kit archive遮蔽证据。12个eval的runtime metadata一致。
- 官方 actor observation无显式接触力，`applied_wrench`为控制器命令力矩/力，不可称force-sensing PPO。官方成功是XY<2.5mm、单侧z_disp<1mm（PegInsert height .025×threshold .04），没有下界、绝对z、接触力、持续时间或旋转约束。保留此语义，不能扩展为机械装配质量保证。

## 训练完成、稳定性与TensorBoard

- audit精确为epochs1…200且无重复，200×8192=**1,638,400 transitions**。里程碑checkpoint epoch/frame为50/409600、100/819200、150/1228800、200/1638400。
- primary选中final是 `last_Factory_ep_200_rew__350.4961_.pth`，SHA256 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`。独立重新计算所有12次eval所引用checkpoint hash，均等于各自metrics与inventory。所有checkpoint inspection为finite；final actor Adam22个state的step=12792，吻合observer。不能依据两个epoch200文件的不同文件hash就宣称不同policy，也不能未经tensor比较声称二者字节等同。
- actor尝试12800，实际Adam调用12792；8次AMP skip（epochs1、9、80、111、126、156、170、170），比例.0625%。记录的scale下降，符合GradScaler自动防护，无配置更改。central value实际Adam调用12800。instrumentation_errors为空。
- observer200次 `epoch_loss_arrays_finite=True`，只涵盖A2CAgent.train_epoch返回的数组；**不直接包含单独训练的central-value loss**。进一步逐值和源码检查纠正了早期泛化提醒：本次 actor 内部 c_loss 不是0占位！pinned `a2c_continuous.py:54` 默认 `use_experimental_cv=True`，59行因此 `has_value_loss=True`（YAML未覆盖），actor-network value head也有真实value loss。`losses/c_loss` 首值1.451882、最后.00426727，均非零。`losses/cval_loss` 则是独立43维central critic的loss，首1.4239548、最后.001241208。应把两者分开；200个epoch observer有限数组覆盖前者，而后者有限证据是199个已落盘scalar与里程碑checkpoint。仅在另行设use_experimental_cv=False并有central value时，c_loss才是0占位，不是本次配置。
- TensorBoard frame严格核对：a_loss/c_loss共199点，从frame0（epoch1）到1622016（epoch199）；cval_loss共199点，从8192（epoch1）到1630208（epoch199）。`info/epochs`最后value199在frame1622016。actor记录epoch开始前frame，central-value训练内部先累加8192再记录；对应同一epoch的横坐标相差8192。**1630208这个最后cval scalar是epoch199，不能拿它冒充epoch200的loss**。200epoch预算应来自audit/checkpoint的1638400；epoch200 actor理论step1630208、central理论step1638400的scalar没有落盘。
- 本次observer说明明确“No gradient tensors are inspected”。因此不能继承D001的逐梯度结论，不能写“所有实际梯度都有限”或“无overflow”。准确写法是已记录loss有限、8次AMP跳过、参数及Adam状态变化、checkpoint finite、无已记录数值崩溃。
- analysis含53个scalar tags，但a_loss与cval_loss各199个点；reward/iter198个点为epoch2…199，最后353.4007对应epoch199，**不是epoch200**。最后checkpoint名中的350.4961是epoch200 reward meter另一个证据源，不能将它偷偷补入原TensorBoard事件。
- pinned RL-Games `common/a2c_common.py:281`创建tensorboardX SummaryWriter；默认异步写入、flush_secs120；训练返回和官方入口没有显式writer.flush/close；SimulationApp默认fast_shutdown=True，close调用native shutdown。tensorboardX有atexit清理，但无法确认此过程执行。现象**与尾端异步buffer未flush相容，具体退出路径仍未知**，不能把可能原因写成已证实根因或声称正常exit0。
- shaped_rewards/iter最高937.9557不违反raw reward上界434.58。`a2c_common.py:830–838`在RNN rollout把 `gamma * value * timeout` 加到shaped_rewards再累加，而raw rewards另计。它是含timeout价值bootstrap的训练量，不是额外物理奖励、成功率，也不能用raw reward的理论上界作判断。
- 原始reward meter最近约100个完成episode的滚动统计，相邻epoch重叠；training extras successes是结束时的当前瞬时成功比例，也不能代替独立ever-success。

## 独立评估独立重算

逐一只读核查12个metrics和对应result：全部status complete、monitor exit0、32环境、primary policy mean、model eval模式、checkpoint normalization buffer、inference_mode、done边界清零LSTM hidden/cell。每seed内部(env_id,episode_id)无重复；全部149 policy steps timeout；最终各seed3轮共96 episode/seed，447向量steps。每个terminal capture与官方extras successes吻合，post-reset计数归0；没有使用ep_success_times缓存作真值。所有144条几何snapshot均满足原官方predicate与独立geometry复核。

| epoch | seed1001 | seed1002 | seed1003 | aggregate ever=terminal | Wilson95 |
|---|---:|---:|---:|---:|---:|
|50|22/32|25/32|22/32|69/96=71.875%|62.1741–79.8926%|
|100|27/32|31/32|23/32|81/96=84.375%|75.8085–90.2963%|
|150|32/32|32/32|28/32|92/96=95.8333%|89.7718–98.3679%|
|200|95/96|96/96|89/96|280/288=97.2222%|94.6154–98.5859%|

全部primary评估中transient-only=0，但只说明这批episode的ever和terminal一致，不能推导policy永远不会暂态成功。

成功episode条件下first-success：

|epoch|n成功|mean step|median step|IQR step|range step|
|---|---:|---:|---:|---:|---:|
|50|69|35.2464|33|26–38|20–102|
|100|81|33.6543|27|23–37|19–141|
|150|92|35.5543|28|23–43.25|18–141|
|200|280|33.5357|29|24–40|18–105|

policy dt=1/15秒，final mean2.235714秒，median1.933333秒；这不含reset/init时间，也不是训练计算wall-clock。

Wilson区间描述episode计数的不确定性，所有episodes共享同一policy且同seed/向量环境不一定独立；三评估seed不等于三训练seed。checkpoint150与200差异小且分母不同、CI重叠，不可宣称200显著优于150。final200为预先指定，不能据评估事后改选best；训练best别名来自训练reward也不是测试集选优。

Final失败8例：seed1001(env11,episode0)；seed1003(env4,0)、(12,0)、(25,0)、(29,0)、(5,1)、(1,2)、(5,2)。原始primary失败轨迹没有充分全时几何，不能仅凭episode统计臆测接触/卡滞原因；G3可逐一诊断重放，单列为额外分析而不并入预注册分母。

## 计算效率、wall与监测空白

- 全程官方PPO total=9093.065246秒，aggregate180.181265FPS；环境stepping8306.779244秒、197.236492FPS；rollout8374.645599秒，optimization718.419592秒。
- 稳态epochs11…200共1556480 transitions/8737.157647秒=178.144891FPS。
- pipeline开始UTC12:12:02.991694至原PID消失UTC14:44:55.718230，约9172.726536秒=2h32m52.727s，端到端178.616466FPS。起点非精确child spawn，应用分析已标误差；KST是2026-10-23 21:12:02.992–23:44:55.718。不可用reattached duration8995.880秒替代总wall。
- 200epochs64env的预算是官方128env同epochs3,276,800 transitions的一半。mini_epochs/minibatch不变时每epoch64次actor尝试，对应128env为128次；不属于相等样本/优化步数实验。
- original CSV542条最后elapsed54.1；reattached89702条；合计90244。实际中位采样间隔0.1/0.100276931秒。分段采样跨度覆盖98.6607%仅统计跨度，不是连续监测保证。
- observed peak used6909.6875MiB/minfree907.9375MiB，是全卡含桌面，driver reserved374.375MiB另计。GPU缺口约122.746851秒；未知区间可能存在更高峰/更低free。不得写全程free>1GiB。
- 逐行重算两个CSV：**仅1个sample free<1024，最大连续1次**，未满足连续3次的停止规则。该sample在UTC2026-10-23 12:31:34.405057（KST21:31:34.405057）；前一sample12:31:34.304763 free1537.5625，后一sample12:31:34.505341 free1212.75。观测段无guard stop，但不能对缺口保证未触线。
- console没有Traceback/CUDA OOM/已记录fatal错误；有已知MaterialX/OmniHub、deprecated dynamic-control/AMP API、render interval1<decimation8告警，应保留。无法回收原退出状态仍是独立限制。

## G3优先级建议（只提案）

1. 先修复任务监督的持久化：父进程/guard独立生存、heartbeat/liveness、完整child退出码、训练结束显式flush并核查事件尾端；工程smoke验证后再开新长跑。不要补改G2原始证据。
2. 保持本次64env/200epochs/官方配置，预注册至少额外training seeds1、2以估计训练随机性；评估seed/episode协议提前固定。仍需用户另行授权G3启动，不自动执行。
3. 对8例失败做几何和playback诊断、扩充独立评估seed/每seedepisode数；保留final200 primary统计不变；评估较大样本下失败/保持指标。
4. 之后才单列等样本预算/128env容量研究。当前内存最小已低于1GiB，不可由64env短/长跑推断128env可行；不自动延至400、扩128、调PPO/奖励或启动Gear/Nut。

公开GitHub API只读检查 `https://api.github.com/repos/dreamerToTOE/isaac-factory-repro/issues?state=open&per_page=100` 返回空列表。仓库docs/OPEN_ISSUES.md仍有G0/G1/D001“未启动200epochs”历史语句，主agent应新增G2当前状态并明确旧段为历史记录，保留既有失败证据。
