# D001 审阅跟进：R1–R4

日期：2026-10-23。上游 Isaac Lab `v2.1.1` / `90b79bb2d44feb8d833f260f2bf37da3487180ba`，RL-Games 1.6.1，RTX 3070 8GB。官方源码、PPO 网络、奖励、控制器、成功条件均未修改。本轮只有有界短测，**没有启动 200 epochs**。

## 结论与下一轮建议

**工程启动 GO：建议后续首次 PegInsert 训练使用 64 环境、固定 seed=0、全新初始化。当前停在 D001，等待审阅。** 64 环境通过真实优化与保存/加载；独立checkpoint评估另在4环境验证。64环境200 epochs按短测外推约 **2.0小时**，实际排期建议 **2–2.5小时**。外推包含观察工具同步开销，不是长期性能保证。

64 环境是已实测的资源适配配置。官方默认128环境未在本轮验证；64×128×200 = **1,638,400 transitions**，只有官方128环境200 epochs样本预算的一半。若以后要求相同样本预算，64环境需400 epochs（短测外推约4小时），必须单独审批。不能将64环境200 epochs称为与默认配置等样本的复现。

checkpoint可用于推理和继续优化；**当前官方保存格式不支持完整可复现的中断续训**。首次正式训练不得加载G1/D001 checkpoint。

## R1：原提交推送

- 原提交：`a77d215b1244b120605eac5c0af59bf0f32c20c5`；父提交及最初远程main：`c54da28f61569ec6cc8249b46ffcea498ff27600`。
- 初始工作树干净，`main` 领先1；检查了status、remote、history和全部103个changed files。总内容430,792 bytes，最大文件38,710 bytes。常见token/private-key/带密码URL签名未命中，无`.pth/.pt/.ckpt/.pkl/.bin`。
- SSH fetch后验证远程main为本地祖先；保留HTTPS fetch URL，仅将origin的push URL设为已验证的`git@github.com:dreamerToTOE/isaac-factory-repro.git`。没有更改SSH配置、读取密钥或令牌。
- 普通`git push origin HEAD:refs/heads/main`首次返回Broken pipe，但随后独立`ls-remote`证明服务端已经接收该提交：远程SHA等于`a77d215...`。未盲目重推、未force push。
- D001增量另作后续提交，最终远程SHA由交付报告及Git历史给出。原始文件和历史均保留。

证据：[完整库存](../evidence/d001/r1_inventory.json)、[推送与独立远程核验](../evidence/d001/r1_push.json)。HTTPS本身仍无本地认证配置；通过既有SSH路径已完成本任务，认证不再是阻塞项。

## R2：独立成功指标

源码核对确认：

1. `extras["successes"]` 在有reset的步取**当下**成功mask / 总环境数。同步timeout baseline中，它是末步成功率，不能解释为曾成功率。
2. `ep_succeeded` 记录曾成功，在reset后的下一pre-physics阶段清零；它没有用于上述日志的分子。
3. `ep_success_times` 未在reset清零，后续失败episode可能保留旧值；`success_times`不能直接当作当前completed episodes的首达均值。
4. 成功不触发done；只按timeout结束，当前实测第149个policy step自动reset。

固定源码：[官方成功统计与缓存](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L458-L484)、[缓存清理](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L292-L300)、[timeout规则](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py#L425-L456)。

独立评估器捕获**官方原predicate返回值**，不重写几何条件。父`step()`会在返回前reset，因此捕获发生在reward计算中、reset之前。每个完整episode独立记录ever、terminal、first-success step、seed和episode身份；失败首达为null。ever/terminal的分母均为完整结束的environment episodes。首次成功时间只对成功episodes取条件均值。详细科学定义、固定行链接和运行命令见 [SUCCESS_METRICS](SUCCESS_METRICS.md)。

CPU确定性测试：**15/15通过**。受控瞬时成功后失败的episode为ever=true、terminal=false；末步成功为两者true；自动reset返回false不会覆盖捕获的末步成功；下一episode不继承旧首达。另一个明确真值序列得到ever=3/4、terminal=2/4、transient-only=1/4。合成证据不是机器人实验结果。

真实checkpoint评估：seed0/1各4个完整episode，共8个；每episode149步，ever=0/8、terminal=0/8、first-success均null。两组都验证了捕获counter149、返回counter0及官方末步日志的一致性。该3-epoch smoke模型没有成功样本，不能估计正式训练性能；瞬时/末步成功的区分由明确真值的合成测试验证。证据：[seed0](../evidence/d001/metrics_seed0.json)、[seed1](../evidence/d001/metrics_seed1.json)。评估采用确定性均值动作，区别于官方player默认随机动作；只改变评估方式，不改训练基线。

## R3：统一短测对比

每组使用seed0，全新训练3 epochs，再从本组epoch3 checkpoint加载至总epoch4（只增加1 epoch）。保持官方2层1024 LSTM、MLP[512,128,64]、horizon/seq128、minibatch512、mini_epochs4、AMP、自适应actor学习率等。仅覆盖环境数量、短测总epoch、实验名及保存频率/起点，后两项用于观察保存行为。

| 环境数 | 全新训练transitions | actor/critic实际更新 | 环境step FPS（epoch2–3） | PPO总FPS（epoch2–3合计） | 全卡峰值MiB | 测试compute峰值MiB | 最小空闲MiB | 200 epochs外推 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 6,144 | 47 / 48 | 70.7–71.2 | 67.31 | 5669.25 | 4238.68 | 2148.38 | 1.69 h |
| 32 | 12,288 | 95 / 96 | 135.7–137.5 | 125.10 | 5679.25 | 4294.68 | 2138.38 | 1.82 h |
| 64 | 24,576 | 191 / 192 | 265.4–265.8 | 227.50 | 5724.94 | 4306.68 | 2092.69 | 2.00 h |

各组3 epochs整个进程墙钟分别110.20/114.60/123.40秒，包含启动、reset、checkpoint I/O。200 epochs外推取epochs2–3平均epoch时间30.43/32.74/36.01秒×200，不计一次冷启动和额外I/O。不同环境数的同epoch不等样本量；64环境更高总吞吐仍每epoch处理更多样本。

FPS定义：环境step FPS=总transitions / RL-Games计量的env.step时间（包括在step内的reset）；rollout FPS另含policy inference；PPO总FPS再含优化。表中FPS不是每个环境的FPS。观察工具的CUDA同步开销包含在时间内，未人为扣除，也未量化无观察时的加速比。

显存使用只读NVML v2以**10 Hz**采样，实测中位间隔0.100秒；全卡used包含桌面，排除约374MiB驱动reserved，与`nvidia-smi memory.used`口径相同。`used + free + reserved = total`。进程列是本次进程组compute占用，不能相加到全卡used。保留独立PyTorch allocator精确峰值，但它不覆盖PhysX/Kit全部内存。采样不能保证捕获100ms以下瞬时峰值。

所有组设置1024MiB空闲停止线（连续3次采样）；未触发，且没有并发GPU测试。64环境完成后停止扩容，没有尝试128/256/512。

### 有限性、AMP与加载

- 六次train/resume运行均exit0；观察工具和NVML错误列表为空。actor损失、central critic损失、policy输出均为有限值；所有**实际应用**的actor/critic梯度为有限值。
- 每个fresh run首次AMP attempt存在scaled及unscaled非有限梯度，GradScaler从65536降至32768并**跳过该次actor更新**。后续attempt全有限，三个fresh run实际更新均比计划少1。不能写成“所有梯度全有限”，也未关闭AMP来隐藏此现象。resume使用恢复的scale，无再次跳步。
- 三个resume分别实际增加16/32/64次actor与critic更新；主Adam累计step47→63、95→127、191→255。每组actor22个参数张量、critic18个参数张量变化，所有checkpoint tensor有限。
- 保存与加载均成功；仍只证明推理兼容/可执行继续优化，不证明完整状态连续恢复。短resume的128步未到149步timeout，出现无episode统计的`rew_-inf`和对应warning，损失及模型有限性检查并未失败。
- 日志保留crash reporter缺失、rendering_modes extension配置缺失、OmniHub不可达、MaterialX提示、render interval<decimation及deprecated Isaac/AMP API等告警；未见OOM、PhysX错误或非有限值导致的实际优化失败。没有证明长期物理稳定或装配收敛。

证据：[结构化汇总](../evidence/d001/scalability_summary.json)、[checkpoint有限性/变化/hash](../evidence/d001/checkpoint_scaling.json)、[六组原始console/10Hz CSV/逐次梯度](../evidence/d001/README.md)。

## R4：checkpoint精确恢复边界

主模型（含actor输入/value RMS）、central critic模型及RMS、主Adam、AMP scaler、actor epoch/frame可保存恢复。**critic独立Adam、其counter、调度使用的last_lr、RNG流位置、仿真状态和运行时LSTM hidden/cell/rollout均不完整**。主Adam参数组LR恢复并不等于`self.last_lr`恢复。官方critic YAML虽写adaptive，1.6.1实现此处实际走IdentityScheduler；保留此行为。

通过实际官方get/set_full_state_weights做CPU受控round trip，10项断言通过，并直接证明critic optimizer与last_lr遗漏。源码文件身份逐项比对，详见 [CHECKPOINT_REPRODUCIBILITY](CHECKPOINT_REPRODUCIBILITY.md)。未来fresh seed0 baseline不受续训状态缺失的起始污染，但若中途断电，当前官方checkpoint不能承诺与不中断run等价。

## 复测入口（只启动短测）

下列示例使用新实验名和不存在的证据目录，避免覆盖既有结果；将`FACTORY_N`改为16/32/64。所有真实执行命令在各组`result.json`。

```bash
export FACTORY_WORKSPACE=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been
export FACTORY_REPRO="$FACTORY_WORKSPACE/outputs/isaac-factory-repro"
export FACTORY_VENV="$FACTORY_WORKSPACE/work/envs/factory-g01"
export FACTORY_N=64
export FACTORY_RUN=d001_repeat64
cd "$FACTORY_WORKSPACE/work/runs"
env OMP_NUM_THREADS=4 \
  FACTORY_AUDIT_OUTPUT="$FACTORY_WORKSPACE/work/logs/$FACTORY_RUN/training_audit.json" \
  python3 "$FACTORY_REPRO/scripts/run_bounded.py" \
  --output "$FACTORY_WORKSPACE/work/logs/$FACTORY_RUN" --timeout 600 --min-free-mib 1024 -- \
  bash "$FACTORY_REPRO/scripts/run_sim.sh" "$FACTORY_REPRO/scripts/train_audited.py" \
  --upstream "$FACTORY_WORKSPACE/work/upstream/IsaacLab/scripts/reinforcement_learning/rl_games/train.py" \
  --headless --device cuda:0 --task Isaac-Factory-PegInsert-Direct-v0 \
  --num_envs "$FACTORY_N" --seed 0 --max_iterations 3 \
  agent.params.config.full_experiment_name="$FACTORY_RUN" \
  agent.params.config.save_frequency=1 agent.params.config.save_best_after=0
```

恢复测试在同样命令中加本组已核验SHA256的`--checkpoint`，改为`--max_iterations 4`、新实验名和输出目录；仅该次进程设置`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。正式训练另行审阅后从头开始，保存频率恢复官方100、save_best_after恢复10。

## 剩余限制

- 短测不能证明长期显存增长、温度/功耗稳定、收敛或成功率；200 epochs估时只是外推。
- 正式128环境默认预算尚未实测；64环境200 epochs是资源适配预算。
- 没有完整可复现续训保证，也没有GPU物理逐位确定性保证。
- 最小依赖安装仍保留pin-pink/dex-retargeting/moviepy声明缺口；已验证Factory headless无视频路径。
- 独立评估的极小episode数只用于工具验证；首轮性能报告需要预先固定更多seeds/episodes及deterministic/stochastic设置。
- 原上游成功时间缓存和其他G0/G1审计问题保持原样；没有以工程补丁替换官方baseline。

**本轮到此停止，不启动200-epoch run。**
