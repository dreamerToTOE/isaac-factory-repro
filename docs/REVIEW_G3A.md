# G3A：训练监督与日志可靠性审阅

日期：2026-10-25（Asia/Seoul）。仓库起点为G2提交 `b89dc08f7421b24480c9a756780cc4f6fc058422`；上游仍为Isaac Lab v2.1.1 / `90b79bb2d44feb8d833f260f2bf37da3487180ba`。

**决策：PASS，限工程可靠性验收。** 最终执行闸门方案通过77项CPU测试及独立64环境、2epoch PPO smoke：真实child exit0、完整采样协议覆盖、checkpoint和末轮TensorBoard均有直接证据。此前两次开发smoke的失败/部分完成结果一并保留。没有执行新的200epoch实验、seed1/2训练或G3B；G2科学PASS及历史数据不变。

## 1. 根因发现与证据边界

G2原supervisor/monitor消失的确切原因仍未知。约122.747秒显存/guard空白、原child退出码未知、TensorBoard缺epoch200尾部均不能事后恢复。原G2事件SHA256与全部Git内G2证据再次核对未变，见[preservation_verification.json](../evidence/g3a/preservation_verification.json)。

直接源码确认：pinned RL-Games 1.6.1创建tensorboardX writer，`Runner.run_train`调用`agent.train()`后没有显式flush/close；官方训练入口随后关闭env/Sim。tensorboardX默认异步flush间隔120秒。该路径与G2缺尾相容，但不能确定历史退出路径或唯一根因。[源码审计](../evidence/g3a/tensorboard_source_audit.json)保留hash和定位；G3A没有合成或回填G2标量。

本轮开发直接发现：手工在child启动后提交guard造成启动空白；旧版每样本同步写盘，且实测采样间隔拖长；没有单独计量每次fsync耗时，不能据此断言唯一原因。最终方案自动先采样、绑定后exec；运行心跳采用快速原子更新，持久化同步留给元数据、事件与最终状态。新smoke成功不能反证G2事故根因。

## 2. 实现

新增[supervisor](../scripts/durable_supervisor.py)、[独立guard](../scripts/gpu_guard.py)、[systemd launcher](../scripts/g3a_launch_smoke.py)、[身份/原子文件工具](../scripts/supervision_common.py)、[执行闸门](../scripts/supervised_exec.py)。准确命令与文件契约见[G3_SUPERVISION_PROTOCOL.md](G3_SUPERVISION_PROTOCOL.md)。

guard和supervisor由独立systemd用户服务拥有，均为`Restart=no`、`KillMode=control-group`。guard真实采样就绪后才创建等待pipe的child；guard绑定不可变boot ID、PID、birth tick后，supervisor才放行原训练argv。闸门只影响开始时间。两服务互查monotonic心跳和身份，丢失、采样错误/gap、低显存均形成可见失败状态。

默认用户GPU0全局`flock`防止重复启动，锁FD由child继承，supervisor单独死亡后仍拒绝第二次启动。新目录/不可变JSON拒绝覆盖；最终JSON原子持久化。分别保存原始wait返回码、普通exit code、signal与termination reason。无法回收wait时保持未知，guard/systemd退出码不能代替child。

显存规则保持 **free<1024MiB连续3次**；0.1秒采样。另有明确工程故障线：样本间隔>0.5秒记gap并fail closed，心跳stale上限2秒。未降低采样频率来掩盖丢样。

`python_entry.py`仅在`FACTORY_G3A_FINALIZE_TENSORBOARD=1`时启用[finalizer](../scripts/tensorboard_finalizer.py)，仅包装官方`Runner.run_train`，在返回/异常收尾时flush→close并记录结果。[tail checker](../scripts/tensorboard_tail.py)独立检查固定必需tags、精确计数、原生step与有限性。官方PPO、奖励、任务、控制器、物理和成功定义均未编辑。

## 3. CPU-only验收

最终源码在GPU运行前通过 **77 passed in18.41s**：监督/guard16项、日志19项、既有42项。[命令](../evidence/g3a/cpu_final.txt)、[JUnit](../evidence/g3a/cpu_final_junit.xml)、[牺牲进程证据](../evidence/g3a/cpu_cases/)保留实际结果。早期16/58项开发记录不替代最终验收。

| 用户要求 | 直接测试 |
| --- | --- |
| exit0、非零、signal | 真实wait为0、7、-SIGTERM；signal普通exit为null |
| 调用端消失 | 仅SIGKILL新建牺牲systemd测试launcher，服务继续并取得child exit0 |
| guard失败 | 缺guard不执行训练；guard SIGKILL/mock NVML异常停止所属child，不伪造final |
| supervisor丢失 | 牺牲supervisor SIGKILL，独立guard检测并停止匹配child，不臆测child退出码 |
| 原子最终状态 | 并发读取无半写JSON；模拟rename失败保留旧文件 |
| 防重复/非覆盖 | supervisor死亡后child仍持锁；第二启动拒绝；原证据不变 |
| 模拟低显存 | 连续3低值触发；非连续低值重置；未耗尽真实GPU |
| 错身份/gap | 错birth不发信号；模拟0.65秒gap记录失败 |
| 缺TensorBoard尾部 | 缺tag/step/非有限值拒绝；fake Runner检查原返回/异常、writer失败及flush→close |

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`禁用无关ROS插件自动加载，沿用既有venv，未安装依赖。开发测试发现并修复身份文件已可见而启动心跳尚未写出的竞态；最终通过后冻结源码，GPU后核对无漂移。

## 4. 最终GPU smoke

run=`g3a_smoke_gated_64env_20261025_final`。[launch.json](../evidence/g3a/smoke_gated/launch.json)保存实际展开argv、环境和两个unit命令。cwd为仓库外`work/runs`；官方train.py，无checkpoint参数，64env、seed0、2epochs、独立名称、save_frequency=1。它是短工程smoke，不是G2完整基线重训或性能评估。

| 项目 | 实测 |
| --- | --- |
| 预算 | 连续epochs1、2；16,384 transitions；官方MAX EPOCHS结束标记 |
| 真实wait | **raw_wait_returncode=0、child_exit_code=0、child_signal=null** |
| 终止原因 | child_exit；supervisor error=null |
| supervisor墙钟 | **82.726370秒**，含等待guard、初始化、保存和退出 |
| guard | **829样本**；child_wait_observed；error=null、gap=[] |
| 采样间隔 | median0.100000303秒，max0.281052809秒；gap线0.5秒 |
| 观测全卡显存 | used峰值6458.0MiB，free最低1359.625MiB；连续低值最大0 |
| checkpoint | epoch2/frame16384、tensor有限、22组非零Adam状态、actor step127 |
| 优化观察 | actor尝试128/实际127，首次官方AMP自动skip1次；central-value128；返回loss数组有限 |
| TensorBoard | flush/close成功；50tags全有限；固定13个必需tags检查通过 |
| console | 无已记录OOM/Traceback；保留API弃用、render interval等官方告警 |

完整覆盖独立核验五个monotonic时间：

```text
first_sample  8604.683965967
child_spawn   8605.226726640
exec_release  8605.583348916
parent_wait   8687.415336563
last_sample   8687.484187956
```

首样本在spawn前0.542761秒，末样本在wait后0.068851秒，三方身份一致。见[coverage](../evidence/g3a/gated_coverage.json)、[运行中健康快照](../evidence/g3a/gated_active_health_snapshot.json)、[supervisor status](../evidence/g3a/smoke_gated/supervisor/status.json)、[guard status](../evidence/g3a/smoke_gated/guard/status.json)。完整覆盖指采样协议执行完整，不是连续物理测量或瞬时峰值保证。

TensorBoard actor/performance各2点（frame0、8192），central-value各2点（8192、16384），`info/epochs=[1,2]`。149步episode与128步horizon使reward首次在epoch2出现，因此1个reward点是正确计数。见[tail](../evidence/g3a/gated_tensorboard_tail.json)、[finalizer](../evidence/g3a/smoke_gated/supervisor/tensorboard_finalizer.json)。

见[checkpoint检查](../evidence/g3a/gated_checkpoint.json)、[外部原始文件清单](../evidence/g3a/raw_artifact_inventory.json)。大checkpoint/事件不提交。YAML与G2只有max_epochs200→2、save_frequency50→1、名称/路径差异；env YAML字节一致，runtime64/19/43/6、horizon128、minibatch512、mini_epochs4一致，见[config_identity](../evidence/g3a/config_identity.json)。

## 5. 开发失败保留

| 尝试 | 实际结果 |
| --- | --- |
| smoke_20261025_0633 | 手工guard迟启超过20秒grace，supervisor fail closed；child SIGKILL、普通exit=null，未完成epoch |
| smoke_20261025_0638 | 2epoch/exit0/checkpoint/TB完整，99.890272秒；guard晚约27.54秒才开始，181样本且多处超旧0.25秒gap线，监控PARTIAL |
| 最终gated run | 最终77项CPU通过后执行，先采样、绑定后exec，无gap；作为最终验收 |

前两次stdout/状态/guard序列见[attempts](../evidence/g3a/attempts/)。没有覆盖、拼接或以成功替代失败。旧schema开发hash只代表当时版本；最终源码以[gated_preflight](../evidence/g3a/gated_preflight.json)为准。

## 6. 限制与G3B建议

- 当前`Linger=no`。直接测试覆盖调用客户端/终端消失；未改变全局登录配置，不保证跨完整注销、user-manager退出或重启。
- 两服务同时SIGKILL、断电、kernel/磁盘故障可能阻止final状态落盘；缺失final/stale heartbeat必须视为未知/不完整，无法保证恢复wait状态。
- 采样不是硬实时，短峰值仍可能漏掉；真实gap超0.5秒即失败，不能计为持续成功。
- 通用命令daemonize、改变PGID、关闭继承锁、绕过launcher超出标准run_sim exec保证；检查身份与发信号之间仍有极小Unix竞态。systemd只清理本次新unit的cgroup。
- 2epoch不证明长时硬件/存储稳定、跨训练seed性能或精确续训；G2历史缺失与D001续训限制不变。

建议G3B先在已审阅协议下，对G2 final的8个失败实例保留完整动作/几何/保持轨迹，再预注册额外训练seed实验与评估预算。长训练前重新核验版本、资源、manifest与guard。本轮在G3A停止，未启动G3B或任何新长训练。
