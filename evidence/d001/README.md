# D001 证据索引

所有机器人/GPU结果来自本机实际运行；synthetic及checkpoint state round trip是明确标记的CPU控制实验。大checkpoint只保留在仓库外的`work/runs`，此处保存hash、路径及检查结果。未改写或删除原G0/G1证据。

| 文件/目录 | 作用 |
| --- | --- |
| [r1_inventory.json](r1_inventory.json) | 原提交103个文件的完整体积清单与常见凭证签名检查 |
| [r1_push.json](r1_push.json) | 原提交SSH推送断连、服务端接收后的独立SHA核验、祖先关系 |
| [scalability_summary.json](scalability_summary.json) | 六次train/resume实际更新、损失、分阶段梯度、FPS、显存、warnings和200epoch外推 |
| [checkpoint_scaling.json](checkpoint_scaling.json) | 16/32/64各epoch3及恢复epoch4的本地路径/SHA256/有限性/Adam/权重差异 |
| [train16](train16/result.json)、[train32](train32/result.json)、[train64](train64/result.json) | 各3epoch命令、返回码、墙钟、10Hz NVML统计；同目录console/gpu.csv/training_audit.json |
| [resume16](resume16/result.json)、[resume32](resume32/result.json)、[resume64](resume64/result.json) | 各checkpoint加载后只增加1epoch；同目录原始观察记录 |
| [metrics_tests.txt](metrics_tests.txt) | 15项CPU指标/Player接口/自动reset控制测试通过 |
| [metrics_synthetic.json](metrics_synthetic.json) | 已知真值：ever3/4，terminal2/4，transient-only1/4；明确非机器人结果 |
| [metrics_seed0.json](metrics_seed0.json)、[metrics_seed1.json](metrics_seed1.json) | 每seed4完整episode，官方pre-reset成功、counter及日志一致性，首达时间、seed、checkpoint hash |
| [metrics_combined.json](metrics_combined.json) | 两seed合计8完整episode；ever/terminal0/8，首达均null |
| [eval_seed0](eval_seed0/result.json)、[eval_seed1](eval_seed1/result.json) | 真实GPU评估命令/console/10Hz显存CSV |
| [checkpoint_state_audit.json](checkpoint_state_audit.json) | 官方保存/恢复方法的CPU控制实验10项断言，证明critic Adam/last_lr等未完整恢复 |
| [checkpoint_state_audit_source.txt](checkpoint_state_audit_source.txt) | 精确本地RL-Games源码摘录和行号 |
| [checkpoint_state_audit_source_identity.json](checkpoint_state_audit_source_identity.json) | 9个本地源码与官方v1.6.1固定SHA逐文件hash一致 |
| [training_observer_notes.md](training_observer_notes.md) | 梯度阶段、AMP skip、FPS与观察开销语义 |
| [script_identity.json](script_identity.json) | 项目Python工具最终SHA256身份 |
| [delivery_verification.json](delivery_verification.json) | 提交前范围/体积/凭证签名、上游干净状态、算法配置一致性及文档链接检查 |

## 重建汇总

在工作区根目录：

```bash
python3 outputs/isaac-factory-repro/scripts/summarize_scaling.py \
  --logs-root work/logs \
  --output outputs/isaac-factory-repro/evidence/d001/scalability_summary.json
```

输出会明确标记缺失证据，不能将缺文件视为通过。各run的原始JSON保存了完整命令和cwd；额外环境设置为`FACTORY_VENV=.../work/envs/factory-g01`、`OMP_NUM_THREADS=4`、`FACTORY_AUDIT_OUTPUT=.../work/logs/d001_{train,resume}N/training_audit.json`。所有resume及eval仅对本项目本地已核验SHA的文件额外设置单次`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。

NVML v2的used排除driver reserved、包含桌面；free单独报告，不能用total减used推算free。所有GPU工作串行；1024MiB空闲停止线未触发。原始scaled/unscaled梯度存在首次AMP overflow，实际应用梯度均有限，不能只用checkpoint有限性抹去该观察。

正式200epochs没有执行，所有估时均为短测外推。详见[完整报告](../../docs/REVIEW_D001.md)。
