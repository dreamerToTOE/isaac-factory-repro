# G4-I1 接受计划（预注册，尚未执行）

本文件属于I0审计交付，不授权自动开始I1。主参考固定G2 epoch200 SHA `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`，R4仅作第二适配检查，R5保留为附加证据。遵循 `evidence/g4_i/i0/interface_spec.json` 和完整RMS/配置指纹；不能悄悄替换checkpoint。

## 独立入口和身份

1. 只创建官方等价actor/normalizer/LSTM和原Factory场景，不创建Runner训练循环、optimizer、PPO更新；不改reward、physics、成功定义或checkpoint。
2. SHA在可信本地pickle加载前核验；strict检查keys/shapes/dtypes/RMS/epoch，model.eval、FP32、inference_mode；初始不启用AMP。
3. 记录Python/Isaac Lab/RL-Games/Isaac Sim版本和文件SHA、唯一run目录、命令、resolved config、依赖、child实际wait/exit和console。

## 固定trace与parity门禁

预注册evaluation/reset seed=1001（非training seed），torch起始seed=1001，B=1。首个trace至少包含完整149步及reset后首动作；CPU合同fixture为确定性固定数值、不消耗随机数。若资源条件要求变更运行协议，须在运行前另存修订，不覆盖I0冻结记录。

- 先用同一非空trace逐步输入官方evaluator和独立入口：原始19D obs、各checkpoint完整RMS、同一zero h/c、batch/device及previous-cache。至少覆盖初态、普通运动、clip饱和、零旋转、位置边界、timeout和reset后的首动作。
- 分阶段比较normalized obs、raw mu、h_next/c_next、clip、EMA、位置target、upright quaternion、controller输入。只比较clip后的动作会掩盖raw mu误差，不足以通过。
- CPU FP32 normalized/mu/h/c：`abs(a-b) <= 1e-6 + 1e-6*abs(reference)`；GPU同device FP32：`<=1e-5 + 1e-5*abs(reference)`。报告max_abs/max_rel、超差元素数、样本数和每阶段结果，阈值不得看结果后调整。
- cache和目标位置FP32同以上容差（m）；旋转target按同一wxyz符号规则比较，并报告sign-invariant quaternion测地角误差<=1e-4rad。超差必须定位和记录，不能调整科学参数使通过。
- 官方Player选mu前会discard一个Normal sample；新入口明确选择复制RNG消耗，或分离/reset RNG并报告随机流变化。只算mu不消耗RNG时，同seed不等于同reset场景；trace parity必须固定观测，episode比较须冻结初始状态/随机流。
- 冻结RMS在所有步骤前后逐bit一致；不是重新估计统计量。hidden/cell `[2,B,1024]` 随步骤延续、done后对应列清零；cache按初始pose重建，不全零、不额外clip。

## 至少一个完整episode

原IsaacFactory PegInsert单env/headless，至少完成149policy步至官方timeout；记录初始reset、pre-reset末态、post-reset obs/cache、清零h/c及新episode首动作。初始和reset物理推进另记，不能混作149步控制时间。

在auto-reset前抓取官方 `_get_curr_successes`，报告ever、terminal和first-success（从未成功则null）及真实分母。一个episode验证运行路径，不证明成功率或重现288episode科学结果。

trace需包括逐步obs、raw/clip/EMA action、每个物理子步target、controller输入、episode step、simulation time、wall time、checkpoint身份及artifact SHA。GPU同步的P50/P95/P99、deadline miss和完整科学分母属于I2，不在I1以单次wall time替代。

## 错误与资源门禁

- CPU mock注入非有限输入/输出、错shape/device、SHA不符、缺RMS或状态batch不符：拒绝提交新命令、保存原错误、非零退出；验证控制提交mock调用次数为0。物理停止/保持策略另在I2明确登记和测试，不声称真机安全认证。
- GPU工作开始前单独只读资源前检及检查其他GPU任务；R4/R5 no-GPU-guard授权不自动继承。不改驱动、不终止无关进程；真实OOM/asset/runtime失败保留原错误后退出，不自动改配置重试。

## 结论规则

独立入口、身份链、分阶段parity、至少一个完整episode、reset/RNG证据、真实exit全部通过才判 `G4_I1_PASS`。只运行官方play.py、仅GUI或仅mock通过为PARTIAL。失败据实为PARTIAL/FAIL。

I1完成后停止等待审阅，不自动运行I2、Sim2Sim、seed2、GearMesh或NutThread。I0本次没有运行上述任何测试或工作负载。
