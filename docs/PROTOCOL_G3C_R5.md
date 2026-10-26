# G3C-R5 — 独立 training seed 1 预注册

起点为 `04af151c7b78c0a3fe4adaf7a646ff151feecc86`。保留整个G3C-R4基线及其失败/成功历史证据。本轮仅改变 training seed 0→1，以及隔离实验所需的名称、标识和输出路径。冻结官方 PegInsert、32 environments、400 epochs、horizon128、minibatch512、4 mini-epochs、官方 PPO/LSTM/网络/物理/奖励/成功谓词及R4所有其他超参数。每epoch4096 transitions，总计划1,638,400。每25epoch保存periodic checkpoint，保持R4设置。

从fresh policy/optimizer开始；没有checkpoint参数，不加载seed0权重，不重试、不恢复、不拼接，不根据性能调整超参数。继续用户明确批准的 `user-authorized-no-gpu-guard` 模式；正常CUDA错误处理、durable supervisor、独占锁、真实child wait、console、TensorBoard flush/close均保持。没有独立GPU guard，也不依赖GPU遥测完整性。12小时wall timeout与R4一致。

只有400连续epochs完成、真实正常退出0、TensorBoard及checkpoint完整性通过后，顺序独立评估epoch100/200/300/400。评价环境32，evaluation seeds1001/1002/1003；早期checkpoint各96episodes，最终checkpoint288episodes；最大step分别160和480。原有deterministic evaluator、checkpoint normalization与RNN reset方式不变。ever/terminal success来自官方pre-reset predicate；first-success timing仅在成功episode条件下统计。

training seed1是一次独立训练；1001/1002/1003是评估初始条件seed，不是额外训练。与32-env training seed0逐checkpoint、逐evaluation seed直接比较，不合并64-env历史基线或跨训练seed的episode分母。两个训练seed只能提供描述性复现证据，不能精确估计训练seed总体变异。本轮停止于seed1；seed2仅给建议，不自动启动。禁止GearMesh、NutThread和外部GPU租用。

精确命令、路径、预运行CPU检查与冻结源SHA见 `evidence/g3c_r5/manifest_frozen.json`。结果在运行结束后独立添加，不修改该预注册历史。
