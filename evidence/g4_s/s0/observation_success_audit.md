# G4-S0：19D 观测、冻结 actor 与官方成功判据审计

本审计只读取本地冻结源码、既有证据与 G2 checkpoint 的 CPU 元数据/RMS，没有运行策略前向、rollout、物理引擎或训练。所有目标 MuJoCo 接口均是 **PROPOSED**；源码语义与下述 checkpoint 复核为 **VERIFIED**。没有因为接口可映射而声称跨引擎等价或 Sim2Sim PASS。

## 冻结身份与数值边界

G2 epoch 200 checkpoint 为 `last_Factory_ep_200_rew__350.4961_.pth`，212,889,225 bytes，SHA-256 `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`，frame 1,638,400，training seed 0、64 environments。本次在 CPU 读取前后复核文件 SHA 相同；`torch.load(map_location='cpu')` 的 epoch/frame、空 `env_state`、所有 recurrent tensor shapes 与归档一致。三项 RMS（完整 mean[19]、var[19]、count）逐项 `torch.equal` 精确匹配 I0 的 float64 数值，未创建 actor 或 optimizer。完整数值与来源哈希见同目录 JSON。

Isaac Lab v2.1.1 / commit `90b79bb2d44feb8d833f260f2bf37da3487180ba`、Isaac Sim 4.5.0、RL-Games 1.6.1 是冻结历史依赖。直接检查现存官方 Factory 源码，env/config/task SHA 与 I0 匹配。配置占位 `observation_space=21` 不是实际 actor 输入：`obs_order` 四组共13项，末尾再拼 EMA cache 六项，共19项；critic43D 不传入 actor。

实际链路为19 → LSTM(input19, hidden1024, layers2, dropout0) → LayerNorm1024(eps1e-5) → Linear512/ELU → Linear128/ELU → Linear64/ELU → mu6(identity)。配置 `concat_input=true` 在 `before_mlp=true` 分支无效，MLP 首权重是[512,1024]，不能擅自追加19维。状态 h/c 各[2,B,1024]、FP32；初始为零，done 后只清零对应 actor 列，在下一回合首动作前完成；不从 checkpoint 恢复运行时状态。

冻结归一化恰为 `clamp((obs_fp32 - mean.float()) / sqrt(var.float() + 1e-5), -5, 5)`；RMS 保存 FP64，但计算 FP32。count 为1,638,401。先按物理量组装19维，再且仅再归一化一次；wrapper 原始观测 clip 是 infinity。保持 `model.eval()`，不更新 RMS；`inference_mode` 本身不能替代 eval。I1 的 mean-only actor 为保留原 Player RNG 消耗仍执行并丢弃一次 `Normal(mu,exp(logstd)).sample()`；同数值 evaluation seed 不能证明两个引擎的 reset 初态逐项配对。

## 19项逐维映射

表中 `p` 是 `panda_fingertip_centered` 的世界位置减 env origin；`q` 是该 body 世界 wxyz 四元数；`f0` 是 reset 时固定物体根位置；`R0` 是该固定物体 reset 朝向；`n` 是每回合一次 `N(0,diag(.001²,.001²,.001²))` 噪声；`b=f0+R0[0,0,.025]+n`，在 reset 缓存。世界轴的差分不是孔局部旋转坐标。目标 TCP 的 exact local transform 必须先由原资产核对，不能把普通 Panda hand/site 当作同一 TCP。

|索引|字段 / 原 PhysX 来源|原坐标与单位|MuJoCo 对应计算（PROPOSED）|必需变换与精度|无 privileged state 可观测性|精确复现结论|
|---|---|---|---|---|---|---|
|0|relative position x；env.py194-201,262|世界轴，m|`(site_xpos[tcp]-origin-b)[0]`|保留 root→tip .025m、reset缓存和减噪声；FP32运算|否；孔 reset 真值来自模拟器|数学映射可行；TCP/孔根帧待核|
|1|relative position y；同上|世界轴，m|同式分量1|不乘固定物体逆旋转；FP32|否|同上|
|2|relative position z；同上|世界轴，m|同式分量2|观测参考是孔顶端，不是成功判据的孔根；FP32|否|同上|
|3|fingertip quaternion w；env.py201,263|世界朝向 wxyz，无量纲|exact TCP body `xquat[...,0]` 或 site矩阵转换|不作 w≥0 规范化；保留原 sign；FP32|编码器FK+准确标定原则可得；原值是模拟器真值|旋转可映射；原始符号一致性 UNKNOWN|
|4|fingertip quaternion x；同上|世界朝向，无量纲|同式分量1|q与−q不能互换，RMS对其不对称|同上|同上|
|5|fingertip quaternion y；同上|世界朝向，无量纲|同式分量2|site_xmat转换会丢失四元数符号信息|同上|同上|
|6|fingertip quaternion z；同上|世界朝向，无量纲|同式分量3|不能误用xyzw或局部朝向|同上|同上|
|7|ee linear velocity x；env.py214-216,264|世界轴，m/s|`(p_now-p_prev_physics)[0]/dt`|dt=1/120，非1/15；每新physics timestamp更新；FP32差分|可由标定FK历史估计；非原真值保证|相同pose样本下语义可复现|
|8|ee linear velocity y；同上|世界轴，m/s|同式分量1|不直接替换成qvel或body速度；reset为0|同上|同上|
|9|ee linear velocity z；同上|世界轴，m/s|同式分量2|必须保留8子步内的前一pose；FP32差分误差乘120|同上|同上|
|10|ee angular velocity x；env.py219-225,265|世界轴，rad/s|`axis_angle(sign(dq.w)*dq)[0]/dt`，`dq=q_now*conj(q_prev)`|左空间差；同源axis-angle eps1e-6；FP32|标定朝向历史原则可估计|数值语义可复现；符号/轨迹一致性未验证|
|11|ee angular velocity y；同上|世界轴，rad/s|同式分量1|不能反乘成body-local差或用原生qvel冒充|同上|同上|
|12|ee angular velocity z；同上|世界轴，rad/s|同式分量2|原`sign(0)=0`在恰180°差时会零化delta；不静默修正|同上|同上|
|13|prev_actions position x；env.py258,266,302-304|内部EMA、归一化动作单位|cache0=`.2*clip(mu0)+.8*cache0`|每policy步一次，reset=`(p-b)[0]/.05`，不是/.02|命令内部状态可知；reset继承孔真值依赖|可按原程序复现，target未实现|
|14|prev_actions position y；同上|内部EMA、归一化动作单位|同式分量1|reset=`(p-b)[1]/.05`，不再clip|同上|同上|
|15|prev_actions position z；同上|内部EMA、归一化动作单位|同式分量2|reset=`(p-b)[2]/.05`，cache可能越[-1,1]|同上|同上|
|16|prev_actions rotation x；同上|内部EMA、归一化动作单位|同式分量3|reset=0；不把它当raw mu或已缩放角速度|是，内部状态|同上|
|17|prev_actions rotation y；同上|内部EMA、归一化动作单位|同式分量4|reset=0|是，内部状态|同上|
|18|prev_actions rotation z；同上|内部EMA、归一化动作单位|同式分量5|reset为下述非零yaw映射；不额外clip|是，内部状态|同上|

MuJoCo 的 [mjData 类型](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata)提供 body `xpos/xquat` 与 site `site_xpos/site_xmat`；`xipos` 是质心位置，不能替代 body 原点。其 [坐标与四元数文档](https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations)给出 scalar-first 单位四元数。这里提出读取匹配 body/site 的接口，不声称模型已经实现。常规 MuJoCo double 状态需显式构建FP32 actor输入；相同几何朝向并不保证四元数符号或引擎数值轨迹相同。

**原actor仍有 privileged dependency。** 0–2维使用固定物体 reset 真值，再加模拟噪声，不是相机测量。13–15维 pose cache reset 继承这个引用。不能用机器人编码器推断孔位置，不能任意估计后宣称观测等价。物理部署若改为外部感知，必须另行注册误差模型；本次没有实现该替换。

**四元数符号风险。** 原程序原样输入 `body_quat_w`，没有 hemisphere canonicalization；q与−q虽是同一SO(3)姿态，对冻结RMS/网络却是不同输入，例如primary RMS的qx均值约0.98157。因此仅检查转矩阵相同不够；S1须在匹配姿态/reset轨迹逐项检查原始符号。角速度的delta才有`sign(w)`处理，不能把此处理移植到原始观测四元数。

**reset与速度。** env.py845-847保存最后TCP pose，879-881将actor FD线/角速度显式置0；首观测不能从reset前pose差得到尖峰。后续 `_apply_action` 按新physics timestamp更新，最后 `_get_dones` 更新最后一个子步，再产出policy观测。dt永远是1/120；不能用两次policy观测之差/1/15冒充。MuJoCo 的 [qpos/qvel 约定](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework)也不能绕过这个原始有限差分定义。

**pose cache reset。** 位置为`(p-b)/[.05,.05,.05]`。旋转x/y为0；令`qrel=quat_from_euler(-pi,0,0)*q`，从原Euler算法取yaw；若yaw>pi/2则减2pi，再若yaw<−pi则加2pi；最后`cache[5]=(yaw+pi)/(3*pi/2)*2−1`。它与LSTM清零是两件独立操作，cache不是全零，也不保证落在[-1,1]。

## 官方成功谓词：必须保留一侧深度比较

直接读取 env.py92-107、132-135、231-237、431-464 和 task config96-109、131，PegInsert held-base local offset与fixed-success local offset**均为零**。所以当前success比较的是两个**当前root frame**，不是孔顶端观测缓存，也不是COM：

\[
p_H=p_{held}+R(q_{held})[0,0,0],\qquad
p_T=p_{fixed}+R(q_{fixed})[0,0,0].
\]
\[
s=\left[\sqrt{(p_{T,x}-p_{H,x})^2+(p_{T,y}-p_{H,y})^2}<0.0025\right]
\land\left[p_{H,z}-p_{T,z}<0.025\cdot0.04\right].
\]

即世界XY严格小于2.5mm，并且**有符号Z**严格小于1mm。边界相等不成功。没有`abs(z)`、没有下限、没有姿态/接触/力/保持时间条件；PegInsert 的`check_rot=False`。这是原定义，不改成更宽松或更严格的几何判据。它不能证明插接的物理稳固性，也不排除下穿透等几何假阳性。

建议目标计算（只是一比一程序映射，不是已实现的MuJoCo rollout）保持FP32及原操作顺序：

```python
# p_held_root, p_fixed_root: current exact root poses in same world/env axes
# PegInsert local offsets are both zero; never use the +.025m obs tip here.
xy_dist = torch.linalg.vector_norm(p_fixed_root[:, :2] - p_held_root[:, :2], dim=1)
z_disp = p_held_root[:, 2] - p_fixed_root[:, 2]
success = (xy_dist < 0.0025) & (z_disp < 0.025 * 0.04)
```

成功检查在每个policy步物理演化之后、自动reset之前。DirectRLEnv先计数+1，然后 `_get_dones`、`_get_rewards`，之后auto-reset，最后返回下一回合观测；仅在返回后拿pose重算会误读末态。已有`PreResetSuccessCapture`只捕获正式success阈值那次调用，排除engagement阈值。ever是每回合pre-reset masks之OR；terminal是done步的pre-reset mask；first-success取首次true的官方1-based counter，乘8/120秒。success不会提前结束。

10秒名义时长给出`max_episode_length=ceil(10/(8/120))=150`，源码timeout为计数`>=150−1`，故149次policy step、9.9333333333秒simulation time，两个terminated/truncated标志都是真。reset的IK/夹持会额外推进physics，不计入149步回合。保持120Hz、decimation8、15Hz模拟策略节拍；G4-I2完整wall command rate约3.470Hz的历史限制仍成立，不能把它变成15Hz墙钟性能。

## S1前置条件与未决项

- VERIFIED：本次source hash、primary checkpoint hash、CPU元数据/RMS精确匹配、actor维度和官方谓词。
- PROPOSED：MuJoCo的19D同语义计算、pose cache与pre-reset计数实现路线。
- UNKNOWN：原资产到MuJoCo后TCP/root局部变换的一致性、对应模型的原始四元数符号、引擎接触/动力学与reset初态流是否可忠实重建。
- S1进入前须固定合法精确资产及TCP/孔根frames，完成CPU matched-pose 19维、符号、physics-FD、reset、严格阈值边界测试。接口可映射不构成策略跨引擎表现证据；本审计支持有条件可行，不支持直接GO或Sim2Sim PASS。

复核依据、逐维八项属性、完整RMS、source绝对路径/行号/哈希及实际CPU metadata probe结果保存在 [observation_success_audit.json](observation_success_audit.json)。
