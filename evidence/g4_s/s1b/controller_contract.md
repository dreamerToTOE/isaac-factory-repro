# G4-S1B 控制器与观测契约

本附录仅验证适配版 Sim2Sim 的数学接口。未导入 Isaac 环境、加载 checkpoint、执行 PPO actor 或运行策略 episode。`scripts/g4s1b_contract.py` 使用 NumPy float64 接收 MuJoCo double 状态；raw observation 与 RMS 输出为 float32。因此这里只声明公式和接口语义的移植，不能声明与 PhysX/Torch float32 逐位一致。

## 源依据与独立验证

原始 Factory `factory_env.py`、`factory_control.py` 的路径与 SHA-256 从冻结的 G4-I0 清单读取并核对。测试通过 AST 选取其实际 `_apply_action`、`_pre_physics_step`、`_get_observations`、`get_pose_error`、完整 `compute_dof_torque` 和增益函数，以及原 reset action-cache 和 FD 语句。函数体不改写，只去掉 JIT decorator；在普通 CPU Torch 张量和最小状态代理上调用，不构建 Isaac 模拟器。原安装的 quaternion utility 同样只在内存中执行，未复制发布其代码。移植文件保留完整 Isaac Lab BSD-3-Clause 声明。

固定种子 20261026 的 24 组合成夹具覆盖非交换朝向、正负原始四元数符号、非对角正定质量矩阵、非零 raw joint velocity、PD/动态一致 nullspace。另有不依赖移植输出的已知数值控制：1 Nm task torque、20 Nm null torque、−19 Nm 阻尼结果、±100 Nm 软件饱和。独立源函数并非仅调用待测实现作为 oracle。

| 量 | 24 组合成夹具最大绝对差或残差 |
|---|---:|
| Cartesian position / quaternion target | 1.888405e−7（混合位置和无量纲分量最大值） |
| 软件限制后 arm torque | 7.020107e−6 Nm |
| PD wrench | 6.673315e−6（N / Nm 对应分量） |
| position / axis-angle error | 2.225294e−7 |
| `||J M⁻¹ tau_null||` | 2.730251e−15 |

这些数值只对应登记夹具，不能作为全状态空间误差证书。原 FP32 世界位置的 FD 在乘 120 后放大舍入差；独立 FD oracle 测试将 linear / angular 绝对容差分别登记为 5e−6 m/s 与 5e−5 rad/s。一次完整量化探针（包括 import）耗时 1.231144 s，峰值 RSS 494292 KiB，真实进程 exit code 0。

## 六维动作与目标

每个 policy update 仅调用一次 `policy_update(raw6)`：先 clip 到 ±1，再以 `.2 * clipped + .8 * old` 更新 EMA；EMA 不再 clip。随后八个 `1/120 s` 物理子步各自使用最新 TCP pose 重建目标：

```text
p* = b + clip(p_current + .02 * EMA[:3] - b, -.05, .05)
r = .097 * EMA[3:]
dq = quat_from_rotvec(r)    # ||r|| <= 1e-6 时 identity
q_pre = dq_world * q_current
yaw = source_get_euler_xyz(q_pre)[2]     # modulo 2pi
q* = source_extrinsic_xyz(3.14159, 0, yaw)
```

这是世界轴位置增量及左乘轴角旋转，不是关节目标、速度或对旧 Cartesian target 累积。source 的 modulo 2pi 及 `roll=3.14159` 常数保留；不因姿态矩阵等价而改写目标 quaternion 原始符号。

## Task PD 与动态一致 nullspace

任务 pose 是 TCP，但原 `J` 特指两 finger **body origin** 几何 Jacobian 的均值，排列为世界平移行在上、旋转行在下，仅七 arm DOF 列。它不等于 TCP site Jacobian，目标实现分别提供这两种 Jacobian。`M` 是全系统质量矩阵按七 arm DOF 地址取出的主子矩阵。nullspace 使用七 raw simulator joint velocity，而 task damping 使用每物理步的 TCP FD。

```text
Kp = [100,100,100,30,30,30]; Kd = 2*sqrt(Kp)
e = [p*-p, axis_angle(q* * inverse(q))]
F = Kp*e - Kd*[linear_FD, angular_FD]
tau_task = J.T @ F
Lambda = inverse(J @ inverse(M) @ J.T)
J_hash = Lambda @ J @ inverse(M)
u_null = M @ (10*wrap(q_default-q) - 6.3246*qdot_raw)
tau_null = (I - J.T @ J_hash) @ u_null
tau_requested = tau_task + tau_null
tau_software = clip(tau_requested, -100, +100)
```

旋转误差先用 `dot(target,current)>=0` 选择 shortest target sign；inverse 使用 conjugate 除以 quaternion norm²。`q_default=[-1.3003,-.4015,1.1791,-2.1493,.4001,1.9425,.4754]`。task wrench 没有额外 Lambda 质量整形。没有新增 gravity、Coriolis、`qfrc_bias` 前馈、伪逆阻尼或调 gain。

实现显式拒绝 NaN/Inf、非单位 quaternion、非对称或非正定 M、rank deficient J，以及 inverse condition number ≥ `1/float64 eps` 的数值不可分辨情况；不会静默发零 torque 或换控制律。该 gate 只拒绝状态，没有 regularization。返回 requested、software limited、task/null torque、wrench 与 condition number，MuJoCo actuator force / generalized contribution 由环境另外测量，不混称实际施加 torque。finger 的零位置目标与物理 PD 属于环境中的明确适配，arm 数学函数不替代接触夹持。

## 19 维观测与 reset

| 索引 | 字段 |
|---|---|
| 0–2 | TCP 世界位置减 noisy reset hole tip，m |
| 3–6 | TCP 世界 quaternion，wxyz，保留输入 raw sign |
| 7–9 | 世界位置按 `1/120 s` FD，m/s |
| 10–12 | `axis_angle(sign(dq.w)*dq)/dt`，`dq=q_now*conj(q_prev)`，rad/s |
| 13–18 | 六维内部 EMA action cache |

固定 reference 为 reset 时 hole root 加旋转后的 `[0,0,.025]` 与每回合噪声；不是孔局部坐标。噪声由调用者显式提供；source 是三轴 sigma .001 m Gaussian、每回合采样一次。`observation_noise=None` 是指定零噪声诊断，不能宣称已重现 source RNG。reset cache xyz=`(TCP-noisy_tip)/.05`、不 clip；旋转 x/y 为零，yaw 保留原 `unrot(-pi)`、wrap 与 270° 映射。reset 第一帧 FD 为零。恰好 `dq.w=0` 时 source `sign(0)=0` 会使 FD 角速度为零，按源分支保留。

`update_physics` 以物理 timestamp 缓存 FD，相同 timestamp 且 pose 未变时复用结果，拒绝跳步或倒退。`normalize_observation` 接受已冻结 mean/variance，转 FP32 后执行 `(obs-mean)/sqrt(var+1e-5)` 再 clip ±5，不更新统计。测试使用归档 primary RMS 标量，未加载 checkpoint。未来获准接 actor 时仍须分别 reset 原两层 1024 LSTM 的 h/c 与 EMA；本模块不执行 LSTM。

这里使用 simulator ground truth 提供 TCP 与 hole reset pose，是 privileged-state 假设。原 PhysX raw quaternion sign 跨引擎一致性、原 reset IK / RNG 轨迹、finger implicit PD、历史实际 torque、接触及惯量等效仍没有被数学夹具证明。

## CPU 检查

最终独立子集真实结果为 `16 passed in 1.40s`、exit code 0。完整回归收据由主任务另行保存。重现子集：

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/envs/factory-g01/bin/python -m pytest -q tests/test_g4s1b_contract.py -p no:cacheprovider
```

机器可读公式、字段、源 SHA、容差、实际统计与限制见同目录 `controller_contract.json`。该文件不代表策略闭环成功或 PhysX 动力学等效。
