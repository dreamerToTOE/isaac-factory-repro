# G4-S1A Panda 静态映射审计

只用归档 S0 Factory 元数据与独立许可的固定 Menagerie Panda；不读取/转换原 USD，
不积分、不运行策略。`mj_kinematics` 输出仅为静态 FK，不能建立 PhysX 动力学/接触或四元数符号一致性。

| Factory joint | MuJoCo joint | qpos / dof | Factory limit (rad/m) | Panda limit | 最大差 |
|---|---|---|---|---|---|
| panda_joint1 | joint1 | 0 / 0 | [-2.8972998215323886, 2.8972998215323886] | [-2.8973, 2.8973] | 1.78467611e-07 |
| panda_joint2 | joint2 | 1 / 1 | [-1.7627998550613584, 1.7627998550613584] | [-1.7628, 1.7628] | 1.44938642e-07 |
| panda_joint3 | joint3 | 2 / 2 | [-2.8972998215323886, 2.8972998215323886] | [-2.8973, 2.8973] | 1.78467611e-07 |
| panda_joint4 | joint4 | 3 / 3 | [-3.0717997235343044, -0.0697999915935665] | [-3.0718, -0.0698] | 2.76465696e-07 |
| panda_joint5 | joint5 | 4 / 4 | [-2.8972998215323886, 2.8972998215323886] | [-2.8973, 2.8973] | 1.78467611e-07 |
| panda_joint6 | joint6 | 5 / 5 | [-0.01749999978815516, 3.752499969733032] | [-0.0175, 3.7525] | 3.0266968e-08 |
| panda_joint7 | joint7 | 6 / 6 | [-2.8972998215323886, 2.8972998215323886] | [-2.8973, 2.8973] | 1.78467611e-07 |
| panda_finger_joint1 | finger_joint1 | 7 / 7 | [0.0, 0.03999999910593033] | [0.0, 0.04] | 8.94069672e-10 |
| panda_finger_joint2 | finger_joint2 | 8 / 8 | [0.0, 0.03999999910593033] | [0.0, 0.04] | 8.94069672e-10 |

完整父子局部帧、轴、mimic、9D 地址在 JSON。右夹指尤其不能按名字直接等同：
Factory `localRot0=localRot1=Rx(pi)` 相消，零位 finger body 与 hand 同向；
Panda `right_finger` 是 `Rz(pi)`。两者 +q 均沿 hand -Y 开口，但 body 朝向相差 pi。

| 静态 fixture / 单侧开口 m | virtual TCP 位置差 m | 朝向差 rad |
|---|---|---|
| zero_algebraic_only_outside_some_limits / 0.0 | 2.14033796e-08 | 1.00153702e-07 |
| zero_algebraic_only_outside_some_limits / 0.02 | 2.14033796e-08 | 1.00153702e-07 |
| zero_algebraic_only_outside_some_limits / 0.04 | 2.14033796e-08 | 1.00153702e-07 |
| menagerie_home / 0.0 | 2.22040848e-08 | 1.00153703e-07 |
| menagerie_home / 0.02 | 2.22040848e-08 | 1.00153703e-07 |
| menagerie_home / 0.04 | 2.22040848e-08 | 1.00153703e-07 |
| frozen_factory_default / 0.0 | 2.04462811e-08 | 1.00153702e-07 |
| frozen_factory_default / 0.02 | 2.04462811e-08 | 1.00153702e-07 |
| frozen_factory_default / 0.04 | 2.04462811e-08 | 1.00153702e-07 |
| interior_asymmetric / 0.0 | 2.18785055e-08 | 1.00153703e-07 |
| interior_asymmetric / 0.02 | 2.18785055e-08 | 1.00153703e-07 |
| interior_asymmetric / 0.04 | 2.18785055e-08 | 1.00153703e-07 |

virtual TCP 只是在数值上使用 hand-local z=0.11207100003957748 的元数据计算；
未发布含此站点的原资产衍生模型。零位 fixture 在 joint4 等限位外，只作代数帧检查。

| Factory body | Panda body | 原 mass kg | Panda mass kg | 对齐 frame 后 authored tensor 最大差 kg·m² |
|---|---|---|---|---|
| panda_link0 | link0 | 0.6297690272 | 0.629769 | 1.4257300215100334e-10 |
| panda_link1 | link1 | 4.970684052 | 4.970684 | 4.579913677332792e-07 |
| panda_link2 | link2 | 0.6469259858 | 0.646926 | 1.7761940737659465e-09 |
| panda_link3 | link3 | 3.228604078 | 3.228604 | 3.95290821603167e-09 |
| panda_link4 | link4 | 3.587894917 | 3.587895 | 3.3764419385406974e-09 |
| panda_link5 | link5 | 1.22594595 | 1.225946 | 1.2851911826428086e-09 |
| panda_link6 | link6 | 1.666555047 | 1.666555 | 3.925193321416276e-10 |
| panda_link7 | link7 | 0.7355219722 | 0.735522 | 4.292715211073883e-10 |
| force_sensor | MISSING | 0.009999999776 | N/A | N/A |
| panda_hand | hand | 0.7300000191 | 0.73 | 1.5023054128011784e-10 |
| panda_leftfinger | left_finger | 0.02969999984 | 0.015 | 5.177000170107923e-06 |
| panda_rightfinger | right_finger | 0.02969999984 | 0.015 | 5.177000170107923e-06 |
| panda_fingertip_centered | MISSING | 9.999999975e-07 | N/A | N/A |

表中张量差先把原 source 惯量旋转到 candidate body 轴后再比较，右夹指的 pi 帧差已计入。
JSON `source_principal_inertia` / `target_principal_inertia` 是 CoM 处主轴对角元；
`*_CoM_tensor_in_body_axes` 才是 `R diag(I) Rᵀ` 恢复的 CoM 处完整3×3矩阵。
`*_tensor_in_target_body_axes` 则进一步转换到 candidate body 轴；这些不是关于 body 原点的张量。
另记源码要求的每体 `+0.01 I` 条件张量；这不是关节 armature，也不是实测 PhysX tensor。
`force_sensor` 仅 authored mass/CoM、无 authored inertia；目标也缺该 0.01 kg 体。
TCP参考体质量约1e-6 kg/inertia1e-9 kg·m² 亦不能静默删除后宣称等价。

Panda 原驱动为 7 个位置伺服及 1 个 split tendon 开口驱动（0..255），并非 7 个 torque motor。
原两夹指 implicit PD=7500/173、40 N、friction0.1、mimic 的实际实现尚未重建。
JSON 列出所有 collision geoms 局部 origins、盒尺寸、mesh、过滤和材料；这些诊断接触不等于 Factory。
Panda gravity=-9.81、gravcomp=0、armature=.1、damping=1，与 Factory robot gravity-disabled/0/.0 不符。

两次相同进程编译的 MJB SHA 相同：`8935ec801d77487589f76880494533751a625c3ce38dda19bdddb307ee2716f4`。
4 个无效输入（缺 mesh、负惯量、无效 joint 引用、惯量三角不等式）均记录实际拒绝错误。
本次耗时 0.546s；进程 peak RSS 541460 KiB；积分步数0。

**结论：静态名字/帧/惯量映射已细化；完整 Factory 重建及接触/动力学等价仍未通过。**
