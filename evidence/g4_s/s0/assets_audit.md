# G4-S0：资产与第二引擎只读审计

本文件仅支持可行性结论 **CONDITIONAL**。`assets_audit.json` 为最终机器证据；两个 `assets_audit_*_failed.json` 保留审计器开发中的失败，不可当作最终验证通过记录。没有安装依赖、下载大型资产、创建 MuJoCo `MjData`、调用 forward/step、运行策略或启动 Isaac Sim/Kit。

## 已执行的范围

- **VERIFIED**：既有环境 `/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python` 可 import MuJoCo，Python 包和实际库均为 **3.5.0**。仅导入和 CPU 编译现有模型。
- **VERIFIED**：本地 Menagerie 干净于 `8161bba264d7fa7c99ca301e91e7fb44737676ad`。Panda XML 编译为 `nq=9,nv=9,nu=8`；FR3 XML 编译为 `nq=7,nv=7,nu=7`。逐关节索引、限位、轴、质量、惯量主轴、驱动参数、运行时与峰值 RSS 均在 JSON。
- **VERIFIED**：已有 USD 0.22.11 库可独立读取缓存文件；未启动 SimulationApp。首先用 `Sdf.Layer` 确认 robot/peg/hole 无外部 reference、payload 或 sublayer，然后只用 `Usd.Stage.LoadNone` 读取。桌面只读层依赖清单，未合成或获取桌面依赖。
- 本地缓存使用不透明文件名；原 basename 搜索没有找到不等于资源缺失。对缓存文件记录 SHA-256，并比对官方 HTTPS **HEAD** 返回的大小、ETag 与本地 MD5。三者吻合是很强的来源关联证据，但这里没有下载远端正文并计算其 SHA-256，不能冒称远端 SHA-256 已独立核验。

## 冻结原资产

来源固定于 Isaac Lab `90b79bb2d44feb8d833f260f2bf37da3487180ba`，Factory 配置的资产根为 Isaac **4.5** 云目录。原机器人是 `franka_mimic.usd` 中的 **Panda**，不是 FR3；其 default prim 为 `panda`。配置显式要求 `panda_joint1` 至 `panda_joint7` 和 Panda 夹指。`factory_env_cfg.py`、`factory_tasks_cfg.py` 和 `factory_env.py` 的散列也在证据中。

| 资产 | 冻结来源 | 本地 SHA-256 | 审计边界 |
| --- | --- | --- | --- |
| 机器人、手、夹指、末端参考体 | [Factory/franka_mimic.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/franka_mimic.usd) | `86d0017a093669b222a8c640bf581cc3e3715035aa7c596600f478da011f6c8e` | 关节、质量惯量、所有 mesh 的顶点/面计数、碰撞属性与包围盒已读 |
| Peg8mm | [Factory/factory_peg_8mm.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/factory_peg_8mm.usd) | `b5040c8c639f3dc0731adfb6022c0d63383b7550354e413dc24e7417b600869d` | 原 mesh 与 SDF 碰撞属性已读，未导出 mesh |
| Hole8mm | [Factory/factory_hole_8mm.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/factory_hole_8mm.usd) | `4bad48cde5a2e523454f9ed584023fae53d619a5ffb3d6ab08f29a6240014c28` | 同上；孔拓扑不能由单个凸包保留 |
| 桌面顶层 USD | [SeattleLabTable/table_instanceable.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd) | `479ed8fd374d8eafdedf3ca3b9d0d5981ffc6d5c5016929347f748be93d5df17` | 仅 Sdf 层元数据与依赖清单；完整桌面依赖、许可及几何仍 UNKNOWN |

任务源码规定 peg 直径 `0.007986 m`、高度 `0.050 m`、覆盖质量 `0.019 kg`；hole 名义孔径 `0.0081 m`、高度 `0.025 m`、覆盖质量 `0.05 kg`。名义直径间隙 `0.000114 m`，径向 `0.000057 m`；这不是对网格内孔、倒角或 SDF 有效边界的测量。peg 原生 `physics:density=7000` 等 USD 参数不能替代任务运行时的显式质量覆盖；惯量需按该质量覆盖和实际 PhysX 导入路径重新核验。原 peg mesh 为1154顶点/1152面，hole为1897顶点/3806面；原 robot 有22个 mesh、13个关节（含固定关节）和13个显式质量体。

原 robot 为 metre/Z-up，USD RevoluteJoint 的角度限位为 **degree**，运行时和 MuJoCo 所需为 radian。原 peg/hole 层有 `metersPerUnit=0.01`、`upAxis=Y` 元数据，但原 mesh 坐标量级与源码米制尺寸相符。审计器把这些层引用到 metre/Z-up 的 CPU USD host，比较逐 mesh 世界包围盒；reference 没有自动执行该 metadata 所暗示的缩放或轴旋转。S1 应保留原 Factory 的实际引用/根变换语义，不能再按 `0.01` 缩放或盲目旋转。

## 候选模型不是自动等价模型

| 项目 | Menagerie Panda | Menagerie FR3 | 对冻结 Factory 的意义 |
| --- | --- | --- | --- |
| 来源 | [固定提交 Panda README](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/README.md)，来自 Franka ROS 的简化 MJCF | [固定提交 FR3 README](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_fr3/README.md)，来自 franka_description | Panda 可作为审查模板；FR3 不是本任务允许的替代品 |
| 关节命名/顺序 | `joint1..7,finger_joint1,finger_joint2` | `fr3_joint1..7` | 未来必须显式名映射，不能按数组位置猜测；FR3 没有原 Panda 夹手 |
| 7 臂关节范围，rad | ±2.8973；±1.7628；±2.8973；[-3.0718,-0.0698]；±2.8973；[-0.0175,3.7525]；±2.8973 | ±2.7437；±1.7837；±2.9007；[-3.0421,-0.1518]；±2.8065；[0.5445,4.5169]；±3.0159 | FR3 限位不同；原 USD 的精确 degree 数值见 JSON，转换要保留浮点差异 |
| 臂驱动 | 有位置伺服增益/bias；damping=1、armature=0.1 | 有位置驱动、摩擦、armature | 均不能原样接收冻结 torque controller；不得靠位置驱动使实验“跑通” |
| 质量/惯量 | 臂参数与原资产有共同来源；但手指每个 0.015kg | 臂质量、惯量不同 | 原 Factory 手指每个约 0.0297kg，额外 force_sensor 约0.01kg 与末端参考体；Panda 也未获整体动力学等价认证 |
| 碰撞与视觉 | link5 凸分解、人工设计 fingertip geoms、视觉 mesh 转换与碰撞过滤 | 独立 FR3 mesh | 原 Factory collision mesh、指垫、接触材料、self-collision 关闭及质量覆盖应独立迁移；视觉相似不证明接触等价 |

模型 XML、README、LICENSE 以及 XML 直接引用的每个现有 mesh 文件均逐文件记录大小和 SHA-256；无需将 36MB/57MB 的模型复制入本仓库。编译所得位置驱动控制范围是候选的默认状态，**不是**本任务对目标控制配置的提议。

## 许可边界

- **VERIFIED**：MuJoCo 3.5.0 的[引擎许可](https://github.com/google-deepmind/mujoco/blob/3.5.0/LICENSE)为 Apache-2.0。Menagerie 的 [Panda 许可](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/LICENSE)和 FR3 自带许可也是 Apache-2.0；再分发需履行许可、通知、保留归属和改动声明等相应条件。候选许可证不会自动覆盖其它 Factory 资产。
- **VERIFIED**：冻结 Isaac Lab framework 是 BSD-3-Clause；[Franka 资产许可文件](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/docs/licenses/assets/franka-license.txt)为 Apache-2.0。原安装中的 `isaac-sim-assets-LICENSE.txt` 明确要求查阅各机器人和 props 的 hosted-folder 许可。
- **UNKNOWN**：该 Franka 许可是否完整覆盖经过 Forge/Factory 修改的 `franka_mimic.usd`、新增手指/传感器/材料，以及 peg/hole/table 的具体许可范围。Factory 根目录常见四个许可名称的 HEAD 均为404；这只证明这些 URL 未发现许可，不能证明资产没有许可或禁止使用。
- **BLOCKED（S1 分发门禁）**：在 per-asset 授权范围被确认前，不进行原 Factory USD/mesh/其衍生文件的公共再分发。没有将资产复制或上传到 Git。当前 NVIDIA 许可网页的存在不能追溯证明原 4.5 云资产适用的具体条款。

## 重建路径与进入条件

**PROPOSED**：取得可核验的 per-asset 使用/转换/再分发依据后，优先由这些冻结 USD 的实际变换、关节、质量与惯量生成专用 MJCF；保留独立视觉和碰撞表示。原 peg/hole 使用 PhysX `sdf`，其 cooked buffer 不具跨引擎直接兼容证明。MuJoCo 的普通 mesh collision 使用凸包，孔会被填实；可以审查[凸分解或 SDF 插件](https://mujoco.readthedocs.io/en/3.5.0/computation/index.html#collision-detection)，但其几何误差、间隙和接触行为必须另作有界验证，不能默认为原形状等价。把 peg 改成理想圆柱、把 hole 扩大、改成其它装配件均不满足冻结任务。

G4-S1 入口至少包括：明确资产许可；冻结完整模型/依赖指纹；CPU 变换、FK/Jacobian/惯量与关节映射对照；保留实际指垫与夹持结构；确认孔拓扑和57µm名义径向间隙未被简化；实现直接 torque actuation 并审计模型附加阻尼/摩擦/重力。S0 的模型读取/编译通过不支持任何 Sim2Sim 成功率或策略可迁移结论。

复现（会拒绝覆盖既有 JSON）：

```bash
python3 scripts/g4s0_asset_audit.py --output /tmp/g4s0_asset_audit_new.json --network-head
```

机器路径仅为本次已有依赖的来源记录；换机器需先提供等价的冻结依赖，不能因此自动下载或安装。
