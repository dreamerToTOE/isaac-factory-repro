# G4-S1A：Factory Panda 资产与几何重建门禁

日期：2026-10-26（Asia/Seoul）。仓库 `dreamerToTOE/isaac-factory-repro`，分支 `main`。
已验证起点：**`554120e2ea001170c8895c6441f831622db7fd91`**。

**工程审计 COMPLETE；忠实 Factory 重建 BLOCKED；当前 S1B 决策 NO-GO。**

本任务完成了前置条件调查、显式机器人映射、原始几何只读测量和独立许可模型的 CPU
诊断。尚未取得四类精确原资产的完整转换许可依据，也未构造忠实 Factory MJCF。
NO-GO 是当前证据门禁，可由明确的新证据解除，不是认定重建在数学或法律上永远不可能。
独立 Panda/合成孔道的成功检查不授予原 Factory 忠实性结论。

最关键的新发现是：**配置的57 µm名义径向间隙与原 raw mesh 不符**。完整直孔 slab
的原边界半径下界约4.498925657 mm，完整 peg 半径上界约3.993004597 mm；在明确的
共轴、直孔段范围内，raw 径向间隙至少505.921060430 µm。原孔还存在孔底和顶部倒角。
实际训练使用的 **cooked PhysX SDF 有效表面仍 UNKNOWN**，不能把 raw 测量当作其碰撞间隙。

## 1. 冻结范围、来源与执行边界

实施前已阅读 [S0报告](REVIEW_G4_S0.md)、S0 的资产、controller、observation/success
JSON/附录、I0 checkpoint/source 清单及相关冻结 Factory 源码。保留所有历史报告、
实验、失败证据和 checkpoint；没有回写其“尚未执行”等历史叙述。

| 对象 | 本次固定身份及操作 |
|---|---|
| Isaac Lab 原参考 | v2.1.1，`90b79bb2d44feb8d833f260f2bf37da3487180ba`；14份源码与6份resolved配置重新哈希 |
| 第二引擎 | **MuJoCo3.5.0**，Python包与实际库版本均核对；复用已有环境，无安装 |
| 起始机器人 | **Menagerie Panda**，commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`；67个直接mesh依赖与S0哈希一致 |
| 原资产 | 本机缓存的 `franka_mimic`、peg、hole、table顶层，完整SHA见许可/几何JSON；读取前后相同 |
| checkpoint | 三份真实历史文件只做字节哈希；不反序列化、不执行actor、不改权重 |
| 原任务 | 不启动Isaac Sim，不构建/运行原任务物理探针；success predicate及所有科学配置未改 |
| 独立诊断 | Panda只编译和`mj_kinematics`；原创annulus在完整性检查后只做静态`mj_forward`/距离检测；积分步数0 |

没有使用FR3，没有PPO inference、policy rollout、训练、微调、外部GPU租用或物理设备
命令。没有启动S1B/S1C/S2、GearMesh或NutThread。学习与理论考核仍为 **PENDING**，
安排在Sim2Sim项目之后。

证据用语：VERIFIED仅指具体字节/源码/数值检查已经完成；UNKNOWN表示尚无证据；
BLOCKED表示本阶段不能完成该项。它们不合并成跨引擎接触或动力学等价。

## 2. 逐资产来源与许可

完整矩阵、法律文本文件SHA、20项有界HTTP记录及版本适用分析见
[licensing_audit.md](../evidence/g4_s/s1a/licensing_audit.md) 和
[licensing_audit.json](../evidence/g4_s/s1a/licensing_audit.json)。没有将原始/衍生资产
复制到仓库，报告仅保存来源、标量和审计元数据。

| 必需资产 | 已验证依据 | 本地跨引擎转换 | 公开原始/衍生资产 |
|---|---|---|---|
| Factory `franka_mimic`、hand/fingers、sensor/TCP | 官方4.5 stock Franka文件有Apache-2.0；Factory/Forge修改版全部内容与该许可的适用关系尚未确立 | NOT_ESTABLISHED / BLOCKED | NOT_ESTABLISHED / BLOCKED |
| `factory_peg_8mm.usd` | 4.5内容包通用协议存在；精确资产/第三方适用范围未闭合 | NOT_ESTABLISHED / BLOCKED | NOT_ESTABLISHED / BLOCKED |
| `factory_hole_8mm.usd` | 同上 | NOT_ESTABLISHED / BLOCKED | NOT_ESTABLISHED / BLOCKED |
| SeattleLabTable与`table.usd`依赖 | 顶层出处已记录；prop许可与完整依赖未闭合 | NOT_ESTABLISHED / BLOCKED | NOT_ESTABLISHED / BLOCKED |
| 固定Menagerie Panda | 精确模型README/LICENSE为Apache-2.0，远端与本地核对 | VERIFIED，遵守条件，仅独立诊断 | VERIFIED，遵守归属/许可/修改通知等条件；本次未分发mesh |

安装的Isaac Sim4.5协议明确对受其管辖的Content提供修改/衍生许可，并对分发方式设置
条件；内容包许可又要求查询各机器人/prop文件夹，component-specific条款优先。
因此不能用通用grant、代码BSD或stock Franka Apache推断四类精确资产均已clear。
14个候选许可URL的404只说明这些位置未找到文本，不证明不存在许可。

本次新增的正面依据是官方
[4.5 stock Franka许可](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Franka/franka-LICENSE.txt)。
原`franka_mimic` layer文档指向构建时Forge目录，尚缺覆盖修改内容的连接证据。
当前June-2025网页没有被追溯当成4.5安装协议。此处为保守实施门禁，不是“原资产无许可”
的法律断言；本地转换权和向公共仓库分发权分别核查。

## 3. 显式关节、body与frame映射

[robot_mapping.md](../evidence/g4_s/s1a/robot_mapping.md) 列出全部9轴精确限位与13个
原body；[robot_mapping.json](../evidence/g4_s/s1a/robot_mapping.json) 保存每轴父子体、
局部joint frame、轴、单位、独立`qposadr`/`dofadr`、mimic关系、每体CoM和完整惯量。

| Factory坐标 | Panda坐标 | qpos/dof地址 | 单位/范围概要 |
|---|---|---|---|
| `panda_joint1` | `joint1` | 0/0 | rad，约±2.8973 |
| `panda_joint2` | `joint2` | 1/1 | rad，约±1.7628 |
| `panda_joint3` | `joint3` | 2/2 | rad，约±2.8973 |
| `panda_joint4` | `joint4` | 3/3 | rad，约[-3.0718,-.0698] |
| `panda_joint5` | `joint5` | 4/4 | rad，约±2.8973 |
| `panda_joint6` | `joint6` | 5/5 | rad，约[-.0175,3.7525] |
| `panda_joint7` | `joint7` | 6/6 | rad，约±2.8973 |
| `panda_finger_joint1` | `finger_joint1` | 7/7 | m，[0,约.04] |
| `panda_finger_joint2` | `finger_joint2` | 8/8 | m，[0,约.04] |

此表的约数不能回写原范围。原USD degree转rad后与候选的最大臂限位差为
2.76465696e-7 rad，finger差为8.94069672e-10 m，不宣称字节相同。

body名映射为`panda_link0..7→link0..7`、`panda_hand→hand`、
`panda_leftfinger/rightfinger→left_finger/right_finger`；原`force_sensor`与
`panda_fingertip_centered`在候选缺失。TCP不能用视觉mesh中心代替：link7到sensor
局部平移z=.10700000077486038，wxyz约(.9238795042,0,0,-.3826834559)；sensor到hand
为identity；hand到TCP局部z=.11207100003957748、identity rotation。

原右指两个joint frame均为Rx(π)，零位时相消，finger body与hand同向；候选右指
body为Rz(π)。两者正开口均沿hand的−Y，但body轴不同，CoM、完整惯量、collision
origin均须显式转换。不能仅因关节名或开口方向相同就认定frame相同。

四组臂姿态×三组单侧开口（0/.02/.04m），共12项静态检查：虚拟TCP位置差约20–22nm、
方向差约1.002e-7rad。零臂姿态明确标为超出joint4等限位的**代数检查**。这些值比较
authored joint FK与候选静态FK，不是实测PhysX姿态、raw quaternion sign、J、M或接触
等价。没有向外发布带原TCP修改的衍生模型。

## 4. 质量、CoM、惯量、驱动与重力

各原body的主惯量/主轴、`I_body=R diag(I_principal) Rᵀ`、CoM、source→target body
frame，以及统一表达后的完整张量差均保存在robot JSON；附录有逐体表，不由collision
geom重新推断质量。

| 维度 | 具体差异/剩余条件 |
|---|---|
| 臂/hand | 臂质量与authored惯量来源接近，但不构成运行时等价；hand原约.730000019kg，候选.73kg |
| 两finger | 原每指.02969999984kg，候选.015kg；右指CoM转换后Y约−.0114、Z约.0231m，候选为0，差约.02576m |
| 附加body | 原sensor约.01kg、TCP参考体约1e-6kg，候选缺失；sensor没有authored inertia，保持UNKNOWN |
| 原运行时惯量 | 源码对每个robot body增加`.01 I` kg·m²；记录为authored+源码规则的**条件张量**，未测PhysX导入后的实际tensor；不是joint armature |
| peg/hole | 配置质量.019/.05kg；完整运行时CoM/惯量及mass override后的cooking结果未建立，不以理想圆柱计算冒充原值 |
| arm actuator | 原stiffness/damping/armature/friction均0，配置effort为前4轴87Nm、后3轴12Nm；软件±100Nm单列，实际backend applied torque仍UNKNOWN |
| finger actuator | 原implicit PD 7500/173、40N、joint friction .1、闭合target0；原mimic gearing−1/offset0需连同两joint frame解释，不能仅复制符号 |
| 候选actuator | 7个position servo及1个split-tendon开口驱动，ctrlrange0..255；不是原7个torque motor+双指PD/mimic；全部gain/bias/force参数已列JSON |
| 重力 | 原scene−9.81，robot/held peg禁用gravity，reset随机化期间scene gravity0；候选scene−9.81、12body gravcomp全0，不匹配 |
| 候选默认 | joint damping1、armature.1、dt=.002s；没有将这些默认值作为原任务科学配置 |

只重建关节frame不能解决运行时惯量、力路由或gravcomp实现。原self-collision关闭；
候选已有过滤/局部排除也不是已匹配的原碰撞规则。本阶段不以修改参数使其“跑通”。

## 5. peg/hole原几何、孔底及误差预算

[geometry_audit.md](../evidence/g4_s/s1a/geometry_audit.md) 给出全部方法、候选比较、公式
与复现命令；[geometry_audit.json](../evidence/g4_s/s1a/geometry_audit.json) 保留原始SHA、
精确有理数、向外sqrt夹界、真实错误及子进程退出码。没有原点/面索引导出。

原peg/hole分别1152/3806个原三角面，无额外fan triangulation；原mesh变换实际为
identity。精确坐标去重后，两者无边界边、非流形边、零面积面或朝向不一致。
层metadata的0.01/Y-up没有导致擅自再次缩放/换轴；构成行为以S0证据为准。

| 量 | 结果及证据范围 |
|---|---|
| 配置名义尺寸 | peg直径.007986m，hole直径.0081m，径向差57µm；这些字段没有缩放USD |
| raw peg直段截面 | 半径3.992046726..3.993004597mm；截面采用整段极值，非有限角度采样 |
| raw hole直段截面 | 内壁半径4.498925657..4.500004881mm，而非4.05mm |
| raw顶部倒角 | z=.0245m处内半径4.998806496..5.000004957mm |
| 完整raw直孔段界 | 对全部588个与`0<z<.024000000208616257m`相交的原三角面精确slab裁剪，XY投影检查内部及完整边极值；半径下界.004498925657431m |
| 完整raw peg界 | 所有原vertex的XY径向范数最大值加凸性，覆盖全部三角面；半径上界.003993004597001m |
| 限定raw径向间隙 | 上述共轴直孔段≥505.921060430µm，约名义值8.8758倍；不是任意姿态/整次插入的3D碰撞证书 |
| 孔底 | 轴线与原边界只在z≈−.003和0m相交，正z孔腔上方接外部；原孔有底座/孔底，不能替换为无底贯通annulus |
| 原转换误差 | raw→converted Hausdorff：UNKNOWN_NO_CONVERSION；raw→cooked PhysX SDF：UNKNOWN；cooked PhysX→MuJoCo：UNKNOWN |

slab径向结果依赖正轴孔腔为空的闭合参考解释，不涉及z=0孔底跨越；详细外部anchor
与轴线证据在几何附录。原cooked buffers尚未解码，也未运行PhysX，因此不能认定训练
碰撞孔径已测得4.5mm。不能缩放原hole/peg以“修正”差异，也不能称未转换所以误差为0。

[MuJoCo3.5普通mesh碰撞](https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html#asset-mesh)
使用凸表示；原凹孔一个普通mesh collider会填孔。保孔凸分解可作为候选，但必须覆盖
原直孔、倒角、孔底、外结构并建立全域误差、编译后hull和接缝接触证据。
[SDF扩展接口](https://mujoco.readthedocs.io/en/3.5.0/programming/extension.html#sdf)
存在不等于PhysX cooked SDF可直接移植；自定义distance/gradient及误差仍需实现和验证。

### 独立原创数学诊断（不是Factory转换）

以解析公式生成`rin=.00405,rout=.02,h=.025m`的独立annulus，仅用名义标量诊断尺度。
N个凸梯形棱柱使用内接内边界/外切外边界，其材料并集保守包含理想annulus；不包含
原孔底/倒角/侧部结构，不能作为原任务替代。

理想数学内侵量`rin*(1−cos(π/N))`、外扩量`rout*(sec(π/N)−1)`，以`π<22/7`等不等式
得到精确有理数上界。N=16内侵上界78.133µm，该界不能保证57µm名义剩余间隙仍为正；N=64为
4.883µm（约8.57%），剩余径向间隙下界52.117µm，外边界误差上界24.144µm。
完整分辨率表见附录。此界属于理想原创实体，不是原Factory转换误差。

实际MuJoCo3.5两次编译64块模型的关键几何数组SHA相同；全部64块编译后顶点双向
测量最大偏差约.999366nm。该数值为浮点测量，未建立向外舍入的编译hull证书，不能把
理想公式直接升级为最终碰撞面严格界。

独立球静态负控：保孔64块轴心0接触；移至内壁2接触；同一原创圆环顶点构成的普通
单mesh collider在轴心产生1接触、signed distance约−.0126m，实际拒绝了填孔方案。
所有静态检查的physics time为0，没有积分、grasp或策略运行。

## 6. gripper、真实夹持与支撑

原hand collider为600点/200面convexHull，两finger各4245点/8202面convexHull。
候选共23个collision geom，其中每指5个人工box pad；原robot有11个collision mesh。
JSON列出各候选geom的body-local origin、类型、尺寸/mesh、材料与过滤。视觉相似或
相同Panda名称不能认证指尖支持面、接触法线、摩擦或实际凸包相同。

原grasp尚未重建。peg必须保留独立6DOF、真实finger/peg接触、原reset随机分布、约
.032392m的held-relative偏置及.25模拟秒的闭合阶段。没有weld，也没有通过替换pad、
改质量、放大孔或缩小peg来消除问题；原材料/摩擦、mimic/PD饱和与接触顺应仍未匹配。

支撑侧只读核对了原hole的fixed-world joint和Factory场景源码：ground z=−1.05m；
table translation=(.55,0,0)，源码wxyz=(.70711,0,0,.70711)。table顶层引用`table.usd`，
其完整几何、collision/material和许可未闭合，不能用任意box/plane冒充已重建桌面。
原任务物理探针在这些前置条件满足前保持未执行。

## 7. CPU验证、回归与保全

最终验证收据：
[cpu_validation_20261026T_g4s1a_r2/summary.json](../evidence/g4_s/s1a/cpu_validation_20261026T_g4s1a_r2/summary.json)。

| 检查 | 最终结果 |
|---|---|
| 新增robot CPU测试 | 20 passed；含实际固定Panda编译、独立FK/frame/惯量oracle、非有限输入、4个真实非法模型拒绝及已有证据拒绝覆盖 |
| 新增geometry CPU测试 | 28 passed；含独立tetrahedron、整边/三角内部极值、有理数sqrt、拓扑/非法索引/NaN、误差预算及真实引擎正负控证据 |
| 选定回归 | 535 collected：534 passed、1 skipped、0 failed；skip为冻结venv没有matplotlib |
| 系统Python补测 | 同一plot测试1 passed、0 skipped；因此允许范围内535个不同case均取得通过，补测不冒充新增case |
| no-policy tripwire | 独立dummy负控在body执行前退出97；正式两组测试guard均active、forbidden_calls=[] |
| 历史保全 | 起点1637个tracked文件内容/执行位、20份外部源码/配置、3个checkpoint、4份原USD、另checkout的5个G3V文件，前后完全相同 |

用户本阶段明确禁止PPO inference，所以没有重跑四个含actor forward的历史测试文件，
共73个S0已收集case。完整排除清单/理由/旧case数量见`excluded_policy_tests.json`。
**没有声称本次跑过全套560项历史回归**；原测试文件全部保留。其余测试包括纯张量
reset/normalization、success边界与mock controller，不执行actor。tripwire覆盖本次pytest
进程及新Python线程的已知入口，reviewed collector子进程另以源码/调用记录审查；不是
通用沙箱或无法绕过的安全证明。三份真实checkpoint仅哈希，没有load。

首轮新runner以脚本启动时将`scripts/`留在import path，形成
`success_metrics`/`scripts.success_metrics`双模块，导致一个历史mock测试失败：
532 passed、1 failed、1 skipped。原始日志/JUnit、runner文本与保全快照完整保存在
[首次验证目录](../evidence/g4_s/s1a/cpu_validation_20261026T_g4s1a/summary.json)。
只修正新增runner，使导入根与`python -m pytest`一致，同时增加tripwire收据校验及负控；
未修改历史实现/测试。修正后的r1已通过533+1补测；随后仅加固robot collector拒绝已有
输出目录，并新增一项保全测试，r2最终回归全数通过。几何helper的custom float→Fraction审阅修正
前证据也保存在`geometry_audit_pre_review.json`；默认数值未改变。

Panda在同一进程两次编译的完整MJB SHA均为
`8935ec801d77487589f76880494533751a625c3ce38dda19bdddb307ee2716f4`，编译bytes42,680,185，
只在内存哈希、未分发MJB；nq/nv/nu=9/9/8。新进程复编译测试也核对该指纹。
原创annulus两次关键几何SHA均为
`92af208595544e148f9d3734a7fae4adb2f4c1663084232de0933c078b86657c`。
这只证明固定环境/依赖下的重复编译，不承诺跨平台、版本或浮点实现逐位相同。

耗时/内存：robot collector约.546s、进程peak RSS541,460KiB；原几何只读子进程和
独立诊断子进程的逐项墙时、peak RSS与stderr均在JSON。RSS为进程生命周期峰值，
不是此次增量内存。回归的最终耗时以summary/JUnit为准，所有退出码实际保存。
非法mesh的Qhull stderr保留，没有用joggle/缩放/吞异常使非法资产通过。

复现命令均拒绝覆盖已有证据；使用新路径：

```bash
python3 scripts/g4s1a_license_audit.py --output /tmp/g4s1a_license_NEW.json --network
/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/g4s1a_robot_audit.py --output /tmp/g4s1a_robot_NEW
python3 scripts/g4s1a_geometry_audit.py --output /tmp/g4s1a_geometry_NEW.json
python3 scripts/audit_g4s1a.py --output evidence/g4_s/s1a/cpu_validation_NEW
```

## 8. S1B门禁与交付

综合机器可读结论见
[reconstruction_gate.json](../evidence/g4_s/s1a/reconstruction_gate.json)。
后续忠实Factory实现至少需先解决：

1. 精确Factory修改机器人、peg、hole、table依赖的版本许可/归属；分别确认本地转换和计划分发方式。
2. raw/config/cooked SDF孔径差异的来源与有效碰撞面，原mesh/底座/倒角/孔底的完整转换及可审查误差预算。
3. 原指尖/手掌碰撞支持面、相对frame、材料、真实6DOF grasp、mimic与PD/effort路由。
4. 全部body/peg/hole运行时mass/CoM/inertia和原重力禁用行为，sensor/TCP附加体的处理依据。
5. 支撑几何/材料/依赖闭合，以及固定版本编译后的全部collision覆盖与非法输入拒绝。

静态关节映射已取得实质进展；运行时PhysX tensor、J/M、raw quaternion sign、实际
applied torque、接触/求解器等价仍不因本阶段审计而自动变成VERIFIED。没有发布可用的
完整Factory目标环境，也没有policy性能或Sim2Sim成功率可报告。

本次只增加S1A脚本、测试、报告和证据。提交前进行独立几何/机器人/许可交叉审阅，
核对staged diff均为新文件、历史不变、无原/衍生mesh或权重。以fast-forward方式推送
`main`并核验远端SHA；具体commit与推送回执在交付消息报告，避免文档自引用commit hash。

**STOP：G4-S1A结束；S1B及之后阶段未启动。**
