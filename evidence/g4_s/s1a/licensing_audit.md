# G4-S1A：逐资产许可与转换门禁

本审计区分**条款文本已读**、**条款适用范围已证**、**本地转换**和**公共分发**。
结论为：**原 Factory 资产的本地跨引擎转换和原始/衍生几何分发许可仍未确立，门禁
BLOCKED；固定 Menagerie Panda 可在 Apache-2.0 条件下用于独立、明确标注的诊断。**
这是未决证据的保守实施门禁，不是断言原资产“没有许可”或法律上绝对禁止转换。

先阅读 `docs/REVIEW_G4_S0.md`、S0 的 `assets_audit.json/.md`，再进行本次有限调查。
没有复制、修改或导出原 USD/mesh，没有下载大型资产，没有调用引擎或物理步。
完整来源、文件 hash、HTTP 状态、日期、耗时与结论位于同目录 `licensing_audit.json`。

## 逐项矩阵

| 必需资产 | 冻结来源/身份 | 已验证的许可依据 | 本地转换 | 原始或衍生资产公共分发 |
|---|---|---|---|---|
| Factory Panda、夹手、指尖、force_sensor、TCP 参考体 | [4.5 Factory/franka_mimic.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/franka_mimic.usd)，SHA `86d0017a…1f6c8e` | stock Franka 有 Apache-2.0；没有可核验依据覆盖此 Forge/Factory 修改版的全部内容 | NOT_ESTABLISHED / BLOCKED | NOT_ESTABLISHED / BLOCKED |
| Factory Peg8mm | [4.5 Factory/factory_peg_8mm.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/factory_peg_8mm.usd)，SHA `b5040c8c…0869d` | 内容包通用条款存在；精确资产许可/归属未发现 | NOT_ESTABLISHED / BLOCKED | NOT_ESTABLISHED / BLOCKED |
| Factory Hole8mm | [4.5 Factory/factory_hole_8mm.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Factory/factory_hole_8mm.usd)，SHA `4bad48cd…14c28` | 同上；不能从代码 BSD 或另一模型 Apache 推断覆盖 | NOT_ESTABLISHED / BLOCKED | NOT_ESTABLISHED / BLOCKED |
| SeattleLabTable 及其 `table.usd` 依赖 | [4.5 table_instanceable.usd](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd)，顶层 SHA `479ed8fd…5df17` | 独立 prop 条款及依赖闭合尚未确立 | NOT_ESTABLISHED / BLOCKED | NOT_ESTABLISHED / BLOCKED |
| Menagerie Panda，仅独立诊断起点 | [固定 `8161bba…` Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda) | 精确模型 README 声明并附 Apache-2.0，许可证本地/远端 SHA 相同 | VERIFIED，遵守许可条件 | VERIFIED，遵守许可条件；本任务未复制其 mesh |

所有完整 SHA、来源路径和许可范围字段以 JSON 为准。原资产继续保持为本机只读参照；
读取既有尺寸、拓扑和质量元数据不能自动授权把这些资产转换并发布为完整 MJCF/OBJ。
没有以 FR3 替代 Panda。

## 较 S0 新增的关键证据

1. **4.5 安装包自带的 Omniverse 协议不是当前 June-2025 页面。**
   `/home/ubuntu2004/isaacsim-4.5.0/LICENSE.txt` 的 SHA 为
   `e8a684e1278cc1de7a4b3fce180228f3195f02c3b208a5722122d95cce29ca20`。
   Section 1.4 定义 NVIDIA 2D/3D Content，Section 2.1 对受其管辖的 Content 明确提供
   修改/衍生权利，并将分发限定于融入产品或服务的使用，排除独立分发 Content；
   Section 3 另有分发条件。该文本的存在和这些条文 **VERIFIED**，但其对每个必需资产
   及第三方部分的完整适用范围 **UNKNOWN**。
2. 同一安装包的 `isaac-sim-assets-LICENSE.txt`（SHA
   `5d10853c8554f71b06b6fe734d059b7abf108b016a9f9ff3e6b225e5598b5f3c`）
   指向内容包通用 Omniverse 条款，同时明确机器人/prop 的许可应查询各 hosted 文件夹。
   主协议的 Components Under Other Licenses 条款也保留 component-specific 条款的优先性。
   因此通用 Content 衍生权利不能单独作为四个精确 Factory/prop 资产的无条件 clearance。
3. 找到真实的 **200 / Apache-2.0** 官方 4.5
   [stock Franka hosted-folder 许可证](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Franka/franka-LICENSE.txt)，
   SHA `94b9224462ffdd31fb184897c2e6c69dfc9f03f667d8e2a458ed57b302df4684`。
   不能把 `Robots/Franka` 的条款范围扩大至 `IsaacLab/Factory`。S0 已记录的
   `franka_mimic` layer documentation 指向生成时的本地 `Robots/Forge/franka_mimic.usd`，
   没有发现可连接全部修改内容与该 stock 许可证的归属声明。peg/hole 的对应 layer
   文档和 customLayerData 也没有许可字段；这不是资产全部属性的全面法律检索。
4. 15 个有明确边界的 hosted-folder 候选地址中，1 个 stock Franka 许可证成功，
   Factory/父级/prop 的其余 14 个返回 404。两个 4.5 官方文档地址也返回 404。
   这些负结果只证明在这些地址没有找到条款，不能证明其它位置不存在条款。
   没有遍历服务器或下载原资产。
5. [当前 NVIDIA Additional Software and Materials 页面](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/)
   标明 June 9, 2025，具有另一组许可限制；本次 HTTP 正文 probe 仅保留有界读取状态，
   超过 64 KiB 的 HTML 没有被当作完整正文哈希。当前页面仅作版本差异的背景资料，
   不能追溯替换原 4.5 安装协议或 January-2025 资产的确切适用条款。

## 独立 Panda 的权利边界

[固定 Panda 的 LICENSE](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/LICENSE)
及冻结 Isaac Lab 的
[一般 Franka LICENSE](https://github.com/isaac-sim/IsaacLab/blob/90b79bb2d44feb8d833f260f2bf37da3487180ba/docs/licenses/assets/franka-license.txt)
都实际取回并与本地字节核对。两份 SHA 同为
`a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9`；
相同许可证文本不证明两个资产的全部内容相同或具有同一授权来源。

独立模型的本地使用、修改及分发应保留相关归属与许可证、标注修改；如果提供了 NOTICE，
应保留其适用内容。在该固定 checkout 的 Panda 目录与根目录没有 `NOTICE` 文件，
此观察不免除其它已有通知。MuJoCo 3.5.0 引擎的 Apache-2.0 也不会许可 NVIDIA USD
或重新许可它们的派生几何。诊断成功不能改写为原 Factory 忠实重建通过。

## 解除门禁所需证据

- 针对精确 `franka_mimic` 修改版、peg、hole、table/必要依赖，取得可核验的资产许可、
  权利人归属及其版本适用关系。
- 明确这些条款允许本地跨引擎转换，且处理了 component-specific / 第三方范围。
- 若未来需要把原始或衍生几何提交公开仓库，另行确认该分发方式与通知要求；
  本地转换权与公共分发权不能混为一谈。

本任务没有联系权利人或发送外部消息。`licensing_audit_network_initial.json` 保留首次
成功网络调查，最终 `licensing_audit.json` 增补已存在的 layer 元数据和 stock Franka
正面发现；两者均不是物理验证，也不包含原始或衍生 mesh。

复现（拒绝覆盖既有证据）：

```bash
python3 scripts/g4s1a_license_audit.py --output /tmp/g4s1a_license_NEW.json --network
```

去掉 `--network` 只重新采集既有许可文件与保守门禁；未执行网络取证时不会伪称 HTTP
探测已运行。collector 只允许固定的少量许可证/文档 URL，每次最多读 64 KiB，无引擎调用。
