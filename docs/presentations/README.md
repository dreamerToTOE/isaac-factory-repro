# PegInsert 项目介绍演示稿

本目录有旧版项目介绍（深色演示）和新版 **学术机制图表风格**。推荐使用新版。

## 新版（三页）

- [可编辑 PPTX](Isaac_Factory_PegInsert_academic_style_3slides.pptx)
- [逐页中文讲解稿](Isaac_Factory_PegInsert_academic_style_speaker_notes.md)
- [再生成脚本](rebuild_peg_insert_academic.py)：需要 `python-pptx`，在仓库根目录执行。脚本根据参考稿的视觉系统重新构图，不复用参考稿的非本项目实验照片。

内容分别为控制架构、PPO训练与独立几何评估、Sim2Sim 失稳排障和2×2惯量对照。

### 证据边界

- PhysX 中 `280/288=97.22%` 是原 Factory 几何成功率，来自一个训练 Seed（0）及三个评估 Seed（1001/1002/1003）。
- MuJoCo 2×2实验支持短窗口内物理惯量调整显著改善关节速度，但未获得完整任务的零样本迁移成功证据。
- 幻灯片里的 Peg/Hole 小图是**机制示意**，不是实物或仿真截图。

本目录早期文件 `Isaac_Factory_PegInsert_3slides.pptx` 与 `Isaac_Factory_PegInsert_讲解稿.md` 保留作历史版本，不覆盖或混同。
