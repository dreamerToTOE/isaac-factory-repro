# 文件整理记录

日期：2026-11-25

本次整理只处理了证据目录中可以确认与同目录正式结果**字节完全相同**的重复文件。

## 已删除

删除 25 个 `metrics.json.partial.json` 文件。每个文件都与同目录下的 `metrics.json` 具有相同 Git blob SHA，因此删除不会丢失新的实验结果、错误信息或额外字段。

涉及：

- `evidence/g2/preflight/known_checkpoint/metrics.json.partial.json`
- `evidence/g3c_r4/completion/evaluation/epoch{100,200,300,400}/seed{1001,1002,1003}/metrics.json.partial.json`
- `evidence/g3c_r5/raw_collect_evaluated/completion/evaluation/epoch{100,200,300,400}/seed{1001,1002,1003}/metrics.json.partial.json`

## 保留

- `evidence/g3c_r5/raw_collect/completion/evaluation/epoch100/seed1001/metrics.json.partial.json` 仍保留。它不是重复文件，内容记录了一次 `incomplete_error`，属于有意义的失败证据。
- 不同实验阶段、不同运行目录中的同名结果文件仍保留。它们虽然有些内容相同，但路径表达了不同的实验或复核上下文。
- `preservation_before.json` / `preservation_after.json`、`sources_before.json` / `sources_after.json` 等成对文件仍保留。它们承担前后状态核验职责。
- 原始日志、图像、测试输出和完整证据目录未做批量删除。

## 判断依据

本次删除依据文件内容和 Git blob SHA 的逐项比较。文件名包含 `partial` 本身不足以作为删除依据，因此未删除仍包含独立失败信息的 partial 文件，也未删除仅因实验路径不同而重复的保全材料。
