# 未决问题、失败及解决状态

日期：2026-10-23（Asia/Seoul）。GPU仿真、Factory初始化与有限PPO已通过；当前停止G0/G1。

## 独立非阻塞问题：GitHub推送

用户已选择继续全部G0/G1工作并本地提交，将认证单独处理。没有请求、读取或输出认证令牌。

- 初始HTTPS公开clone/read成功；最终一次HTTPS读取发生TLS握手中断，存在瞬时网络波动。
- HTTPS origin的`git push --dry-run origin main`仍报`could not read Username for https://github.com`，没有credential.helper，gh不在PATH。当前HTTPS写入路径缺少认证。
- 早期SSH握手失败，但最终复查`ssh -T git@github.com`已通过既有配置`ssh.github.com:443`认证为dreamerToTOE。GitHub不提供shell所以该命令exit1伴随成功消息是预期行为。
- 使用现有SSH的一次性`git push --dry-run git@github.com:dreamerToTOE/isaac-factory-repro.git main`成功（exit0）。没有实际推送、没有改origin或SSH配置。故当前可用SSH，无需为此任务请求令牌；原HTTPS认证缺失才是origin路径的实际问题。
- 认证诊断及瞬时失败均见[git_access.txt](../evidence/2026-10-23/git_access.txt)。未读取或输出密钥、令牌，未绕过限制。

### 使用现有SSH完成推送的确切步骤（本次未执行）

用户完成审阅后可执行；一次性SSH地址保留origin原HTTPS配置：

```bash
cd /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro
ssh -o BatchMode=yes -T git@github.com
git fetch git@github.com:dreamerToTOE/isaac-factory-repro.git main
git merge --ff-only FETCH_HEAD
git log -1 --format='%H %s'
git push git@github.com:dreamerToTOE/isaac-factory-repro.git main
git ls-remote git@github.com:dreamerToTOE/isaac-factory-repro.git refs/heads/main
```

若网络再次关闭SSH连接，修复网络后重试即可；现有SSH认证已在本次最终检查通过。若remote已有新提交导致ff-only失败，应先检查差异并正常合并，保留历史，不强推。最后远程SHA应与本地`git rev-parse HEAD`一致。

如果选择继续使用HTTPS而不使用已验证SSH，可以安装gh后按[官方浏览器登录说明](https://cli.github.com/manual/gh_auth_login)执行`gh auth login --hostname github.com --git-protocol https --web`，再执行[`gh auth setup-git`](https://cli.github.com/manual/gh_auth_setup-git)和`git push origin main`。本次未安装gh或变更凭证。

## 已解决：保留失败证据

| 问题 | 证据 | 处理与状态 |
| --- | --- | --- |
| 首次冷启动240秒超时 | gpu_gate | 保留日志，gpu_gate_retry通过GPU物理240步 |
| Factory缺少flatdict | peg_probe_1 / peg_probe_1_retry | 在独立venv补依赖；未改系统或Sim包 |
| flatdict构建缺pkg_resources | install_support.txt | 固定setuptools80.10.2，no-build-isolation安装成功 |
| uv同时安装锁等待告警 | install_learning.txt | 顺序重跑完成 |
| Kit archive遮蔽学习包 | peg_probe_16运行时版本、早期train_peg4 | python_entry预载与启动后断言；最终pinned2.7环境重测通过 |
| Torch2.7默认weights_only与RL-Games1.6.1不兼容 | resume_peg4_pinned | 自有已知SHA checkpoint命令级兼容env；resume_peg4_pinned_compat通过 |

均见 [证据目录](../evidence/2026-10-23/README.md)，没有删去失败后只报告成功。

## 仍存在的验证边界

- 有意最小安装未包含pin-pink/dex-retargeting/moviepy，pip check声明检查不完整；Factory headless无视频路径已实测。
- GPU显存为约1Hz采样峰值；16环境仅1epoch，未验证长时增长或大规模容量。桌面仍占用显存，初始空闲随时间变化。
- 16环境及恢复TensorBoard scalar tags为0，事件检查返回非零（无数据）；4环境3epoch有50项有限标量，所有最终checkpoint tensor有限。不能把无数据当loss检查通过。
- checkpoint加载/继续更新通过，actor Adam状态递增；独立central-value Adam未被RL-Games1.6.1原样checkpoint保存，RNG/物理/RNN rollout逐位续接也未验证。后续精确续训应单列工程协议，当前不改基线。
- 源码`success_times`缓存、Nut动作切片、无上限IK重试、timeout双标记等详见[SOURCE_AUDIT](SOURCE_AUDIT.md)。保留上游行为，后续修补必须独立标记。
- Gear/Nut只做单环境reset+32步，不代表其完整episode或PPO验证；本阶段训练仅PegInsert。
- 未进行200epoch正式训练、成功率评价、收敛或多seed实验。缺少长期结果是本阶段边界，不是伪造“已复现成功”的理由。

已完成的最小测试没有剩余GPU权限阻塞。后续训练配置见[EXPERIMENT_PROTOCOL](EXPERIMENT_PROTOCOL.md)，等待审阅后再执行。
