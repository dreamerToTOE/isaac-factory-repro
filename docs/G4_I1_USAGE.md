# G4-I1 独立策略推理使用说明

入口 `scripts/g4i1_runtime.py` 直接构造冻结 actor/RMS/LSTM，使用原始 Factory 环境、动作映射和控制器；不创建 PPO Runner、optimizer 或训练更新。`--validate` 额外构造与原 evaluator 相同的官方 `PpoPlayerContinuous` 作为只读 oracle，运行相同 observation 的中间值对照。原 evaluator 和 Isaac Lab 源码不改动。

## CPU 前检

从仓库根目录，以项目固定环境执行。所有 output 必须是新路径，保护历史证据：

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../work/envs/factory-g01/bin/python -m pytest -q tests --basetemp=/dev/shm/g4i1_new_cpu_tests
CUDA_VISIBLE_DEVICES='' ../../work/envs/factory-g01/bin/python -m scripts.g4i1_cpu_check --output /tmp/g4i1_new_cpu_parity.json
CUDA_VISIBLE_DEVICES='' ../../work/envs/factory-g01/bin/python -m scripts.g4i1_edge_cpu --output /tmp/g4i1_new_edge_parity.json
```

CPU 实际 checkpoint 检查使用 G2 primary 和 R4 secondary，各自冻结权重与 RMS；不会混用模型。CPU synthetic input 结果不冒充真实仿真 episode。

## GPU 资源和运行

先保存 `nvidia-smi` 的 GPU 型号、driver、显存及 compute process 清单。不得关闭桌面或其它用户进程；没有足够 headroom 时停止。以下命令只展示执行方式，须换用新的 run ID、路径和 systemd unit name。独立 GPU guard 的1024MiB/3样本、0.1s采样、0.5s gap保护保持原实现，不继承旧训练 no-GPU-guard 例外。

```bash
G4_REPO=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro
G4_BASE=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been
G4_VENV="$G4_BASE/work/envs/factory-g01"
python3 "$G4_REPO/scripts/g3a_launch_smoke.py" \
  --output /dev/shm/g4i1_new_preflight_supervision \
  --unit-prefix g4i1-new-preflight \
  --cwd "$G4_BASE/work/runs" --timeout 600 --wait \
  --setenv "FACTORY_VENV=$G4_VENV" \
  --setenv ISAACSIM_PATH=/home/ubuntu2004/isaacsim-4.5.0 \
  --setenv OMP_NUM_THREADS=4 \
  --setenv FACTORY_G2_AUDIT_OUTPUT= \
  --setenv FACTORY_G3A_FINALIZE_TENSORBOARD= \
  -- bash "$G4_REPO/scripts/run_sim.sh" "$G4_REPO/scripts/g4i1_runtime.py" \
  --headless --device cuda:0 --mode preflight --validate \
  --output /tmp/g4i1_new_preflight_runtime
```

2步 preflight 的 CPU/GPU parity、fixture、真实child wait和guard全部通过后，才用另外的新目录和unit运行 `--mode episode`：150次policy action，覆盖149步完整timeout episode及下一episode首动作。科学参数仍为单env、evaluation/reset seed1001、torch seed1001、FP32和G2 epoch200 primary。

不使用 `--validate` 时入口只运行standalone actor；本次验收必须带 `--validate` 才产生完整官方oracle比较证据。即使wrapper/launcher为0，也须检查 `supervisor/status.json.raw_wait_returncode`、`termination_reason`、`guard/status.json` 和 runtime/result.json；原始异常保留为 runtime_error.json 与console，不能用wrapper的成功掩盖失败。

## 产物与结果

- `trace.jsonl`：原始obs/normalized/mu/h/c/clip/EMA、逐physics substep target及controller inputs/effort、pre-reset终态、post-reset obs/cache、h/c清零、下一episode首动作。
- `paired_trace.npz`：`reference__<stage>` 与 `actual__<stage>`，不包含pickle object数组。可用 `python -m scripts.g4i1_parity --help` 查看独立复算接口。
- `parity.json`：冻结CPU/GPU绝对/相对容差下的逐stage比较数、最大误差、违规数，以及四元数符号不变角误差。
- `edge_fixtures.json`：normal、clipping、near-zero rotation和position saturation。调用原Factory方法的tensor proxy，不推进physics，不向simulator提交命令。
- `result.json`：真实episode数/分母、ever/terminal、first-success、149步timeout、辅助physics推进、RNG和RMS状态。

官方Player选mu前仍会产生被丢弃的Normal sample；standalone复制同一次采样。oracle在fork_rng中运行并恢复live RNG，随后逐步核验standalone的RNG输出状态与oracle一致。固定trace相等和真实随机场景重现的证据范围分别报告。

一个episode只验收运行链，不复现原288episode总体成功率。墙钟total runtime不代表15Hz硬实时或P95/P99延迟；这些留待另行授权的I2。本任务完成即停止，不启动I2、Sim2Sim、seed2、GearMesh、NutThread或训练。

最终运行曾修复两项工程问题：参考控制器重算必须保留原PhysX tensor stride；异常时必须在Kit关闭前调用 `app.app.post_quit(1)`，避免fast shutdown的默认0掩盖失败。`--mode exit-probe` 为有意的异常退出验证，不创建任务环境或提交机器人命令。所有旧失败记录保留。

用 `python -m scripts.g4i1_acceptance --run-root RAW_RUN_ROOT --phase episode_verified --output NEW_ACCEPTANCE.json` 可独立检查原始trace、固定case覆盖、真实wait、guard和runtime_error。只有child0、无错误且最终结果齐全才可通过；CPU回归和checkpoint/source核验另见本阶段报告。
