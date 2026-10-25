# Supplemental rendered playback of G2 epoch 200

The completed playback is **1 successful timeout-completed episode / 1 requested episode**, with seed 1001 and one environment. It is a separate illustration and is excluded from the primary G2 evaluation denominator. It does not claim to reproduce one of the primary 32-environment evaluation episodes.

The checkpoint SHA-256 is `f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f`, matching the selected epoch-200 checkpoint. Inference uses the checkpoint's policy mean and normalization buffers in evaluation mode. The official task, reward, success predicate, physics, observation and action settings remain unchanged.

| Capture | Policy step | XY offset | Z displacement | Official predicate |
|---|---:|---:|---:|---|
| Start | 1 | 16.7675 mm | +45.4671 mm | false |
| First success | 37 | 0.491052 mm | +0.0348762 mm | true |
| Terminal before reset | 149 | 0.397044 mm | +0.0443906 mm | true |

The official limits are **XY < 2.5 mm** and **Z < 1.0 mm**. An independent geometric computation agrees with the official result at all 149 policy steps. First success occurs at 2.4667 simulated episode seconds; terminal time is 9.9333 episode seconds. Simulation clock values in the JSON include initialization/reset settling and therefore differ from episode time.

The capture hook runs during the official reward-phase success call. The terminal image is saved before the automatic reset. Rendering advances no physics: robot joint state, held/fixed asset root states, geometry, episode counters, simulation time and physics-step counter are checked before and after each screenshot. The completed step then verifies official `extras['successes'] == 1` and the returned post-reset episode counter is zero.

- [Completed run JSON and complete geometry trace](completed/playback.json)
- [Start screenshot](completed/episode_00_step_001_start.png)
- [First-success screenshot](completed/episode_00_step_037_first_success.png)
- [Terminal screenshot before reset](completed/episode_00_step_149_terminal_success.png)
- [Validation checks](validation.json)
- [Exact presentation configuration differences](render_config_diff.json)
- [Completed GPU guard result](completed/guard_result.json)

The final run used the official `performance` rendering preset and 640×480 images. Its grainy shading is a rendering quality limitation; the peg/socket and motion are visible. These images were visually inspected. The completed 60.35-second process had a sampled minimum GPU free memory of **1043.19 MiB** with the unchanged **1024 MiB** guard, return code 0 and no monitor error. This margin is narrow and is not evidence that every rendering run will fit; sampling can miss shorter peaks.

## Preserved attempts

1. Initial launch omitted the existing D001 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` compatibility environment flag. RL-Games checkpoint restoration failed before any episode. Only that playback child was terminated. Kit reported exit 0, but no completed playback was claimed. See [failure classification](attempt_1_failure.json), [partial state](attempt_1_partial.json), and [guard result](attempt_1_guard_result.json).
2. With the compatibility flag restored, balanced rendering at 800×600 crossed the guard floor: **931.81 MiB** free. The guard terminated its own process group. A successful pre-reset terminal image was already saved, but the run did not finish post-reset episode accounting; it is incomplete and supplies no completed-episode denominator. See [failure classification](attempt_2_failure.json), [partial state](attempt_2_partial.json), [guard result](attempt_2_guard_result.json), [console-derived frame records](attempt_2_logged_frames.json), and [higher-quality terminal image](attempt_2/episode_00_step_149_terminal_success.png). The 125 geometry rows persisted before interruption match the completed run exactly.
3. The final attempt selected the official performance preset and smaller presentation resolution, and requested one complete episode with a 200-step cap. This completed successfully. The preset differences, camera and resolution are fully recorded in `render_config_diff.json`. No installation or upstream code replacement was performed.

All raw logs remain under `/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g2_playback/`. Large videos were not created or committed. The frozen quantitative evaluator was not edited by the playback work.

## Executed completed-run command

The command was launched through a user systemd service to survive the parent terminal lifetime. `FACTORY_VENV` and `ISAACSIM_PATH` select the existing installation; `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` applies the established D001 compatibility handling for the trusted local checkpoint.

```bash
systemd-run --user --unit=g2-playback-seed1001-performance \
  --property=WorkingDirectory=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/outputs/isaac-factory-repro \
  --setenv=FACTORY_VENV=/home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/envs/factory-g01 \
  --setenv=ISAACSIM_PATH=/home/ubuntu2004/isaacsim-4.5.0 \
  --setenv=TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /usr/bin/python3 scripts/run_bounded.py \
  --output /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g2_playback/seed1001_performance_guard \
  --timeout 900 --min-free-mib 1024 -- \
  bash scripts/run_sim.sh scripts/playback_g2_checkpoint.py \
  --checkpoint /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/runs/logs/rl_games/Factory/g2_peginsert_64env_seed0_baseline_rerun/nn/last_Factory_ep_200_rew__350.4961_.pth \
  --output-dir /home/ubuntu2004/Documents/Codex/2026-10-23/the-github-repository-has-now-been/work/logs/g2_playback/seed1001_performance \
  --seed 1001 --episodes 1 --max-steps 200 \
  --width 640 --height 480 --rendering_mode performance \
  --device cuda:0 --headless --enable_cameras
```

Use a new service name and new output paths for a rerun: the harness refuses to overwrite evidence.
