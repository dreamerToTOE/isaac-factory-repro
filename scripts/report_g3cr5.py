#!/usr/bin/env python3
"""Create a fresh Chinese R5 report/result from actual CPU analysis evidence.

Read-only inputs; no simulation, checkpoint loading, process control, or GPU API.
Both output files must be new.  Numeric seed-0/seed-1 results are always read
from the supplied analysis reports, never embedded in this report generator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def number(value, digits=3):
    if type(value) not in (int, float) or not math.isfinite(value):
        return '未记录'
    return f'{value:,.{digits}f}' if digits else f'{value:,}'


def rate(aggregate, key):
    aggregate = aggregate or {}
    count, denominator, value = (aggregate.get(key + '_count'), aggregate.get('episodes_completed'),
                                 aggregate.get(key + '_rate'))
    if value is None or denominator is None or not denominator:
        return '未评估'
    return f'{count}/{denominator}（{100 * value:.2f}%）'


def timing(aggregate):
    stats = (aggregate or {}).get('first_success_time_seconds_conditional') or {}
    return f"n={stats.get('n', 0)}；均值 {number(stats.get('mean'))} s；中位数 {number(stats.get('median'))} s"


def final_evaluation(report):
    return next((row for row in report.get('evaluation', {}).get('checkpoints', [])
                 if row.get('checkpoint_epoch') == 400), {})


def seed2_recommendation(report, seed0):
    if report.get('engineering_status') != 'COMPLETE':
        return ('本次未达到完整训练与评估验收条件。建议先记录并解决已观察到的工程原因，'
                '保留本次全部证据；待另行授权后再用完全相同科学配置从头运行 training seed 2。'
                '不因本次表现更改超参数，不自动重试、续训或启动 seed 2。')
    current = final_evaluation(report).get('aggregate', {})
    previous = final_evaluation(seed0).get('aggregate', {})
    now, old = current.get('ever_success_rate'), previous.get('ever_success_rate')
    difference = (f'本次最终 ever-success 相对 seed 0 的观察差值为 {(now-old)*100:+.2f} 个百分点。'
                  if now is not None and old is not None else '')
    return (difference + '建议在独立授权后继续一个完全相同配置的全新 training seed 2，'
            '以增加训练随机性证据；保持 32 env、400 epochs、horizon 128、同一评估协议和全部超参数。'
            '两个训练种子只能给出描述性重现性证据，尚不足以稳定估计训练种子总体分布。'
            '本任务到 seed 1 为止，不自动启动 seed 2。')


def make_result(report, seed0, manifest, identities, provenance=None):
    train = report.get('training', {})
    progress = train.get('verified_progress', {})
    return {'schema_version': 'g3cr5_result_v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'engineering_status': report.get('engineering_status'),
            'scientific_status': report.get('scientific_status'),
            'training_seed': 1, 'evaluation_seeds': [1001, 1002, 1003],
            'training_configuration': manifest.get('training'),
            'completed_epochs': progress.get('completed_epochs_verified'),
            'completed_transitions': progress.get('completed_transitions_verified'),
            'training_duration_seconds': train.get('training_duration_seconds'),
            'official_ppo_total_seconds': train.get('official_ppo_total_seconds'),
            'official_ppo_aggregate_fps': train.get('official_ppo_aggregate_fps'),
            'stable_ppo_fps_epochs_11_onward': train.get('stable_aggregate_total_fps'),
            'child_exit': train.get('exit'), 'loss_stability': train.get('loss_stability'),
            'amp_statistics': train.get('amp_statistics'),
            'checkpoints': report.get('checkpoint_inspections'),
            'evaluations': report.get('evaluation'),
            'seed0_comparison': report.get('seed0_comparison'),
            'training_comparison': {'seed0': seed0.get('training'), 'seed1': train},
            'seed2_recommendation': seed2_recommendation(report, seed0),
            'seed2_launched': False, 'historical_64env_included_in_32env_statistics': False,
            'input_identities': identities, 'issues': report.get('issues'),
            'completion_gates': report.get('completion_gates'),
            'monitoring_policy': report.get('monitoring_policy'),
            'evaluation_recovery_provenance': provenance or manifest.get('evaluation_recovery')}


def render(report, seed0, manifest, result):
    train = report.get('training', {}); base_train = seed0.get('training', {})
    progress = train.get('verified_progress', {}); exit_record = train.get('exit') or {}
    amp = train.get('amp_statistics') or {}; scales = amp.get('completed_epoch_amp_scale') or {}
    losses = train.get('loss_stability') or {}; runtime = report.get('audit_runtime') or {}
    final = final_evaluation(report).get('aggregate', {}); base_final = final_evaluation(seed0).get('aggregate', {})
    lines = [
        '# G3C-R5：32 环境独立训练种子 1 完整报告', '',
        f"工程完成状态：**{report.get('engineering_status')}**；科学评估状态：**{report.get('scientific_status')}**。",
        f"本次从头训练使用 training seed **1**；独立评估使用 seeds **1001、1002、1003**。评估种子控制评估初始条件，不能当作额外的训练重复。", '',
        f"经连续 epoch 审计验证完成 **{progress.get('completed_epochs_verified', 0)}/400 epochs**、**{progress.get('completed_transitions_verified', 0):,}/1,638,400 transitions**。"
        f"最终 checkpoint 的 ever-success 为 **{rate(final, 'ever_success')}**，terminal-success 为 **{rate(final, 'terminal_success')}**。", '',
        '## 配置冻结与历史保留', '',
        f"起始提交：`{manifest.get('starting_commit')}`。以完整 G3C-R4 的 32-env training seed 0 作为唯一直接对照。",
        '科学配置固定为原始 Factory PegInsert、32 个环境、400 epochs、horizon 128、官方 PPO 与 LSTM。训练命令仅把 seed 0 改为 seed 1，并更换实验名称和输出目录。奖励、物理、成功判据、观测/动作空间、网络、PPO 超参数、checkpoint 周期与评估协议保持相同。',
        f"运行名称：`{manifest.get('training', {}).get('experiment_name')}`。checkpoint 每 {manifest.get('training', {}).get('save_frequency', '未记录')} epochs 保存；本次未载入 seed-0 checkpoint，不续训、不拼接旧训练。",
        'R4 源文件、报告、checkpoint 和失败记录保留；历史 64-env G2、G3B 诊断 episode 及不完整训练均未进入本报告的 32-env 统计。保留校验与冻结 SHA 记录见同目录的 manifest、preflight 与 preservation evidence。', '',
        '## 监督与退出证据', '',
        '按照用户在 G3C-R5 中继续给予的授权，本次沿用显式 `user-authorized-no-gpu-guard` 模式。独立 GPU guard 不运行；缺失 GPU telemetry 或采样间隙不触发训练停止。CUDA 正常异常处理保持不变，未更换驱动、未终止无关用户进程。GPU 连续利用率/显存覆盖未测量，因此没有连续覆盖或真实显存峰值声明。',
        'durable supervisor 保持 GPU0 独占锁、console、实际 child wait 与退出码记录。systemd `Restart=no`；TensorBoard 在官方训练退出路径执行 flush/close。控制器仅在完整 400-epoch 训练通过验收后，依次评估 4 个 checkpoint；不自动启动、恢复、终止或重试训练。',
        f"实际 child wait returncode：`{exit_record.get('raw_wait_returncode')}`；child exit code：`{exit_record.get('child_exit_code')}`；child signal：`{exit_record.get('child_signal')}`；退出原因：`{exit_record.get('termination_reason')}`。",
        f"supervisor error：`{exit_record.get('error')}`；stop signals：`{exit_record.get('stop_signals')}`；supervisor 收到的 signals：`{exit_record.get('received_supervisor_signals')}`。", '',
        '## 训练耗时与吞吐', '',
        '| 指标 | training seed 0（R4） | training seed 1（R5） |',
        '|---|---:|---:|',
        f"| Supervisor 训练用时（s） | {number(base_train.get('training_duration_seconds'))} | {number(train.get('training_duration_seconds'))} |",
        f"| 官方 PPO 累计时间（s） | {number(base_train.get('official_ppo_total_seconds'))} | {number(train.get('official_ppo_total_seconds'))} |",
        f"| 官方 PPO aggregate transitions/s | {number(base_train.get('official_ppo_aggregate_fps'))} | {number(train.get('official_ppo_aggregate_fps'))} |",
        f"| Epochs 11 起稳定段 transitions/s | {number(base_train.get('stable_aggregate_total_fps'))} | {number(train.get('stable_aggregate_total_fps'))} |",
        f"| 完成 transitions / 全程 wall s | {number(base_train.get('end_to_end_transitions_per_wall_second'))} | {number(train.get('end_to_end_transitions_per_wall_second'))} |", '',
        '官方 PPO 吞吐的分子和分母来自同一连续完成 epoch 前缀。Supervisor 用时包含进程启动、训练与实际 child wait/finalization；它与官方 PPO rollout/optimization 累计时间不同。若训练不完整，未完成 rollout 的交互数量未知，未计入完成 transition 数。', '',
        '## Loss、AMP 与优化器', '',
        f"观测 epoch loss 数组全部有限：`{losses.get('observed_epoch_loss_arrays_finite')}`；记录的 loss/optimization scalars 全部有限：`{losses.get('logged_loss_and_optimization_scalars_finite')}`。",
        f"实际 actor optimizer attempts：**{amp.get('actor_attempts')}**；实际 actor Adam steps：**{amp.get('actor_optimizer_steps')}**；实际 central-value Adam steps：**{amp.get('central_value_optimizer_steps')}**。",
        f"AMP skip 事件：**{amp.get('amp_skip_count')}**；attempts−actor steps：`{amp.get('actor_attempts_minus_actor_steps')}`；与事件数一致：`{amp.get('skip_count_matches_attempt_step_difference')}`。",
        f"完成 epoch 的 AMP scale：首次 `{scales.get('first')}`；末次 `{scales.get('last')}`；统计 `{scales.get('statistics')}`。",
        '计数来自只读训练 observer 的实际事件，未用预期 minibatch 数补齐。顶层优化器计数可能包含不完整 epoch 中已记录的工作；完成 epoch 的累计值单独保留在机器证据。未直接测量逐梯度有限性。有限 loss/参数只支持数值稳定性描述，不能证明收敛、任务成功或所有梯度有限。', '',
        '| TensorBoard 标量 | n | 最小值 | 中位数 | 最大值 |', '|---|---:|---:|---:|---:|']
    recovery = result.get('evaluation_recovery_provenance')
    if recovery:
        # Keep the initial incomplete controller visible even when a later,
        # explicitly selected evaluation-only recovery supplies final metrics.
        boundary = lines.index('## 配置冻结与历史保留')
        section = [
            '## 评估基础设施故障与独立恢复的来源', '',
            '这份最终报告保留并区分最初的评估控制器与随后人工选择的仅评估恢复。原始训练没有重新运行、恢复或拼接；完成的 training seed 1 checkpoint 及其 SHA 保持不变。',
            '首个评估进程虽然 child wait 为 0，但没有产生 metrics，不能认定为科学评估完成。原始控制器因此记录为 INCOMPLETE。Kit/OmniClient 记录在解析官方 S3 的 `default_environment.usd` 时出现 libcurl SSL connect error 35；该故障不属于训练失败或 CUDA OOM。',
            '随后在新的 completion/evaluation 输出目录中，显式执行仅评估恢复。训练 checkpoint、环境、task/reward/success、确定性推理、评估种子与 episode 分母沿用冻结协议。最初的失败 console、controller status、部分分析和 result 仍保留，不被最终结果覆盖；不存在选择表现更好的训练重跑。',
            '最终工程/科学状态描述本次训练和登记的评估恢复所获得的完整证据；不回填或改写原始失败控制器状态。具体恢复 manifest、原始来源与身份由下列机器证据保留：', '',
            '```json', json.dumps(recovery, ensure_ascii=False, indent=2, sort_keys=True), '```', '']
        lines[boundary:boundary] = section
    for tag, stats in losses.get('scalar_statistics', {}).items():
        lines.append(f"| `{tag}` | {stats.get('n')} | {number(stats.get('min'), 6)} | {number(stats.get('median'), 6)} | {number(stats.get('max'), 6)} |")
    lines += ['', '## Checkpoint 完整性', '',
              '| Epoch | Frame | 大小（bytes） | CPU 数值有限 | SHA-256 |', '|---|---:|---:|---|---|']
    for epoch in (100, 200, 300, 400):
        ckpt = report.get('checkpoint_inspections', {}).get(str(epoch), {})
        lines.append(f"| {epoch} | {ckpt.get('frame', '未记录')} | {ckpt.get('size_bytes', '未记录')} | `{ckpt.get('all_tensors_and_numeric_arrays_finite')}` | `{ckpt.get('sha256', '未记录')}` |")
    lines += ['', '只对新产生且受信任的本地 checkpoint 在 CPU 上解包，核验 epoch/frame、actor/central-value 模型、Adam 状态与数值有限性；SHA 用于绑定独立评估所使用的同一策略。Epoch 400 按审计保存顺序选择官方最终 checkpoint，周期保存和最终保存均保留。', '',
              '## 独立评估及 training seed 0 对照', '',
              '评估采用确定性策略均值，恢复 checkpoint observation RMS，推理模式运行，LSTM 在 episode 边界重置。官方成功判据在自动 reset 前捕获。每个 checkpoint 使用相同 evaluation seeds 1001/1002/1003：epochs 100/200/300 每 seed 32 个完成 episodes、总分母 96；epoch 400 每 seed 96 个完成 episodes、总分母 288。未完成 episode 不补零，也不并入分母。', '',
              '| Epoch / transitions | seed 0 ever-success | seed 1 ever-success | seed 0 terminal-success | seed 1 terminal-success | seed 1−0 ever / terminal（百分点） |',
              '|---|---|---|---|---|---|']
    for row in report.get('seed0_comparison', {}).get('rows', []):
        old, now, delta = row.get('seed0_evaluation') or {}, row.get('seed1_evaluation') or {}, row.get('observed_rate_differences') or {}
        lines.append(f"| {row['checkpoint_epoch']} / {row['matched_training_transitions']:,} | {rate(old, 'ever_success')} | {rate(now, 'ever_success')} | {rate(old, 'terminal_success')} | {rate(now, 'terminal_success')} | {number(delta.get('ever_success_rate_percentage_points'))} / {number(delta.get('terminal_success_rate_percentage_points'))} |")
    lines += ['', '首次成功时间只针对 ever-successful 的已完成 episodes：', '',
              '| Epoch | training seed 0（模拟时间） | training seed 1（模拟时间） |', '|---|---|---|']
    for row in report.get('seed0_comparison', {}).get('rows', []):
        lines.append(f"| {row['checkpoint_epoch']} | {timing(row.get('seed0_evaluation'))} | {timing(row.get('seed1_evaluation'))} |")
    lines += ['', '失败 episodes 不被当作 0 秒或 timeout 秒数；因此首次成功均值/中位数会受成功样本组成影响。', '',
              '## 每个 evaluation seed 的表现', '',
              '| Checkpoint epoch | Evaluation seed | Training seed 0 ever / terminal | Training seed 1 ever / terminal | R5 首次成功（条件） |',
              '|---|---:|---|---|---|']
    for row in report.get('seed0_comparison', {}).get('rows', []):
        for pair in row.get('per_evaluation_seed', []):
            old, now = pair.get('seed0') or {}, pair.get('seed1') or {}
            lines.append(f"| {row['checkpoint_epoch']} | {pair['evaluation_seed']} | {rate(old, 'ever_success')} / {rate(old, 'terminal_success')} | {rate(now, 'ever_success')} / {rate(now, 'terminal_success')} | {timing(now)} |")
    lines += ['', '## 解释、局限及 seed 2 建议', '',
              '本次比较在相同 32-env 科学配置下，仅改变 training seed。相同 transition 预算和相同评估 seeds 提供直接描述性对照；它们不保证随机轨迹相同。两个训练策略仍不足以可靠估计训练种子总体均值、方差或失败概率。每个策略的 episode 共享策略和部分环境机制，episode 层面的 Wilson 区间仅作描述，不能解释为训练种子间的不确定性。',
              '训练种子与评估种子分开报告；没有把历史 64-env baseline 混入 32-env 成功率，也没有把两个训练种子的 episodes 合并为单个性能分母。观察到的差异不触发任何超参数适配。',
              result['seed2_recommendation'], '', '## 验收、机器证据与可追溯性', '',
              '机器证据包括 `analysis/report.json`、TensorBoard scalars/CSV/图、完整 console/audit/finalizer/supervisor status、checkpoint SHA/CPU inspection、12 次独立评估的逐 episode metrics、source freeze、配置差异和保留审计。CPU regression 与 fast-forward push 验证随最终交付记录保存。',
              f"验收失败项：`{report.get('completion_gates', {}).get('failed_requirements', [])}`。",
              f"分析错误：`{report.get('issues', {}).get('errors', [])}`；分析警告：`{report.get('issues', {}).get('warnings', [])}`。"]
    error_lines = train.get('console_evidence', {}).get('error_lines', [])
    if report.get('engineering_status') != 'COMPLETE' and error_lines:
        lines += ['', '实际错误摘录：', '', '```text', '\n'.join(error_lines), '```']
    lines += ['', '| 输入 | SHA-256 |', '|---|---|']
    for name, record in result['input_identities'].items():
        lines.append(f"| `{name}` | `{record['sha256']}` |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--provenance', type=Path,
                        help='Optional immutable JSON recording evaluation-only recovery provenance')
    args = parser.parse_args(argv)
    if args.output.exists() or args.result.exists():
        parser.error('报告和 result 必须使用全新路径；禁止覆盖历史证据')
    report, seed0, manifest = load(args.report), load(args.baseline), load(args.manifest)
    if manifest.get('training_seed') != 1:
        parser.error('R5 manifest 必须明确 training_seed=1')
    identities = {'analysis_report': identity(args.report), 'seed0_analysis_report': identity(args.baseline),
                  'manifest': identity(args.manifest), 'report_generator': identity(__file__)}
    provenance = None
    if args.provenance:
        provenance = load(args.provenance)
        if not isinstance(provenance, dict):
            parser.error('--provenance must contain a JSON object')
        identities['evaluation_recovery_provenance'] = identity(args.provenance)
    expected_seed0 = manifest.get('seed0_baseline', {}).get('analysis_sha256')
    if expected_seed0 and identities['seed0_analysis_report']['sha256'] != expected_seed0:
        parser.error('seed-0 baseline SHA 与冻结 manifest 不一致')
    result = make_result(report, seed0, manifest, identities, provenance)
    markdown = render(report, seed0, manifest, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(markdown)
    with args.result.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'result': str(args.result.resolve()),
                      'engineering_status': result['engineering_status'],
                      'scientific_status': result['scientific_status']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
