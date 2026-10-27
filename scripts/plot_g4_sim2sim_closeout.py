"""Charts from archived numbers only. No model compilation or physics/inference."""
import json
import sys

from g4_sim2sim_closeout import ROOT, OUT, sha, write_new

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def records(path):
    return [json.loads(x) for x in (ROOT / path).read_text().splitlines()]


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    source = 'evidence/g4_s/s2a_r2/numerical_review_v2/factorial_effects.json'
    metrics = json.loads((ROOT / source).read_text())['I1']['common_prefix_metrics']
    labels = ['I1_A\nP0 / C0', 'I1_B\nP_FULL / C0', 'I1_C\nP0 / C1', 'I1_D\nP_FULL / C1']
    speed = [metrics[k]['max_arm_velocity_rad_s'] for k in 'ABCD']
    slip = [1000 * metrics[k]['max_translation_slip_m'] for k in 'ABCD']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
    for ax, values, unit, title in zip(axes, [speed, slip], ['rad/s', 'mm'],
                                     ['Maximum arm speed', 'Maximum peg-hand translation slip']):
        bars = ax.bar(labels, values, color=['#b54a41', '#287b77', '#b54a41', '#287b77'])
        ax.set_yscale('log')
        ax.set_ylabel(unit + ' (log scale)')
        ax.set_title(title)
        ax.bar_label(bars, fmt='%.4g', padding=4)
        ax.set_ylim(min(values) * .5, max(values) * 2.5)
    fig.suptitle('R2 factorial: identical I1 initial state; matched first 5 steps (41.67 ms)\n'
                 'P = physical plant; C = arm-controller mass matrix. Local effect only.', fontsize=11)
    fig.savefig(OUT / 'inertia_common_prefix.png')
    plt.close(fig)

    paths = {
        'V1_PREP_I2': 'evidence/g4_s/gripper_v1/matrix_v1/runs/PREP_I2/output/physics.jsonl',
        'R1_PREP_I2_RATE_LIMITED': 'evidence/g4_s/gripper_v1_r1/matrix_v1/runs/PREP_I2_RATE_LIMITED/output/physics.jsonl',
        'R1_PREP_I3_RATE_LIMITED': 'evidence/g4_s/gripper_v1_r1/matrix_v1/runs/PREP_I3_RATE_LIMITED/output/physics.jsonl',
    }
    traces = {}
    for name, path in paths.items():
        rows = records(path)
        traces[name] = {'steps': [r['step'] for r in rows],
            'target_per_finger_mm': [1000 * (r['finger_target_m'] if 'finger_target_m' in r else r['state']['finger_command']['target_opening_per_finger_m']) for r in rows],
            'actual_left_mm': [1000 * r['state']['qpos'][7] for r in rows],
            'actual_right_mm': [1000 * r['state']['qpos'][8] for r in rows],
            'actual_left_step_delta_mm': [1000 * (r['state']['qpos'][7] - r['pre_state']['qpos'][7]) for r in rows],
            'initial_per_finger_mm': 1000 * rows[0]['pre_state']['qpos'][7]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    a, b = axes
    names = ['V1_PREP_I2', 'R1_PREP_I2_RATE_LIMITED']
    xs = [0, 1]
    targets = [traces[n]['target_per_finger_mm'][0] - traces[n]['initial_per_finger_mm'] for n in names]
    actuals = [traces[n]['actual_left_step_delta_mm'][0] for n in names]
    a.bar([x - .18 for x in xs], targets, .36, label='target delta', color='#737b89')
    a.bar([x + .18 for x in xs], actuals, .36, label='actual left delta', color='#b54a41')
    a.set_xticks(xs, ['V1 PREP_I2\nabrupt (1 step)', 'R1 PREP_I2\nrate limited (step 1)'])
    a.set_ylabel('per-finger displacement (mm)')
    a.set_title('Same registered I2 initial state: first step only')
    a.legend(fontsize=8)
    for n, color, style in [('R1_PREP_I2_RATE_LIMITED', '#b54a41', '-'),
                            ('R1_PREP_I3_RATE_LIMITED', '#287b77', '--')]:
        t = traces[n]
        b.plot(t['steps'], t['actual_left_mm'], style, color=color, marker='.', label=n.replace('R1_PREP_', 'R1 '))
    t = traces['R1_PREP_I2_RATE_LIMITED']
    b.plot(t['steps'], t['target_per_finger_mm'], color='#737b89', label='target (I2; I3 same)')
    b.axhline(0, color='black', linewidth=.6)
    b.set_yscale('symlog', linthresh=8)
    b.set_xlabel('physics step (dt = 1/120 s)')
    b.set_ylabel('per-finger position (mm; symlog outside +/-8)')
    b.set_title('R1: bounded target, unstable actual trajectory')
    b.legend(fontsize=7, loc='lower left')
    fig.suptitle('Native tendon V1: archived preparation failures, no policy\n'
                 'V1 stopped at step 1; R1 I2/I3 stopped at step 23. Unequal total windows.', fontsize=11)
    fig.savefig(OUT / 'gripper_target_vs_actual.png')
    plt.close(fig)
    data = {'scope': 'archived numerical data only; no new integration',
            'factorial': {'source': source, 'sha256': sha(ROOT / source), 'I1_common_prefix_steps': 5,
                'A_B_C_D_max_arm_velocity_rad_s': speed, 'A_B_C_D_max_translation_slip_mm': slip},
            'gripper': {'sources': {n: {'path': p, 'sha256': sha(ROOT / p)} for n, p in paths.items()},
                'traces': traces,
                'comparison_limit': 'same I2 registered initial state and first step; full histories have unequal windows; not a success-rate or global-stability comparison'},
            'outputs': {p: sha(OUT / p) for p in ['inertia_common_prefix.png', 'gripper_target_vs_actual.png']},
            'runtime': {'python': sys.version, 'matplotlib': matplotlib.__version__,
                'forbidden_modules_loaded': sorted(set(sys.modules) & {'mujoco', 'torch', 'isaacsim', 'isaaclab', 'rl_games'})}}
    write_new(OUT / 'chart_data.json', data)


if __name__ == '__main__':
    main()
