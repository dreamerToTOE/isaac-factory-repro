"""Read-only historical closeout audit; stdlib only, never loads a simulator/actor.

Run from the repository root. snapshot/build/check write only their explicitly
new closeout receipts. Hash checkpoints as bytes; never deserialize them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import re
import subprocess
import sys


class ForbiddenRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'torch', 'isaacsim', 'isaaclab', 'rl_games'}:
            raise ImportError('closeout forbids simulator/actor import: ' + fullname)


sys.meta_path.insert(0, ForbiddenRuntime())
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/g4_s/closeout'
BASE = '85017ba626e123ed95c397c32c4b852e9118abe4'
CP = 'f5e9da77553f51c8fad430eb254fbeabc8963425d89cdc28af8f071944b08f2f'
PREVIOUS = 'evidence/g4_s/gripper_v1_r1/regression_final_v1/preservation_after.json'
EDITABLE = {'README.md', 'docs/case_studies/README.md',
            'docs/case_studies/SIM2SIM_PANDA_CONTROL_INSTABILITY.md'}
DOCS = sorted(EDITABLE | {'docs/REVIEW_G4_SIM2SIM_CLOSEOUT.md',
                         'docs/case_studies/SIM2SIM_INTERVIEW_GUIDE.md',
                         'evidence/g4_s/closeout/README.md'})
FINAL = {
    'G4_SIM2SIM_MILESTONE': 'PARTIAL',
    'G4_CURRENT_GRIPPER_ROUTE': 'CLOSED_WITH_NEGATIVE_RESULT',
    'G4_S2A_R3': 'NO_GO', 'G4_S2B': 'NO_GO',
    'ROOT_CAUSE': 'PARTIALLY_CHARACTERIZED_NOT_UNIQUELY_IDENTIFIED',
    'TASK_LEVEL_ZERO_SHOT_TRANSFER': 'NOT_DEMONSTRATED',
}
# Copy these exact fields from the original machine gates, without normalizing
# historical PASS/PARTIAL or replacing them with the final milestone decision.
GATES = {
    'S0': ('s0/summary.json', ['engineering_status', 'decision', 'sim2sim_status']),
    'S1A': ('s1a/reconstruction_gate.json', ['engineering_status', 'faithful_Factory_reconstruction', 'S1B_decision']),
    'S1B': ('s1b/reconstruction_gate.json', ['status', 'scientific_zero_shot_policy_effectiveness']),
    'S2A': ('s2a/completion_gate.json', ['status', 'scientific_effectiveness', 'policy_inferences', 'policy_physics_steps', 'technically_valid_trajectories', 'official_ever_success', 'official_terminal_success', 'exact_experiment_exit_code']),
    'S2A_R1': ('s2a_r1/completion_gate.json', ['status', 'PASS_scope', 'scientific']),
    'S2A_R2': ('s2a_r2/completion_gate.json', ['engineering_status', 'candidate_decision', 'R3', 'S2B', 'full_horizon_all_gate_pass_runs']),
    'S2A_R2_R1': ('s2a_r2_r1/completion_gate.json', ['final_engineering_status', 'candidate_decision', 'scientific_outcome', 'decision_basis']),
    'S2A_R2_R2': ('s2a_r2_r2/completion_gate.json', ['engineering_status', 'new_correction_candidate', 'historical_adaptation_candidate', 'new_physics_integrations', 'new_policy_inferences']),
    'S2A_R2_R3': ('s2a_r2_r3/completion_gate.json', ['engineering_status', 'existing_p_full', 'new_correction', 'r3', 's2b', 'integration_steps', 'candidate_executed']),
    'GRIPPER_V1': ('gripper_v1/completion_gate.json', ['engineering_status', 'candidate_status', 'i2_status', 'i2_max_pad_penetration_m', 'i3_status', 'new_gripper_static_mapping']),
    'GRIPPER_V1_R1': ('gripper_v1_r1/completion_gate.json', ['engineering_status', 'candidate_status', 'candidate_scope', 'r3', 's2b', 'historical_v1_i3_cause', 'state_preflight', 'preparations', 'new_physics_steps', 'request_result_pairs', 'unstarted', 'policy_calls']),
}


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')


def snapshot():
    tracked = git('ls-tree', '-r', '--name-only', BASE).splitlines()
    old = read(PREVIOUS)
    external_expected = {}
    for group in ['frozen_sources_and_configurations', 'independent_pinned_Panda_files',
                  'four_original_USD_hashes', 'unrelated_original_checkout_g3v']:
        external_expected.update(old[group])
    for item in old['checkpoints_hash_only'].values():
        external_expected[item['path']] = item['sha256']
    checkpoint_root = Path(old['checkpoints_hash_only']['g2_epoch200_primary']['path']).parents[5]
    assert checkpoint_root.name == 'runs', checkpoint_root
    checkpoints = sorted(checkpoint_root.rglob('*.pth'))
    # This inventory covers every .pth below the archived work/runs root, in
    # addition to the individually referenced external scientific sources.
    external = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in sorted({Path(p) for p in external_expected} | set(checkpoints))}
    mismatches = [p for p, expected in external_expected.items()
                  if external[p]['sha256'] != expected]
    if mismatches:
        raise ValueError('archived external hash mismatch: ' + repr(mismatches))
    return {'starting_commit': BASE,
            'tracked': {p: sha(ROOT / p) for p in tracked},
            'external': external, 'prior_manifest_reference': PREVIOUS,
            'prior_manifest_expected_hashes_match': True,
            'checkpoint_inventory': {'root': str(checkpoint_root),
                                    'files': [str(p) for p in checkpoints],
                                    'count': len(checkpoints),
                                    'bytes': sum(p.stat().st_size for p in checkpoints)},
            'checkpoint_access': 'SHA256_BYTES_ONLY_NO_DESERIALIZATION'}


def make_summary():
    gates = {}
    for name, (p, keys) in GATES.items():
        p = 'evidence/g4_s/' + p
        d = read(p)
        gates[name] = {'reference': p, 'sha256': sha(ROOT / p),
                       'original_fields': {k: d[k] for k in keys}}
    source_path = 'evidence/g2/analysis/evaluation_summary.json'
    source = next(x for x in read(source_path)['checkpoints'] if x['checkpoint_epoch'] == 200)
    assert source['aggregate']['episodes_completed'] == 288
    # Independent inference results are separately retained, not pooled with G2.
    inference_path = 'evidence/g4_i/i2/results/summary.json'
    inference = read(inference_path)['evaluation']
    factorial_path = 'evidence/g4_s/s2a_r2/numerical_review_v2/factorial_effects.json'
    factorial = read(factorial_path)
    inference_gates = {}
    for name, p, keys in [
        ('I0_CPU', 'evidence/g4_i/i0/g4i0_acceptance_cpu.json', ['exit_code', 'unfiltered_tests']),
        ('I1', 'evidence/g4_i/i1/runtime_acceptance_strict_coverage.json', ['status', 'passed', 'checks']),
        ('I2', 'evidence/g4_i/i2/acceptance.json', ['status', 'scope', 'checks'])]:
        original = read(p)
        inference_gates[name] = {'reference': p, 'sha256': sha(ROOT / p),
                                 'original_fields': {k: original[k] for k in keys}}
    refs = sorted({v['reference'] for v in gates.values()} | {
        source_path, inference_path, factorial_path,
        'evidence/g4_i/i0/checkpoint_and_source_identity.json',
        'evidence/g4_i/i1/independent_parity_review.json',
        'evidence/g4_i/i2/scientific_protocol_review.json',
        'evidence/g4_s/gripper_v1_r1/image_qa.json',
        'evidence/g4_s/gripper_v1_r1/numerical_review_v1.json',
    } | {v['reference'] for v in inference_gates.values()}
      | {str(p.relative_to(ROOT)) for p in ROOT.glob('docs/REVIEW_G4_*.md')
         if p.name != 'REVIEW_G4_SIM2SIM_CLOSEOUT.md'})
    hashes = {p: sha(ROOT / p) for p in refs}
    stages_complete = [n for n, v in gates.items() if any(
        isinstance(x, str) and (x == 'COMPLETE' or x.endswith('_PASS'))
        for k, x in v['original_fields'].items() if k in
        {'engineering_status', 'status', 'final_engineering_status'})]
    return {
        'schema_version': 'g4_sim2sim_closeout_v1', 'starting_commit': BASE,
        'frozen_checkpoint_sha256': CP, 'final_status': FINAL,
        'final_engineering_status': FINAL['G4_SIM2SIM_MILESTONE'],
        'final_scientific_status': FINAL['TASK_LEVEL_ZERO_SHOT_TRANSFER'],
        'original_source_baseline': {'reference': source_path, 'original_epoch200_fields': source,
            'metric_scope': 'PhysX Factory original geometric success; NOT MuJoCo transfer'},
        'standalone_source_inference': {'reference': inference_path, 'original_evaluation': inference,
            'comparison': 'descriptive unpaired; do not pool denominators with G2'},
        'historical_gates': gates, 'engineering_completed_stages': stages_complete,
        'inference_engineering_evidence': inference_gates,
        'partial_engineering_stages': ['S2A', 'S2A_R2_R1', 'S2A_R2_R3', 'GRIPPER_V1'],
        'failed_stages': {'task_transfer': ['S2A'], 'physical_preparation': ['GRIPPER_V1', 'GRIPPER_V1_R1'],
                          'meaning': 'scientific/physical acceptance not achieved; historical engineering statuses above remain unchanged'},
        'failed_experiments': {
            'S2A': gates['S2A']['original_fields'],
            'GRIPPER_V1': gates['GRIPPER_V1']['original_fields'],
            'GRIPPER_V1_R1': gates['GRIPPER_V1_R1']['original_fields']['preparations'],
            'scope': 'physical failures / invalid execution; not blanket G4 engineering failure'},
        'rejected_candidates': {'native_tendon_v1': gates['GRIPPER_V1_R1']['original_fields']['candidate_status'],
            'scope': gates['GRIPPER_V1_R1']['original_fields']['candidate_scope']},
        'verified_local_findings': {
            'reference': factorial_path,
            'I1_common_prefix_steps': factorial['I1']['common_prefix_steps'],
            'I1_A_B_C_D_common_prefix_metrics': factorial['I1']['common_prefix_metrics'],
            'interpretation': 'P_FULL plant inertia locally stabilizes initial motion; controller-only substitution does not; no global stability or task-success claim',
            'state_integrity_R1': gates['GRIPPER_V1_R1']['original_fields']['state_preflight']},
        'remaining_UNKNOWN': ['unique contact/grasp root cause', 'historical V1 I3 initial-state mutation cause',
            'historical I3_FULL_C1 exact child exit after unmatched request 212',
            'source PhysX actual applied motor torque', 'exact original cooked PhysX collision representation',
            'original NVIDIA asset conversion/distribution permission',
            'target-engine task success probability', 'continuous collision behavior between recorded states'],
        'historical_reference_sha256': hashes,
        'authorization_boundaries': {'allowed': ['documentation', 'archived-data charts', 'CPU consistency and hash checks', 'commit and fast-forward push'],
            'executed_new_physics_steps': 0, 'executed_new_policy_calls': 0,
            'checkpoint_deserializations': 0, 'training': False,
            'forbidden': ['MuJoCo integration', 'PPO inference/training', 'new gripper tests', 'R3', 'S2B', 'GearMesh', 'NutThread'],
            'future_architecture': 'UNTESTED_RECOMMENDATION_REQUIRES_SEPARATE_AUTHORIZATION',
            'learning_and_theoretical_examination': 'PENDING'},
    }


def validate_summary(summary):
    if summary != make_summary():
        raise ValueError('summary diverges from archived gates / final authorized contract')
    if summary['original_source_baseline']['original_epoch200_fields']['aggregate']['ever_success_count'] != 280:
        raise ValueError('source denominator/count mismatch')
    return True


def check_links(text, path, expected_outputs=()):
    missing = []
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
        link = link.strip('<>').split('#')[0]
        if not link or re.match(r'^[a-z]+:', link):
            continue
        target = (path.parent / link).resolve()
        if not target.exists() and target not in expected_outputs:
            missing.append(link)
    return missing


def preservation_diff(before, after):
    if set(before['tracked']) != set(after['tracked']):
        raise ValueError('baseline inventory changed')
    changed = [p for p in before['tracked'] if before['tracked'][p] != after['tracked'][p]]
    illegal = sorted(set(changed) - EDITABLE)
    if illegal or before['external'] != after['external']:
        raise ValueError('preservation failure: ' + repr(illegal))
    return changed


def check():
    before = read('evidence/g4_s/closeout/preservation_before.json')
    after = snapshot()
    changed = preservation_diff(before, after)
    summary = read('evidence/g4_s/closeout/milestone_summary.json')
    validate_summary(summary)
    # These two receipts are produced below. Resolve all other references now,
    # then check every link again, without exceptions, after actual creation.
    pending = {(OUT / p).resolve() for p in ['preservation_after.json', 'consistency_checks.json']}
    missing = {p: check_links((ROOT / p).read_text(), ROOT / p, pending) for p in DOCS}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise ValueError('broken documentation links: ' + repr(missing))
    # Original case narrative after section 1 is append-only preserved.
    case = 'docs/case_studies/SIM2SIM_PANDA_CONTROL_INSTABILITY.md'
    old = git('show', BASE + ':' + case)
    marker = '## 1. 背景：为什么要做 Sim2Sim'
    if old[old.index(marker):] not in (ROOT / case).read_text():
        raise ValueError('original case narrative changed')
    for k, v in FINAL.items():
        if f'{k}={v}' not in (ROOT / 'docs/REVIEW_G4_SIM2SIM_CLOSEOUT.md').read_text():
            raise ValueError('report missing final contract ' + k)
    forbidden = sorted(set(sys.modules) & {'mujoco', 'torch', 'isaacsim', 'isaaclab', 'rl_games'})
    if forbidden:
        raise ValueError('forbidden runtime imported')
    write_new(OUT / 'preservation_after.json', after)
    result = {'status': 'PASS', 'starting_commit': BASE,
              'starting_tracked_count': len(before['tracked']),
              'modified_historical_files': changed,
              'all_other_historical_files_byte_identical': True,
              'external_scientific_files_and_all_checkpoint_hashes_identical': True,
              'checkpoint_count': after['checkpoint_inventory']['count'],
              'checkpoint_bytes': after['checkpoint_inventory']['bytes'],
              'summary_matches_original_gates': True, 'local_document_links': 'PASS',
              'original_case_narrative_preserved': True,
              'forbidden_runtime_imports': forbidden,
              'physics_steps': 0, 'policy_calls': 0, 'checkpoint_deserializations': 0,
              'scope': 'documentation and integrity only; historical simulator tests intentionally not invoked'}
    write_new(OUT / 'consistency_checks.json', result)
    assert not any(check_links((ROOT / p).read_text(), ROOT / p) for p in DOCS)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['snapshot', 'build', 'check'])
    mode = p.parse_args().mode
    if mode == 'snapshot':
        if git('rev-parse', 'HEAD') != BASE:
            raise ValueError('snapshot requires verified starting HEAD')
        write_new(OUT / 'preservation_before.json', snapshot())
    elif mode == 'build':
        write_new(OUT / 'milestone_summary.json', make_summary())
    else:
        check()
