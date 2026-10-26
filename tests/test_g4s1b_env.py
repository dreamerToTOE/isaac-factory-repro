"""CPU tests for the adapted MuJoCo environment, via the pinned 3.5.0 process."""
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
MUJOCO = Path('/home/ubuntu2004/miniconda3/envs/unitree_rl_mjlab/bin/python')


@pytest.mark.skipif(not MUJOCO.is_file(), reason='pinned MuJoCo 3.5.0 environment unavailable')
def test_environment_collector_real_exit_and_no_policy(tmp_path):
    output = tmp_path/'environment.json'
    command = [str(MUJOCO), str(ROOT/'scripts/g4s1b_env.py'), '--output', str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report['mujoco_version'] == '3.5.0'
    assert report['menagerie_commit'] == '8161bba264d7fa7c99ca301e91e7fb44737676ad'
    assert report['structure']['nq'] == 16 and report['structure']['nv'] == 15 and report['structure']['nu'] == 9
    assert not report['structure']['has_weld']
    assert report['geometry']['nominal_radial_clearance_m'] == pytest.approx(57e-6)
    assert report['geometry']['margin_m'] == 0 and report['geometry']['gap_m'] == 0
    assert report['probes']['robot_kinematics']['mass_symmetry_max_error'] == 0
    assert min(report['probes']['robot_kinematics']['mass_eigenvalues']) > 0
    assert report['probes']['gentle_reference_insertion']['passed']
    assert not report['probes']['aligned_drop_insertion']['passed']
    assert report['probes']['physical_grasp']['no_weld']
    bridge = report['probes']['reference_controller_bridge']
    assert bridge['initial_observation_size'] == 19
    assert bridge['policy_updates'] == 3
    assert all(x['substep_count'] == 8 and x['state_is_pre_reset'] for x in bridge['actions'])


def test_environment_source_contains_no_policy_or_checkpoint_imports():
    source = (ROOT/'scripts/g4s1b_env.py').read_text()
    forbidden = ('g4i1_policy', 'torch.load', 'from_checkpoint', 'PpoPlayer', 'SimulationApp')
    assert not any(token in source for token in forbidden)
