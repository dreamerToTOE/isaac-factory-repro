"""CPU regression of the actual runtime's nested batch reset-cache verifier.

Only its AST FunctionDef is executed; importing the runtime or Isaac is avoided.
Quaternion helpers deliberately reject broadcasting. Known zero recovered yaw
implies a reset yaw cache of 1/3 independently of the implementation's formula.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.g4i1_policy import CommandBoundary


RUNTIME = Path(__file__).resolve().parents[1] / "scripts/g4i1_runtime.py"


class StrictQuaternionMath:
    def __init__(self):
        self.multiplied_shapes = []

    def quat_from_euler_xyz(self, roll, pitch, yaw):
        assert roll.shape == pitch.shape == yaw.shape == (1,)
        torch.testing.assert_close(roll, torch.tensor([-np.pi]), atol=1e-6, rtol=0)
        assert torch.count_nonzero(pitch) == torch.count_nonzero(yaw) == 0
        # Exact known -pi rotation about X, in Isaac wxyz convention.
        return torch.tensor([[0., -1., 0., 0.]])

    def quat_mul(self, left, right):
        assert left.shape == right.shape, "quat_mul does not broadcast across actor batches"
        assert left.ndim == 2 and left.shape[1] == 4
        self.multiplied_shapes.append((tuple(left.shape), tuple(right.shape)))
        # Independently evaluate Hamilton multiplication for the known X pair.
        lw, lx, ly, lz = left.unbind(-1)
        rw, rx, ry, rz = right.unbind(-1)
        return torch.stack((lw*rw-lx*rx-ly*ry-lz*rz,
                            lw*rx+lx*rw+ly*rz-lz*ry,
                            lw*ry-lx*rz+ly*rw+lz*rx,
                            lw*rz+lx*ry-ly*rx+lz*rw), dim=-1)

    def get_euler_xyz(self, quat):
        expected = torch.zeros_like(quat)
        expected[:, 0] = 1.
        torch.testing.assert_close(quat, expected, atol=0, rtol=0)
        zeros = torch.zeros(quat.shape[0])
        return zeros.clone(), zeros.clone(), zeros.clone()


def extract_actual_verifier(env, count, boundary, quaternion_math):
    tree = ast.parse(RUNTIME.read_text())
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == "verify_reset_cache"]
    assert len(nodes) == 1
    node = copy.deepcopy(nodes[0])
    assert not node.decorator_list
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {"torch": torch, "np": np, "tu": quaternion_math, "env": env,
             "count": count, "boundary": boundary,
             "plain": lambda value: value.detach().cpu().tolist()}
    exec(compile(module, str(RUNTIME), "exec"), scope)
    return scope["verify_reset_cache"]


def fixture(count):
    position = torch.arange(count*3, dtype=torch.float32).reshape(count, 3) / 100
    actions = torch.zeros(count, 6)
    actions[:, 5] = 1/3  # independent known yaw=0 -> normalized yaw reset cache
    env = SimpleNamespace(actions=actions, device="cpu",
        fingertip_midpoint_pos=position.clone(), fixed_pos_action_frame=position.clone(),
        fingertip_midpoint_quat=torch.tensor([[0., 1., 0., 0.]]).repeat(count, 1),
        cfg=SimpleNamespace(ctrl=SimpleNamespace(pos_action_bounds=[.1, .2, .3])))
    obs = torch.zeros(count, 19)
    obs[:, 13:19] = actions
    boundary = CommandBoundary(SimpleNamespace(device=torch.device("cpu")),
                               lambda _action: pytest.fail("no command is allowed in this CPU verifier test"),
                               batch_size=count)
    math = StrictQuaternionMath()
    return env, obs, boundary, math, extract_actual_verifier(env, count, boundary, math)


@pytest.mark.parametrize("count", [1, 32])
def test_actual_reset_cache_uses_matching_quaternion_batch_and_known_yaw(count):
    env, obs, boundary, math, verify = fixture(count)
    before = env.actions.clone()
    report = verify(obs)
    assert math.multiplied_shapes == [((count, 4), (count, 4))]
    assert boundary.cache_initialized is True
    assert torch.equal(env.actions, before)
    expected = torch.tensor([[0., 0., 0., 0., 0., 1/3]]).repeat(count, 1)
    torch.testing.assert_close(torch.tensor(report["expected_cache"]), expected, atol=1e-7, rtol=0)
    assert report["actual_cache"] == before.tolist()
    assert report["max_absolute_error"] <= 1e-7


def test_partial_done_checks_only_reset_rows_but_initializes_full_batch_cache():
    env, obs, boundary, math, verify = fixture(32)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[[0, 11, 31]] = True
    # Active actors retain their live EMA cache; a pose-derived reset formula
    # must not be imposed upon these non-reset actors.
    env.actions[~mask] = torch.tensor([.7, -.8, .9, .4, -.3, -.2])
    obs[:, 13:19] = env.actions
    before = env.actions.clone()
    report = verify(obs, mask=mask)
    assert boundary.cache_initialized and math.multiplied_shapes == [((32, 4), (32, 4))]
    assert torch.equal(env.actions, before)
    expected = torch.tensor(report["expected_cache"])
    torch.testing.assert_close(expected[mask], env.actions[mask], atol=1e-7, rtol=0)
    assert not torch.allclose(expected[~mask], env.actions[~mask])
    assert report["max_absolute_error"] > .5  # diagnostic includes live rows
    # Sensitivity control: omitting the mask would incorrectly reject this same
    # valid partial-reset batch because active caches intentionally differ.
    with pytest.raises(AssertionError): verify(obs)


@pytest.mark.parametrize("bad_row", [0, 11, 31])
def test_partial_done_still_rejects_wrong_cache_for_any_reset_actor(bad_row):
    env, obs, boundary, _math, verify = fixture(32)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[[0, 11, 31]] = True
    env.actions[bad_row, 5] = -.5
    obs[:, 13:19] = env.actions
    with pytest.raises(AssertionError): verify(obs, mask=mask)
    assert not boundary.cache_initialized


@pytest.mark.parametrize("bad_row", [0, 5, 31])
def test_partial_mask_does_not_hide_observation_cache_mismatch_in_any_actor(bad_row):
    env, obs, boundary, _math, verify = fixture(32)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[0] = True
    env.actions[1:, :] = .7
    obs[:, 13:19] = env.actions
    obs[bad_row, 13] += .01
    with pytest.raises(ValueError, match="EMA cache must equal"):
        verify(obs, mask=mask)
    assert not boundary.cache_initialized


def test_strict_math_control_rejects_the_original_single_quaternion_batch_bug():
    math = StrictQuaternionMath()
    one = math.quat_from_euler_xyz(torch.tensor([-np.pi]), torch.zeros(1), torch.zeros(1))
    batch = torch.tensor([[0., 1., 0., 0.]]).repeat(32, 1)
    with pytest.raises(AssertionError, match="does not broadcast"):
        math.quat_mul(one, batch)
