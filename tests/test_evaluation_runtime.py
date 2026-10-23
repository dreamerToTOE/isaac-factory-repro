from types import SimpleNamespace

import pytest
import torch
from rl_games.common.player import BasePlayer

from scripts.evaluation_runtime import reset_player_rnn_states, tensor_tree_is_finite


def test_full_observation_finite_check_includes_privileged_states():
    obs = {"obs": torch.ones(2, 19), "states": torch.ones(2, 43)}
    assert tensor_tree_is_finite(obs)
    obs["states"][1, 3] = float("nan")
    player = BasePlayer.__new__(BasePlayer)
    player.device = torch.device("cpu")
    # Verify the installed API strips the outer obs/states structure. Checking
    # only this output would miss the deliberately nonfinite critic state.
    converted = player.obs_to_torch(obs)
    assert torch.is_tensor(converted)
    assert torch.isfinite(converted).all()
    assert not tensor_tree_is_finite(obs)


def test_finite_check_handles_nested_tensor_observations():
    assert tensor_tree_is_finite({"obs": [torch.ones(2), (torch.zeros(1),)]})
    assert not tensor_tree_is_finite({"obs": {"camera": torch.tensor([float("inf")])}})
    with pytest.raises(TypeError, match="Unsupported"):
        tensor_tree_is_finite("not an observation tensor")


def test_actual_player_batch_and_rnn_state_api():
    player = BasePlayer.__new__(BasePlayer)
    player.device = torch.device("cpu")
    player.obs_shape = (19,)
    player.has_batch_dimension = False
    player.is_rnn = True
    player.model = SimpleNamespace(get_default_rnn_state=lambda: [torch.zeros(2, 1, 7)] * 2)
    assert player.get_batch_size(torch.ones(4, 19), 1) == 4
    assert player.has_batch_dimension is True
    player.init_rnn()
    assert len(player.states) == 2
    assert all(tuple(state.shape) == (2, 4, 7) for state in player.states)


def test_timeout_clears_only_done_player_states_inside_inference_mode():
    with torch.inference_mode():
        states = [torch.ones(2, 3, 7), torch.full((2, 3, 7), 2.0)]
    player = SimpleNamespace(is_rnn=True, states=states)
    assert all(state.is_inference() for state in player.states)
    reset_player_rnn_states(player, [False, True, False])
    assert all(torch.count_nonzero(state[:, 1, :]) == 0 for state in player.states)
    assert torch.all(player.states[0][:, 0, :] == 1)
    assert torch.all(player.states[1][:, 2, :] == 2)


def test_nonrecurrent_player_needs_no_hidden_state_reset():
    reset_player_rnn_states(SimpleNamespace(is_rnn=False, states=None), [True])
