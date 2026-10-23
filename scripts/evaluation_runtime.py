"""Small CPU-testable helpers for the evaluation harness; no Isaac Sim imports."""
from collections.abc import Mapping

import torch


def tensor_tree_is_finite(value):
    """Check all tensors, including privileged states stripped by obs_to_torch."""
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(tensor_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensor_tree_is_finite(item) for item in value)
    raise TypeError(f"Unsupported observation/reward value type: {type(value).__name__}")


def reset_player_rnn_states(player, dones):
    """Follow RL-Games Player.states API, safely handling inference tensors."""
    if not player.is_rnn or player.states is None:
        return
    # States returned by a model inside inference_mode are inference tensors.
    # Mutating them after leaving that context raises even under no_grad.
    with torch.inference_mode():
        for state in player.states:
            done_mask = torch.as_tensor(dones, device=state.device, dtype=torch.bool)
            state[:, done_mask, :] = 0.0
