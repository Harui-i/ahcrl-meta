import argparse

import pytest
import torch
from torch import nn

from ahcrl.contests.ahc061.encoder import BOARD_SIZE, NUM_PLANES
from ahcrl.contests.ahc061.train_ppo import (
    _transform_actions_d4,
    _transform_board_d4,
    _transform_flat_board_d4,
    update_model,
)


def test_d4_action_transforms_are_permutations() -> None:
    actions = torch.arange(BOARD_SIZE * BOARD_SIZE)

    for transform_id in range(8):
        transformed = _transform_actions_d4(actions, transform_id)

        assert sorted(transformed.tolist()) == list(range(BOARD_SIZE * BOARD_SIZE))


@pytest.mark.parametrize("transform_id", range(8))
def test_d4_flat_board_transform_matches_action_transform(transform_id: int) -> None:
    actions = torch.arange(BOARD_SIZE * BOARD_SIZE)
    flat_board = torch.arange(BOARD_SIZE * BOARD_SIZE).unsqueeze(0)

    transformed_actions = _transform_actions_d4(actions, transform_id)
    transformed_board = _transform_flat_board_d4(flat_board, transform_id).squeeze(0)

    for source, destination in enumerate(transformed_actions.tolist()):
        assert transformed_board[destination].item() == source


def test_d4_board_transform_rejects_unknown_transform() -> None:
    with pytest.raises(ValueError, match="transform_id"):
        _transform_board_d4(torch.zeros(1, 1, BOARD_SIZE, BOARD_SIZE), 8)


class TinyActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = nn.Parameter(torch.zeros(BOARD_SIZE * BOARD_SIZE))
        self.value_scale = nn.Parameter(torch.tensor(0.1))
        self.value_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        logits = self.policy.expand(batch_size, -1)
        value = x.mean(dim=(1, 2, 3)) * self.value_scale + self.value_bias
        return logits, value


def test_update_model_runs_with_full_d4_augmentation() -> None:
    model = TinyActorCritic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rollout = {
        "obs": torch.randn(1, 2, NUM_PLANES, BOARD_SIZE, BOARD_SIZE),
        "actions": torch.tensor([[0, BOARD_SIZE * BOARD_SIZE - 1]]),
        "logprobs": torch.zeros(1, 2),
        "advantages": torch.tensor([[1.0, -1.0]]),
        "returns": torch.tensor([[0.5, -0.5]]),
        "masks": torch.ones(1, 2, BOARD_SIZE * BOARD_SIZE, dtype=torch.bool),
    }
    args = argparse.Namespace(
        epochs=1,
        minibatch_size=2,
        clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )

    stats = update_model(
        model,
        model,
        optimizer,
        rollout,
        args,
        torch.device("cpu"),
    )

    assert set(stats) == {
        "policy_loss",
        "value_loss",
        "entropy",
        "normalized_entropy",
        "approx_kl",
        "clip_frac",
    }
    assert torch.isfinite(model.policy).all()
