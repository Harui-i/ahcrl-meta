from typing import Any, cast

import pytest
import torch

from ahcrl.contests.ahc061.encoder import BOARD_SIZE, NUM_PLANES
from ahcrl.contests.ahc061.model import ActorCritic


@pytest.mark.parametrize("block_type", ["convnext", "residual"])
def test_actor_critic_output_shapes(block_type: str) -> None:
    model = ActorCritic(channels=8, blocks=2, block_type=block_type)
    x = torch.randn(3, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

    logits, value = model(x)

    assert logits.shape == (3, BOARD_SIZE * BOARD_SIZE)
    assert value.shape == (3,)


def test_actor_critic_can_be_traced() -> None:
    model = ActorCritic(channels=8, blocks=1).eval()
    x = torch.randn(1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

    traced = cast(Any, torch.jit.trace(model, x, strict=True))
    logits, value = traced(x)

    assert logits.shape == (1, BOARD_SIZE * BOARD_SIZE)
    assert value.shape == (1,)


def test_value_head_receives_gradients() -> None:
    model = ActorCritic(channels=8, blocks=1)
    x = torch.randn(3, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    _, value = model(x)
    loss = value.square().mean()

    loss.backward()

    grad_norm = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in model.named_parameters()
        if name.startswith("value.") and parameter.grad is not None
    )
    assert grad_norm > 0.0


def test_actor_critic_rejects_unknown_block_type() -> None:
    with pytest.raises(ValueError, match="unknown block_type"):
        ActorCritic(block_type="unknown")
