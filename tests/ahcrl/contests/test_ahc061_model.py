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


def test_actor_critic_rejects_unknown_block_type() -> None:
    with pytest.raises(ValueError, match="unknown block_type"):
        ActorCritic(block_type="unknown")
