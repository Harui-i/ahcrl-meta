import torch

from ahcrl.contests.ahc063.encoder import ACTION_COUNT, MAX_BOARD_SIZE, NUM_PLANES
from ahcrl.contests.ahc063.model import ActorCritic


def test_actor_critic_convnext() -> None:
    model = ActorCritic(NUM_PLANES, 16, 4, "convnext")
    batch_size = 5
    x = torch.randn(batch_size, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
    policy, value = model(x)

    assert policy.shape == (batch_size, ACTION_COUNT)
    assert value.shape == (batch_size,)
    return
