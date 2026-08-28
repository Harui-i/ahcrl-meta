import torch
from torch import nn

from ahcrl.contests.ahc063.encoder import ACTION_COUNT, MAX_BOARD_SIZE, NUM_PLANES
from ahcrl.contests.ahc063.model import ActorCritic


def test_actor_critic_convnext() -> None:
    model = ActorCritic(NUM_PLANES, 16, 4, "convnext")
    batch_size = 5
    x = torch.randn(batch_size, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
    policy, value = model(x)

    assert policy.shape == (batch_size, ACTION_COUNT)
    assert value.shape == (batch_size,)


def test_actor_critic_uses_explicit_max_pool_for_compile_safety() -> None:
    model = ActorCritic(NUM_PLANES, 16, 1, "spherical_attention_simba")
    features = torch.randn(2, 16, MAX_BOARD_SIZE, MAX_BOARD_SIZE)

    assert isinstance(model.max_pool, nn.AdaptiveMaxPool2d)
    torch.testing.assert_close(model.max_pool(features).flatten(1), features.amax(dim=(-2, -1)))
