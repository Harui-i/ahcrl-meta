from typing import Any, cast

import pytest
import torch

from ahcrl.contests.ahc061.encoder import BOARD_SIZE, CRITIC_FEATURE_SHAPE, NUM_PLANES
from ahcrl.contests.ahc061.model import ActorCritic


@pytest.mark.parametrize(
    "block_type",
    [
        "convnext",
        "per_cell_mlp",
        "residual",
        "simbav2_block",
        "spherical_depthwise_simba",
        "spherical_global_context",
        "spherical_attention_simba",
    ],
)
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


def test_critic_actor_critic_can_be_traced_without_critic_features() -> None:
    model = ActorCritic(channels=8, blocks=1, critic_feature_mode="oracle").eval()
    x = torch.randn(1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

    traced = cast(Any, torch.jit.trace(model, x, strict=True))
    logits, value = traced(x)

    assert logits.shape == (1, BOARD_SIZE * BOARD_SIZE)
    assert value.shape == (1,)


def test_actor_critic_runs_with_bfloat16_weights_and_inputs() -> None:
    model = ActorCritic(channels=8, blocks=1).to(dtype=torch.bfloat16).eval()
    x = torch.randn(1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE, dtype=torch.bfloat16)

    logits, value = model(x)

    assert logits.dtype == torch.bfloat16
    assert value.dtype == torch.bfloat16


def test_actor_critic_reports_trunk_feature_norm_stats() -> None:
    model = ActorCritic(channels=8, blocks=1)
    x = torch.randn(3, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

    stats = model.trunk_feature_norm_stats(x)

    assert set(stats) == {
        "trunk_feature_norm_mean",
        "trunk_feature_norm_std",
        "trunk_feature_norm_max",
    }
    assert all(value >= 0.0 for value in stats.values())


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


def test_critic_features_only_affect_value_head() -> None:
    model = ActorCritic(channels=8, blocks=1, critic_feature_mode="oracle")
    x = torch.randn(3, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    zero_features = torch.zeros(3, *CRITIC_FEATURE_SHAPE)
    oracle_features = torch.rand(3, *CRITIC_FEATURE_SHAPE)

    zero_logits, zero_value = model(x, zero_features)
    oracle_logits, oracle_value = model(x, oracle_features)
    oracle_value.square().mean().backward()

    assert torch.equal(zero_logits, oracle_logits)
    assert zero_value.shape == oracle_value.shape == (3,)
    critic_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in model.named_parameters()
        if name.startswith("value.critic_") and parameter.grad is not None
    )
    assert critic_grad_norm > 0.0


def test_actor_critic_rejects_unknown_block_type() -> None:
    with pytest.raises(ValueError, match="unknown block_type"):
        ActorCritic(block_type="unknown")
