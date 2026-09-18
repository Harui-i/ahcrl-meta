from typing import Any, cast

import torch
from torch import nn

from ahcrl.contests.ahc063.encoder import (
    ACTION_COUNT,
    CATEGORICAL_EXCLUDED_CHANNELS,
    MAX_BOARD_SIZE,
    NUM_PLANES,
    TYPED_NUM_PLANES,
)
from ahcrl.contests.ahc063.model import AHC063_CATEGORICAL_GROUPS, ActorCritic
from ahcrl.nn.modula import ModularEmbedding, build_modula_parameter_specs
from ahcrl.nn.observation import RunningObservationNormalizer


def test_actor_critic_convnext() -> None:
    model = ActorCritic(NUM_PLANES, 16, 4)
    batch_size = 5
    x = torch.randn(batch_size, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
    policy, value = model(x)

    assert policy.shape == (batch_size, ACTION_COUNT)
    assert value.shape == (batch_size,)


def test_actor_critic_with_two_blocks() -> None:
    model = ActorCritic(NUM_PLANES, 16, 2)
    x = torch.randn(3, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)

    policy, value = model(x)

    assert policy.shape == (3, ACTION_COUNT)
    assert value.shape == (3,)


def test_actor_critic_can_be_traced() -> None:
    model = ActorCritic(NUM_PLANES, 8, 1).eval()
    x = torch.randn(1, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)

    traced = cast(Any, torch.jit.trace(model, x, strict=True))
    policy, value = traced(x)

    assert policy.shape == (1, ACTION_COUNT)
    assert value.shape == (1,)


def test_actor_critic_runs_with_bfloat16_weights_and_inputs() -> None:
    model = ActorCritic(NUM_PLANES, 8, 1).to(dtype=torch.bfloat16).eval()
    x = torch.randn(1, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE, dtype=torch.bfloat16)

    policy, value = model(x)

    assert policy.dtype == torch.bfloat16
    assert value.dtype == torch.bfloat16


def test_actor_critic_uses_explicit_max_pool_for_compile_safety() -> None:
    model = ActorCritic(NUM_PLANES, 16, 1)
    features = torch.randn(2, 16, MAX_BOARD_SIZE, MAX_BOARD_SIZE)

    assert isinstance(model.max_pool, nn.AdaptiveMaxPool2d)
    torch.testing.assert_close(model.max_pool(features).flatten(1), features.amax(dim=(-2, -1)))


def test_actor_critic_reports_feature_norm_stats_after_cell_encoder() -> None:
    model = ActorCritic(NUM_PLANES, 8, 1)
    x = torch.randn(2, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)

    stats = model.feature_norm_stats(x)

    assert set(stats) == {
        "trunk_feature_norm_mean",
        "trunk_feature_norm_std",
        "trunk_feature_norm_max",
    }
    assert all(value >= 0.0 for value in stats.values())


def test_categorical_adapter_expands_all_semantic_groups_and_handles_none() -> None:
    model = ActorCritic(NUM_PLANES, 16, 1)
    x = torch.zeros(1, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
    for group in AHC063_CATEGORICAL_GROUPS:
        x[0, group.start, 1, 1] = 1.0

    adapted = model.input_adapter(x)
    assert adapted.shape == (1, TYPED_NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)

    raw_cursor = 0
    output_cursor = 0
    for group, embedding_module in zip(
        AHC063_CATEGORICAL_GROUPS, model.input_adapter.embeddings, strict=True
    ):
        output_cursor += group.start - raw_cursor
        embedding = embedding_module
        assert isinstance(embedding, ModularEmbedding)
        none = embedding.weight[0]
        active = embedding.weight[1]
        torch.testing.assert_close(
            adapted[0, output_cursor : output_cursor + group.output_dim, 0, 0], none
        )
        torch.testing.assert_close(
            adapted[0, output_cursor : output_cursor + group.output_dim, 1, 1], active
        )
        raw_cursor = group.end
        output_cursor += group.output_dim


def test_cell_encoder_is_per_cell_and_projects_to_trunk_width() -> None:
    model = ActorCritic(NUM_PLANES, 16, 1).eval()
    x = torch.zeros(2, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
    adapted = model.input_adapter(x)
    encoded = model.cell_encoder(adapted.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    assert encoded.shape == (2, 16, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
    perturbed = x.clone()
    perturbed[:, 31, 0, 0] = 1.0
    perturbed_encoded = model.cell_encoder(
        model.input_adapter(perturbed).permute(0, 2, 3, 1)
    ).permute(0, 3, 1, 2)
    unaffected = torch.ones((MAX_BOARD_SIZE, MAX_BOARD_SIZE), dtype=torch.bool)
    unaffected[0, 0] = False
    torch.testing.assert_close(
        encoded.permute(0, 2, 3, 1)[:, unaffected],
        perturbed_encoded.permute(0, 2, 3, 1)[:, unaffected],
    )


def test_embedding_and_cell_encoder_receive_gradients() -> None:
    model = ActorCritic(NUM_PLANES, 8, 1)
    x = torch.randn(3, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
    logits, _ = model(x)

    logits.square().mean().backward()

    embedding_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in model.named_parameters()
        if name.startswith("input_adapter.embeddings") and parameter.grad is not None
    )
    cell_encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in model.named_parameters()
        if name.startswith("cell_encoder") and parameter.grad is not None
    )
    assert embedding_grad_norm > 0.0
    assert cell_encoder_grad_norm > 0.0


def test_embedding_and_cell_encoder_have_dedicated_modula_geometries() -> None:
    model = ActorCritic(NUM_PLANES, 8, 1)
    specs = {spec.name: spec for spec in build_modula_parameter_specs(model)}

    for index in range(len(AHC063_CATEGORICAL_GROUPS)):
        geometry = specs[f"input_adapter.embeddings.{index}.weight"].geometry
        assert geometry is not None and geometry.name == "embedding"
    for name in ("cell_encoder.0.weight", "cell_encoder.0.bias", "cell_encoder.2.weight"):
        geometry = specs[name].geometry
        assert geometry is not None
        expected = "linear" if name.endswith("weight") else "bounded_rms_vector"
        assert geometry.name == expected


def test_observation_normalizer_excludes_semantic_planes() -> None:
    normalizer = RunningObservationNormalizer(
        NUM_PLANES,
        excluded_channels=CATEGORICAL_EXCLUDED_CHANNELS,
    )
    observations = torch.randn(4, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
    categorical = list(CATEGORICAL_EXCLUDED_CHANNELS)
    observations[:, categorical] = (observations[:, categorical] > 0).float()

    normalized = normalizer.update_and_normalize(observations)
    torch.testing.assert_close(normalized[:, categorical], observations[:, categorical])
    normalized_channels = [
        channel for channel in range(NUM_PLANES) if channel not in CATEGORICAL_EXCLUDED_CHANNELS
    ]
    assert torch.allclose(
        normalized[:, normalized_channels].mean(dim=(0, 2, 3)),
        torch.zeros(len(normalized_channels)),
        atol=1e-5,
    )
