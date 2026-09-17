from typing import Any, cast

import torch

from ahcrl.contests.ahc061.encoder import (
    BOARD_SIZE,
    CRITIC_FEATURE_SHAPE,
    NUM_PLANES,
    TYPED_NUM_PLANES,
)
from ahcrl.contests.ahc061.model import ActorCritic, RunningObservationNormalizer
from ahcrl.nn.modula import build_modula_parameter_specs


def test_actor_critic_output_shapes() -> None:
    model = ActorCritic(channels=8, blocks=2)
    x = torch.randn(3, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

    logits, value = model(x)

    assert logits.shape == (3, BOARD_SIZE * BOARD_SIZE)
    assert value.shape == (3,)
    assert model.input_adapter.output_channels == TYPED_NUM_PLANES


def test_actor_and_critic_have_independent_representations() -> None:
    model = ActorCritic(channels=8, blocks=1)

    assert (
        model.input_adapter.embeddings[0].weight
        is not model.value.input_adapter.embeddings[0].weight
    )
    assert model.cell_encoder[0].weight is not model.value.cell_encoder[0].weight
    assert model.trunk[0].weight is not model.value.trunk[0].weight

    x = torch.randn(2, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    critic_loss = model.value_predictions(x).square().mean()
    critic_loss.backward()

    actor_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("value.")
    ]
    critic_gradients = [
        parameter.grad for name, parameter in model.named_parameters() if name.startswith("value.")
    ]
    assert all(gradient is None for gradient in actor_gradients)
    assert all(gradient is not None for gradient in critic_gradients)


def test_actor_only_and_critic_only_apis_match_forward() -> None:
    model = ActorCritic(channels=8, blocks=1).eval()
    x = torch.randn(2, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    critic_features = torch.randn(2, *CRITIC_FEATURE_SHAPE)

    logits, values = model(x, critic_features)
    actor_logits = model.policy_logits(x)
    critic_values = model.value_predictions(x, critic_features)

    torch.testing.assert_close(actor_logits, logits)
    torch.testing.assert_close(critic_values, values)


def test_cell_encoder_projects_each_cell_to_trunk_width_without_spatial_mixing() -> None:
    model = ActorCritic(channels=8, blocks=1).eval()
    x = torch.randn(2, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

    adapted = model.input_adapter(x)
    encoded = model.cell_encoder(adapted.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    assert adapted.shape == (2, TYPED_NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert encoded.shape == (2, 8, BOARD_SIZE, BOARD_SIZE)

    perturbed = x.clone()
    perturbed[:, 0, 0, 0] += 1.0
    perturbed_encoded = model.cell_encoder(
        model.input_adapter(perturbed).permute(0, 2, 3, 1)
    ).permute(0, 3, 1, 2)
    unaffected = torch.ones((BOARD_SIZE, BOARD_SIZE), dtype=torch.bool)
    unaffected[0, 0] = False
    torch.testing.assert_close(
        encoded.permute(0, 2, 3, 1)[:, unaffected],
        perturbed_encoded.permute(0, 2, 3, 1)[:, unaffected],
    )


def test_embedding_and_cell_encoder_receive_gradients() -> None:
    model = ActorCritic(channels=8, blocks=1)
    x = torch.randn(3, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
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


def test_cell_encoder_is_in_modula_graph_without_changing_embedding_geometry() -> None:
    model = ActorCritic(channels=8, blocks=1)
    specs = {spec.name: spec for spec in build_modula_parameter_specs(model)}
    embedding_geometry = specs["input_adapter.embeddings.0.weight"].geometry
    first_cell_geometry = specs["cell_encoder.0.weight"].geometry
    second_cell_geometry = specs["cell_encoder.2.weight"].geometry

    assert embedding_geometry is not None and embedding_geometry.name == "embedding"
    assert first_cell_geometry is not None and first_cell_geometry.name == "linear"
    assert second_cell_geometry is not None and second_cell_geometry.name == "linear"


def test_actor_critic_can_be_traced() -> None:
    model = ActorCritic(channels=8, blocks=1).eval()
    x = torch.randn(1, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)

    traced = cast(Any, torch.jit.trace(model, x, strict=True))
    logits, value = traced(x)

    assert logits.shape == (1, BOARD_SIZE * BOARD_SIZE)
    assert value.shape == (1,)


def test_critic_actor_critic_can_be_traced_without_critic_features() -> None:
    model = ActorCritic(channels=8, blocks=1).eval()
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


def test_observation_normalizer_is_model_state_and_normalizes_raw_inputs() -> None:
    model = ActorCritic(channels=8, blocks=1)
    normalizer = RunningObservationNormalizer(NUM_PLANES, epsilon=1e-8)
    model.observation_normalizer = normalizer
    observations = torch.randn(2, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    normalized = normalizer.update_and_normalize(observations)
    state = model.state_dict()

    assert "observation_normalizer.count" in state
    assert "observation_normalizer.mean" in state
    assert torch.allclose(normalized.mean(dim=(0, 2, 3)), torch.zeros(NUM_PLANES), atol=1e-5)
    assert torch.equal(state["observation_normalizer.mean"], normalizer.mean)


def test_observation_normalizer_excludes_categorical_planes() -> None:
    excluded = tuple(range(1, 23))
    normalizer = RunningObservationNormalizer(NUM_PLANES, excluded_channels=excluded)
    observations = torch.randn(4, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    observations[:, 1:23] = (observations[:, 1:23] > 0).float()

    normalized = normalizer.update_and_normalize(observations)
    assert torch.equal(normalized[:, 1:23], observations[:, 1:23])
    assert torch.allclose(
        normalized[:, 23:].mean(dim=(0, 2, 3)),
        torch.zeros(NUM_PLANES - 23),
        atol=1e-5,
    )
    assert not any(name == "normalize_mask" for name in normalizer.state_dict())
    mean, invstd = normalizer.effective_affine()
    assert torch.equal(mean[:, 1:23], torch.zeros_like(mean[:, 1:23]))
    assert torch.equal(invstd[:, 1:23], torch.ones_like(invstd[:, 1:23]))
    stats = normalizer.stats()
    assert stats["obs_norm_std_min"] >= 0.0


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
    model = ActorCritic(channels=8, blocks=1)
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
