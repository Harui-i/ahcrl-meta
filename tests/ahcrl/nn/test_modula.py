import math

import pytest
import torch
from torch import nn

from ahcrl.contests.ahc061.model import ActorCritic as AHC061ActorCritic
from ahcrl.contests.ahc063.model import ActorCritic as AHC063ActorCritic
from ahcrl.nn.modula import (
    ModulaGraphNode,
    ModularConv2d,
    ModularDepthwiseConv2d,
    ModularEmbedding,
    ModularLinear,
    ModularSequential,
    build_modula_parameter_specs,
)


@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (1, 9)])
def test_linear_projection_matches_scaled_polar_factor(shape: tuple[int, int]) -> None:
    layer = ModularLinear(shape[1], shape[0], bias=False)
    weight = torch.randn(shape)

    projected = layer.geometry.project(weight)

    expected = math.sqrt(shape[0] / shape[1])
    assert torch.linalg.svdvals(projected) == pytest.approx(
        torch.full((min(shape),), expected), rel=0.02, abs=0.02
    )


def test_embedding_geometry_leaves_zero_gradient_rows_untouched() -> None:
    embedding = ModularEmbedding(4, 3)
    gradient = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    update = embedding.geometry.dualize(gradient, target_norm=0.5)

    assert update[[1, 3]].count_nonzero() == 0
    assert update[[0, 2]].norm(dim=1) == pytest.approx(torch.full((2,), math.sqrt(3) * 0.5))


def test_linear_geometry_maps_zero_gradient_to_zero() -> None:
    layer = ModularLinear(3, 2, bias=False)

    update = layer.geometry.dualize(torch.zeros_like(layer.weight), target_norm=1.0)

    assert update.count_nonzero() == 0


def test_one_hot_embedding_is_not_classified_as_linear() -> None:
    class SemanticModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = ModularEmbedding(5, 3)
            self.linear = ModularLinear(5, 3)

        def modula_graph(self) -> ModulaGraphNode:
            return ModulaGraphNode(
                "parallel",
                (
                    ModulaGraphNode("atom", parameter_name="embedding.weight", own_mass=1.0),
                    ModulaGraphNode("atom", parameter_name="linear.weight", own_mass=1.0),
                ),
            )

    specs = build_modula_parameter_specs(SemanticModel())
    geometries = {spec.name: spec.geometry.name for spec in specs if spec.geometry is not None}

    assert geometries == {"embedding.weight": "embedding", "linear.weight": "linear"}


def test_grouped_and_depthwise_conv_projection_uses_separate_kernels() -> None:
    grouped = ModularConv2d(4, 6, kernel_size=3, groups=2, bias=False)
    depthwise = ModularDepthwiseConv2d(4, kernel_size=3, bias=False)

    grouped_projected = grouped.geometry.project(torch.randn_like(grouped.weight))
    depthwise_projected = depthwise.geometry.project(torch.randn_like(depthwise.weight))

    assert grouped_projected.shape == grouped.weight.shape
    assert depthwise_projected.shape == depthwise.weight.shape
    expected_depthwise_norm = math.sqrt(1 / 9)
    assert depthwise_projected.flatten(1).norm(dim=1) == pytest.approx(
        torch.full((4,), expected_depthwise_norm), rel=0.02, abs=0.02
    )


def test_graph_allocates_serial_parallel_and_residual_target_norms() -> None:
    first = ModulaGraphNode("atom", parameter_name="first", own_mass=1.0, own_sensitivity=2.0)
    second = ModulaGraphNode("atom", parameter_name="second", own_mass=3.0, own_sensitivity=4.0)

    assert ModulaGraphNode("sequence", (first, second)).allocate() == pytest.approx(
        {"first": 1 / 16, "second": 3 / 4}
    )
    assert ModulaGraphNode("parallel", (first, second)).allocate() == pytest.approx(
        {"first": 1 / 4, "second": 3 / 4}
    )
    residual = ModulaGraphNode("residual", (ModulaGraphNode("bond", own_sensitivity=1.0), first))
    assert residual.allocate() == {"first": 1.0}


def test_unclassified_trainable_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unclassified trainable parameters"):
        build_modula_parameter_specs(nn.Linear(3, 2))


def test_duplicate_parameter_in_graph_is_rejected() -> None:
    atom = ModulaGraphNode("atom", parameter_name="weight", own_mass=1.0)

    with pytest.raises(ValueError, match="duplicate parameters"):
        ModulaGraphNode("parallel", (atom, atom)).allocate()


@pytest.mark.parametrize("model_type", [AHC061ActorCritic, AHC063ActorCritic])
def test_contest_models_classify_every_trainable_parameter(
    model_type: type[nn.Module],
) -> None:
    model = model_type(channels=8, blocks=1)

    specs = build_modula_parameter_specs(model)

    assert {spec.name for spec in specs} == {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert {spec.role for spec in specs} == {"modular", "adamw"}


def test_modular_sequential_keeps_standard_state_dict_names() -> None:
    standard = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    modular = ModularSequential(ModularLinear(3, 4), nn.ReLU(), ModularLinear(4, 2))

    modular.load_state_dict(standard.state_dict())

    x = torch.randn(5, 3)
    assert torch.equal(modular(x), standard(x))
