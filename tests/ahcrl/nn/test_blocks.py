import pytest
import torch

from ahcrl.nn import (
    ConvNeXtBlock,
    HyperEmbedder2d,
    HyperLinear,
    HyperMLP,
    HypersphericalFeatureNorm,
    LinearScaler,
    PerCellMLPBlock,
    ResidualBlock,
    Scaler,
    ShiftL2Norm,
    SimbaV2Block,
    SphericalDepthwiseSimbaBlock,
    l2_normalize,
    make_group_norm,
    project_hyperspherical_weights_,
    project_weight_to_unit_norm_,
)


def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock(channels=8)
    x = torch.randn(4, 8, 12, 16)

    y = block(x)

    assert y.shape == x.shape


def test_residual_block_uses_group_norm() -> None:
    block = ResidualBlock(channels=8)

    norms = [module for module in block.modules() if isinstance(module, torch.nn.GroupNorm)]

    assert len(norms) == 2
    assert all(norm.num_groups == 8 for norm in norms)


def test_residual_block_rejects_non_positive_channels() -> None:
    with pytest.raises(ValueError):
        ResidualBlock(channels=0)


def test_convnext_block_preserves_shape() -> None:
    block = ConvNeXtBlock(channels=8)
    x = torch.randn(4, 8, 12, 16)

    y = block(x)

    assert y.shape == x.shape


def test_convnext_block_uses_3x3_depthwise_by_default() -> None:
    block = ConvNeXtBlock(channels=8)

    assert block.depthwise.kernel_size == (3, 3)
    assert block.depthwise.padding == (1, 1)


def test_convnext_block_rejects_non_positive_channels() -> None:
    with pytest.raises(ValueError, match="channels"):
        ConvNeXtBlock(channels=0)


def test_convnext_block_rejects_non_positive_expansion() -> None:
    with pytest.raises(ValueError, match="expansion"):
        ConvNeXtBlock(channels=8, expansion=0)


@pytest.mark.parametrize("kernel_size", [0, 4])
def test_convnext_block_rejects_invalid_kernel_size(kernel_size: int) -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        ConvNeXtBlock(channels=8, kernel_size=kernel_size)


def test_convnext_block_rejects_negative_layer_scale() -> None:
    with pytest.raises(ValueError, match="layer_scale_init"):
        ConvNeXtBlock(channels=8, layer_scale_init=-1e-6)


def test_per_cell_mlp_block_preserves_shape() -> None:
    block = PerCellMLPBlock(channels=8)
    x = torch.randn(4, 8, 12, 16)

    y = block(x)

    assert y.shape == x.shape


def test_per_cell_mlp_block_has_no_spatial_mixing_modules() -> None:
    block = PerCellMLPBlock(channels=8)

    convolutions = [module for module in block.modules() if isinstance(module, torch.nn.Conv2d)]

    assert convolutions == []


def test_per_cell_mlp_block_rejects_non_positive_channels() -> None:
    with pytest.raises(ValueError, match="channels"):
        PerCellMLPBlock(channels=0)


def test_per_cell_mlp_block_rejects_non_positive_expansion() -> None:
    with pytest.raises(ValueError, match="expansion"):
        PerCellMLPBlock(channels=8, expansion=0)


def test_per_cell_mlp_block_rejects_negative_layer_scale() -> None:
    with pytest.raises(ValueError, match="layer_scale_init"):
        PerCellMLPBlock(channels=8, layer_scale_init=-1e-6)


def test_hyperspherical_feature_norm_preserves_shape_and_returns_finite_values() -> None:
    norm = HypersphericalFeatureNorm(channels=8)
    x = torch.randn(4, 8, 12, 16)

    y = norm(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_hyperspherical_feature_norm_initializes_norm_near_sqrt_channels() -> None:
    norm = HypersphericalFeatureNorm(channels=8)
    x = torch.randn(4, 8, 12, 16)

    y = norm(x)
    feature_norm = y.float().pow(2).sum(dim=1).sqrt()

    assert feature_norm.mean().item() == pytest.approx(8**0.5, rel=1e-4)


def test_l2_normalize_projects_along_requested_dimension() -> None:
    x = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    y = l2_normalize(x, dim=-1)

    assert torch.allclose(torch.linalg.vector_norm(y, dim=-1), torch.ones(2))


def test_shift_l2_norm_appends_constant_axis_and_preserves_magnitude_information() -> None:
    norm = ShiftL2Norm(shift_const=3.0)
    x = torch.tensor([[4.0, 0.0], [8.0, 0.0]])

    y = norm(x)

    assert y.shape == (2, 3)
    assert torch.allclose(torch.linalg.vector_norm(y, dim=-1), torch.ones(2))
    assert y[0, -1] != pytest.approx(y[1, -1])


def test_shift_l2_norm_rejects_non_positive_shift() -> None:
    with pytest.raises(ValueError, match="shift_const"):
        ShiftL2Norm(shift_const=0.0)


def test_scaler_decouples_parameter_scale_from_forward_initialization() -> None:
    scaler = Scaler(dim=3, init=2.0, scale=10.0)
    x = torch.ones(2, 3)

    y = scaler(x)

    assert torch.allclose(scaler.scaler, torch.full((3,), 10.0))
    assert torch.allclose(y, torch.full((2, 3), 2.0))


def test_linear_scaler_preserves_shape_and_can_normalize_output() -> None:
    layer = LinearScaler(4, 6, scaler_init=1.0, scaler_scale=1.0, normalize_output=True)
    x = torch.randn(5, 4)

    y = layer(x)

    assert y.shape == (5, 6)
    assert torch.allclose(torch.linalg.vector_norm(y.float(), dim=-1), torch.ones(5), rtol=1e-5)


def test_linear_scaler_initializes_weight_vectors_on_unit_sphere() -> None:
    layer = LinearScaler(4, 6)

    norms = torch.linalg.vector_norm(layer.linear.weight.detach().float(), dim=1)

    assert torch.allclose(norms, torch.ones(6), rtol=1e-5)


def test_project_weight_to_unit_norm_projects_rows_in_place() -> None:
    weight = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    returned = project_weight_to_unit_norm_(weight)

    assert returned is weight
    assert torch.allclose(torch.linalg.vector_norm(weight, dim=1), torch.ones(2))


def test_project_hyperspherical_weights_projects_only_hyper_linear_weights() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.Conv2d(2, 5, kernel_size=3),
        HyperLinear(4, 3),
    )
    linear = model[0]
    conv = model[1]
    hyper = model[2]
    assert isinstance(linear, torch.nn.Linear)
    assert isinstance(conv, torch.nn.Conv2d)
    assert isinstance(hyper, HyperLinear)
    with torch.no_grad():
        regular_linear_norm_before = torch.linalg.vector_norm(linear.weight.detach().float(), dim=1)
        regular_conv_norm_before = torch.linalg.vector_norm(
            conv.weight.detach().float().flatten(1),
            dim=1,
        )
        hyper.weight.mul_(3.0)

    projected = project_hyperspherical_weights_(model)

    hyper_norm = torch.linalg.vector_norm(hyper.weight.detach().float(), dim=1)
    regular_linear_norm_after = torch.linalg.vector_norm(linear.weight.detach().float(), dim=1)
    regular_conv_norm_after = torch.linalg.vector_norm(
        conv.weight.detach().float().flatten(1),
        dim=1,
    )
    assert projected == 1
    assert torch.allclose(hyper_norm, torch.ones(3), rtol=1e-5)
    assert torch.allclose(regular_linear_norm_after, regular_linear_norm_before)
    assert torch.allclose(regular_conv_norm_after, regular_conv_norm_before)


def test_hyper_mlp_preserves_shape_and_normalizes_last_dimension() -> None:
    mlp = HyperMLP(channels=8, expansion=2)
    x = torch.randn(4, 12, 16, 8)

    y = mlp(x)

    assert y.shape == x.shape
    assert torch.allclose(
        torch.linalg.vector_norm(y.float(), dim=-1),
        torch.ones(4, 12, 16),
        rtol=1e-5,
    )


def test_hyper_embedder_2d_projects_each_cell_to_unit_sphere() -> None:
    embedder = HyperEmbedder2d(in_channels=5, out_channels=8)
    x = torch.randn(4, 5, 12, 16)

    y = embedder(x)

    assert y.shape == (4, 8, 12, 16)
    assert torch.allclose(
        torch.linalg.vector_norm(y.float(), dim=1),
        torch.ones(4, 12, 16),
        rtol=1e-5,
    )


def test_simbav2_block_preserves_shape() -> None:
    block = SimbaV2Block(channels=8)
    x = torch.randn(4, 8, 12, 16)

    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.allclose(
        torch.linalg.vector_norm(y.float(), dim=1),
        torch.ones(4, 12, 16),
        rtol=1e-5,
    )


def test_simbav2_block_alpha_scaler_initializes_to_alpha_init() -> None:
    block = SimbaV2Block(channels=8, alpha_init=0.25)
    x = torch.ones(2, 3, 4, 8)

    y = block.alpha_scaler(x)

    assert torch.allclose(y, torch.full_like(x, 0.25))


def test_simbav2_block_rejects_non_positive_channels() -> None:
    with pytest.raises(ValueError, match="channels"):
        SimbaV2Block(channels=0)


def test_spherical_depthwise_simba_block_preserves_shape_and_normalizes_channels() -> None:
    block = SphericalDepthwiseSimbaBlock(channels=8)
    x = torch.randn(4, 8, 12, 16)

    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.allclose(
        torch.linalg.vector_norm(y.float(), dim=1),
        torch.ones(4, 12, 16),
        rtol=1e-5,
    )


def test_spherical_depthwise_simba_block_uses_3x3_depthwise_by_default() -> None:
    block = SphericalDepthwiseSimbaBlock(channels=8)

    assert block.depthwise.kernel_size == (3, 3)
    assert block.depthwise.padding == (1, 1)
    assert block.depthwise.groups == 8
    assert block.depthwise.bias is None


def test_spherical_depthwise_simba_block_alpha_scalers_initialize_to_init_values() -> None:
    block = SphericalDepthwiseSimbaBlock(
        channels=8,
        local_alpha_init=0.05,
        alpha_init=0.25,
    )
    x = torch.ones(2, 3, 4, 8)

    local_y = block.local_alpha_scaler(x)
    y = block.alpha_scaler(x)

    assert torch.allclose(local_y, torch.full_like(x, 0.05))
    assert torch.allclose(y, torch.full_like(x, 0.25))


def test_spherical_depthwise_simba_block_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="channels"):
        SphericalDepthwiseSimbaBlock(channels=0)
    with pytest.raises(ValueError, match="expansion"):
        SphericalDepthwiseSimbaBlock(channels=8, expansion=0)
    with pytest.raises(ValueError, match="kernel_size"):
        SphericalDepthwiseSimbaBlock(channels=8, kernel_size=4)
    with pytest.raises(ValueError, match="local_alpha_scale"):
        SphericalDepthwiseSimbaBlock(channels=8, local_alpha_scale=0.0)
    with pytest.raises(ValueError, match="alpha_scale"):
        SphericalDepthwiseSimbaBlock(channels=8, alpha_scale=0.0)


@pytest.mark.parametrize(
    ("channels", "expected_groups"),
    [(64, 8), (10, 5), (7, 7)],
)
def test_make_group_norm_selects_divisible_group_count(
    channels: int,
    expected_groups: int,
) -> None:
    norm = make_group_norm(channels)

    assert norm.num_channels == channels
    assert norm.num_groups == expected_groups
