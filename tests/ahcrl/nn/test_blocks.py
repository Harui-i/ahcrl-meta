import pytest
import torch

from ahcrl.nn import ConvNeXtBlock, ResidualBlock, make_group_norm


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
