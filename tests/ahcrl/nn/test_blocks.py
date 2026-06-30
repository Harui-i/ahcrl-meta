import pytest
import torch

from ahcrl.nn import ResidualBlock


def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock(channels=8)
    x = torch.randn(4, 8, 12, 16)

    y = block(x)

    assert y.shape == x.shape


def test_residual_block_rejects_non_positive_channels() -> None:
    with pytest.raises(ValueError):
        ResidualBlock(channels=0)

