"""Reusable PyTorch neural network blocks."""

from __future__ import annotations

import torch
from torch import nn


def make_group_norm(channels: int, *, max_groups: int = 8) -> nn.GroupNorm:
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    if max_groups <= 0:
        raise ValueError(f"max_groups must be positive, got {max_groups}")

    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)

    raise AssertionError("unreachable")


class ResidualBlock(nn.Module):
    """Shape-preserving residual block for 2D convolutional features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ConvNeXtBlock(nn.Module):
    """Shape-preserving ConvNeXt block for 2D convolutional features."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        kernel_size: int = 3,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if layer_scale_init < 0.0:
            raise ValueError(f"layer_scale_init must be non-negative, got {layer_scale_init}")

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        hidden_channels = channels * expansion
        self.pointwise = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(x)
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = self.pointwise(y)
        y = self.layer_scale * y
        y = y.permute(0, 3, 1, 2)
        return residual + y


class HypersphericalFeatureNorm(nn.Module):
    """Normalize per-cell channel features onto a learnable-radius sphere."""

    def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.eps = eps
        self.scale = nn.Parameter(torch.full((channels,), channels**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).sum(dim=1, keepdim=True).clamp_min(self.eps**2).sqrt()
        scale = self.scale.view(1, -1, 1, 1).to(dtype=x.dtype)
        return x / norm.to(dtype=x.dtype) * scale


class SphericalConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with hyperspherical channel-feature normalization."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        kernel_size: int = 3,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if layer_scale_init < 0.0:
            raise ValueError(f"layer_scale_init must be non-negative, got {layer_scale_init}")

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.pre_norm = HypersphericalFeatureNorm(channels)
        hidden_channels = channels * expansion
        self.pointwise = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale_init))
        self.post_norm = HypersphericalFeatureNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(x)
        y = self.pre_norm(y)
        y = y.permute(0, 2, 3, 1)
        y = self.pointwise(y)
        y = self.layer_scale * y
        y = y.permute(0, 3, 1, 2)
        return self.post_norm(residual + y)
