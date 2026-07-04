"""Reusable PyTorch neural network blocks."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Shape-preserving residual block for 2D convolutional features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
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
