"""Reusable PyTorch neural network blocks."""

from __future__ import annotations

import math

import torch
from torch import nn

from ahcrl.nn.components import (
    GlobalContextLERP2d,
    HyperEmbedder2d,
    HyperLinear,
    HyperMLP,
    HypersphericalFeatureNorm,
    LinearScaler,
    Scaler,
    ShiftL2Norm,
    SphericalSelfAttentionLERP2d,
    l2_normalize,
    make_group_norm,
    project_hyperspherical_weights_,
    project_weight_to_unit_norm_,
)

__all__ = [
    "ConvNeXtBlock",
    "GlobalContextLERP2d",
    "HyperEmbedder2d",
    "HyperLinear",
    "HyperMLP",
    "HypersphericalFeatureNorm",
    "LinearScaler",
    "PerCellMLPBlock",
    "ResidualBlock",
    "Scaler",
    "ShiftL2Norm",
    "SimbaV2Block",
    "SphericalAttentionSimbaBlock",
    "SphericalDepthwiseSimbaBlock",
    "SphericalGlobalContextBlock",
    "SphericalSelfAttentionLERP2d",
    "l2_normalize",
    "make_group_norm",
    "project_hyperspherical_weights_",
    "project_weight_to_unit_norm_",
]


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


class PerCellMLPBlock(nn.Module):
    """Shape-preserving per-cell channel MLP block without spatial mixing."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if layer_scale_init < 0.0:
            raise ValueError(f"layer_scale_init must be non-negative, got {layer_scale_init}")

        self.norm = nn.LayerNorm(channels)
        hidden_channels = channels * expansion
        self.pointwise = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )
        self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        residual = x
        y = x.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = self.pointwise(y)
        y = self.layer_scale * y
        y = y.permute(0, 3, 1, 2)
        return residual + y


class SimbaV2Block(nn.Module):
    """SimbaV2 HyperLERPBlock for per-cell 2D channel features."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        alpha_init: float = 0.2,
        alpha_scale: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if alpha_scale == 0.0:
            raise ValueError("alpha_scale must be non-zero")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        default_alpha_scale = 1.0 / math.sqrt(channels)
        self.mlp = HyperMLP(
            channels,
            expansion=expansion,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            eps=eps,
        )
        self.alpha_scaler = Scaler(
            channels,
            init=alpha_init,
            scale=default_alpha_scale if alpha_scale is None else alpha_scale,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        residual = x
        y = x.permute(0, 2, 3, 1)
        y = self.mlp(y)
        y = y.permute(0, 3, 1, 2)
        y = residual + self.alpha_scaler((y - residual).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return l2_normalize(y, dim=1, eps=self.eps)


class SphericalDepthwiseSimbaBlock(nn.Module):
    """SimbaV2-style block with local depthwise spatial mixing before HyperMLP."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        kernel_size: int = 3,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        local_alpha_init: float = 0.05,
        local_alpha_scale: float | None = None,
        alpha_init: float = 0.2,
        alpha_scale: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if local_alpha_scale == 0.0:
            raise ValueError("local_alpha_scale must be non-zero")
        if alpha_scale == 0.0:
            raise ValueError("alpha_scale must be non-zero")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        default_alpha_scale = 1.0 / math.sqrt(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.local_alpha_scaler = Scaler(
            channels,
            init=local_alpha_init,
            scale=default_alpha_scale if local_alpha_scale is None else local_alpha_scale,
        )
        self.mlp = HyperMLP(
            channels,
            expansion=expansion,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            eps=eps,
        )
        self.alpha_scaler = Scaler(
            channels,
            init=alpha_init,
            scale=default_alpha_scale if alpha_scale is None else alpha_scale,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        local_delta = self.depthwise(x)
        local_delta = self.local_alpha_scaler(local_delta.permute(0, 2, 3, 1))
        local = l2_normalize(x + local_delta.permute(0, 3, 1, 2), dim=1, eps=self.eps)
        y = local.permute(0, 2, 3, 1)
        y = self.mlp(y)
        y = y.permute(0, 3, 1, 2)
        y = x + self.alpha_scaler((y - x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return l2_normalize(y, dim=1, eps=self.eps)


class SphericalGlobalContextBlock(nn.Module):
    """Per-cell SimbaV2 block followed by broadcast global context mixing."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        alpha_init: float = 0.2,
        alpha_scale: float | None = None,
        global_alpha_init: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.per_cell = SimbaV2Block(
            channels,
            expansion=expansion,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            eps=eps,
        )
        self.global_context = GlobalContextLERP2d(
            channels,
            expansion=expansion,
            alpha_init=global_alpha_init,
            alpha_scale=alpha_scale,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_context(self.per_cell(x))


class SphericalAttentionSimbaBlock(nn.Module):
    """Per-cell SimbaV2 block followed by hyperspherical self-attention LERP."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        heads: int = 4,
        max_spatial_size: int = 10,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        alpha_init: float = 0.2,
        alpha_scale: float | None = None,
        attention_alpha_init: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.per_cell = SimbaV2Block(
            channels,
            expansion=expansion,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            eps=eps,
        )
        self.attention = SphericalSelfAttentionLERP2d(
            channels,
            heads=heads,
            max_spatial_size=max_spatial_size,
            alpha_init=attention_alpha_init,
            alpha_scale=alpha_scale,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.per_cell(x))
