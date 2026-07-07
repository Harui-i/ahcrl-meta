"""Reusable PyTorch neural network blocks."""

from __future__ import annotations

import torch
from torch import nn


def l2_normalize(x: torch.Tensor, *, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    norm = x.float().pow(2).sum(dim=dim, keepdim=True).clamp_min(eps**2).sqrt()
    return x / norm.to(dtype=x.dtype)


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
        scale = self.scale.view(1, -1, 1, 1).to(dtype=x.dtype)
        return l2_normalize(x, dim=1, eps=self.eps) * scale


class ShiftL2Norm(nn.Module):
    """Append a constant coordinate and project the result onto the unit sphere."""

    def __init__(self, *, shift_const: float = 3.0, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        if shift_const <= 0.0:
            raise ValueError(f"shift_const must be positive, got {shift_const}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.shift_const = shift_const
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim if self.dim >= 0 else x.ndim + self.dim
        if dim < 0 or dim >= x.ndim:
            raise ValueError(f"dim out of range for input with {x.ndim} dimensions: {self.dim}")

        shift_shape = list(x.shape)
        shift_shape[dim] = 1
        shift = torch.full(
            shift_shape,
            self.shift_const,
            dtype=x.dtype,
            device=x.device,
        )
        return l2_normalize(torch.cat([x, shift], dim=dim), dim=dim, eps=self.eps)


class Scaler(nn.Module):
    """Learnable element-wise scale with decoupled forward initialization."""

    def __init__(self, dim: int, *, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if scale == 0.0:
            raise ValueError("scale must be non-zero")

        self.dim = dim
        self.init = init
        self.scale_value = scale
        self.scaler = nn.Parameter(torch.full((dim,), scale))
        self.forward_scaler = init / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        return x * self.scaler.to(dtype=x.dtype) * self.forward_scaler


class LinearScaler(nn.Module):
    """Bias-free linear layer followed by Scaler, optionally followed by l2 norm."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        normalize_output: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.scaler = Scaler(out_features, init=scaler_init, scale=scaler_scale)
        self.normalize_output = normalize_output
        self.eps = eps
        nn.init.orthogonal_(self.linear.weight)
        project_weight_to_unit_norm_(self.linear.weight, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.scaler(self.linear(x))
        if self.normalize_output:
            y = l2_normalize(y, dim=-1, eps=self.eps)
        return y


@torch.no_grad()
def project_weight_to_unit_norm_(weight: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Project each output weight vector onto the unit l2 sphere in-place."""

    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if weight.ndim == 0:
        raise ValueError("weight must have at least one dimension")

    if weight.ndim == 1:
        flat = weight.view(1, -1)
    else:
        flat = weight.view(weight.shape[0], -1)
    norms = flat.float().pow(2).sum(dim=1, keepdim=True).clamp_min(eps**2).sqrt()
    flat.div_(norms.to(dtype=weight.dtype))
    return weight


@torch.no_grad()
def project_hyperspherical_weights_(module: nn.Module, *, eps: float = 1e-12) -> int:
    """Project Linear/Conv weights in a module tree and return the number projected."""

    projected = 0
    for child in module.modules():
        if isinstance(child, nn.Linear | nn.Conv1d | nn.Conv2d | nn.Conv3d):
            project_weight_to_unit_norm_(child.weight, eps=eps)
            projected += 1
    return projected


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
