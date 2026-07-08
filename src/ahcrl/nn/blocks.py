"""Reusable PyTorch neural network blocks."""

from __future__ import annotations

import math

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


class HyperLinear(nn.Linear):
    """Bias-free linear layer whose rows are constrained to the unit sphere."""

    def __init__(self, in_features: int, out_features: int, *, eps: float = 1e-12) -> None:
        super().__init__(in_features, out_features, bias=False)
        self.eps = eps
        nn.init.orthogonal_(self.weight)
        project_weight_to_unit_norm_(self.weight, eps=eps)


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
    """Project hyperspherical linear weights in a module tree and return the count."""

    projected = 0
    for child in module.modules():
        if isinstance(child, HyperLinear):
            project_weight_to_unit_norm_(child.weight, eps=eps)
            projected += 1
    return projected


class HyperMLP(nn.Module):
    """SimbaV2 HyperMLP applied to the last dimension."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        hidden_channels = channels * expansion
        default_scale = math.sqrt(2.0 / channels) / math.sqrt(expansion)
        self.w1 = HyperLinear(channels, hidden_channels, eps=eps)
        self.scaler = Scaler(
            hidden_channels,
            init=default_scale if scaler_init is None else scaler_init,
            scale=default_scale if scaler_scale is None else scaler_scale,
        )
        self.activation = nn.ReLU()
        self.w2 = HyperLinear(hidden_channels, channels, eps=eps)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        y = self.scaler(y)
        y = self.activation(y) + self.eps
        y = self.w2(y)
        return l2_normalize(y, dim=-1, eps=self.eps)


class HyperEmbedder2d(nn.Module):
    """SimbaV2 HyperEmbedder for per-cell 2D channel features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        shift_const: float = 3.0,
        scaler_init: float | None = None,
        scaler_scale: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if shift_const <= 0.0:
            raise ValueError(f"shift_const must be positive, got {shift_const}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        default_scale = math.sqrt(2.0 / out_channels)
        self.shift = ShiftL2Norm(shift_const=shift_const, dim=-1, eps=eps)
        self.linear = HyperLinear(in_channels + 1, out_channels, eps=eps)
        self.scaler = Scaler(
            out_channels,
            init=default_scale if scaler_init is None else scaler_init,
            scale=default_scale if scaler_scale is None else scaler_scale,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        y = x.permute(0, 2, 3, 1)
        y = self.shift(y)
        y = self.scaler(self.linear(y))
        y = l2_normalize(y, dim=-1, eps=self.eps)
        return y.permute(0, 3, 1, 2)


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


class SphericalSelfAttentionLERP2d(nn.Module):
    """Global self-attention that LERPs hyperspherical per-cell features."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 4,
        max_spatial_size: int = 10,
        alpha_init: float = 0.05,
        alpha_scale: float | None = None,
        output_scaler_init: float | None = None,
        output_scaler_scale: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if channels % heads != 0:
            raise ValueError(f"channels must be divisible by heads, got {channels} and {heads}")
        if max_spatial_size <= 0:
            raise ValueError(f"max_spatial_size must be positive, got {max_spatial_size}")
        if alpha_scale == 0.0:
            raise ValueError("alpha_scale must be non-zero")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        default_output_scale = math.sqrt(2.0 / channels)
        default_alpha_scale = 1.0 / math.sqrt(channels)
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.max_spatial_size = max_spatial_size
        self.qkv = HyperLinear(channels, channels * 3, eps=eps)
        self.output = HyperLinear(channels, channels, eps=eps)
        self.output_scaler = Scaler(
            channels,
            init=default_output_scale if output_scaler_init is None else output_scaler_init,
            scale=default_output_scale if output_scaler_scale is None else output_scaler_scale,
        )
        self.alpha_scaler = Scaler(
            channels,
            init=alpha_init,
            scale=default_alpha_scale if alpha_scale is None else alpha_scale,
        )
        table_size = max_spatial_size * 2 - 1
        self.relative_position_bias = nn.Parameter(torch.zeros(heads, table_size, table_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        if height > self.max_spatial_size or width > self.max_spatial_size:
            raise ValueError(
                "input spatial size exceeds relative position bias table: "
                f"got {height}x{width}, max {self.max_spatial_size}"
            )

        tokens = height * width
        y = x.permute(0, 2, 3, 1).reshape(batch, tokens, channels)
        qkv = self.qkv(y).view(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = l2_normalize(q, dim=-1, eps=self.eps).transpose(1, 2)
        k = l2_normalize(k, dim=-1, eps=self.eps).transpose(1, 2)
        v = v.transpose(1, 2)

        logits = torch.matmul(q.float(), k.float().transpose(-2, -1))
        logits = logits * math.sqrt(self.head_dim)
        logits = logits + self._relative_position_bias(height, width).float()
        attn = torch.softmax(logits, dim=-1).to(dtype=x.dtype)
        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        target = self.output_scaler(self.output(attended))
        target = l2_normalize(target, dim=-1, eps=self.eps)
        target = target.view(batch, height, width, channels).permute(0, 3, 1, 2)
        delta = (target - x).permute(0, 2, 3, 1)
        mixed = x + self.alpha_scaler(delta).permute(0, 3, 1, 2)
        return l2_normalize(mixed, dim=1, eps=self.eps)

    def _relative_position_bias(self, height: int, width: int) -> torch.Tensor:
        y = torch.arange(height, device=self.relative_position_bias.device)
        x = torch.arange(width, device=self.relative_position_bias.device)
        coords = torch.stack(torch.meshgrid(y, x, indexing="ij")).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative_y = relative[0] + self.max_spatial_size - 1
        relative_x = relative[1] + self.max_spatial_size - 1
        bias = self.relative_position_bias[:, relative_y, relative_x]
        return bias.unsqueeze(0)


class GlobalContextLERP2d(nn.Module):
    """Broadcast pooled global context and LERP per-cell features toward it."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        alpha_init: float = 0.1,
        alpha_scale: float | None = None,
        shift_const: float = 3.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if alpha_scale == 0.0:
            raise ValueError("alpha_scale must be non-zero")
        if shift_const <= 0.0:
            raise ValueError(f"shift_const must be positive, got {shift_const}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        hidden_channels = channels * expansion
        default_hidden_scale = math.sqrt(2.0 / channels) / math.sqrt(expansion)
        default_alpha_scale = 1.0 / math.sqrt(channels)
        self.shift = ShiftL2Norm(shift_const=shift_const, dim=-1, eps=eps)
        self.w1 = HyperLinear(channels * 2 + 1, hidden_channels, eps=eps)
        self.hidden_scaler = Scaler(
            hidden_channels,
            init=default_hidden_scale,
            scale=default_hidden_scale,
        )
        self.activation = nn.ReLU()
        self.w2 = HyperLinear(hidden_channels, channels, eps=eps)
        self.alpha_scaler = Scaler(
            channels,
            init=alpha_init,
            scale=default_alpha_scale if alpha_scale is None else alpha_scale,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        avg_features = x.mean(dim=(-2, -1))
        # Avoid x.amax(dim=(-2, -1)): on CUDA bfloat16 under torch.compile, its backward
        # has produced NaN gradients through the upstream HyperEmbedder2d.
        max_features = x.flatten(2).max(dim=2).values
        global_features = torch.cat([avg_features, max_features], dim=1)
        y = self.shift(global_features)
        y = self.w1(y)
        y = self.hidden_scaler(y)
        y = self.activation(y) + self.eps
        y = self.w2(y)
        y = l2_normalize(y, dim=-1, eps=self.eps)
        target = y[:, :, None, None]
        delta = (target - x).permute(0, 2, 3, 1)
        mixed = x + self.alpha_scaler(delta).permute(0, 3, 1, 2)
        return l2_normalize(mixed, dim=1, eps=self.eps)


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
