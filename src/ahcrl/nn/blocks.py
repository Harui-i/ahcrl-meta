"""Reusable PyTorch neural network blocks."""

import torch
from jaxtyping import Float
from torch import nn

from ahcrl.nn.modula import (
    ModulaGraphNode,
    ModularDepthwiseConv2d,
    ModularLinear,
    ModularSequential,
    mark_adaptive_parameter,
    module_to_modula_graph,
)

__all__ = ["ConvNeXtBlock"]


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

        self.depthwise = ModularDepthwiseConv2d(
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(channels)
        hidden_channels = channels * expansion
        self.pointwise = ModularSequential(
            ModularLinear(channels, hidden_channels),
            nn.GELU(),
            ModularLinear(hidden_channels, channels),
        )
        self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale_init))
        mark_adaptive_parameter(self, "layer_scale")

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        def child(name: str, module: nn.Module) -> ModulaGraphNode:
            path = f"{prefix}.{name}" if prefix else name
            return module_to_modula_graph(module, path)

        branch = ModulaGraphNode(
            "sequence",
            (
                child("depthwise", self.depthwise),
                child("norm", self.norm),
                child("pointwise", self.pointwise),
                ModulaGraphNode(
                    "atom",
                    parameter_name=(f"{prefix}.layer_scale" if prefix else "layer_scale"),
                    own_mass=1.0,
                    own_sensitivity=1.0,
                ),
            ),
        )
        return ModulaGraphNode(
            "residual",
            (ModulaGraphNode("bond", own_sensitivity=1.0), branch),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch channels H W"]
    ) -> Float[torch.Tensor, "batch channels H W"]:
        residual = x
        y = self.depthwise(x)
        y = y.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = self.pointwise(y)
        y = self.layer_scale * y
        y = y.permute(0, 3, 1, 2)
        return residual + y
