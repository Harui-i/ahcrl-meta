"""Small Modula-aware blocks used by the AHC063 slot-fusion model."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from ahcrl.nn.modula import (
    ModulaGraphNode,
    ModularConv2d,
    ModularDepthwiseConv2d,
    ModularLinear,
    ModularResidual,
    ModularSequential,
    ModularTare,
    module_to_modula_graph,
)


class _UnitBond(nn.Module):
    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        return ModulaGraphNode("bond", own_sensitivity=1.0)


class UnitGELU(_UnitBond):
    """GELU with Modula's unit-sensitivity normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x) / 1.1289


class RMSDivide(_UnitBond):
    def __init__(self, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x.square(), dim=1, keepdim=True) + self.eps)
        return x / rms


class MeanSubtract(_UnitBond):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=1, keepdim=True)


class Abs(_UnitBond):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs()


class ConvexFusion(nn.Module):
    """A parameter-free convex sum with an explicit Modula graph bond."""

    def __init__(self, *branches: nn.Module) -> None:
        super().__init__()
        if not branches:
            raise ValueError("ConvexFusion requires at least one branch")
        self.branches = nn.ModuleList(branches)
        self.weight = 1.0 / len(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.branches[0](x) * self.weight
        for branch in self.branches[1:]:
            output = output + branch(x) * self.weight
        return output

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        children = []
        for index, branch in enumerate(self.branches):
            node = module_to_modula_graph(
                branch, f"{prefix}.branches.{index}" if prefix else f"branches.{index}"
            )
            children.append(
                ModulaGraphNode(
                    "sequence", (ModulaGraphNode("bond", own_sensitivity=self.weight), node)
                )
            )
        return ModulaGraphNode("parallel", tuple(children))


class SequenceResidue(nn.Module):
    """Official Modula ResNet-style bidirectional sequence convolution."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.layers = ModularSequential(
            RMSDivide(),
            ModularConv2d(
                channels,
                channels,
                kernel_size=(1, 3),
                padding=(0, dilation),
                dilation=(1, dilation),
                bias=False,
            ),
            Abs(),
            MeanSubtract(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SequenceStack(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("sequence stack must contain at least one block")
        self.stack = ModularTare(
            ModularSequential(
                *(
                    ModularResidual(
                        SequenceResidue(channels, 2**index),
                        branch_scale=1.0 / blocks,
                    )
                    for index in range(blocks)
                )
            ),
            absolute_mass=5.0,
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Conv2d sees (batch, channel, singleton-height, sequence-length).
        x = x.transpose(1, 2).unsqueeze(2)
        valid = valid[:, None, None, :].to(dtype=x.dtype)
        x = x * valid
        for block in cast(ModularSequential, self.stack.module):
            x = block(x) * valid
        return x.squeeze(2).transpose(1, 2)

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        return module_to_modula_graph(self.stack, f"{prefix}.stack" if prefix else "stack")


class SpatialConvNeXtResidue(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        hidden = channels * 4
        self.depthwise = ModularDepthwiseConv2d(
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = ModularLinear(channels, hidden, bias=False)
        self.gelu = UnitGELU()
        self.project = ModularLinear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = self.gelu(x)
        x = self.project(x)
        return x.permute(0, 3, 1, 2)


class SpatialStack(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("spatial stack must contain at least one block")
        self.stack = ModularTare(
            ModularSequential(
                *(
                    ModularResidual(
                        SpatialConvNeXtResidue(channels, 2**index),
                        branch_scale=1.0 / blocks,
                    )
                    for index in range(blocks)
                )
            ),
            absolute_mass=5.0,
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid[:, None].to(dtype=x.dtype)
        x = x * valid
        for block in cast(ModularSequential, self.stack.module):
            x = block(x) * valid
        return x

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        return module_to_modula_graph(self.stack, f"{prefix}.stack" if prefix else "stack")


class ResidualMLPStack(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("MLP stack must contain at least one block")
        self.stack = ModularTare(
            ModularSequential(
                *(
                    ModularResidual(
                        ModularSequential(
                            ModularLinear(channels, channels, bias=False),
                            UnitGELU(),
                            ModularLinear(channels, channels, bias=False),
                        ),
                        branch_scale=0.5,
                    )
                    for _ in range(blocks)
                )
            ),
            absolute_mass=5.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        return module_to_modula_graph(self.stack, f"{prefix}.stack" if prefix else "stack")


__all__ = [
    "Abs",
    "ConvexFusion",
    "MeanSubtract",
    "RMSDivide",
    "ResidualMLPStack",
    "SequenceStack",
    "SpatialStack",
    "UnitGELU",
]
