"""Shared ConvNeXt trunk construction for contest models."""

from torch import nn

from ahcrl.nn.blocks import ConvNeXtBlock
from ahcrl.nn.components import make_group_norm
from ahcrl.nn.modula import ModularConv2d, ModularSequential


def make_trunk(*, in_channels: int, channels: int, blocks: int) -> nn.Sequential:
    """Build the shared ConvNeXt spatial feature trunk used by contest models."""
    return ModularSequential(
        ModularConv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
        make_group_norm(channels),
        nn.ReLU(inplace=True),
        *[ConvNeXtBlock(channels, residual_branch_scale=1.0 / blocks) for _ in range(blocks)],
    )
