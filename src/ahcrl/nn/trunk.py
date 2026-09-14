"""Shared ConvNeXt trunk construction for contest models."""

from torch import nn

from ahcrl.nn.blocks import ConvNeXtBlock
from ahcrl.nn.components import make_group_norm


def make_trunk(*, in_channels: int, channels: int, blocks: int) -> nn.Sequential:
    """Build the shared ConvNeXt spatial feature trunk used by contest models."""
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
        make_group_norm(channels),
        nn.ReLU(inplace=True),
        *[ConvNeXtBlock(channels) for _ in range(blocks)],
    )
