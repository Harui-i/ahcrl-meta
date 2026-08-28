"""Shared convolutional trunk construction for contest models."""

from collections.abc import Callable

from torch import nn

from ahcrl.nn.blocks import (
    ConvNeXtBlock,
    PerCellMLPBlock,
    ResidualBlock,
    SimbaV2Block,
    SpatialSelfAttentionBlock,
    SphericalAttentionSimbaBlock,
)
from ahcrl.nn.components import HyperEmbedder2d, make_group_norm


def make_trunk(
    *,
    block_type: str,
    in_channels: int,
    channels: int,
    blocks: int,
    block_factory: Callable[[], nn.Module] | None = None,
) -> nn.Sequential:
    """Build the shared spatial feature trunk used by contest models."""
    if block_factory is None:
        block_factory = make_block_factory(block_type, channels=channels, blocks=blocks)

    if block_type in ("simbav2_block", "spherical_attention_simba"):
        return nn.Sequential(
            HyperEmbedder2d(in_channels, channels),
            *[block_factory() for _ in range(blocks)],
        )
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
        make_group_norm(channels),
        nn.ReLU(inplace=True),
        *[block_factory() for _ in range(blocks)],
    )


def make_block_factory(block_type: str, *, channels: int, blocks: int) -> Callable[[], nn.Module]:
    """Create a factory for the supported shape-preserving feature blocks."""
    if block_type == "convnext":
        return lambda: ConvNeXtBlock(channels)
    if block_type == "per_cell_mlp":
        return lambda: PerCellMLPBlock(channels)
    if block_type == "simbav2_block":
        alpha_init = 1.0 / (blocks + 1)
        return lambda: SimbaV2Block(channels, alpha_init=alpha_init)
    if block_type == "spherical_attention_simba":
        alpha_init = 1.0 / (blocks + 1)
        return lambda: SphericalAttentionSimbaBlock(channels, alpha_init=alpha_init)
    if block_type == "residual":
        return lambda: ResidualBlock(channels)
    if block_type == "spatial-att":
        return lambda: SpatialSelfAttentionBlock(channels, heads=4)
    raise ValueError(f"unknown block_type: {block_type}")
