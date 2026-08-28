"""Neural network modules for PyTorch models."""

from ahcrl.nn.blocks import (
    ConvNeXtBlock,
    PerCellMLPBlock,
    ResidualBlock,
    SimbaV2Block,
    SpatialSelfAttentionBlock,
    SphericalAttentionSimbaBlock,
)
from ahcrl.nn.components import (
    HyperEmbedder2d,
    HyperLinear,
    HyperMLP,
    HypersphericalFeatureNorm,
    LinearScaler,
    Scaler,
    ShiftL2Norm,
    SpatialSelfAttention2d,
    SphericalSelfAttentionLERP2d,
    l2_normalize,
    make_group_norm,
    project_hyperspherical_weights_,
    project_weight_to_unit_norm_,
)

__all__ = [
    "ConvNeXtBlock",
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
    "SpatialSelfAttention2d",
    "SpatialSelfAttentionBlock",
    "SphericalAttentionSimbaBlock",
    "SphericalSelfAttentionLERP2d",
    "l2_normalize",
    "make_group_norm",
    "project_hyperspherical_weights_",
    "project_weight_to_unit_norm_",
]
