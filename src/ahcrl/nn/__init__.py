"""Neural network modules for PyTorch models."""

from ahcrl.nn.blocks import ConvNeXtBlock
from ahcrl.nn.components import make_group_norm
from ahcrl.nn.modula import (
    ModularConv2d,
    ModularDepthwiseConv2d,
    ModularEmbedding,
    ModularLinear,
    ModularParallel,
    ModularResidual,
    ModularSequential,
    build_modula_parameter_specs,
    mark_adaptive_parameter,
)
from ahcrl.nn.trunk import make_trunk

__all__ = [
    "ConvNeXtBlock",
    "ModularConv2d",
    "ModularDepthwiseConv2d",
    "ModularEmbedding",
    "ModularLinear",
    "ModularParallel",
    "ModularResidual",
    "ModularSequential",
    "build_modula_parameter_specs",
    "make_group_norm",
    "make_trunk",
    "mark_adaptive_parameter",
]
