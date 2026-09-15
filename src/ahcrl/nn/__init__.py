"""Neural network modules for PyTorch models."""

from ahcrl.nn.blocks import ConvNeXtBlock
from ahcrl.nn.components import make_group_norm
from ahcrl.nn.modula import (
    BoundedDiagonalGeometry,
    BoundedRMSVectorGeometry,
    ModularConv2d,
    ModularDepthwiseConv2d,
    ModularEmbedding,
    ModularLinear,
    ModularParallel,
    ModularReadoutConv2d,
    ModularReadoutLinear,
    ModularResidual,
    ModularSequential,
    build_modula_parameter_specs,
    mark_bounded_diagonal_parameter,
    mark_bounded_rms_parameter,
)
from ahcrl.nn.observation import (
    CategoricalPlaneAdapter,
    CategoricalPlaneGroup,
    RunningObservationNormalizer,
)
from ahcrl.nn.trunk import make_trunk

__all__ = [
    "ConvNeXtBlock",
    "CategoricalPlaneAdapter",
    "CategoricalPlaneGroup",
    "BoundedDiagonalGeometry",
    "BoundedRMSVectorGeometry",
    "ModularConv2d",
    "ModularDepthwiseConv2d",
    "ModularEmbedding",
    "ModularLinear",
    "ModularReadoutConv2d",
    "ModularReadoutLinear",
    "ModularParallel",
    "ModularResidual",
    "ModularSequential",
    "RunningObservationNormalizer",
    "build_modula_parameter_specs",
    "make_group_norm",
    "make_trunk",
    "mark_bounded_diagonal_parameter",
    "mark_bounded_rms_parameter",
]
