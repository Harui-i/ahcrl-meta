"""Neural network modules for PyTorch models."""

from ahcrl.nn.blocks import ConvNeXtBlock
from ahcrl.nn.components import make_group_norm
from ahcrl.nn.fusion_blocks import (
    Abs,
    ConvexFusion,
    MeanSubtract,
    ResidualMLPStack,
    RMSDivide,
    SequenceStack,
    SpatialStack,
    UnitGELU,
)
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
    ModularTare,
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
    "Abs",
    "ConvexFusion",
    "MeanSubtract",
    "RMSDivide",
    "ResidualMLPStack",
    "SequenceStack",
    "SpatialStack",
    "UnitGELU",
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
    "ModularTare",
    "RunningObservationNormalizer",
    "build_modula_parameter_specs",
    "make_group_norm",
    "make_trunk",
    "mark_bounded_diagonal_parameter",
    "mark_bounded_rms_parameter",
]
