"""Architecture-aware optimization geometry for PyTorch modules.

Convolutional geometries use a kernel-matrix approximation; they are not exact
operator norms of the spatial convolution.  This module intentionally requires
trainable parameters to be classified by semantic module type instead of shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "BoundedDiagonalGeometry",
    "BoundedRMSVectorGeometry",
    "ModularReadoutConv2d",
    "ModularReadoutLinear",
    "DEFAULT_LAYER_SCALE_RADIUS",
    "ModulaGraphNode",
    "ModulaParameterSpec",
    "ModularConv2d",
    "ModularDepthwiseConv2d",
    "ModularEmbedding",
    "ModularLinear",
    "ModularParallel",
    "ModularResidual",
    "ModularSequential",
    "build_modula_parameter_specs",
    "mark_bounded_diagonal_parameter",
    "mark_bounded_rms_parameter",
    "modula_parameter_node",
    "module_to_modula_graph",
    "validate_modula_graph",
]

DEFAULT_AFFINE_MASS = 1.0
DEFAULT_BOUND_RADIUS = 1.0
DEFAULT_VECTOR_RADIUS = 8.0
DEFAULT_NORM_GAIN_RADIUS = 1.0
DEFAULT_LAYER_SCALE_RADIUS = 1.0
DEFAULT_READOUT_GAIN_MIN = 0.125
DEFAULT_READOUT_GAIN_MAX = 8.0


def _validate_positive_finite(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _validate_nonnegative_finite(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def _polar_newton_schulz(matrix: torch.Tensor) -> torch.Tensor:
    """Approximate the polar factor with Jiacheng's fixed six-step iteration."""
    if matrix.ndim < 2:
        raise ValueError(f"polar input must contain matrices, got shape={tuple(matrix.shape)}")
    transpose = matrix.shape[-2] > matrix.shape[-1]
    x = matrix.mT if transpose else matrix
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    coefficients = (
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    )
    for a, b, c in coefficients:
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transpose else x


def _row_normalize(matrix: torch.Tensor, *, scale: float) -> torch.Tensor:
    norms = matrix.norm(dim=1, keepdim=True)
    normalized = matrix / norms.clamp_min(torch.finfo(matrix.dtype).tiny)
    return torch.where(norms > 0, normalized * scale, torch.zeros_like(matrix))


def _spectral_norm_power(matrix: torch.Tensor, *, steps: int = 4) -> torch.Tensor:
    vector = torch.ones(
        (*matrix.shape[:-2], matrix.shape[-1], 1),
        device=matrix.device,
        dtype=matrix.dtype,
    )
    vector = vector / vector.norm(dim=-2, keepdim=True)
    for _ in range(steps):
        left = matrix @ vector
        left = left / left.norm(dim=-2, keepdim=True).clamp_min(torch.finfo(matrix.dtype).tiny)
        vector = matrix.mT @ left
        vector = vector / vector.norm(dim=-2, keepdim=True).clamp_min(
            torch.finfo(matrix.dtype).tiny
        )
    return (matrix @ vector).norm(dim=(-2, -1))


@runtime_checkable
class WeightGeometry(Protocol):
    @property
    def name(self) -> str: ...

    def dualize(self, gradient: torch.Tensor, *, target_norm: float) -> torch.Tensor: ...

    def project(self, weight: torch.Tensor) -> torch.Tensor: ...

    def initialize(self, weight: torch.Tensor) -> torch.Tensor: ...

    def spectral_norm(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def orthogonality_residual(self, tensor: torch.Tensor) -> torch.Tensor: ...

    def boundary_saturation_fraction(self, tensor: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class MatrixGeometry:
    name: str = "linear"

    @staticmethod
    def _scale(matrix: torch.Tensor) -> float:
        return math.sqrt(matrix.shape[0] / matrix.shape[1])

    def dualize(self, gradient: torch.Tensor, *, target_norm: float) -> torch.Tensor:
        return _polar_newton_schulz(gradient) * (self._scale(gradient) * target_norm)

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        return _polar_newton_schulz(weight) * self._scale(weight)

    def initialize(self, weight: torch.Tensor) -> torch.Tensor:
        return self.project(torch.randn_like(weight))

    def spectral_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return _spectral_norm_power(tensor)

    def orthogonality_residual(self, tensor: torch.Tensor) -> torch.Tensor:
        q = tensor / self._scale(tensor)
        gram = q @ q.mT if q.shape[-2] <= q.shape[-1] else q.mT @ q
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        return (gram - identity).norm(dim=(-2, -1)) / math.sqrt(gram.shape[-1])

    def boundary_saturation_fraction(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=tensor.device, dtype=tensor.dtype)


@dataclass(frozen=True)
class ReadoutMatrixGeometry(MatrixGeometry):
    """Unit-spectral-norm geometry for a semantic scalar/vector readout."""

    name: str = "readout_linear"

    @staticmethod
    def _scale(matrix: torch.Tensor) -> float:
        # The output gain is a separate semantic parameter.  Keeping the
        # direction at unit norm avoids the sqrt(out/in) shrinkage of a
        # regular hidden linear layer when out=1.
        return 1.0


@dataclass(frozen=True)
class EmbeddingGeometry:
    embedding_dim: int
    name: str = "embedding"

    @property
    def scale(self) -> float:
        return math.sqrt(self.embedding_dim)

    def dualize(self, gradient: torch.Tensor, *, target_norm: float) -> torch.Tensor:
        return _row_normalize(gradient, scale=self.scale * target_norm)

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        return _row_normalize(weight, scale=self.scale)

    def initialize(self, weight: torch.Tensor) -> torch.Tensor:
        return self.project(torch.randn_like(weight))

    def spectral_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(dim=1).max()

    def orthogonality_residual(self, tensor: torch.Tensor) -> torch.Tensor:
        active = tensor.norm(dim=1) > 0
        if not bool(active.any().item()):
            return torch.zeros((), device=tensor.device, dtype=tensor.dtype)
        return (tensor[active].norm(dim=1) / self.scale - 1.0).abs().mean()

    def boundary_saturation_fraction(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=tensor.device, dtype=tensor.dtype)


@dataclass(frozen=True)
class BoundedRMSVectorGeometry:
    """RMS geometry for bias and normalization shifts, bounded in an RMS ball."""

    radius: float = DEFAULT_BOUND_RADIUS
    name: str = "bounded_rms_vector"

    def __post_init__(self) -> None:
        _validate_positive_finite(self.radius, name="radius")

    def dualize(self, gradient: torch.Tensor, *, target_norm: float) -> torch.Tensor:
        rms = gradient.square().mean().sqrt()
        normalized = gradient / rms.clamp_min(torch.finfo(gradient.dtype).tiny)
        return torch.where(rms > 0, normalized * target_norm, torch.zeros_like(gradient))

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        rms = weight.square().mean().sqrt()
        scale = self.radius / rms.clamp_min(torch.finfo(weight.dtype).tiny)
        return weight * scale.clamp(max=1.0)

    def initialize(self, weight: torch.Tensor) -> torch.Tensor:
        return self.project(weight)

    def spectral_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.square().mean().sqrt()

    def orthogonality_residual(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=tensor.device, dtype=tensor.dtype)

    def boundary_saturation_fraction(self, tensor: torch.Tensor) -> torch.Tensor:
        rms = tensor.square().mean().sqrt()
        return (rms >= self.radius * (1.0 - 1e-6)).to(tensor.dtype)


@dataclass(frozen=True)
class BoundedDiagonalGeometry:
    """L-infinity geometry for gains and elementwise residual gates."""

    radius: float = DEFAULT_BOUND_RADIUS
    center: float = 0.0
    name: str = "bounded_diagonal"

    def __post_init__(self) -> None:
        _validate_positive_finite(self.radius, name="radius")
        if not math.isfinite(self.center):
            raise ValueError(f"center must be finite, got {self.center}")

    def dualize(self, gradient: torch.Tensor, *, target_norm: float) -> torch.Tensor:
        return gradient.sign() * min(target_norm, self.radius)

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.clamp(min=self.center - self.radius, max=self.center + self.radius)

    def initialize(self, weight: torch.Tensor) -> torch.Tensor:
        return self.project(weight)

    def spectral_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.abs().max()

    def orthogonality_residual(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=tensor.device, dtype=tensor.dtype)

    def boundary_saturation_fraction(self, tensor: torch.Tensor) -> torch.Tensor:
        distance = (tensor - self.center).abs()
        return (distance >= self.radius * (1.0 - 1e-6)).to(tensor.dtype).mean()


@dataclass(frozen=True)
class ConvKernelGeometry:
    groups: int
    out_channels: int
    depthwise: bool = False
    name: str = "conv"

    def _matrices(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"convolution weight must be OIHW, got {tuple(tensor.shape)}")
        if self.depthwise:
            return tensor.reshape(tensor.shape[0], 1, -1)
        out_per_group = tensor.shape[0] // self.groups
        return tensor.reshape(self.groups, out_per_group, -1)

    def _restore(self, matrices: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return matrices.reshape(reference.shape)

    def dualize(self, gradient: torch.Tensor, *, target_norm: float) -> torch.Tensor:
        matrices = self._matrices(gradient)
        scale = math.sqrt(matrices.shape[-2] / matrices.shape[-1]) * target_norm
        dualized = _polar_newton_schulz(matrices) * scale
        return self._restore(dualized, gradient)

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        matrices = self._matrices(weight)
        scale = math.sqrt(matrices.shape[-2] / matrices.shape[-1])
        projected = _polar_newton_schulz(matrices) * scale
        return self._restore(projected, weight)

    def initialize(self, weight: torch.Tensor) -> torch.Tensor:
        return self.project(torch.randn_like(weight))

    def spectral_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return _spectral_norm_power(self._matrices(tensor)).max()

    def orthogonality_residual(self, tensor: torch.Tensor) -> torch.Tensor:
        matrices = self._matrices(tensor)
        scale = math.sqrt(matrices.shape[-2] / matrices.shape[-1])
        q = matrices / scale
        gram = q @ q.mT if q.shape[-2] <= q.shape[-1] else q.mT @ q
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        residual = (gram - identity).norm(dim=(-2, -1)) / math.sqrt(gram.shape[-1])
        return residual.mean()

    def boundary_saturation_fraction(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=tensor.device, dtype=tensor.dtype)


@dataclass(frozen=True)
class ReadoutConvGeometry(ConvKernelGeometry):
    """Unit row/spectral geometry for a semantic convolutional readout."""

    name: str = "readout_conv"

    def dualize(self, gradient: torch.Tensor, *, target_norm: float) -> torch.Tensor:
        matrices = self._matrices(gradient)
        dualized = _polar_newton_schulz(matrices) * target_norm
        return self._restore(dualized, gradient)

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        return self._restore(_polar_newton_schulz(self._matrices(weight)), weight)


class _ModularAtom:
    weight: nn.Parameter
    bias: nn.Parameter | None
    geometry: WeightGeometry
    mass: float
    sensitivity: float

    def _init_modula_atom(
        self, *, geometry: WeightGeometry, mass: float, sensitivity: float
    ) -> None:
        self.geometry = geometry
        self.mass = _validate_nonnegative_finite(mass, name="mass")
        self.sensitivity = _validate_positive_finite(sensitivity, name="sensitivity")


class ModularLinear(_ModularAtom, nn.Linear):
    def __init__(
        self,
        *args: object,
        mass: float = 1.0,
        sensitivity: float = 1.0,
        **kwargs: object,
    ) -> None:
        nn.Linear.__init__(self, *args, **kwargs)  # type: ignore[arg-type]
        self._init_modula_atom(geometry=MatrixGeometry(), mass=mass, sensitivity=sensitivity)


class ModularReadoutLinear(_ModularAtom, nn.Linear):
    """Semantic readout with unit-norm direction and an explicit output gain."""

    def __init__(
        self,
        *args: object,
        mass: float = 1.0,
        sensitivity: float = 1.0,
        gain_init: float = 1.0,
        **kwargs: object,
    ) -> None:
        nn.Linear.__init__(self, *args, **kwargs)  # type: ignore[arg-type]
        self._init_modula_atom(geometry=ReadoutMatrixGeometry(), mass=mass, sensitivity=sensitivity)
        self.output_gain = nn.Parameter(torch.tensor(float(gain_init)))
        _mark_bounded_parameter(
            self,
            "output_gain",
            geometry=BoundedDiagonalGeometry(
                radius=(DEFAULT_READOUT_GAIN_MAX - DEFAULT_READOUT_GAIN_MIN) / 2.0,
                center=(DEFAULT_READOUT_GAIN_MAX + DEFAULT_READOUT_GAIN_MIN) / 2.0,
            ),
            mass=DEFAULT_AFFINE_MASS,
            sensitivity=DEFAULT_READOUT_GAIN_MAX,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias) * self.output_gain


class ModularEmbedding(_ModularAtom, nn.Embedding):
    def __init__(
        self,
        *args: object,
        mass: float = 1.0,
        sensitivity: float = 1.0,
        **kwargs: object,
    ) -> None:
        nn.Embedding.__init__(self, *args, **kwargs)  # type: ignore[arg-type]
        self._init_modula_atom(
            geometry=EmbeddingGeometry(self.embedding_dim), mass=mass, sensitivity=sensitivity
        )

    def forward_one_hot(self, one_hot: torch.Tensor) -> torch.Tensor:
        if one_hot.shape[-1] != self.num_embeddings:
            raise ValueError(
                f"one-hot width must be {self.num_embeddings}, got {one_hot.shape[-1]}"
            )
        return one_hot @ self.weight


class ModularConv2d(_ModularAtom, nn.Conv2d):
    def __init__(
        self,
        *args: object,
        mass: float = 1.0,
        sensitivity: float = 1.0,
        **kwargs: object,
    ) -> None:
        nn.Conv2d.__init__(self, *args, **kwargs)  # type: ignore[arg-type]
        if (
            self.groups > 1
            and self.groups == self.in_channels
            and self.out_channels == self.in_channels
        ):
            raise ValueError("use ModularDepthwiseConv2d for depthwise convolution")
        self._init_modula_atom(
            geometry=ConvKernelGeometry(self.groups, self.out_channels),
            mass=mass,
            sensitivity=sensitivity,
        )


class ModularReadoutConv2d(_ModularAtom, nn.Conv2d):
    """Semantic convolutional readout with an explicit positive output gain."""

    def __init__(
        self,
        *args: object,
        mass: float = 1.0,
        sensitivity: float = 1.0,
        gain_init: float = 1.0,
        **kwargs: object,
    ) -> None:
        nn.Conv2d.__init__(self, *args, **kwargs)  # type: ignore[arg-type]
        self._init_modula_atom(
            geometry=ReadoutConvGeometry(self.groups, self.out_channels),
            mass=mass,
            sensitivity=sensitivity,
        )
        self.output_gain = nn.Parameter(torch.tensor(float(gain_init)))
        _mark_bounded_parameter(
            self,
            "output_gain",
            geometry=BoundedDiagonalGeometry(
                radius=(DEFAULT_READOUT_GAIN_MAX - DEFAULT_READOUT_GAIN_MIN) / 2.0,
                center=(DEFAULT_READOUT_GAIN_MAX + DEFAULT_READOUT_GAIN_MIN) / 2.0,
            ),
            mass=DEFAULT_AFFINE_MASS,
            sensitivity=DEFAULT_READOUT_GAIN_MAX,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            * self.output_gain
        )


class ModularDepthwiseConv2d(_ModularAtom, nn.Conv2d):
    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int],
        *,
        mass: float = 1.0,
        sensitivity: float = 1.0,
        **kwargs: object,
    ) -> None:
        if "groups" in kwargs or "in_channels" in kwargs or "out_channels" in kwargs:
            raise ValueError("depthwise channels determine in_channels, out_channels and groups")
        nn.Conv2d.__init__(
            self,
            channels,
            channels,
            kernel_size,
            groups=channels,
            **kwargs,  # type: ignore[arg-type]
        )
        self._init_modula_atom(
            geometry=ConvKernelGeometry(channels, channels, depthwise=True, name="depthwise_conv"),
            mass=mass,
            sensitivity=sensitivity,
        )


@dataclass(frozen=True)
class ModulaGraphNode:
    kind: str
    children: tuple[ModulaGraphNode, ...] = ()
    parameter_name: str | None = None
    own_mass: float = 0.0
    own_sensitivity: float = 1.0

    @property
    def mass(self) -> float:
        if self.kind == "atom":
            return self.own_mass
        return sum((child.mass for child in self.children), 0.0)

    @property
    def sensitivity(self) -> float:
        if self.kind in {"atom", "bond", "parameter_group"}:
            return self.own_sensitivity
        if self.kind == "sequence":
            return math.prod(child.sensitivity for child in self.children)
        if self.kind in {"parallel", "residual"}:
            return sum((child.sensitivity for child in self.children), 0.0)
        raise ValueError(f"unknown Modula graph node kind: {self.kind}")

    def allocate(self, target_norm: float = 1.0) -> dict[str, float]:
        if self.kind == "bond":
            return {}
        if self.kind == "atom":
            if self.parameter_name is None:
                raise ValueError("atom node is missing parameter_name")
            return {self.parameter_name: target_norm if self.own_mass > 0 else 0.0}
        if not self.children:
            return {}
        allocations: dict[str, float] = {}
        if self.mass <= 0:
            child_targets = ((child, 0.0) for child in self.children)
        elif self.kind == "sequence":
            targets = []
            for index, child in enumerate(self.children):
                downstream = math.prod(item.sensitivity for item in self.children[index + 1 :])
                child_target = target_norm * child.mass / self.mass / downstream
                targets.append((child, child_target))
            child_targets = iter(targets)
        elif self.kind in {"parallel", "residual", "parameter_group"}:
            child_targets = (
                (child, target_norm * child.mass / self.mass) for child in self.children
            )
        else:
            raise ValueError(f"unknown Modula graph node kind: {self.kind}")
        for child, child_target in child_targets:
            child_allocations = child.allocate(child_target)
            duplicates = set(allocations) & set(child_allocations)
            if duplicates:
                raise ValueError(
                    "duplicate parameters in Modula graph: " + ", ".join(sorted(duplicates))
                )
            allocations.update(child_allocations)
        return allocations


class ModularSequential(nn.Sequential):
    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        return ModulaGraphNode(
            "sequence",
            tuple(
                module_to_modula_graph(module, _join_name(prefix, name))
                for name, module in self.named_children()
            ),
        )


class ModularResidual(nn.Module):
    def __init__(self, branch: nn.Module, *, branch_scale: float = 1.0) -> None:
        super().__init__()
        if not 0.0 < branch_scale <= 1.0:
            raise ValueError(f"branch_scale must be in (0, 1], got {branch_scale}")
        self.branch = branch
        self.branch_scale = branch_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.branch_scale) * x + self.branch_scale * self.branch(x)

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        branch = module_to_modula_graph(self.branch, _join_name(prefix, "branch"))
        if self.branch_scale != 1.0:
            branch = ModulaGraphNode(
                "sequence",
                (branch, ModulaGraphNode("bond", own_sensitivity=self.branch_scale)),
            )
        children = (branch,)
        if self.branch_scale != 1.0:
            children = (ModulaGraphNode("bond", own_sensitivity=1.0 - self.branch_scale), branch)
        return ModulaGraphNode(
            "residual",
            children,
        )


class ModularParallel(nn.Module):
    def __init__(self, *branches: nn.Module) -> None:
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(branch(x) for branch in self.branches)

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        return ModulaGraphNode(
            "parallel",
            tuple(
                module_to_modula_graph(branch, _join_name(prefix, f"branches.{index}"))
                for index, branch in enumerate(self.branches)
            ),
        )


def _join_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def module_to_modula_graph(module: nn.Module, prefix: str = "") -> ModulaGraphNode:
    if isinstance(module, _ModularAtom):
        weight = ModulaGraphNode(
            "atom",
            parameter_name=_join_name(prefix, "weight"),
            own_mass=module.mass,
            own_sensitivity=module.sensitivity,
        )
        parameters: list[ModulaGraphNode] = [weight]
        declarations = _bounded_parameter_declarations(module)
        for name, parameter in module.named_parameters(recurse=False):
            if name == "weight" or not parameter.requires_grad:
                continue
            if name not in declarations:
                raise ValueError(f"parameter {name} has no bounded Modula declaration")
            parameters.append(modula_parameter_node(module, name, prefix))
        if len(parameters) == 1:
            return weight
        group_sensitivity = max(
            module.sensitivity,
            *(declarations[name].sensitivity for name in declarations if name != "weight"),
        )
        return ModulaGraphNode(
            "parameter_group", tuple(parameters), own_sensitivity=group_sensitivity
        )
    if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        norm_parameters: tuple[ModulaGraphNode, ...] = tuple(
            modula_parameter_node(module, name, prefix)
            for name, parameter in module.named_parameters(recurse=False)
            if parameter.requires_grad
        )
        if not norm_parameters:
            return ModulaGraphNode("bond", own_sensitivity=1.0)
        return ModulaGraphNode("parameter_group", norm_parameters, own_sensitivity=1.0)
    node_factory = getattr(module, "modula_node", None)
    if callable(node_factory):
        node = node_factory(prefix)
        if not isinstance(node, ModulaGraphNode):
            raise TypeError("modula_node() must return ModulaGraphNode")
        return node
    children = tuple(
        module_to_modula_graph(child, _join_name(prefix, name))
        for name, child in module.named_children()
    )
    declarations = _bounded_parameter_declarations(module)
    declared = tuple(
        modula_parameter_node(module, name, prefix)
        for name, parameter in module.named_parameters(recurse=False)
        if name in declarations and parameter.requires_grad
    )
    children = declared + children
    if children:
        return ModulaGraphNode("sequence", children)
    return ModulaGraphNode("bond", own_sensitivity=1.0)


def validate_modula_graph(graph: ModulaGraphNode) -> float:
    """Validate a Modula graph and return its declared input sensitivity.

    The returned value is the graph's structural input-to-output sensitivity:
    sequences multiply sensitivities while parallel and residual branches add
    them.  It is the value used by ``allocate()``, not a data-dependent
    Jacobian measurement of the initialized PyTorch model.
    """

    def visit(node: ModulaGraphNode) -> float:
        if node.kind not in {"atom", "bond", "parameter_group", "sequence", "parallel", "residual"}:
            raise ValueError(f"unknown Modula graph node kind: {node.kind}")
        if node.kind == "atom" and node.parameter_name is None:
            raise ValueError("atom node is missing parameter_name")
        if node.kind in {"atom", "bond", "parameter_group"}:
            return _validate_positive_finite(node.own_sensitivity, name=f"{node.kind} sensitivity")

        child_sensitivities = tuple(visit(child) for child in node.children)
        sensitivity = (
            math.prod(child_sensitivities)
            if node.kind == "sequence"
            else sum(child_sensitivities, 0.0)
        )
        return _validate_positive_finite(sensitivity, name=f"{node.kind} sensitivity")

    return visit(graph)


@dataclass(frozen=True)
class _DeclaredParameterGeometry:
    geometry: WeightGeometry
    mass: float
    sensitivity: float


def _mark_bounded_parameter(
    module: nn.Module,
    parameter_name: str,
    *,
    geometry: WeightGeometry,
    mass: float,
    sensitivity: float,
) -> None:
    parameter = module._parameters.get(parameter_name)
    if parameter is None:
        raise ValueError(f"module has no direct parameter named {parameter_name}")
    declarations = dict(getattr(module, "_modula_parameter_geometries", {}))
    declarations[parameter_name] = _DeclaredParameterGeometry(
        geometry=geometry,
        mass=_validate_nonnegative_finite(mass, name="mass"),
        sensitivity=_validate_positive_finite(sensitivity, name="sensitivity"),
    )
    object.__setattr__(module, "_modula_parameter_geometries", declarations)


def mark_bounded_rms_parameter(
    module: nn.Module,
    parameter_name: str,
    *,
    mass: float = DEFAULT_AFFINE_MASS,
    radius: float = DEFAULT_BOUND_RADIUS,
    sensitivity: float = 1.0,
) -> None:
    """Classify a direct parameter as a bounded RMS vector."""

    _mark_bounded_parameter(
        module,
        parameter_name,
        geometry=BoundedRMSVectorGeometry(radius=radius),
        mass=mass,
        sensitivity=sensitivity,
    )


def mark_bounded_diagonal_parameter(
    module: nn.Module,
    parameter_name: str,
    *,
    mass: float = DEFAULT_AFFINE_MASS,
    radius: float = DEFAULT_BOUND_RADIUS,
    center: float = 0.0,
    sensitivity: float = 1.0,
) -> None:
    """Classify a direct parameter as a bounded diagonal operator."""

    _mark_bounded_parameter(
        module,
        parameter_name,
        geometry=BoundedDiagonalGeometry(radius=radius, center=center),
        mass=mass,
        sensitivity=sensitivity,
    )


def _bounded_parameter_declarations(
    module: nn.Module,
) -> dict[str, _DeclaredParameterGeometry]:
    declarations: dict[str, _DeclaredParameterGeometry] = dict(
        getattr(module, "_modula_parameter_geometries", {})
    )
    bias = module._parameters.get("bias") if isinstance(module, _ModularAtom) else None
    if bias is not None and bias.requires_grad:
        declarations.setdefault(
            "bias",
            _DeclaredParameterGeometry(
                BoundedRMSVectorGeometry(radius=DEFAULT_VECTOR_RADIUS),
                DEFAULT_AFFINE_MASS,
                1.0,
            ),
        )
    if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        if module.weight is not None and module.weight.requires_grad:
            declarations.setdefault(
                "weight",
                _DeclaredParameterGeometry(
                    BoundedDiagonalGeometry(radius=DEFAULT_NORM_GAIN_RADIUS, center=1.0),
                    DEFAULT_AFFINE_MASS,
                    1.0,
                ),
            )
        if module.bias is not None and module.bias.requires_grad:
            declarations.setdefault(
                "bias",
                _DeclaredParameterGeometry(
                    BoundedRMSVectorGeometry(radius=DEFAULT_VECTOR_RADIUS),
                    DEFAULT_AFFINE_MASS,
                    1.0,
                ),
            )
    return declarations


def modula_parameter_node(
    module: nn.Module, parameter_name: str, prefix: str = ""
) -> ModulaGraphNode:
    declaration = _bounded_parameter_declarations(module).get(parameter_name)
    if declaration is None:
        raise ValueError(f"parameter {parameter_name} has no bounded Modula declaration")
    return ModulaGraphNode(
        "atom",
        parameter_name=_join_name(prefix, parameter_name),
        own_mass=declaration.mass,
        own_sensitivity=declaration.sensitivity,
    )


@dataclass(frozen=True)
class ModulaParameterSpec:
    name: str
    parameter: nn.Parameter
    role: str
    geometry: WeightGeometry | None
    target_norm: float


def build_modula_parameter_specs(
    model: nn.Module, graph: ModulaGraphNode | None = None
) -> tuple[ModulaParameterSpec, ...]:
    modular: dict[str, tuple[nn.Parameter, WeightGeometry]] = {}
    bounded: dict[str, tuple[nn.Parameter, WeightGeometry]] = {}
    for module_name, module in model.named_modules():
        direct = dict(module.named_parameters(recurse=False))
        if isinstance(module, _ModularAtom):
            modular[_join_name(module_name, "weight")] = (module.weight, module.geometry)
            direct.pop("weight", None)
        declarations = _bounded_parameter_declarations(cast(nn.Module, module))
        for name, declaration in declarations.items():
            parameter = direct.pop(name, None)
            if parameter is None:
                raise ValueError(
                    f"declared Modula parameter is not a direct parameter: "
                    f"{_join_name(module_name, name)}"
                )
            if parameter.requires_grad:
                bounded[_join_name(module_name, name)] = (parameter, declaration.geometry)
        remaining = [name for name, parameter in direct.items() if parameter.requires_grad]
        if remaining:
            qualified = [_join_name(module_name, name) for name in remaining]
            raise ValueError("unclassified trainable parameters: " + ", ".join(qualified))

    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    classified = set(modular) | set(bounded)
    if actual != classified:
        missing = sorted(actual - classified)
        extra = sorted(classified - actual)
        raise ValueError(f"parameter classification mismatch: missing={missing}, extra={extra}")
    if set(modular) & set(bounded):
        raise ValueError("parameters cannot have both matrix and bounded geometry")

    if graph is None:
        graph_factory = getattr(model, "modula_graph", None)
        candidate = graph_factory() if callable(graph_factory) else module_to_modula_graph(model)
        if not isinstance(candidate, ModulaGraphNode):
            raise TypeError("modula_graph() must return ModulaGraphNode")
        graph = candidate
    validate_modula_graph(graph)
    allocations = graph.allocate()
    classified_geometries = set(modular) | set(bounded)
    if set(allocations) != classified_geometries:
        raise ValueError(
            "Modula graph mismatch: "
            f"missing={sorted(classified_geometries - set(allocations))}, "
            f"extra={sorted(set(allocations) - classified_geometries)}"
        )

    specs = [
        ModulaParameterSpec(name, parameter, "modular", geometry, allocations[name])
        for name, (parameter, geometry) in modular.items()
    ]
    specs.extend(
        ModulaParameterSpec(name, parameter, "bounded", geometry, allocations[name])
        for name, (parameter, geometry) in bounded.items()
    )
    return tuple(sorted(specs, key=lambda spec: spec.name))
