"""Architecture-aware optimization geometry for PyTorch modules.

Convolutional geometries use a kernel-matrix approximation; they are not exact
operator norms of the spatial convolution.  This module intentionally requires
trainable parameters to be classified by semantic module type instead of shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn

__all__ = [
    "AdaptiveRMSGeometry",
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
    "mark_adaptive_parameter",
    "module_to_modula_graph",
]


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


@dataclass(frozen=True)
class AdaptiveRMSGeometry:
    """RMS geometry for bias, normalization affine, and residual gates.

    Adam moments determine the direction while this geometry gives that
    direction a parameterization-independent RMS update budget.  These
    parameters keep their module-defined initialization and are not projected.
    """

    name: str = "adaptive_rms"

    def dualize(self, gradient: torch.Tensor, *, target_norm: float) -> torch.Tensor:
        rms = gradient.square().mean().sqrt()
        normalized = gradient / rms.clamp_min(torch.finfo(gradient.dtype).tiny)
        return torch.where(rms > 0, normalized * target_norm, torch.zeros_like(gradient))

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def initialize(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def spectral_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.square().mean().sqrt()

    def orthogonality_residual(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=tensor.device, dtype=tensor.dtype)


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
    def __init__(self, branch: nn.Module) -> None:
        super().__init__()
        self.branch = branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        return ModulaGraphNode(
            "residual",
            (
                ModulaGraphNode("bond", own_sensitivity=1.0),
                module_to_modula_graph(self.branch, _join_name(prefix, "branch")),
            ),
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
        if module.bias is None or not module.bias.requires_grad:
            return weight
        bias = ModulaGraphNode(
            "atom",
            parameter_name=_join_name(prefix, "bias"),
            own_mass=module.mass,
        )
        return ModulaGraphNode(
            "parameter_group", (weight, bias), own_sensitivity=module.sensitivity
        )
    if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        parameters = tuple(
            ModulaGraphNode(
                "atom",
                parameter_name=_join_name(prefix, name),
                own_mass=1.0,
            )
            for name, parameter in module.named_parameters(recurse=False)
            if parameter.requires_grad
        )
        if not parameters:
            return ModulaGraphNode("bond", own_sensitivity=1.0)
        return ModulaGraphNode("parameter_group", parameters, own_sensitivity=1.0)
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
    if children:
        return ModulaGraphNode("sequence", children)
    return ModulaGraphNode("bond", own_sensitivity=1.0)


def mark_adaptive_parameter(module: nn.Module, parameter_name: str) -> None:
    parameter = module._parameters.get(parameter_name)
    if parameter is None:
        raise ValueError(f"module has no direct parameter named {parameter_name}")
    names = set(getattr(module, "_modula_adaptive_parameters", set()))
    names.add(parameter_name)
    object.__setattr__(module, "_modula_adaptive_parameters", names)


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
    adaptive: dict[str, nn.Parameter] = {}
    for module_name, module in model.named_modules():
        direct = dict(module.named_parameters(recurse=False))
        if isinstance(module, _ModularAtom):
            modular[_join_name(module_name, "weight")] = (module.weight, module.geometry)
            bias = direct.get("bias")
            if bias is not None and bias.requires_grad:
                adaptive[_join_name(module_name, "bias")] = bias
            direct.pop("weight", None)
            direct.pop("bias", None)
        declared_adaptive = getattr(module, "_modula_adaptive_parameters", set())
        if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            declared_adaptive = declared_adaptive | set(direct)
        for name in declared_adaptive:
            if name in direct:
                adaptive[_join_name(module_name, name)] = direct.pop(name)
        remaining = [name for name, parameter in direct.items() if parameter.requires_grad]
        if remaining:
            qualified = [_join_name(module_name, name) for name in remaining]
            raise ValueError("unclassified trainable parameters: " + ", ".join(qualified))

    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    classified = set(modular) | set(adaptive)
    if actual != classified:
        missing = sorted(actual - classified)
        extra = sorted(classified - actual)
        raise ValueError(f"parameter classification mismatch: missing={missing}, extra={extra}")
    if set(modular) & set(adaptive):
        raise ValueError("parameters cannot have both matrix and adaptive RMS geometry")

    if graph is None:
        graph_factory = getattr(model, "modula_graph", None)
        candidate = graph_factory() if callable(graph_factory) else module_to_modula_graph(model)
        if not isinstance(candidate, ModulaGraphNode):
            raise TypeError("modula_graph() must return ModulaGraphNode")
        graph = candidate
    allocations = graph.allocate()
    classified_geometries = set(modular) | set(adaptive)
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
        ModulaParameterSpec(name, parameter, "adaptive", AdaptiveRMSGeometry(), allocations[name])
        for name, parameter in adaptive.items()
    )
    return tuple(sorted(specs, key=lambda spec: spec.name))
