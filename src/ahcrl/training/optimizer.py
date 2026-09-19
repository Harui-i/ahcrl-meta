"""Shared fp32 master weights and architecture-aware optimizers."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from ahcrl.nn.modula import ModulaParameterSpec, build_modula_parameter_specs

MODULA_OPTIMIZER_FORMAT_VERSION = 4
_BOUNDED_BETA1 = 0.9
_BOUNDED_BETA2 = 0.999
_BOUNDED_EPSILON = 1e-8


@runtime_checkable
class OptimizerLike(Protocol):
    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def step(self) -> Any: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class FP32MasterWeights:
    """Keep named fp32 optimizer parameters synchronized with a model."""

    def __init__(self, model: nn.Module) -> None:
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.names = [name for name, _ in named]
        self.model_parameters = [parameter for _, parameter in named]
        self.parameters = [
            nn.Parameter(parameter.detach().float().clone()) for parameter in self.model_parameters
        ]

    def named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(zip(self.names, self.parameters, strict=True))

    @torch.no_grad()
    def copy_model_to_master(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.parameters, strict=True
        ):
            master_parameter.copy_(model_parameter.float())

    @torch.no_grad()
    def copy_master_to_model(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.parameters, strict=True
        ):
            model_parameter.copy_(master_parameter.to(dtype=model_parameter.dtype))

    def copy_gradients_from_model(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.parameters, strict=True
        ):
            master_parameter.grad = (
                None
                if model_parameter.grad is None
                else model_parameter.grad.detach().float().clone()
            )

    def first_nonfinite_gradient(self) -> str | None:
        for name, parameter in self.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                return name
        return None

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "parameters": [parameter.detach().cpu().clone() for parameter in self.parameters],
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any] | list[torch.Tensor]) -> None:
        # The list form keeps existing AdamW checkpoints resumable.
        if isinstance(state, list):
            saved_names = self.names
            saved_parameters = state
        else:
            saved_names = state.get("names")
            saved_parameters = state.get("parameters")
        if saved_names != self.names:
            raise ValueError("master weight parameter names mismatch")
        if not isinstance(saved_parameters, list) or len(saved_parameters) != len(self.parameters):
            raise ValueError("master weight count mismatch")
        for parameter, saved in zip(self.parameters, saved_parameters, strict=True):
            if not isinstance(saved, torch.Tensor):
                raise ValueError("master weight state must contain tensors")
            parameter.copy_(saved.to(device=parameter.device, dtype=torch.float32))


class HybridModularOptimizer:
    """Polar matrix updates plus Adam-preconditioned bounded affine updates."""

    def __init__(
        self,
        specs: tuple[ModulaParameterSpec, ...],
        master_parameters: Mapping[str, nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        project: bool,
        diagnostics_interval: int,
    ) -> None:
        if lr <= 0 or not math.isfinite(lr):
            raise ValueError("lr must be finite and positive")
        if weight_decay < 0 or not math.isfinite(weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.specs = specs
        self.parameters_by_name = dict(master_parameters)
        expected = {spec.name for spec in specs}
        if expected != set(self.parameters_by_name):
            raise ValueError("master parameters do not match Modula parameter specifications")
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.project_enabled = project
        if diagnostics_interval <= 0:
            raise ValueError("diagnostics_interval must be positive")
        self.diagnostics_interval = diagnostics_interval
        self.step_count = 0
        self.momentum_buffers: dict[str, torch.Tensor] = {}
        self.bounded_first_moments: dict[str, torch.Tensor] = {}
        self.bounded_second_moments: dict[str, torch.Tensor] = {}
        self.bounded_steps: dict[str, int] = {}
        self.last_metrics: dict[str, float] = {}
        # Kept separately from diagnostic metrics so projection timing is
        # available on every step, not only on diagnostics_interval steps.
        self.last_timing: dict[str, float] = {}
        self._gradients_validated = False

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._gradients_validated = False
        for parameter in self.parameters_by_name.values():
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def validate_gradients(self) -> float:
        gradients = [
            parameter.grad
            for parameter in self.parameters_by_name.values()
            if parameter.grad is not None
        ]
        if not gradients:
            self._gradients_validated = True
            return 0.0
        total = torch.linalg.vector_norm(torch.stack([gradient.norm() for gradient in gradients]))
        total_value = float(total.item())
        if not math.isfinite(total_value):
            for name, parameter in self.parameters_by_name.items():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all().item()
                ):
                    raise FloatingPointError(f"non-finite gradient at {name}")
            raise FloatingPointError("non-finite aggregate gradient norm")
        self._gradients_validated = True
        return total_value

    @torch.no_grad()
    def initialize_modular_parameters(self) -> None:
        for spec in self.specs:
            if spec.geometry is None:
                continue
            parameter = self.parameters_by_name[spec.name]
            parameter.copy_(spec.geometry.initialize(parameter))

    @torch.no_grad()
    def step(self) -> None:
        if not self._gradients_validated:
            self.validate_gradients()
        collect_diagnostics = self.step_count % self.diagnostics_interval == 0
        started = time.perf_counter()
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        dualize_seconds = 0.0
        project_seconds = 0.0
        for spec in self.specs:
            if spec.role != "modular" or spec.geometry is None:
                continue
            parameter = self.parameters_by_name[spec.name]
            gradient = parameter.grad
            if gradient is None:
                continue
            buffer = self.momentum_buffers.get(spec.name)
            if buffer is None:
                buffer = torch.zeros_like(parameter)
                self.momentum_buffers[spec.name] = buffer
            buffer.mul_(self.momentum).add_(gradient)
            direction = gradient.add(buffer, alpha=self.momentum) if self.nesterov else buffer
            dualize_started = time.perf_counter()
            dualized = spec.geometry.dualize(direction, target_norm=spec.target_norm)
            dualize_seconds += time.perf_counter() - dualize_started
            parameter.mul_(1.0 - self.lr * self.weight_decay).add_(dualized, alpha=-self.lr)
            projection_displacement_rms = 0.0
            if self.project_enabled:
                before_projection = parameter.clone() if collect_diagnostics else None
                projection_started = time.perf_counter()
                parameter.copy_(spec.geometry.project(parameter))
                project_seconds += time.perf_counter() - projection_started
                if collect_diagnostics:
                    assert before_projection is not None
                    projection_displacement_rms = float(
                        (parameter - before_projection).square().mean().sqrt().item()
                    )
            if not collect_diagnostics:
                continue
            geometry = spec.geometry.name
            update = self.lr * dualized
            parameter_rms = float(parameter.square().mean().sqrt().item())
            update_rms = float(update.square().mean().sqrt().item())
            values = {
                "raw_grad_rms": float(gradient.square().mean().sqrt().item()),
                "update_rms": update_rms,
                "update_spectral_norm": float(spec.geometry.spectral_norm(update).item()),
                "parameter_rms": parameter_rms,
                "parameter_natural_norm": float(spec.geometry.spectral_norm(parameter).item()),
                "parameter_max_abs": float(parameter.abs().max().item()),
                "update_parameter_ratio": update_rms
                / max(parameter_rms, torch.finfo(parameter.dtype).tiny),
                "orthogonality_residual": float(
                    spec.geometry.orthogonality_residual(parameter).item()
                ),
                "projection_displacement_rms": projection_displacement_rms,
                "boundary_saturation_fraction": float(
                    spec.geometry.boundary_saturation_fraction(parameter).item()
                ),
            }
            for key, value in values.items():
                metric = f"optimizer/{geometry}/{key}"
                totals[metric] = totals.get(metric, 0.0) + value
            counts[geometry] = counts.get(geometry, 0) + 1

        for spec in self.specs:
            if spec.role != "bounded" or spec.geometry is None:
                continue
            parameter = self.parameters_by_name[spec.name]
            gradient = parameter.grad
            if gradient is None:
                continue
            first = self.bounded_first_moments.get(spec.name)
            second = self.bounded_second_moments.get(spec.name)
            if first is None or second is None:
                first = torch.zeros_like(parameter)
                second = torch.zeros_like(parameter)
                self.bounded_first_moments[spec.name] = first
                self.bounded_second_moments[spec.name] = second
            first.lerp_(gradient, 1.0 - _BOUNDED_BETA1)
            second.mul_(_BOUNDED_BETA2).addcmul_(gradient, gradient, value=1.0 - _BOUNDED_BETA2)
            bounded_step = self.bounded_steps.get(spec.name, 0) + 1
            self.bounded_steps[spec.name] = bounded_step
            first_hat = first / (1.0 - _BOUNDED_BETA1**bounded_step)
            second_hat = second / (1.0 - _BOUNDED_BETA2**bounded_step)
            direction = first_hat / (second_hat.sqrt() + _BOUNDED_EPSILON)
            dualize_started = time.perf_counter()
            dualized = spec.geometry.dualize(direction, target_norm=spec.target_norm)
            dualize_seconds += time.perf_counter() - dualize_started
            parameter.add_(dualized, alpha=-self.lr)
            before_projection = parameter.clone() if collect_diagnostics else None
            projection_started = time.perf_counter()
            parameter.copy_(spec.geometry.project(parameter))
            projection_displacement_rms = 0.0
            project_seconds += time.perf_counter() - projection_started
            if collect_diagnostics:
                assert before_projection is not None
                projection_displacement_rms = float(
                    (parameter - before_projection).square().mean().sqrt().item()
                )
            if not bool(torch.isfinite(parameter).all().item()):
                raise FloatingPointError(f"non-finite parameter after bounded update: {spec.name}")
            if not collect_diagnostics:
                continue
            geometry = spec.geometry.name
            update = self.lr * dualized
            parameter_rms = float(parameter.square().mean().sqrt().item())
            update_rms = float(update.square().mean().sqrt().item())
            values = {
                "raw_grad_rms": float(gradient.square().mean().sqrt().item()),
                "update_rms": update_rms,
                "update_natural_norm": float(spec.geometry.spectral_norm(update).item()),
                "parameter_rms": parameter_rms,
                "parameter_natural_norm": float(spec.geometry.spectral_norm(parameter).item()),
                "parameter_max_abs": float(parameter.abs().max().item()),
                "update_parameter_ratio": update_rms
                / max(parameter_rms, torch.finfo(parameter.dtype).tiny),
                "projection_displacement_rms": projection_displacement_rms,
                "boundary_saturation_fraction": float(
                    spec.geometry.boundary_saturation_fraction(parameter).item()
                ),
            }
            for key, value in values.items():
                metric = f"optimizer/{geometry}/{key}"
                totals[metric] = totals.get(metric, 0.0) + value
            counts[geometry] = counts.get(geometry, 0) + 1

        self.last_timing = {
            "timing/optimizer_step_seconds": time.perf_counter() - started,
            "timing/optimizer_dualize_seconds": dualize_seconds,
            "timing/optimizer_project_seconds": project_seconds,
        }
        if collect_diagnostics:
            for geometry, count in counts.items():
                for metric in list(totals):
                    if metric.startswith(f"optimizer/{geometry}/"):
                        totals[metric] /= count
            elapsed = time.perf_counter() - started
            totals["timing/optimizer_seconds"] = elapsed
            totals["timing/dualize_seconds"] = dualize_seconds
            totals["timing/project_seconds"] = project_seconds
            self.last_metrics = totals
        self.step_count += 1
        self._gradients_validated = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": MODULA_OPTIMIZER_FORMAT_VERSION,
            "optimizer_type": "modula",
            "names": [spec.name for spec in self.specs],
            "roles": [spec.role for spec in self.specs],
            "geometries": [
                None if spec.geometry is None else spec.geometry.name for spec in self.specs
            ],
            "step_count": self.step_count,
            "momentum_buffers": {
                name: buffer.detach().cpu().clone()
                for name, buffer in self.momentum_buffers.items()
            },
            "bounded_first_moments": {
                name: moment.detach().cpu().clone()
                for name, moment in self.bounded_first_moments.items()
            },
            "bounded_second_moments": {
                name: moment.detach().cpu().clone()
                for name, moment in self.bounded_second_moments.items()
            },
            "bounded_steps": dict(self.bounded_steps),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("optimizer_type") != "modula":
            raise ValueError("cannot resume a non-Modula optimizer as Modula")
        if state_dict.get("format_version") != MODULA_OPTIMIZER_FORMAT_VERSION:
            raise ValueError("unsupported Modula optimizer state format")
        expected_names = [spec.name for spec in self.specs]
        expected_roles = [spec.role for spec in self.specs]
        expected_geometries = [
            None if spec.geometry is None else spec.geometry.name for spec in self.specs
        ]
        if state_dict.get("names") != expected_names:
            raise ValueError("Modula optimizer parameter names mismatch")
        if state_dict.get("roles") != expected_roles:
            raise ValueError("Modula optimizer parameter roles mismatch")
        if state_dict.get("geometries") != expected_geometries:
            raise ValueError("Modula optimizer geometries mismatch")
        self.step_count = int(state_dict.get("step_count", 0))
        buffers = state_dict.get("momentum_buffers")
        if not isinstance(buffers, dict):
            raise ValueError("Modula optimizer momentum state must be an object")
        self.momentum_buffers = {
            name: saved.to(device=self.parameters_by_name[name].device, dtype=torch.float32)
            for name, saved in buffers.items()
        }
        bounded_names = {spec.name for spec in self.specs if spec.role == "bounded"}
        self.bounded_first_moments = self._load_bounded_moments(
            state_dict, key="bounded_first_moments", expected_names=bounded_names
        )
        self.bounded_second_moments = self._load_bounded_moments(
            state_dict, key="bounded_second_moments", expected_names=bounded_names
        )
        saved_steps = state_dict.get("bounded_steps")
        if not isinstance(saved_steps, dict) or not set(saved_steps) <= bounded_names:
            raise ValueError("invalid bounded geometry step state")
        self.bounded_steps = {}
        for name, step in saved_steps.items():
            if not isinstance(name, str) or not isinstance(step, int) or step <= 0:
                raise ValueError("invalid bounded geometry step entry")
            self.bounded_steps[name] = step
        state_names = set(self.bounded_steps)
        if (
            set(self.bounded_first_moments) != state_names
            or set(self.bounded_second_moments) != state_names
        ):
            raise ValueError("inconsistent bounded geometry optimizer state")

    def _load_bounded_moments(
        self, state_dict: Mapping[str, Any], *, key: str, expected_names: set[str]
    ) -> dict[str, torch.Tensor]:
        saved_moments = state_dict.get(key)
        if not isinstance(saved_moments, dict) or not set(saved_moments) <= expected_names:
            raise ValueError(f"invalid {key} state")
        loaded: dict[str, torch.Tensor] = {}
        for name, saved in saved_moments.items():
            if not isinstance(name, str) or not isinstance(saved, torch.Tensor):
                raise ValueError(f"invalid {key} entry")
            loaded[name] = saved.to(
                device=self.parameters_by_name[name].device, dtype=torch.float32
            )
        return loaded


def build_optimizer(
    *,
    model: nn.Module,
    master_weights: FP32MasterWeights,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer | HybridModularOptimizer:
    optimizer_name = str(config["optimizer"])
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            master_weights.parameters,
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
    if optimizer_name != "modula":
        raise ValueError(f"unknown optimizer: {optimizer_name}")
    specs = build_modula_parameter_specs(model)
    return HybridModularOptimizer(
        specs,
        dict(master_weights.named_parameters()),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        momentum=float(config["modula_momentum"]),
        nesterov=bool(config["modula_nesterov"]),
        project=bool(config["modula_project"]),
        diagnostics_interval=int(config["modula_diagnostics_interval"]),
    )


def optimizer_metrics(optimizer: OptimizerLike) -> dict[str, float]:
    if not isinstance(optimizer, HybridModularOptimizer):
        return {}
    return dict(optimizer.last_metrics) | dict(optimizer.last_timing)
