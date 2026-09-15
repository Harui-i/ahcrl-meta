"""Reusable, preallocated storage for PPO rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutFieldSpec:
    """Shape and placement of one field in a :class:`RolloutBuffer`."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class RolloutBuffer:
    """Fixed-size ``[steps, envs, *field.shape]`` rollout storage.

    The tensors are allocated once and reused after :meth:`reset`.  ``store``
    deliberately copies into the destination so callers may safely reuse their
    temporary model/environment tensors immediately afterwards.
    """

    def __init__(
        self,
        steps: int,
        num_envs: int,
        fields: Mapping[str, RolloutFieldSpec],
    ) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not fields:
            raise ValueError("fields must not be empty")
        self.steps = steps
        self.num_envs = num_envs
        self._fields = dict(fields)
        self._tensors: dict[str, torch.Tensor] = {}
        for name, spec in self._fields.items():
            if not name:
                raise ValueError("field names must not be empty")
            if any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in spec.shape
            ):
                raise ValueError(f"field {name!r} has an invalid shape {spec.shape!r}")
            self._tensors[name] = torch.empty(
                (steps, num_envs, *spec.shape), dtype=spec.dtype, device=spec.device
            )
        self._written = torch.zeros((steps, len(self._fields)), dtype=torch.bool)

    @property
    def fields(self) -> Mapping[str, RolloutFieldSpec]:
        return self._fields

    def store(self, step: int, **values: torch.Tensor) -> None:
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < self.steps:
            raise IndexError(f"step must be in [0, {self.steps}), got {step}")
        expected = set(self._fields)
        actual = set(values)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)!r}")
            if extra:
                details.append(f"unexpected={sorted(extra)!r}")
            raise ValueError("rollout fields do not match: " + ", ".join(details))
        if bool(self._written[step].any().item()):
            raise ValueError(f"rollout step {step} was already written")
        for index, name in enumerate(self._fields):
            source = values[name]
            destination = self._tensors[name][step]
            expected_shape = destination.shape
            if source.shape != expected_shape:
                raise ValueError(
                    f"field {name!r} has shape {tuple(source.shape)}, "
                    f"expected {tuple(expected_shape)}"
                )
            if source.dtype != destination.dtype:
                raise TypeError(
                    f"field {name!r} has dtype {source.dtype}, expected {destination.dtype}"
                )
            destination.copy_(source)
            self._written[step, index] = True

    def as_dict(self) -> dict[str, torch.Tensor]:
        if not bool(self._written.all().item()):
            missing = (~self._written).nonzero(as_tuple=False)
            first_step, first_field = (int(value) for value in missing[0])
            name = tuple(self._fields)[first_field]
            raise RuntimeError(f"rollout field {name!r} at step {first_step} was not written")
        return dict(self._tensors)

    def reset(self) -> None:
        self._written.zero_()
