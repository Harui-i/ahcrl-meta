"""Reusable, semantic adapters for plane-based observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from ahcrl.nn.modula import ModulaGraphNode, ModularEmbedding, module_to_modula_graph

__all__ = [
    "CategoricalPlaneAdapter",
    "CategoricalPlaneGroup",
    "RunningObservationNormalizer",
]


@dataclass(frozen=True)
class CategoricalPlaneGroup:
    """A contiguous one-hot field in an NCHW observation tensor."""

    name: str
    start: int
    channels: int
    implicit_zero: bool = False
    embedding_dim: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("categorical plane group name must not be empty")
        if self.start < 0:
            raise ValueError("categorical plane group start must be non-negative")
        if self.channels <= 0:
            raise ValueError("categorical plane group channels must be positive")
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("categorical plane group embedding_dim must be positive")

    @property
    def num_embeddings(self) -> int:
        return self.channels + int(self.implicit_zero)

    @property
    def output_dim(self) -> int:
        return self.embedding_dim or self.num_embeddings

    @property
    def end(self) -> int:
        return self.start + self.channels


class CategoricalPlaneAdapter(nn.Module):
    """Replace selected one-hot plane groups with semantic embeddings.

    The replacement is performed at every board cell.  Groups with
    ``implicit_zero`` receive an additional leading class whose value is
    ``1 - sum(source_channels)``; this makes all categorical fields explicit
    one-hot vectors even when the source encoder uses an all-zero sentinel.
    """

    def __init__(self, in_channels: int, groups: Sequence[CategoricalPlaneGroup]) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        ordered_groups = tuple(groups)
        previous_end = 0
        for group in ordered_groups:
            if group.start < previous_end:
                raise ValueError("categorical plane groups must be sorted and non-overlapping")
            if group.end > in_channels:
                raise ValueError(
                    f"categorical plane group {group.name!r} exceeds input width {in_channels}"
                )
            previous_end = group.end

        self.in_channels = in_channels
        self.groups = ordered_groups
        self.embeddings = nn.ModuleList(
            ModularEmbedding(group.num_embeddings, group.output_dim) for group in ordered_groups
        )
        self.output_channels = in_channels + sum(
            group.output_dim - group.channels for group in ordered_groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        if not self.groups:
            return x

        outputs: list[torch.Tensor] = []
        cursor = 0
        for group, embedding_module in zip(self.groups, self.embeddings, strict=True):
            embedding = cast(ModularEmbedding, embedding_module)
            if cursor < group.start:
                outputs.append(x[:, cursor : group.start])
            source = x[:, group.start : group.end]
            if group.implicit_zero:
                zero = 1.0 - source.sum(dim=1, keepdim=True)
                source = torch.cat((zero, source), dim=1)
            encoded = embedding.forward_one_hot(source.permute(0, 2, 3, 1))
            outputs.append(encoded.permute(0, 3, 1, 2))
            cursor = group.end
        if cursor < self.in_channels:
            outputs.append(x[:, cursor:])
        return torch.cat(outputs, dim=1)

    def modula_node(self, prefix: str = "") -> ModulaGraphNode:
        # Concatenation has one parameter-free pass-through branch and one
        # semantic atom per field.  A parallel node gives each embedding its
        # own natural geometry while retaining the adapter's input sensitivity.
        if not self.groups:
            return ModulaGraphNode("bond", own_sensitivity=1.0)
        children: list[ModulaGraphNode] = []
        has_passthrough = (
            self.groups[0].start > 0
            or any(
                left.end < right.start
                for left, right in zip(self.groups, self.groups[1:], strict=True)
            )
            or self.groups[-1].end < self.in_channels
        )
        if has_passthrough:
            children.append(ModulaGraphNode("bond", own_sensitivity=1.0))
        for index, embedding in enumerate(self.embeddings):
            embedding_prefix = f"{prefix}.embeddings.{index}" if prefix else f"embeddings.{index}"
            children.append(module_to_modula_graph(embedding, embedding_prefix))
        return ModulaGraphNode("parallel", tuple(children))


class RunningObservationNormalizer(nn.Module):
    """Checkpointed channel-wise Welford normalizer with optional exclusions."""

    def __init__(
        self,
        channels: int,
        epsilon: float = 1e-8,
        excluded_channels: Sequence[int] = (),
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        excluded = tuple(sorted(set(excluded_channels)))
        if any(channel < 0 or channel >= channels for channel in excluded):
            raise ValueError("excluded channel is outside the normalizer width")
        self.epsilon = epsilon
        normalize_mask = torch.ones(1, channels, 1, 1, dtype=torch.bool)
        if excluded:
            normalize_mask[:, excluded] = False
        self.normalize_mask: torch.Tensor
        self.register_buffer("normalize_mask", normalize_mask, persistent=False)
        self.count: torch.Tensor
        self.mean: torch.Tensor
        self.m2: torch.Tensor
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("m2", torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)

    @torch.no_grad()
    def update_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        values = x.detach().float()
        batch_count = values.shape[0] * values.shape[2] * values.shape[3]
        if batch_count:
            batch_mean = values.mean((0, 2, 3), keepdim=True).to(device=self.mean.device)
            batch_m2 = (
                values.sub(batch_mean.to(values.device))
                .square()
                .sum((0, 2, 3), keepdim=True)
                .to(device=self.mean.device)
            )
            # Excluded fields are semantic one-hot inputs.  Keep their
            # checkpointed statistics at the neutral affine values instead of
            # accumulating irrelevant moments for them.
            mask = self.normalize_mask.to(device=batch_mean.device)
            batch_mean = torch.where(mask, batch_mean, torch.zeros_like(batch_mean))
            batch_m2 = torch.where(mask, batch_m2, torch.zeros_like(batch_m2))
            current = int(self.count.item())
            if current == 0:
                self.count.fill_(batch_count)
                self.mean.copy_(batch_mean)
                self.m2.copy_(batch_m2)
            else:
                total = current + batch_count
                delta = batch_mean - self.mean
                self.mean.add_(delta * batch_count / total)
                self.m2.add_(batch_m2 + delta.square() * current * batch_count / total)
                self.count.fill_(total)
        return self.normalize(x)

    def _variance(self) -> torch.Tensor:
        count = self.count.to(dtype=self.m2.dtype).clamp_min(1)
        return torch.where(self.count > 0, self.m2 / count, torch.ones_like(self.m2))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        variance = self._variance()
        normalized = (x.float() - self.mean.to(x.device)) / torch.sqrt(
            variance.to(x.device) + self.epsilon
        )
        mask = self.normalize_mask.to(device=x.device)
        return torch.where(mask, normalized, x.float()).to(dtype=x.dtype)

    def effective_affine(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and inverse std suitable for a fused inference kernel."""
        variance = self._variance()
        mask = self.normalize_mask.to(device=self.mean.device)
        mean = torch.where(mask, self.mean, torch.zeros_like(self.mean))
        invstd = torch.where(mask, torch.rsqrt(variance + self.epsilon), torch.ones_like(variance))
        return mean, invstd

    def stats(self) -> dict[str, float]:
        std = self._variance().sqrt()[self.normalize_mask]
        if std.numel() == 0:
            return {
                "obs_norm_count": float(self.count.item()),
                "obs_norm_std_min": 0.0,
                "obs_norm_std_max": 0.0,
            }
        return {
            "obs_norm_count": float(self.count.item()),
            "obs_norm_std_min": float(std.min().item()),
            "obs_norm_std_max": float(std.max().item()),
        }
