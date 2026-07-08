from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from ahcrl.contests.ahc061.encoder import MAX_LEVEL, MAX_PLAYERS, NUM_PLANES, PLANE_M, PLANE_U
from ahcrl.nn.blocks import (
    ConvNeXtBlock,
    HyperEmbedder2d,
    PerCellMLPBlock,
    ResidualBlock,
    SimbaV2Block,
    SphericalDepthwiseSimbaBlock,
    SphericalGlobalContextBlock,
    make_group_norm,
)


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
        block_type: str = "convnext",
    ) -> None:
        super().__init__()
        block_factory = _block_factory(block_type, channels=channels, blocks=blocks)
        self.trunk = _make_trunk(
            block_type=block_type,
            in_channels=in_channels,
            channels=channels,
            blocks=blocks,
            block_factory=block_factory,
        )
        self.policy = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Flatten(),
        )
        self.value = RichValueHead(
            in_channels=in_channels,
            channels=channels,
            block_factory=block_factory,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy(h)
        value = self.value(h, x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def trunk_feature_norm_stats(self, x: torch.Tensor) -> dict[str, float]:
        h = self.trunk(x)
        feature_norm = h.float().pow(2).sum(dim=1).sqrt()
        return {
            "trunk_feature_norm_mean": float(feature_norm.mean().item()),
            "trunk_feature_norm_std": float(feature_norm.std(unbiased=False).item()),
            "trunk_feature_norm_max": float(feature_norm.max().item()),
        }


class ObservationNormalizedActorCritic(nn.Module):
    """Apply frozen observation RSNorm before an ActorCritic model."""

    def __init__(
        self,
        model: ActorCritic,
        *,
        mean: torch.Tensor,
        variance: torch.Tensor,
        epsilon: float,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.model = model
        self.register_buffer("mean", mean.detach().float().clone())
        self.register_buffer("variance", variance.detach().float().clone())
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = cast(torch.Tensor, self.mean)
        variance = cast(torch.Tensor, self.variance)
        y = (x.float() - mean) / torch.sqrt(variance + self.epsilon)
        return self.model(y.to(dtype=x.dtype))


class GroupedObservationNormalizedActorCritic(nn.Module):
    """Apply frozen per-(M,U) observation RSNorm before an ActorCritic model."""

    def __init__(
        self,
        model: ActorCritic,
        *,
        global_mean: torch.Tensor,
        global_variance: torch.Tensor,
        group_mean: torch.Tensor,
        group_variance: torch.Tensor,
        group_count: torch.Tensor,
        epsilon: float,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.model = model
        self.register_buffer("global_mean", global_mean.detach().float().clone())
        self.register_buffer("global_variance", global_variance.detach().float().clone())
        self.register_buffer("group_mean", group_mean.detach().float().clone())
        self.register_buffer("group_variance", group_variance.detach().float().clone())
        self.register_buffer("group_count", group_count.detach().long().clone())
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m_values = torch.round(x[:, PLANE_M, 0, 0].float() * MAX_PLAYERS).long()
        u_values = torch.round(x[:, PLANE_U, 0, 0].float() * MAX_LEVEL).long()
        m_values = torch.clamp(m_values, 0, MAX_PLAYERS)
        u_values = torch.clamp(u_values, 0, MAX_LEVEL)

        group_count = cast(torch.Tensor, self.group_count)
        has_group = group_count[m_values, u_values] > 0
        group_mean = cast(torch.Tensor, self.group_mean)[m_values, u_values]
        group_variance = cast(torch.Tensor, self.group_variance)[m_values, u_values]
        global_mean = cast(torch.Tensor, self.global_mean)
        global_variance = cast(torch.Tensor, self.global_variance)
        mean = torch.where(has_group.view(-1, 1, 1, 1), group_mean, global_mean)
        variance = torch.where(
            has_group.view(-1, 1, 1, 1),
            group_variance,
            global_variance,
        )
        y = (x.float() - mean) / torch.sqrt(variance + self.epsilon)
        y[:, PLANE_M] = x[:, PLANE_M].float()
        y[:, PLANE_U] = x[:, PLANE_U].float()
        return self.model(y.to(dtype=x.dtype))


class RichValueHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        channels: int,
        block_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(block_factory(), block_factory())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        stats_channels = in_channels * 2
        pooled_channels = channels * 2
        hidden_channels = channels * 2
        self.mlp = nn.Sequential(
            nn.Linear(pooled_channels + stats_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, trunk_features: torch.Tensor, raw_planes: torch.Tensor) -> torch.Tensor:
        h = self.blocks(trunk_features)
        avg_features = self.avg_pool(h).flatten(1)
        max_features = self.max_pool(h).flatten(1)
        plane_mean = raw_planes.mean(dim=(-2, -1))
        plane_max = raw_planes.amax(dim=(-2, -1))
        features = torch.cat([avg_features, max_features, plane_mean, plane_max], dim=1)
        return self.mlp(features)


def _make_trunk(
    *,
    block_type: str,
    in_channels: int,
    channels: int,
    blocks: int,
    block_factory: Callable[[], nn.Module],
) -> nn.Sequential:
    if block_type in ("simbav2_block", "spherical_depthwise_simba", "spherical_global_context"):
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


def _block_factory(block_type: str, *, channels: int, blocks: int) -> Callable[[], nn.Module]:
    if block_type == "convnext":
        return lambda: ConvNeXtBlock(channels)
    if block_type == "per_cell_mlp":
        return lambda: PerCellMLPBlock(channels)
    if block_type == "simbav2_block":
        alpha_init = 1.0 / (blocks + 1)
        return lambda: SimbaV2Block(channels, alpha_init=alpha_init)
    if block_type == "spherical_global_context":
        alpha_init = 1.0 / (blocks + 1)
        return lambda: SphericalGlobalContextBlock(channels, alpha_init=alpha_init)
    if block_type == "spherical_depthwise_simba":
        alpha_init = 1.0 / (blocks + 1)
        return lambda: SphericalDepthwiseSimbaBlock(channels, alpha_init=alpha_init)
    if block_type == "residual":
        return lambda: ResidualBlock(channels)
    raise ValueError(f"unknown block_type: {block_type}")
