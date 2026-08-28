from collections.abc import Callable

import torch
from torch import nn

from ahcrl.contests.ahc061.encoder import (
    CRITIC_FEATURE_SHAPE,
    MAX_PLAYERS,
    NUM_PLANES,
)
from ahcrl.nn.components import make_group_norm
from ahcrl.nn.trunk import make_block_factory, make_trunk


class RunningObservationNormalizer(nn.Module):
    """Checkpoint に保存する channel-wise Welford normalizer。"""

    def __init__(self, channels: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
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

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        count = self.count.to(dtype=self.m2.dtype).clamp_min(1)
        variance = torch.where(self.count > 0, self.m2 / count, torch.ones_like(self.m2))
        normalized = (x.float() - self.mean.to(x.device)) / torch.sqrt(
            variance.to(x.device) + self.epsilon
        )
        return normalized.to(dtype=x.dtype)

    def stats(self) -> dict[str, float]:
        count = self.count.to(dtype=self.m2.dtype).clamp_min(1)
        variance = torch.where(self.count > 0, self.m2 / count, torch.ones_like(self.m2))
        return {
            "obs_norm_count": float(self.count.item()),
            "obs_norm_std_min": float(variance.sqrt().min().item()),
            "obs_norm_std_max": float(variance.sqrt().max().item()),
        }


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
        block_type: str = "convnext",
    ) -> None:
        super().__init__()
        self.observation_normalizer: RunningObservationNormalizer | None = None
        block_factory = make_block_factory(block_type, channels=channels, blocks=blocks)
        self.trunk = make_trunk(
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

    def forward(
        self,
        x: torch.Tensor,
        critic_features: torch.Tensor | None = None,
        normalize_input: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normalize_input and self.observation_normalizer is not None:
            x = self.observation_normalizer(x)
        h = self.trunk(x)
        logits = self.policy(h)
        value = self.value(h, x, critic_features).squeeze(-1)
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
        player_embedding_channels = max(8, channels // MAX_PLAYERS)
        self.critic_player_encoder = nn.Sequential(
            nn.Linear(CRITIC_FEATURE_SHAPE[1], player_embedding_channels),
            nn.ReLU(inplace=True),
        )
        self.critic_encoder = nn.Sequential(
            nn.Linear(MAX_PLAYERS * player_embedding_channels, channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(pooled_channels + stats_channels + channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        trunk_features: torch.Tensor,
        raw_planes: torch.Tensor,
        critic_features: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.blocks(trunk_features)
        avg_features = self.avg_pool(h).flatten(1)
        max_features = self.max_pool(h).flatten(1)
        plane_mean = raw_planes.mean(dim=(-2, -1))
        plane_max = raw_planes.amax(dim=(-2, -1))
        features = [avg_features, max_features, plane_mean, plane_max]
        if critic_features is None:
            critic_features = torch.zeros(
                (trunk_features.shape[0], *CRITIC_FEATURE_SHAPE),
                device=trunk_features.device,
                dtype=trunk_features.dtype,
            )
        critic_embedding = self.critic_player_encoder(critic_features).flatten(1)
        features.append(self.critic_encoder(critic_embedding))
        features = torch.cat(features, dim=1)
        return self.mlp(features)
