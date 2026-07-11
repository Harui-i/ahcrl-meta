from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from ahcrl.contests.ahc061.encoder import (
    CRITIC_FEATURE_SHAPE,
    MAX_LEVEL,
    MAX_PLAYERS,
    NUM_PLANES,
    PLANE_M,
    PLANE_U,
)
from ahcrl.nn.blocks import (
    ConvNeXtBlock,
    PerCellMLPBlock,
    ResidualBlock,
    SimbaV2Block,
    SphericalAttentionSimbaBlock,
)
from ahcrl.nn.components import (
    HyperEmbedder2d,
    make_group_norm,
)


class KeyedObservationNormalizer(nn.Module):
    """Running channel-wise observation normalization stored in a model state."""

    def __init__(
        self,
        channels: int,
        *,
        epsilon: float,
        grouping: str = "none",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if grouping not in ("none", "m_u"):
            raise ValueError(f"unsupported observation normalization grouping: {grouping}")
        self.channels = channels
        self.epsilon = epsilon
        self.grouping = grouping
        self.count: torch.Tensor
        self.mean: torch.Tensor
        self.m2: torch.Tensor
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros((1, channels, 1, 1), dtype=torch.float32))
        self.register_buffer("m2", torch.zeros((1, channels, 1, 1), dtype=torch.float32))
        if grouping == "m_u":
            num_keys = (MAX_PLAYERS + 1) * (MAX_LEVEL + 1)
            self.group_count: torch.Tensor
            self.group_mean: torch.Tensor
            self.group_m2: torch.Tensor
            self.register_buffer("group_count", torch.zeros(num_keys, dtype=torch.long))
            self.register_buffer(
                "group_mean",
                torch.zeros((num_keys, channels, 1, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "group_m2",
                torch.zeros((num_keys, channels, 1, 1), dtype=torch.float32),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.normalize(observations)

    @torch.no_grad()
    def update_and_normalize(
        self,
        observations: torch.Tensor,
        *,
        m_values: torch.Tensor | None = None,
        u_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.update(observations, m_values=m_values, u_values=u_values)
        return self.normalize(observations, m_values=m_values, u_values=u_values)

    @torch.no_grad()
    def update(
        self,
        observations: torch.Tensor,
        *,
        m_values: torch.Tensor | None = None,
        u_values: torch.Tensor | None = None,
    ) -> None:
        self._validate(observations, m_values, u_values)
        m_values, u_values = self._resolve_keys(observations, m_values, u_values)
        values = observations.detach().float()
        batch_count = int(values.shape[0] * values.shape[2] * values.shape[3])
        if batch_count == 0:
            return
        batch_mean = values.mean(dim=(0, 2, 3), keepdim=True).to(device=self.mean.device)
        batch_m2 = (
            values.sub(batch_mean.to(device=values.device))
            .pow(2)
            .sum(dim=(0, 2, 3), keepdim=True)
            .to(device=self.mean.device)
        )
        self._merge(self.count, self.mean, self.m2, batch_count, batch_mean, batch_m2)

        if self.grouping != "m_u":
            return
        assert m_values is not None and u_values is not None
        keys = self._keys(m_values, u_values).cpu()
        for key in torch.unique(keys).tolist():
            selector = keys == key
            group_values = values[selector.to(device=values.device)]
            group_count = int(group_values.shape[0] * group_values.shape[2] * group_values.shape[3])
            group_mean = group_values.mean(dim=(0, 2, 3), keepdim=True).to(device=self.mean.device)
            group_m2 = (
                group_values.sub(group_mean.to(device=group_values.device))
                .pow(2)
                .sum(dim=(0, 2, 3), keepdim=True)
                .to(device=self.mean.device)
            )
            self._merge(
                self.group_count[key : key + 1],
                self.group_mean[key : key + 1],
                self.group_m2[key : key + 1],
                group_count,
                group_mean,
                group_m2,
            )

    def normalize(
        self,
        observations: torch.Tensor,
        *,
        m_values: torch.Tensor | None = None,
        u_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate(observations, m_values, u_values)
        m_values, u_values = self._resolve_keys(observations, m_values, u_values)
        if self.grouping == "m_u":
            assert m_values is not None and u_values is not None
            keys = self._keys(m_values, u_values).to(device=observations.device)
            group_count = self.group_count.to(device=observations.device)[keys]
            group_mean = self.group_mean.to(device=observations.device)[keys]
            group_variance = self._variance(self.group_m2, self.group_count)[keys]
            global_mean = self.mean.to(device=observations.device)
            global_variance = self._variance(self.m2, self.count).to(device=observations.device)
            use_group = (group_count > 0).view(-1, 1, 1, 1)
            mean = torch.where(use_group, group_mean, global_mean)
            variance = torch.where(use_group, group_variance, global_variance)
        else:
            mean = self.mean.to(device=observations.device)
            variance = self._variance(self.m2, self.count).to(device=observations.device)
        normalized = (observations.float() - mean) / torch.sqrt(variance + self.epsilon)
        if self.grouping == "m_u":
            normalized[:, PLANE_M] = observations[:, PLANE_M].float()
            normalized[:, PLANE_U] = observations[:, PLANE_U].float()
        return normalized.to(dtype=observations.dtype)

    def stats(self) -> dict[str, float]:
        variance = self._variance(self.m2, self.count)
        stats = {
            "obs_norm_count": float(self.count.item()),
            "obs_norm_mean_abs_max": float(self.mean.abs().max().item()),
            "obs_norm_std_min": float(torch.sqrt(variance).min().item()),
            "obs_norm_std_max": float(torch.sqrt(variance).max().item()),
        }
        if self.grouping == "m_u":
            for key, count in enumerate(self.group_count.tolist()):
                if count > 0:
                    m_value, u_value = divmod(key, MAX_LEVEL + 1)
                    stats[f"obs_norm_count_by_m_u/m_{m_value}_u_{u_value}"] = float(count)
        return stats

    @property
    def variance(self) -> torch.Tensor:
        return self._variance(self.m2, self.count)

    @staticmethod
    def _variance(m2: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        if count.numel() == 1:
            safe_count = count.to(dtype=m2.dtype).clamp_min(1)
            return torch.where(count > 0, m2 / safe_count, torch.ones_like(m2))
        safe_count = count.clamp_min(1).view(-1, 1, 1, 1)
        return torch.where(count.view(-1, 1, 1, 1) > 0, m2 / safe_count, torch.ones_like(m2))

    def _merge(
        self,
        count: torch.Tensor,
        mean: torch.Tensor,
        m2: torch.Tensor,
        batch_count: int,
        batch_mean: torch.Tensor,
        batch_m2: torch.Tensor,
    ) -> None:
        current_count = int(count.item())
        if current_count == 0:
            count.fill_(batch_count)
            mean.copy_(batch_mean)
            m2.copy_(batch_m2)
            return
        total_count = current_count + batch_count
        delta = batch_mean - mean
        mean.add_(delta * batch_count / total_count)
        m2.add_(batch_m2 + delta.pow(2) * current_count * batch_count / total_count)
        count.fill_(total_count)

    def _keys(self, m_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        m_values = torch.clamp(m_values.long(), 0, MAX_PLAYERS)
        u_values = torch.clamp(u_values.long(), 0, MAX_LEVEL)
        return m_values * (MAX_LEVEL + 1) + u_values

    def _validate(
        self,
        observations: torch.Tensor,
        m_values: torch.Tensor | None,
        u_values: torch.Tensor | None,
    ) -> None:
        if observations.ndim != 4:
            raise ValueError(f"expected NCHW observations, got {observations.ndim} dimensions")
        if observations.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {observations.shape[1]}")
        if m_values is not None or u_values is not None:
            if m_values is None or u_values is None:
                raise ValueError("m_values and u_values must be provided together")
            if m_values.shape != (observations.shape[0],) or u_values.shape != m_values.shape:
                raise ValueError("m_values and u_values must have shape (batch,)")

    def _resolve_keys(
        self,
        observations: torch.Tensor,
        m_values: torch.Tensor | None,
        u_values: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.grouping != "m_u":
            return m_values, u_values
        if m_values is None or u_values is None:
            m_values = torch.round(observations[:, PLANE_M, 0, 0].float() * MAX_PLAYERS).long()
            u_values = torch.round(observations[:, PLANE_U, 0, 0].float() * MAX_LEVEL).long()
        return m_values, u_values


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
        block_type: str = "convnext",
        critic_feature_mode: str = "oracle",
    ) -> None:
        super().__init__()
        self.observation_normalizer: KeyedObservationNormalizer | None = None
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
            critic_feature_mode=critic_feature_mode,
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

    def forward(
        self,
        x: torch.Tensor,
        critic_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = cast(torch.Tensor, self.mean)
        variance = cast(torch.Tensor, self.variance)
        y = (x.float() - mean) / torch.sqrt(variance + self.epsilon)
        return self.model(y.to(dtype=x.dtype), critic_features)


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

    def forward(
        self,
        x: torch.Tensor,
        critic_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return self.model(y.to(dtype=x.dtype), critic_features)


class RichValueHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        channels: int,
        block_factory: Callable[[], nn.Module],
        critic_feature_mode: str,
    ) -> None:
        super().__init__()
        if critic_feature_mode not in ("none", "posterior", "oracle"):
            raise ValueError(f"unknown critic feature mode: {critic_feature_mode}")
        self.critic_feature_mode = critic_feature_mode
        self.blocks = nn.Sequential(block_factory(), block_factory())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        stats_channels = in_channels * 2
        pooled_channels = channels * 2
        hidden_channels = channels * 2
        critic_channels = 0
        if critic_feature_mode != "none":
            player_embedding_channels = max(8, channels // MAX_PLAYERS)
            self.critic_player_encoder = nn.Sequential(
                nn.Linear(CRITIC_FEATURE_SHAPE[1], player_embedding_channels),
                nn.ReLU(inplace=True),
            )
            self.critic_encoder = nn.Sequential(
                nn.Linear(MAX_PLAYERS * player_embedding_channels, channels),
                nn.ReLU(inplace=True),
            )
            critic_channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(pooled_channels + stats_channels + critic_channels, hidden_channels),
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
        if self.critic_feature_mode != "none":
            if critic_features is None:
                critic_features = torch.zeros(
                    (trunk_features.shape[0], *CRITIC_FEATURE_SHAPE),
                    device=trunk_features.device,
                    dtype=trunk_features.dtype,
                )
            player_encoder = cast(nn.Module, self.critic_player_encoder)
            critic_encoder = cast(nn.Module, self.critic_encoder)
            critic_embedding = player_encoder(critic_features).flatten(1)
            features.append(critic_encoder(critic_embedding))
        features = torch.cat(features, dim=1)
        return self.mlp(features)


def _make_trunk(
    *,
    block_type: str,
    in_channels: int,
    channels: int,
    blocks: int,
    block_factory: Callable[[], nn.Module],
) -> nn.Sequential:
    if block_type in (
        "simbav2_block",
        "spherical_attention_simba",
    ):
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
    if block_type == "spherical_attention_simba":
        alpha_init = 1.0 / (blocks + 1)
        return lambda: SphericalAttentionSimbaBlock(channels, alpha_init=alpha_init)
    if block_type == "residual":
        return lambda: ResidualBlock(channels)
    raise ValueError(f"unknown block_type: {block_type}")
