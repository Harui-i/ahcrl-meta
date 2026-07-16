from typing import cast

import torch
from torch import nn

from ahcrl.nn.blocks import ConvNeXtBlock, ResidualBlock
from ahcrl.nn.components import make_group_norm

from .encoder import ACTION_COUNT, NUM_PLANES


class ActorCritic(nn.Module):
    """A shared spherical-attention trunk with four directional actions."""

    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
        block_type: str = "convnext",
    ) -> None:
        super().__init__()
        if channels <= 0 or blocks <= 0:
            raise ValueError("channels and blocks must be positive")
        if block_type not in ("convnext", "residual"):
            raise ValueError(f"unknown block_type: {block_type}")

        self.observation_normalizer: nn.Module | None = None
        block_class = ConvNeXtBlock if block_type == "convnext" else ResidualBlock
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            *[block_class(channels) for _ in range(blocks)],
        )
        self.policy = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, ACTION_COUNT),
        )
        self.value = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        h = self.trunk(x)
        pooled = torch.cat(
            [h.mean(dim=(-2, -1)), h.amax(dim=(-2, -1))],
            dim=1,
        )
        return self.policy(pooled), self.value(pooled).squeeze(-1)

    @torch.no_grad()
    def feature_norm_stats(self, x: torch.Tensor) -> dict[str, float]:
        h = cast(torch.Tensor, self.trunk(x))
        norm = h.float().pow(2).sum(dim=1).sqrt()
        return {
            "trunk_feature_norm_mean": float(norm.mean().item()),
            "trunk_feature_norm_std": float(norm.std(unbiased=False).item()),
            "trunk_feature_norm_max": float(norm.max().item()),
        }


class RunningObservationNormalizer(nn.Module):
    """Channel-wise Welford statistics kept inside the checkpoint."""

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
        y = (x.float() - self.mean.to(x.device)) / torch.sqrt(variance.to(x.device) + self.epsilon)
        return y.to(dtype=x.dtype)

    def stats(self) -> dict[str, float]:
        count = self.count.to(dtype=self.m2.dtype).clamp_min(1)
        variance = torch.where(self.count > 0, self.m2 / count, torch.ones_like(self.m2))
        return {
            "obs_norm_count": float(self.count.item()),
            "obs_norm_std_min": float(variance.sqrt().min().item()),
            "obs_norm_std_max": float(variance.sqrt().max().item()),
        }
