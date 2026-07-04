import torch
from torch import nn

from ahcrl.contests.ahc061.encoder import NUM_PLANES
from ahcrl.nn.blocks import ConvNeXtBlock, ResidualBlock


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
        block_type: str = "convnext",
    ) -> None:
        super().__init__()
        block_cls = _block_class(block_type)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *[block_cls(channels) for _ in range(blocks)],
        )
        self.policy = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Flatten(),
        )
        self.value = RichValueHead(
            in_channels=in_channels,
            channels=channels,
            block_cls=block_cls,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy(h)
        value = self.value(h, x).squeeze(-1)
        return logits, value


class RichValueHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        channels: int,
        block_cls: type[nn.Module],
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(block_cls(channels), block_cls(channels))
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


def _block_class(block_type: str) -> type[nn.Module]:
    if block_type == "convnext":
        return ConvNeXtBlock
    if block_type == "residual":
        return ResidualBlock
    raise ValueError(f"unknown block_type: {block_type}")
