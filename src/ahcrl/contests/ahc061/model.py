import torch
from torch import nn

from ahcrl.contests.ahc061.encoder import NUM_PLANES
from ahcrl.nn.blocks import ResidualBlock


class ActorCritic(nn.Module):
    def __init__(self, in_channels: int = NUM_PLANES, channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *[ResidualBlock(channels) for _ in range(blocks)],
        )
        self.policy = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Flatten(),
        )
        self.value = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value
