"""Reusable PyTorch neural network components."""

from torch import nn

__all__ = ["make_group_norm"]


def make_group_norm(channels: int, *, max_groups: int = 8) -> nn.GroupNorm:
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    if max_groups <= 0:
        raise ValueError(f"max_groups must be positive, got {max_groups}")

    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)

    raise AssertionError("unreachable")
