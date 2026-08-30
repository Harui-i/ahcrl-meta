"""Explained variance に基づく PPO policy warm-up。"""

import math
from dataclasses import dataclass
from typing import Any

import torch


def calculate_explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """Value prediction が return の分散をどれだけ説明するかを返す。"""
    values = values.float()
    returns = returns.float()
    return_variance = returns.var(unbiased=False)
    if float(return_variance.item()) <= 1e-8:
        return float("nan")
    return float((1.0 - (returns - values).var(unbiased=False) / return_variance).item())


@dataclass
class ExplainedVariancePolicyWarmup:
    """閾値を一度超えるまで policy update を止めるラッチ。"""

    threshold: float
    policy_unfrozen: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold >= 1.0:
            raise ValueError("policy unfreeze explained variance threshold must be finite and < 1")

    def observe(self, values: torch.Tensor, returns: torch.Tensor) -> float:
        """最新 rollout を観測し、条件を満たせば以後の policy update を許可する。"""
        explained_variance = calculate_explained_variance(values, returns)
        if math.isfinite(explained_variance) and explained_variance > self.threshold:
            self.policy_unfrozen = True
        return explained_variance

    @property
    def policy_updates_enabled(self) -> bool:
        return self.policy_unfrozen

    def state_dict(self) -> dict[str, float | bool]:
        return {
            "threshold": self.threshold,
            "policy_unfrozen": self.policy_unfrozen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_threshold = float(state["threshold"])
        if not math.isfinite(saved_threshold) or saved_threshold >= 1.0:
            raise ValueError("saved policy warm-up threshold must be finite and < 1")
        self.policy_unfrozen = bool(state["policy_unfrozen"])
