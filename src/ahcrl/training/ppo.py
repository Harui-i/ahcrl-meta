"""Shared numerically guarded PPO policy-ratio calculations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolicySurrogateResult:
    loss: torch.Tensor | None
    clipping_ratio: torch.Tensor | None
    current_behavior_log_ratio: torch.Tensor
    current_behavior_ratio: torch.Tensor | None
    proximal_behavior_ratio: torch.Tensor | None
    current_behavior_approx_kl: torch.Tensor | None
    max_abs_log_ratio: torch.Tensor
    stop_reason: str | None


def tensor_range(tensor: torch.Tensor) -> str:
    """Return a compact range including the number of non-finite values."""

    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return f"no finite values nonfinite_count={tensor.numel()}"
    nonfinite_count = tensor.numel() - finite.numel()
    return (
        f"finite_min={finite.min().item():.6g} finite_max={finite.max().item():.6g} "
        f"nonfinite_count={nonfinite_count}"
    )


def policy_surrogate(
    *,
    new_logprob: torch.Tensor,
    behavior_logprob: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
    proximal_logprob: torch.Tensor | None = None,
    max_abs_log_ratio: float | None = None,
    target_kl: float | None = None,
) -> PolicySurrogateResult:
    """Return a PPO surrogate, or a trust-region stop before unsafe exponentiation."""

    if max_abs_log_ratio is not None and max_abs_log_ratio <= 0.0:
        raise ValueError("max_abs_log_ratio must be positive")
    if target_kl is not None and target_kl <= 0.0:
        raise ValueError("target_kl must be positive")

    current_behavior_log_ratio = new_logprob - behavior_logprob
    log_ratios = [current_behavior_log_ratio]
    current_proximal_log_ratio = None
    proximal_behavior_log_ratio = None
    if proximal_logprob is not None:
        current_proximal_log_ratio = new_logprob - proximal_logprob
        proximal_behavior_log_ratio = proximal_logprob - behavior_logprob
        log_ratios.extend((current_proximal_log_ratio, proximal_behavior_log_ratio))
    for name, log_ratio in zip(
        ("current/behavior", "current/proximal", "proximal/behavior"),
        log_ratios,
        strict=False,
    ):
        if not bool(torch.isfinite(log_ratio).all().item()):
            raise FloatingPointError(
                f"non-finite {name} log-ratio before exp: {tensor_range(log_ratio)}"
            )
    observed_max_abs = torch.stack([log_ratio.abs().max() for log_ratio in log_ratios]).max()
    if max_abs_log_ratio is not None and float(observed_max_abs.item()) > max_abs_log_ratio:
        return PolicySurrogateResult(
            None,
            None,
            current_behavior_log_ratio.detach(),
            None,
            None,
            None,
            observed_max_abs.detach(),
            "max_abs_log_ratio",
        )

    current_behavior_ratio = current_behavior_log_ratio.exp()
    if not bool(torch.isfinite(current_behavior_ratio).all().item()):
        raise FloatingPointError(
            "non-finite current/behavior policy ratio after exp: "
            f"log_ratio_min={current_behavior_log_ratio.min().item():.6g} "
            f"log_ratio_max={current_behavior_log_ratio.max().item():.6g}"
        )
    approx_kl = (current_behavior_ratio - 1.0 - current_behavior_log_ratio).mean()
    if target_kl is not None and float(approx_kl.item()) > target_kl:
        return PolicySurrogateResult(
            None,
            None,
            current_behavior_log_ratio.detach(),
            current_behavior_ratio.detach(),
            None,
            approx_kl.detach(),
            observed_max_abs.detach(),
            "target_kl",
        )

    if proximal_logprob is None:
        clipping_ratio = current_behavior_ratio
        importance_weight = torch.ones_like(clipping_ratio)
    else:
        assert current_proximal_log_ratio is not None
        assert proximal_behavior_log_ratio is not None
        clipping_ratio = current_proximal_log_ratio.exp()
        importance_weight = proximal_behavior_log_ratio.exp()
        if not bool(torch.isfinite(clipping_ratio).all().item()):
            raise FloatingPointError("non-finite current/proximal policy ratio after exp")
        if not bool(torch.isfinite(importance_weight).all().item()):
            raise FloatingPointError("non-finite proximal/behavior importance ratio after exp")

    surrogate = importance_weight * torch.min(
        clipping_ratio * advantages,
        clipping_ratio.clamp(1.0 - clip, 1.0 + clip) * advantages,
    )
    return PolicySurrogateResult(
        -surrogate.mean(),
        clipping_ratio.detach(),
        current_behavior_log_ratio.detach(),
        current_behavior_ratio.detach(),
        importance_weight.detach(),
        approx_kl.detach(),
        observed_max_abs.detach(),
        None,
    )
