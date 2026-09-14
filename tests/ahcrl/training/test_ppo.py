import pytest
import torch

from ahcrl.training.ppo import policy_surrogate


def test_policy_surrogate_matches_standard_clipped_ppo() -> None:
    new = torch.tensor([-0.2, -1.4])
    old = torch.tensor([-0.3, -1.0])
    advantages = torch.tensor([1.5, -0.5])

    result = policy_surrogate(
        new_logprob=new,
        behavior_logprob=old,
        advantages=advantages,
        clip=0.2,
    )
    ratio = (new - old).exp()
    expected = -torch.min(
        ratio * advantages,
        ratio.clamp(0.8, 1.2) * advantages,
    ).mean()

    assert result.stop_reason is None
    assert result.loss is not None
    assert torch.equal(result.loss, expected)


def test_policy_surrogate_stops_before_extreme_ratio_exp() -> None:
    result = policy_surrogate(
        new_logprob=torch.tensor([1000.0]),
        behavior_logprob=torch.tensor([0.0]),
        advantages=torch.ones(1),
        clip=0.2,
        max_abs_log_ratio=20.0,
        target_kl=0.03,
    )

    assert result.loss is None
    assert result.current_behavior_ratio is None
    assert result.stop_reason == "max_abs_log_ratio"


def test_policy_surrogate_stops_at_target_kl() -> None:
    result = policy_surrogate(
        new_logprob=torch.tensor([0.5]),
        behavior_logprob=torch.tensor([0.0]),
        advantages=torch.ones(1),
        clip=0.2,
        max_abs_log_ratio=20.0,
        target_kl=0.03,
    )

    assert result.loss is None
    assert result.current_behavior_approx_kl == pytest.approx(torch.tensor(0.14872127))
    assert result.stop_reason == "target_kl"
