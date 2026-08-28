from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ahcrl.training.metrics import build_completed_episode_score_metrics, build_standard_ppo_metrics
from ahcrl.training.wandb import WandbConfig, finish_wandb, init_wandb


def test_standard_ppo_metrics_contains_only_common_diagnostics() -> None:
    rollout = {
        "rewards": torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        "scaled_rewards": torch.tensor([[0.5, 1.5], [2.5, 3.5]]),
        "dones": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        "values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "advantages": torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
        "returns": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        "masks": torch.tensor([[[True, False], [True, True]]]),
    }
    stats = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "weighted_policy_loss": 0.1,
        "weighted_value_loss": 0.1,
        "entropy_loss": -0.03,
        "total_loss": 0.17,
        "entropy": 0.3,
        "clip_frac": 0.4,
        "grad_norm": 0.5,
    }

    metrics = build_standard_ppo_metrics(
        update=2,
        global_step=20,
        elapsed=4.0,
        rollout=rollout,
        update_stats=stats,
    )

    assert metrics["summary/fps"] == 5.0
    assert metrics["summary/updates"] == 2
    assert metrics["train/mean_reward"] == 4.0
    assert metrics["train/mean_scaled_reward"] == 2.0
    assert metrics["train/std_advantage"] == 1.0
    assert metrics["train/explained_variance"] == pytest.approx(1.0)
    assert metrics["train/valid_action_fraction"] == pytest.approx(0.75)
    assert metrics["loss/total"] == 0.17
    assert metrics["model/grad_norm"] == 0.5
    assert not any("score" in key or "prefix" in key for key in metrics)


def test_completed_episode_score_metrics_selects_only_done_episodes() -> None:
    metrics = build_completed_episode_score_metrics(
        scores=torch.tensor([[10, 99], [30, 40]]),
        dones=torch.tensor([[False, True], [True, False]]),
    )

    assert metrics == {
        "episode/completed_count": 2,
        "episode/score_mean": 64.5,
        "episode/score_min": 30.0,
        "episode/score_max": 99.0,
        "episode/score_std": 34.5,
    }


def test_standard_ppo_metrics_includes_completed_episode_scores_when_available() -> None:
    rollout = {
        "rewards": torch.tensor([[1.0, 3.0]]),
        "dones": torch.tensor([[0.0, 1.0]]),
        "scores": torch.tensor([[10, 20]]),
        "values": torch.tensor([[1.0, 2.0]]),
        "advantages": torch.tensor([[-1.0, 1.0]]),
        "returns": torch.tensor([[2.0, 3.0]]),
    }
    stats = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "weighted_policy_loss": 0.1,
        "weighted_value_loss": 0.1,
        "entropy_loss": -0.03,
        "total_loss": 0.17,
        "entropy": 0.3,
        "clip_frac": 0.4,
        "grad_norm": 0.5,
    }

    metrics = build_standard_ppo_metrics(
        update=1,
        global_step=2,
        elapsed=1.0,
        rollout=rollout,
        update_stats=stats,
    )

    assert metrics["episode/completed_count"] == 1
    assert metrics["episode/score_mean"] == 20.0


def test_wandb_initialization_and_resume_are_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {"defined": []}

    class FakeRun:
        id = "new-run"

        def finish(self) -> None:
            calls["finished"] = True

    def init(**kwargs: Any) -> FakeRun:
        calls["init"] = kwargs
        return FakeRun()

    fake_wandb = SimpleNamespace(
        init=init,
        define_metric=lambda *args, **kwargs: calls["defined"].append((args, kwargs)),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    run = init_wandb(
        WandbConfig(
            enabled=True,
            project="project",
            entity="entity",
            name="name",
            mode="offline",
            tags=["tag"],
        ),
        resolved_config={"value": 1},
        run_id="previous-run",
    )
    finish_wandb(run)

    assert calls["init"] == {
        "project": "project",
        "entity": "entity",
        "name": "name",
        "mode": "offline",
        "tags": ["tag"],
        "config": {"value": 1},
        "id": "previous-run",
        "resume": "must",
    }
    assert calls["defined"] == [
        (("summary/cumulative_env_steps",), {}),
        (("*",), {"step_metric": "summary/cumulative_env_steps"}),
    ]
    assert calls["finished"] is True
