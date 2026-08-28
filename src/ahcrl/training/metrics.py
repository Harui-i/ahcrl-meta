"""PPO trainer 共通のログメトリクス。"""

from collections.abc import Mapping

import torch


def build_completed_episode_score_metrics(
    scores: torch.Tensor,
    dones: torch.Tensor,
) -> dict[str, float | int]:
    """終了したエピソードだけの score 集計を返す。

    ``score`` は RustVecEnv の共通プロトコルで提供される、環境ごとの
    コンテスト定義スコアである。未終了エピソードの途中経過を混ぜないため、
    ``done`` の位置だけを W&B に記録する。
    """
    completed_scores = scores.double()[dones.bool()]
    count = int(completed_scores.numel())
    metrics: dict[str, float | int] = {"episode/completed_count": count}
    if count == 0:
        return metrics
    metrics |= {
        "episode/score_mean": float(completed_scores.mean().item()),
        "episode/score_min": float(completed_scores.min().item()),
        "episode/score_max": float(completed_scores.max().item()),
        "episode/score_std": float(completed_scores.std(unbiased=False).item()),
    }
    return metrics


def build_standard_ppo_metrics(
    *,
    update: int,
    global_step: int,
    elapsed: float,
    rollout: Mapping[str, torch.Tensor],
    update_stats: Mapping[str, float],
) -> dict[str, float | int]:
    """コンテスト評価に依存しない PPO の診断メトリクスを返す。"""
    rewards = rollout["rewards"].float()
    scaled_rewards = rollout.get("scaled_rewards", rollout["rewards"]).float()
    dones = rollout["dones"].float()
    values = rollout["values"].float()
    advantages = rollout["advantages"].float()
    returns = rollout["returns"].float()
    masks = rollout.get("masks")
    return_variance = returns.var(unbiased=False)
    explained_variance = float("nan")
    if float(return_variance.item()) > 1e-8:
        explained_variance = float(
            (1.0 - (returns - values).var(unbiased=False) / return_variance).item()
        )
    valid_action_fraction = 1.0 if masks is None else float(masks.float().mean().item())

    metrics: dict[str, float | int] = {
        "summary/cumulative_env_steps": global_step,
        "summary/updates": update,
        "summary/elapsed_sec": elapsed,
        "summary/fps": global_step / max(elapsed, 1e-6),
        "train/mean_reward": float(rewards.mean().item()),
        "train/sum_reward": float(rewards.sum().item()),
        "train/mean_scaled_reward": float(scaled_rewards.mean().item()),
        "train/done_count": int(dones.sum().item()),
        "train/mean_value": float(values.mean().item()),
        "train/mean_return": float(returns.mean().item()),
        "train/explained_variance": explained_variance,
        "train/mean_advantage": float(advantages.mean().item()),
        "train/std_advantage": float(advantages.std(unbiased=False).item()),
        "train/valid_action_fraction": valid_action_fraction,
        "loss/policy": update_stats["policy_loss"],
        "loss/value": update_stats["value_loss"],
        "loss/weighted_policy": update_stats["weighted_policy_loss"],
        "loss/weighted_value": update_stats["weighted_value_loss"],
        "loss/entropy": update_stats["entropy_loss"],
        "loss/total": update_stats["total_loss"],
        "train/entropy": update_stats["entropy"],
        "train/clip_fraction": update_stats["clip_frac"],
        "model/grad_norm": update_stats["grad_norm"],
    }
    scores = rollout.get("scores")
    if scores is not None:
        metrics |= build_completed_episode_score_metrics(scores, dones)
    return metrics
