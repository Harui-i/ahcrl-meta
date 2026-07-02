import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .encoder import encode_batch
from .model import ActorCritic
from .rust_vec_env import RustVecEnv

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG: dict[str, Any] = {
    "num_envs": 64,
    "total_steps": 200_000,
    "rollout_steps": 128,
    "seed_start": 0,
    "seed_stride": 1,
    "fixed_m": None,
    "fixed_u": None,
    "device": "auto",
    "lr": 3e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "epochs": 4,
    "minibatch_size": 1024,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "checkpoint_dir": ROOT / "contests/ahc-061/artifacts/ppo",
    "checkpoint_interval_updates": 1,
    "model_channels": 64,
    "model_blocks": 4,
    "model_block_type": "convnext",
    "wandb_enabled": False,
    "wandb_project": "ahcrl-meta",
    "wandb_entity": None,
    "wandb_name": None,
    "wandb_mode": "online",
    "wandb_tags": [],
}


def main() -> None:
    args = parse_args()
    print_resolved_config(args)

    torch.manual_seed(args.seed_start)
    np.random.seed(args.seed_start)
    device = torch.device(args.device)
    model = ActorCritic(
        channels=args.model_channels,
        blocks=args.model_blocks,
        block_type=args.model_block_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    env = RustVecEnv(
        num_envs=args.num_envs,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        fixed_m=args.fixed_m,
        fixed_u=args.fixed_u,
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    obs = env.obs
    next_seed_start = _initial_next_seed_start(args)
    global_step = 0
    update = 0
    started = time.time()
    wandb_run = None
    try:
        wandb_run = init_wandb(args)
        while global_step < args.total_steps:
            rollout = collect_rollout(model, env, obs, next_seed_start, args, device)
            obs = rollout.pop("last_obs")
            next_seed_start = rollout.pop("next_seed_start")
            global_step += args.num_envs * args.rollout_steps
            update += 1
            stats = update_model(model, optimizer, rollout, args, device)

            elapsed = max(time.time() - started, 1e-6)
            checkpoint_path = None
            is_last_update = global_step >= args.total_steps
            should_save = update % args.checkpoint_interval_updates == 0 or is_last_update
            if should_save:
                checkpoint_path = args.checkpoint_dir / f"ppo_step{global_step}.pt"
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "step": global_step},
                    checkpoint_path,
                )
            log_metrics = build_log_metrics(
                update=update,
                global_step=global_step,
                elapsed=elapsed,
                rollout=rollout,
                stats=stats,
                checkpoint_path=checkpoint_path,
            )
            if wandb_run is not None:
                wandb_run.log(log_metrics, step=global_step)
                wandb_run.summary["summary/cumulative_env_steps"] = global_step
                wandb_run.summary["summary/updates"] = update
                wandb_run.summary["summary/fps"] = log_metrics["summary/fps"]
                wandb_run.summary["train/mean_score"] = log_metrics["train/mean_score"]
            print(
                " ".join(
                    [
                        f"update={update}",
                        f"step={global_step}",
                        f"fps={log_metrics['summary/fps']:.1f}",
                        f"mean_score={log_metrics['train/mean_score']:.1f}",
                        f"mean_reward={log_metrics['train/mean_reward']:.5f}",
                        f"policy_loss={stats['policy_loss']:.5f}",
                        f"value_loss={stats['value_loss']:.5f}",
                        f"entropy={stats['entropy']:.5f}",
                        f"approx_kl={stats['approx_kl']:.5f}",
                        f"clip_frac={stats['clip_frac']:.5f}",
                        f"checkpoint={checkpoint_path}",
                    ]
                ),
                flush=True,
            )
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


def collect_rollout(
    model: ActorCritic,
    env: RustVecEnv,
    obs: dict[str, np.ndarray],
    next_seed_start: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    obs_buf = []
    action_buf = []
    logprob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    mask_buf = []
    score_buf = []

    for _ in range(args.rollout_steps):
        encoded = torch.from_numpy(encode_batch(obs)).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        with torch.no_grad():
            logits, value = model(encoded)
            logits = logits.masked_fill(~mask, -1e9)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        step = env.step(action.cpu().numpy().astype(np.int64))
        obs_buf.append(encoded.cpu())
        action_buf.append(action.cpu())
        logprob_buf.append(logprob.cpu())
        reward_buf.append(torch.from_numpy(step.reward.copy()))
        done_buf.append(torch.from_numpy(step.done.astype(np.float32)))
        value_buf.append(value.cpu())
        mask_buf.append(mask.cpu())
        score_buf.append(torch.from_numpy(step.score.copy()))

        obs = step.obs
        if step.done.any():
            obs = env.reset(
                next_seed_start,
                args.seed_stride,
                args.fixed_m,
                args.fixed_u,
            )
            next_seed_start = _advance_seed_start(next_seed_start, args)

    with torch.no_grad():
        next_value = model(torch.from_numpy(encode_batch(obs)).to(device))[1].cpu()

    rewards = torch.stack(reward_buf)
    dones = torch.stack(done_buf)
    values = torch.stack(value_buf)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(args.num_envs)
    for t in reversed(range(args.rollout_steps)):
        next_non_terminal = 1.0 - dones[t]
        next_values = next_value if t == args.rollout_steps - 1 else values[t + 1]
        delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values

    return {
        "obs": torch.stack(obs_buf),
        "actions": torch.stack(action_buf),
        "logprobs": torch.stack(logprob_buf),
        "rewards": rewards,
        "dones": dones,
        "values": values,
        "advantages": advantages,
        "returns": returns,
        "masks": torch.stack(mask_buf),
        "scores": torch.stack(score_buf),
        "last_obs": obs,
        "next_seed_start": next_seed_start,
    }


def _initial_next_seed_start(args: argparse.Namespace) -> int:
    return _advance_seed_start(args.seed_start, args)


def _advance_seed_start(seed_start: int, args: argparse.Namespace) -> int:
    return seed_start + args.num_envs * args.seed_stride


def update_model(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    obs = rollout["obs"].flatten(0, 1).to(device)  # type: ignore[union-attr]
    actions = rollout["actions"].flatten().to(device)  # type: ignore[union-attr]
    old_logprobs = rollout["logprobs"].flatten().to(device)  # type: ignore[union-attr]
    advantages = rollout["advantages"].flatten().to(device)  # type: ignore[union-attr]
    returns = rollout["returns"].flatten().to(device)  # type: ignore[union-attr]
    masks = rollout["masks"].flatten(0, 1).to(device)  # type: ignore[union-attr]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    batch_size = obs.shape[0]
    indices = torch.arange(batch_size)
    last_stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
    }
    for _ in range(args.epochs):
        perm = indices[torch.randperm(batch_size)]
        for start in range(0, batch_size, args.minibatch_size):
            mb = perm[start : start + args.minibatch_size].to(device)
            logits, value = model(obs[mb])
            logits = logits.masked_fill(~masks[mb], -1e9)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()
            ratio = (new_logprobs - old_logprobs[mb]).exp()
            pg1 = -advantages[mb] * ratio
            pg2 = -advantages[mb] * torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
            policy_loss = torch.max(pg1, pg2).mean()
            value_loss = F.mse_loss(value, returns[mb])
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_logprobs - old_logprobs[mb]
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_frac = ((ratio - 1.0).abs() > args.clip).float().mean()
            last_stats = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "approx_kl": float(approx_kl.item()),
                "clip_frac": float(clip_frac.item()),
            }
    return last_stats


def init_wandb(args: argparse.Namespace) -> Any | None:
    if not args.wandb_enabled:
        return None

    import wandb

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=config,
    )
    wandb.define_metric("summary/cumulative_env_steps")
    wandb.define_metric("*", step_metric="summary/cumulative_env_steps")
    return run


def build_log_metrics(
    *,
    update: int,
    global_step: int,
    elapsed: float,
    rollout: dict[str, Any],
    stats: dict[str, float],
    checkpoint_path: Path | None,
) -> dict[str, float | int | str]:
    scores = rollout["scores"].float()
    rewards = rollout["rewards"].float()
    dones = rollout["dones"].float()
    values = rollout["values"].float()
    advantages = rollout["advantages"].float()
    returns = rollout["returns"].float()
    masks = rollout["masks"].float()
    checkpoint = "" if checkpoint_path is None else str(checkpoint_path)

    return {
        "summary/cumulative_env_steps": global_step,
        "summary/updates": update,
        "summary/elapsed_sec": elapsed,
        "summary/fps": global_step / max(elapsed, 1e-6),
        "train/mean_score": float(scores.mean().item()),
        "train/max_score": float(scores.max().item()),
        "train/min_score": float(scores.min().item()),
        "train/final_mean_score": float(scores[-1].mean().item()),
        "train/mean_reward": float(rewards.mean().item()),
        "train/sum_reward": float(rewards.sum().item()),
        "train/done_count": int(dones.sum().item()),
        "train/mean_value": float(values.mean().item()),
        "train/mean_return": float(returns.mean().item()),
        "train/mean_advantage": float(advantages.mean().item()),
        "train/std_advantage": float(advantages.std(unbiased=False).item()),
        "train/valid_action_fraction": float(masks.mean().item()),
        "loss/policy": stats["policy_loss"],
        "loss/value": stats["value_loss"],
        "train/entropy": stats["entropy"],
        "train/approx_kl": stats["approx_kl"],
        "train/clip_fraction": stats["clip_frac"],
        "checkpoint/path": checkpoint,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to a TOML config file.")
    parser.add_argument("--num-envs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--total-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--rollout-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed-start", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed-stride", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fixed-m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fixed-u", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gae-lambda", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--clip", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--minibatch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--entropy-coef", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--value-coef", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max-grad-norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--model-channels", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--model-blocks", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--model-block-type",
        choices=("convnext", "residual"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wandb",
        dest="wandb_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--wandb-project", default=argparse.SUPPRESS)
    parser.add_argument("--wandb-entity", default=argparse.SUPPRESS)
    parser.add_argument("--wandb-name", default=argparse.SUPPRESS)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wandb-tag",
        dest="wandb_tags",
        action="append",
        default=argparse.SUPPRESS,
    )
    parsed = parser.parse_args(argv)

    cli_config = vars(parsed).copy()
    config_path = cli_config.pop("config", None)
    config = DEFAULT_CONFIG.copy()
    if config_path is not None:
        config.update(load_toml_config(config_path))
    config.update(cli_config)
    config["checkpoint_dir"] = Path(config["checkpoint_dir"])
    if config["checkpoint_interval_updates"] <= 0:
        raise ValueError("checkpoint_interval_updates must be positive")
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return argparse.Namespace(**config)


def load_toml_config(path: Path) -> dict[str, Any]:
    if path.suffix != ".toml":
        raise ValueError(f"config must be a .toml file: {path}")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if "train" in raw:
        raw = raw["train"]
    unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    return raw


def print_resolved_config(args: argparse.Namespace) -> None:
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
    }
    print("config=" + json.dumps(printable, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
