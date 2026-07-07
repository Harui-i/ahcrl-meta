import argparse
import json
import shutil
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from ahcrl.nn import HypersphericalFeatureNorm, Scaler, project_hyperspherical_weights_

from .encoder import BOARD_SIZE, MAX_LEVEL, MAX_PLAYERS, PLANE_M, PLANE_U
from .model import ActorCritic
from .rust_vec_env import RustVecEnv

ROOT = Path(__file__).resolve().parents[4]
MODEL_DTYPE = torch.bfloat16
DEFAULT_CONFIG: dict[str, Any] = {
    "num_envs": 64,
    "total_steps": 200_000,
    "rollout_steps": 128,
    "seed_start": 0,
    "seed_stride": 1,
    "fixed_m": None,
    "fixed_u": None,
    "pf_particles": 16,
    "device": "auto",
    "compile": True,
    "lr": 3e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "epochs": 4,
    "minibatch_size": 1024,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "reward_scale_mode": "none",
    "reward_scale_epsilon": 1e-8,
    "weight_projection": False,
    "symmetry_augmentation": "none",
    "artifact_dir": ROOT / "contests/ahc-061/artifacts/ppo",
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
RESUME_ALLOWED_OVERRIDE_KEYS = {"total_steps"}
RUNTIME_CONFIG_KEYS = {"config", "resume_dir", "run_dir", "init_checkpoint"}
LATEST_CHECKPOINT_NAME = "checkpoint_latest.pt"
STATE_FILE_NAME = "state.json"
CONFIG_FILE_NAME = "config.json"


class RunningRewardScaler:
    def __init__(self, *, epsilon: float) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update_and_scale(self, rewards: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        values = rewards.float().flatten()
        batch_count = int(values.numel())
        if batch_count == 0:
            return rewards, self.stats()

        batch_mean = float(values.mean().item())
        batch_m2 = float(values.sub(batch_mean).pow(2).sum().item())
        self._merge(batch_count=batch_count, batch_mean=batch_mean, batch_m2=batch_m2)

        scale = self.scale
        return rewards / scale, self.stats()

    @property
    def variance(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.m2 / self.count

    @property
    def scale(self) -> float:
        return max(self.variance**0.5, self.epsilon)

    def stats(self) -> dict[str, float]:
        return {
            "reward_scale": self.scale,
            "reward_running_mean": self.mean,
            "reward_running_std": self.variance**0.5,
        }

    def state_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.count = int(state["count"])
        self.mean = float(state["mean"])
        self.m2 = float(state["m2"])
        self.epsilon = float(state.get("epsilon", self.epsilon))

    def _merge(self, *, batch_count: int, batch_mean: float, batch_m2: float) -> None:
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total_count
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total_count
        self.count = total_count


def main() -> None:
    args = parse_args()
    args.run_dir = prepare_run_dir(args)
    print_resolved_config(args)

    torch.manual_seed(args.seed_start)
    np.random.seed(args.seed_start)
    device = torch.device(args.device)
    raw_model = ActorCritic(
        channels=args.model_channels,
        blocks=args.model_blocks,
        block_type=args.model_block_type,
    ).to(device=device, dtype=MODEL_DTYPE)
    if args.init_checkpoint is not None:
        load_initial_model(args.init_checkpoint, raw_model, device)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)
    reward_scaler = (
        RunningRewardScaler(epsilon=args.reward_scale_epsilon)
        if args.reward_scale_mode == "running_std"
        else None
    )
    env = RustVecEnv(
        num_envs=args.num_envs,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        fixed_m=args.fixed_m,
        fixed_u=args.fixed_u,
        pf_particles=args.pf_particles,
    )

    obs = env.obs
    next_seed_start = _initial_next_seed_start(args)
    global_step = 0
    update = 0
    if args.resume_dir is not None:
        resume_state = load_training_state(args.resume_dir, raw_model, optimizer, device)
        global_step = resume_state["global_step"]
        update = resume_state["update"]
        if reward_scaler is not None and resume_state["reward_scaler_state"] is not None:
            reward_scaler.load_state_dict(resume_state["reward_scaler_state"])
        resume_seed_start = resume_state["next_seed_start"]
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        np.random.set_state(resume_state["numpy_rng_state"])
        env.reset(resume_seed_start, args.seed_stride, args.fixed_m, args.fixed_u)
        obs = env.obs
        next_seed_start = _advance_seed_start(resume_seed_start, args)
        if args.total_steps <= global_step:
            raise ValueError(
                f"total_steps ({args.total_steps}) must be greater than resumed "
                f"global_step ({global_step})"
            )

    model = cast(nn.Module, torch.compile(raw_model)) if args.compile else raw_model
    started = time.time()
    wandb_run = None
    try:
        wandb_run = init_wandb(args, wandb_run_id=get_wandb_run_id(args.run_dir))
        write_config(args)
        update_run_state(
            args.run_dir,
            global_step=global_step,
            update=update,
            next_seed_start=next_seed_start,
            wandb_run_id=None if wandb_run is None else wandb_run.id,
        )
        while global_step < args.total_steps:
            rollout = collect_rollout(
                model,
                env,
                obs,
                next_seed_start,
                args,
                device,
                reward_scaler=reward_scaler,
            )
            obs = rollout.pop("last_obs")
            next_seed_start = rollout.pop("next_seed_start")
            global_step += args.num_envs * args.rollout_steps
            update += 1
            stats = update_model(model, raw_model, optimizer, rollout, args, device)

            elapsed = max(time.time() - started, 1e-6)
            checkpoint_path = None
            is_last_update = global_step >= args.total_steps
            should_save = update % args.checkpoint_interval_updates == 0 or is_last_update
            if should_save:
                checkpoint_seed_start = next_seed_start
                checkpoint_path = save_training_state(
                    args,
                    raw_model,
                    optimizer,
                    global_step=global_step,
                    update=update,
                    next_seed_start=checkpoint_seed_start,
                    wandb_run_id=None if wandb_run is None else wandb_run.id,
                    reward_scaler=reward_scaler,
                )
                obs = env.reset(
                    checkpoint_seed_start,
                    args.seed_stride,
                    args.fixed_m,
                    args.fixed_u,
                )
                next_seed_start = _advance_seed_start(checkpoint_seed_start, args)
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
                        f"mean_final_score={log_metrics['train/final_mean_score']:.1f}",
                        f"min_final_score={log_metrics['train/final_min_score']:.1f}",
                        f"max_final_score={log_metrics['train/final_max_score']:.1f}",
                        f"mean_reward={log_metrics['train/mean_reward']:.5f}",
                        f"reward_scale={log_metrics['train/reward_scale']:.5f}",
                        f"policy_loss={stats['policy_loss']:.5f}",
                        f"value_loss={stats['value_loss']:.5f}",
                        f"entropy={stats['entropy']:.5f}",
                        f"normalized_entropy={stats['normalized_entropy']:.5f}",
                        f"approx_kl={stats['approx_kl']:.5f}",
                        f"clip_frac={stats['clip_frac']:.5f}",
                        f"grad_norm={stats['grad_norm']:.5f}",
                        f"weight_norm={stats['weight_norm']:.5f}",
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
    model: nn.Module,
    env: RustVecEnv,
    obs: dict[str, np.ndarray],
    next_seed_start: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    reward_scaler: RunningRewardScaler | None = None,
) -> dict[str, Any]:
    obs_buf = []
    action_buf = []
    logprob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    mask_buf = []
    score_buf = []
    m_buf = []
    u_buf = []

    for _ in range(args.rollout_steps):
        m_values, u_values = _extract_m_u_from_obs(obs)
        encoded = torch.from_numpy(obs["planes"]).to(device=device, dtype=MODEL_DTYPE)
        mask = torch.from_numpy(obs["mask"]).to(device)
        with torch.inference_mode():
            logits, value = model(encoded)
            logits = logits.float()
            value = value.float()
            logits = logits.masked_fill(~mask, -1e9)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        step = env.step(action.cpu().numpy())
        obs_buf.append(encoded.cpu())
        action_buf.append(action.cpu())
        logprob_buf.append(logprob.cpu())
        reward_buf.append(torch.from_numpy(step.reward.copy()))
        done_buf.append(torch.from_numpy(step.done.astype(np.float32)))
        value_buf.append(value.cpu())
        mask_buf.append(mask.cpu().clone())
        score_buf.append(torch.from_numpy(step.score.copy()))
        m_buf.append(torch.from_numpy(m_values.copy()))
        u_buf.append(torch.from_numpy(u_values.copy()))

        obs = step.obs
        if step.done.any():
            obs = env.reset(
                next_seed_start,
                args.seed_stride,
                args.fixed_m,
                args.fixed_u,
            )
            next_seed_start = _advance_seed_start(next_seed_start, args)

    with torch.inference_mode():
        next_encoded = torch.from_numpy(obs["planes"]).to(
            device=device,
            dtype=MODEL_DTYPE,
        )
        next_value = model(next_encoded)[1].float().cpu()

    rewards = torch.stack(reward_buf)
    if reward_scaler is None:
        scaled_rewards = rewards
        reward_scale_stats = {
            "reward_scale": 1.0,
            "reward_running_mean": float(rewards.float().mean().item()),
            "reward_running_std": float(rewards.float().std(unbiased=False).item()),
        }
    else:
        scaled_rewards, reward_scale_stats = reward_scaler.update_and_scale(rewards)
    dones = torch.stack(done_buf)
    values = torch.stack(value_buf)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(args.num_envs)
    for t in reversed(range(args.rollout_steps)):
        next_non_terminal = 1.0 - dones[t]
        next_values = next_value if t == args.rollout_steps - 1 else values[t + 1]
        delta = scaled_rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values

    return {
        "obs": torch.stack(obs_buf),
        "actions": torch.stack(action_buf),
        "logprobs": torch.stack(logprob_buf),
        "rewards": rewards,
        "scaled_rewards": scaled_rewards,
        "dones": dones,
        "values": values,
        "advantages": advantages,
        "returns": returns,
        "masks": torch.stack(mask_buf),
        "scores": torch.stack(score_buf),
        "m_values": torch.stack(m_buf),
        "u_values": torch.stack(u_buf),
        "last_obs": obs,
        "next_seed_start": next_seed_start,
        **reward_scale_stats,
    }


def _extract_m_u_from_obs(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    m_values = np.rint(obs["planes"][:, PLANE_M, 0, 0] * MAX_PLAYERS).astype(np.int64)
    u_values = np.rint(obs["planes"][:, PLANE_U, 0, 0] * MAX_LEVEL).astype(np.int64)
    return m_values, u_values


def _initial_next_seed_start(args: argparse.Namespace) -> int:
    return _advance_seed_start(args.seed_start, args)


def _advance_seed_start(seed_start: int, args: argparse.Namespace) -> int:
    return seed_start + args.num_envs * args.seed_stride


def prepare_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_dir is not None:
        return args.resume_dir

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.artifact_dir / f"run_{timestamp}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix:02d}")
        suffix += 1
    (run_dir / "checkpoints").mkdir(parents=True)
    return run_dir


def config_for_save(args: argparse.Namespace) -> dict[str, Any]:
    config = {}
    for key, value in sorted(vars(args).items()):
        if key in RUNTIME_CONFIG_KEYS:
            continue
        config[key] = _jsonable(value)
    return config


def write_config(args: argparse.Namespace) -> None:
    config_path = args.run_dir / CONFIG_FILE_NAME
    config_path.write_text(json.dumps(config_for_save(args), indent=2, sort_keys=True) + "\n")


def update_run_state(
    run_dir: Path,
    *,
    global_step: int,
    update: int,
    next_seed_start: int,
    wandb_run_id: str | None,
) -> None:
    state_path = run_dir / STATE_FILE_NAME
    previous: dict[str, Any] = {}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
    now = datetime.now().isoformat(timespec="seconds")
    state = {
        "created_at": previous.get("created_at", now),
        "updated_at": now,
        "global_step": global_step,
        "update": update,
        "next_seed_start": next_seed_start,
        "wandb_run_id": wandb_run_id if wandb_run_id is not None else previous.get("wandb_run_id"),
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def get_wandb_run_id(run_dir: Path) -> str | None:
    state_path = run_dir / STATE_FILE_NAME
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    run_id = state.get("wandb_run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def save_training_state(
    args: argparse.Namespace,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    update: int,
    next_seed_start: int,
    wandb_run_id: str | None,
    reward_scaler: RunningRewardScaler | None = None,
) -> Path:
    checkpoints_dir = args.run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"step_{global_step}.pt"
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_for_save(args),
        "global_step": global_step,
        "update": update,
        "next_seed_start": next_seed_start,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "reward_scaler": None if reward_scaler is None else reward_scaler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    shutil.copy2(checkpoint_path, args.run_dir / LATEST_CHECKPOINT_NAME)
    update_run_state(
        args.run_dir,
        global_step=global_step,
        update=update,
        next_seed_start=next_seed_start,
        wandb_run_id=wandb_run_id,
    )
    return checkpoint_path


def load_training_state(
    resume_dir: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = resume_dir / LATEST_CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"latest checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return {
        "global_step": int(checkpoint["global_step"]),
        "update": int(checkpoint["update"]),
        "next_seed_start": int(checkpoint["next_seed_start"]),
        "torch_rng_state": checkpoint["torch_rng_state"],
        "numpy_rng_state": checkpoint["numpy_rng_state"],
        "reward_scaler_state": checkpoint.get("reward_scaler"),
    }


def load_initial_model(
    checkpoint_path: Path,
    model: ActorCritic,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def update_model(
    model: nn.Module,
    grad_model: nn.Module,
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
    stat_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "normalized_entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "grad_norm": 0.0,
    }
    stat_weight = 0
    for _ in range(args.epochs):
        perm = indices[torch.randperm(batch_size)]
        for start in range(0, batch_size, args.minibatch_size):
            mb = perm[start : start + args.minibatch_size].to(device)
            mb_obs = obs[mb]
            mb_actions = actions[mb]
            mb_masks = masks[mb]
            mb_old_logprobs = old_logprobs[mb]
            mb_advantages = advantages[mb]
            mb_returns = returns[mb]
            transform_ids = range(8) if args.symmetry_augmentation == "full_d4" else range(1)
            for transform_id in transform_ids:
                aug_obs = _transform_board_d4(mb_obs, transform_id)
                aug_masks = _transform_flat_board_d4(mb_masks, transform_id)
                aug_actions = _transform_actions_d4(mb_actions, transform_id)
                logits, value = model(aug_obs)
                logits = logits.float()
                value = value.float()
                logits = logits.masked_fill(~aug_masks, -1e9)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(aug_actions)
                entropy_values = dist.entropy()
                entropy = entropy_values.mean()
                normalized_entropy = _normalized_entropy(entropy_values, aug_masks)
                ratio = (new_logprobs - mb_old_logprobs).exp()
                pg1 = -mb_advantages * ratio
                pg2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = F.mse_loss(value, mb_returns)
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    grad_model.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()
                if (
                    args.weight_projection
                    or getattr(args, "model_block_type", "") == "spherical_convnext"
                ):
                    project_hyperspherical_weights_(grad_model)

                with torch.no_grad():
                    log_ratio = new_logprobs - mb_old_logprobs
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > args.clip).float().mean()
                weight = int(mb_actions.numel())
                stat_weight += weight
                stat_sums["policy_loss"] += float(policy_loss.item()) * weight
                stat_sums["value_loss"] += float(value_loss.item()) * weight
                stat_sums["entropy"] += float(entropy.item()) * weight
                stat_sums["normalized_entropy"] += float(normalized_entropy.item()) * weight
                stat_sums["approx_kl"] += float(approx_kl.item()) * weight
                stat_sums["clip_frac"] += float(clip_frac.item()) * weight
                stat_sums["grad_norm"] += float(grad_norm.item()) * weight
    if stat_weight == 0:
        stat_sums.update(_parameter_norm_stats(grad_model))
        return stat_sums
    stats = {key: value / stat_weight for key, value in stat_sums.items()}
    stats.update(_parameter_norm_stats(grad_model))
    if isinstance(grad_model, ActorCritic):
        feature_stats = grad_model.trunk_feature_norm_stats(obs[: args.minibatch_size])
        stats.update(feature_stats)
    return stats


def _parameter_l2_norm(model: nn.Module) -> float:
    return _parameter_norm_stats(model)["weight_norm"]


def _parameter_norm_stats(model: nn.Module) -> dict[str, float]:
    with torch.no_grad():
        device = next(model.parameters()).device
        total_sq = torch.zeros((), device=device)
        linear_conv_sq = torch.zeros((), device=device)
        norm_affine_sq = torch.zeros((), device=device)
        hyperspherical_scale_sq = torch.zeros((), device=device)
        parameter_count = 0

        linear_conv_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        norm_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
        )
        for module in model.modules():
            for name, parameter in module.named_parameters(recurse=False):
                values = parameter.detach().float()
                parameter_sq = values.pow(2).sum()
                total_sq += parameter_sq
                parameter_count += parameter.numel()

                if isinstance(module, HypersphericalFeatureNorm | Scaler):
                    hyperspherical_scale_sq += parameter_sq
                elif isinstance(module, linear_conv_modules) and name == "weight":
                    linear_conv_sq += parameter_sq
                elif isinstance(module, norm_modules):
                    norm_affine_sq += parameter_sq

        if parameter_count == 0:
            param_rms = 0.0
        else:
            param_rms = float((total_sq / parameter_count).sqrt().item())
        return {
            "weight_norm": float(total_sq.sqrt().item()),
            "linear_conv_weight_norm": float(linear_conv_sq.sqrt().item()),
            "norm_affine_norm": float(norm_affine_sq.sqrt().item()),
            "hyperspherical_scale_norm": float(hyperspherical_scale_sq.sqrt().item()),
            "param_rms": param_rms,
        }


def _transform_board_d4(x: torch.Tensor, transform_id: int) -> torch.Tensor:
    if transform_id < 0 or transform_id >= 8:
        raise ValueError(f"transform_id must be in [0, 8), got {transform_id}")
    transformed = torch.flip(x, dims=(-1,)) if transform_id >= 4 else x
    rotations = transform_id % 4
    if rotations:
        transformed = torch.rot90(transformed, k=rotations, dims=(-2, -1))
    return transformed


def _transform_flat_board_d4(x: torch.Tensor, transform_id: int) -> torch.Tensor:
    original_shape = x.shape
    if original_shape[-1] != BOARD_SIZE * BOARD_SIZE:
        raise ValueError(
            f"flat board last dimension must be {BOARD_SIZE * BOARD_SIZE}, got {original_shape[-1]}"
        )
    board = x.reshape(*original_shape[:-1], BOARD_SIZE, BOARD_SIZE)
    return _transform_board_d4(board, transform_id).reshape(original_shape)


def _transform_actions_d4(actions: torch.Tensor, transform_id: int) -> torch.Tensor:
    if transform_id < 0 or transform_id >= 8:
        raise ValueError(f"transform_id must be in [0, 8), got {transform_id}")
    x = torch.div(actions, BOARD_SIZE, rounding_mode="floor")
    y = actions.remainder(BOARD_SIZE)
    if transform_id >= 4:
        y = BOARD_SIZE - 1 - y
    for _ in range(transform_id % 4):
        x, y = BOARD_SIZE - 1 - y, x
    return x * BOARD_SIZE + y


def _normalized_entropy(entropy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_action_count = mask.sum(dim=-1).to(entropy.dtype)
    max_entropy = valid_action_count.log()
    normalized = torch.where(
        valid_action_count > 1,
        entropy / max_entropy.clamp_min(1e-8),
        torch.zeros_like(entropy),
    )
    return normalized.mean()


def init_wandb(args: argparse.Namespace, *, wandb_run_id: str | None = None) -> Any | None:
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
        id=wandb_run_id,
        resume="must" if wandb_run_id is not None else None,
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
    final_scores = scores[-1]
    rewards = rollout["rewards"].float()
    scaled_rewards = rollout.get("scaled_rewards", rollout["rewards"]).float()
    dones = rollout["dones"].float()
    values = rollout["values"].float()
    advantages = rollout["advantages"].float()
    returns = rollout["returns"].float()
    masks = rollout["masks"].float()
    checkpoint = "" if checkpoint_path is None else str(checkpoint_path)
    return_variance = returns.var(unbiased=False)
    explained_variance = (
        torch.tensor(float("nan"), device=returns.device)
        if return_variance <= 1e-8
        else 1.0 - (returns - values).var(unbiased=False) / return_variance
    )

    metrics: dict[str, float | int | str] = {
        "summary/cumulative_env_steps": global_step,
        "summary/updates": update,
        "summary/elapsed_sec": elapsed,
        "summary/fps": global_step / max(elapsed, 1e-6),
        "train/mean_score": float(scores.mean().item()),
        "train/max_score": float(scores.max().item()),
        "train/min_score": float(scores.min().item()),
        "train/final_mean_score": float(final_scores.mean().item()),
        "train/final_max_score": float(final_scores.max().item()),
        "train/final_min_score": float(final_scores.min().item()),
        "train/mean_reward": float(rewards.mean().item()),
        "train/sum_reward": float(rewards.sum().item()),
        "train/mean_scaled_reward": float(scaled_rewards.mean().item()),
        "train/reward_scale": float(rollout.get("reward_scale", 1.0)),
        "train/reward_running_mean": float(
            rollout.get("reward_running_mean", rewards.mean().item())
        ),
        "train/reward_running_std": float(
            rollout.get("reward_running_std", rewards.std(unbiased=False).item())
        ),
        "train/done_count": int(dones.sum().item()),
        "train/mean_value": float(values.mean().item()),
        "train/mean_return": float(returns.mean().item()),
        "train/explained_variance": float(explained_variance.item()),
        "train/mean_advantage": float(advantages.mean().item()),
        "train/std_advantage": float(advantages.std(unbiased=False).item()),
        "train/valid_action_fraction": float(masks.mean().item()),
        "loss/policy": stats["policy_loss"],
        "loss/value": stats["value_loss"],
        "train/entropy": stats["entropy"],
        "train/normalized_entropy": stats["normalized_entropy"],
        "train/approx_kl": stats["approx_kl"],
        "train/clip_fraction": stats["clip_frac"],
        "model/grad_norm": stats["grad_norm"],
        "model/weight_norm": stats["weight_norm"],
        "model/linear_conv_weight_norm": stats["linear_conv_weight_norm"],
        "model/norm_affine_norm": stats["norm_affine_norm"],
        "model/hyperspherical_scale_norm": stats["hyperspherical_scale_norm"],
        "model/param_rms": stats["param_rms"],
        "model/trunk_feature_norm_mean": stats.get("trunk_feature_norm_mean", float("nan")),
        "model/trunk_feature_norm_std": stats.get("trunk_feature_norm_std", float("nan")),
        "model/trunk_feature_norm_max": stats.get("trunk_feature_norm_max", float("nan")),
        "checkpoint/path": checkpoint,
    }
    if "m_values" in rollout:
        final_m_values = rollout["m_values"][-1].long()
        for m_value in torch.unique(final_m_values).tolist():
            selector = final_m_values == m_value
            metrics[f"train/final_mean_score_by_m/m_{m_value}"] = float(
                final_scores[selector].mean().item()
            )
    if "u_values" in rollout:
        final_u_values = rollout["u_values"][-1].long()
        for u_value in torch.unique(final_u_values).tolist():
            selector = final_u_values == u_value
            metrics[f"train/final_mean_score_by_u/u_{u_value}"] = float(
                final_scores[selector].mean().item()
            )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to a TOML config file.")
    parser.add_argument("--resume-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--init-checkpoint", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--num-envs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--total-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--rollout-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed-start", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed-stride", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fixed-m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fixed-u", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--pf-particles", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gae-lambda", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--clip", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--minibatch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--entropy-coef", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--value-coef", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max-grad-norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--reward-scale-mode",
        choices=("none", "running_std"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--reward-scale-epsilon", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--weight-projection",
        dest="weight_projection",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-weight-projection",
        dest="weight_projection",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--symmetry-augmentation",
        choices=("none", "full_d4"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--artifact-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--model-channels", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--model-blocks", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--model-block-type",
        choices=("convnext", "per_cell_mlp", "residual", "spherical_convnext"),
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
    resume_dir = cli_config.get("resume_dir")
    if resume_dir is not None and "init_checkpoint" in cli_config:
        raise ValueError("resume_dir and init_checkpoint are mutually exclusive")

    config = load_saved_config(resume_dir) if resume_dir is not None else DEFAULT_CONFIG.copy()
    file_config = {}
    if config_path is not None:
        file_config = load_toml_config(config_path)
    if resume_dir is not None:
        validate_resume_overrides(file_config)
        validate_resume_overrides(
            {key: value for key, value in cli_config.items() if key != "resume_dir"}
        )
    config.update(file_config)
    config.update(cli_config)
    config["artifact_dir"] = Path(config["artifact_dir"])
    config["resume_dir"] = None if config.get("resume_dir") is None else Path(config["resume_dir"])
    config["init_checkpoint"] = (
        None if config.get("init_checkpoint") is None else Path(config["init_checkpoint"])
    )
    if config["checkpoint_interval_updates"] <= 0:
        raise ValueError("checkpoint_interval_updates must be positive")
    if config["pf_particles"] <= 0:
        raise ValueError("pf_particles must be positive")
    if config["reward_scale_mode"] not in ("none", "running_std"):
        raise ValueError("reward_scale_mode must be one of: none, running_std")
    if config["reward_scale_epsilon"] <= 0.0:
        raise ValueError("reward_scale_epsilon must be positive")
    if config["symmetry_augmentation"] not in ("none", "full_d4"):
        raise ValueError("symmetry_augmentation must be one of: none, full_d4")
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return argparse.Namespace(**config)


def validate_resume_overrides(overrides: dict[str, Any]) -> None:
    disallowed = sorted(set(overrides) - RESUME_ALLOWED_OVERRIDE_KEYS)
    if disallowed:
        raise ValueError(
            "resume only allows overriding total_steps; disallowed keys: "
            + ", ".join(disallowed)
        )


def load_saved_config(resume_dir: Path) -> dict[str, Any]:
    config_path = resume_dir / CONFIG_FILE_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"saved config not found: {config_path}")
    raw = json.loads(config_path.read_text())
    unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown saved config keys: {', '.join(unknown)}")
    config = DEFAULT_CONFIG.copy()
    config.update(raw)
    return config


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
        key: _jsonable(value)
        for key, value in sorted(vars(args).items())
    }
    print("config=" + json.dumps(printable, sort_keys=True), flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
