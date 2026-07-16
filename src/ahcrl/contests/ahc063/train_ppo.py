"""PPO trainer for AHC063 using a SphericalAttentionSimba policy."""

import argparse
import json
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .encoder import NUM_PLANES
from .model import ActorCritic, RunningObservationNormalizer
from .vec_env import OuroborosVecEnv

ROOT = Path(__file__).resolve().parents[4]
MODEL_DTYPE = torch.bfloat16
DEFAULT_CONFIG: dict[str, Any] = {
    "num_envs": 64,
    "total_steps": 2_000_000,
    "rollout_steps": 128,
    "seed_start": 0,
    "seed_stride": 1,
    "fixed_n": None,
    "fixed_m": None,
    "fixed_c": None,
    "max_episode_steps": 100_000,
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
    "reward_scale": True,
    "obs_norm": True,
    "obs_norm_epsilon": 1e-8,
    "artifact_dir": ROOT / "contests/ahc-063/artifacts/ppo",
    "checkpoint_interval_updates": 20,
    "model_channels": 128,
    "model_blocks": 3,
    "model_block_type": "convnext",
    "wandb_enabled": False,
    "wandb_project": "ahcrl-meta",
    "wandb_name": None,
}
RUNTIME_KEYS = {"config", "resume_dir", "run_dir", "init_checkpoint"}
CHECKPOINT_NAME = "checkpoint_latest.pt"


class RunningRewardScaler:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def scale(self, rewards: torch.Tensor) -> torch.Tensor:
        values = rewards.detach().float().flatten()
        if values.numel():
            batch_count = int(values.numel())
            batch_mean = float(values.mean())
            batch_m2 = float((values - batch_mean).square().sum())
            if self.count == 0:
                self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            else:
                total = self.count + batch_count
                delta = batch_mean - self.mean
                self.mean += delta * batch_count / total
                self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
                self.count = total
        std = max((self.m2 / max(self.count, 1)) ** 0.5, self.epsilon)
        return rewards / std

    def state_dict(self) -> dict[str, float | int]:
        return {"epsilon": self.epsilon, "count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epsilon = float(state["epsilon"])
        self.count = int(state["count"])
        self.mean = float(state["mean"])
        self.m2 = float(state["m2"])


class FP32MasterWeights:
    """Keep bf16 model parameters with fp32 optimizer parameters and states."""

    def __init__(self, model: nn.Module) -> None:
        self.model_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        self.parameters = [
            nn.Parameter(parameter.detach().float().clone()) for parameter in self.model_parameters
        ]

    @torch.no_grad()
    def copy_model_to_master(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.parameters, strict=True
        ):
            master_parameter.copy_(model_parameter.float())

    @torch.no_grad()
    def copy_master_to_model(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.parameters, strict=True
        ):
            model_parameter.copy_(master_parameter.to(dtype=model_parameter.dtype))

    def copy_gradients_from_model(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.parameters, strict=True
        ):
            if model_parameter.grad is None:
                master_parameter.grad = None
            else:
                master_parameter.grad = model_parameter.grad.detach().float().clone()

    def first_nonfinite_gradient(self) -> int | None:
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                return index
        return None

    def state_dict(self) -> list[torch.Tensor]:
        return [parameter.detach().cpu().clone() for parameter in self.parameters]

    @torch.no_grad()
    def load_state_dict(self, state: list[torch.Tensor]) -> None:
        if len(state) != len(self.parameters):
            raise ValueError(
                f"master weight count mismatch: checkpoint has {len(state)}, "
                f"model has {len(self.parameters)}"
            )
        for parameter, saved in zip(self.parameters, state, strict=True):
            parameter.copy_(saved.to(device=parameter.device, dtype=torch.float32))


def create_model(args: argparse.Namespace, device: torch.device) -> ActorCritic:
    model = ActorCritic(
        channels=args.model_channels,
        blocks=args.model_blocks,
        block_type=args.model_block_type,
    ).to(device=device)
    if device.type == "cuda":
        model = model.to(dtype=MODEL_DTYPE)
    if args.obs_norm:
        model.observation_normalizer = RunningObservationNormalizer(
            NUM_PLANES, args.obs_norm_epsilon
        ).to(device=device)
    return model


def _model_forward(
    model: nn.Module, observations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(observations)  # type: ignore[call-arg]


def _observation_normalizer(model: nn.Module) -> RunningObservationNormalizer | None:
    original = getattr(model, "_orig_mod", model)
    normalizer = getattr(original, "observation_normalizer", None)
    return normalizer if isinstance(normalizer, RunningObservationNormalizer) else None


def _first_nonfinite_model_output(logits: torch.Tensor, value: torch.Tensor) -> str | None:
    if not bool(torch.isfinite(logits).all().item()):
        return "policy logits"
    if not bool(torch.isfinite(value).all().item()):
        return "value output"
    return None


def _first_nonfinite_gradient(model: nn.Module) -> str | None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            return name
    return None


def collect_rollout(
    model: nn.Module,
    env: OuroborosVecEnv,
    obs: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    reward_scaler: RunningRewardScaler | None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    prefix_match_ratios: list[torch.Tensor] = []
    for step in range(args.rollout_steps):
        encoded = torch.from_numpy(obs["planes"]).to(device=device)
        encoded = encoded.to(dtype=MODEL_DTYPE if device.type == "cuda" else torch.float32)
        normalizer = _observation_normalizer(model)
        if normalizer is not None:
            encoded = normalizer.update_and_normalize(encoded)
        mask = torch.from_numpy(obs["mask"]).to(device=device)
        with torch.inference_mode():
            logits, value = _model_forward(model, encoded)
            nonfinite_output = _first_nonfinite_model_output(logits, value)
            if nonfinite_output is not None:
                raise FloatingPointError(
                    f"non-finite {nonfinite_output} during rollout; "
                    "the model weights became non-finite"
                )
            dist = Categorical(logits=logits.float().masked_fill(~mask, -1e9))
            action = dist.sample()
            logprob = dist.log_prob(action)
        result = env.step(action.cpu().numpy())
        observations.append(encoded.cpu())
        actions.append(action.cpu())
        logprobs.append(logprob.cpu())
        rewards.append(torch.from_numpy(result.reward.copy()))
        dones.append(torch.from_numpy(result.done.astype(np.float32)))
        values.append(value.float().cpu())
        masks.append(mask.cpu())
        scores.append(torch.from_numpy(result.score.copy()))
        prefix_match_ratios.append(torch.from_numpy(result.prefix_match_ratio.copy()))
        obs = result.obs
        if result.done.any():
            obs = env.reset_done(
                result.done,
                args.seed_start + step + 1,
                args.seed_stride,
                args.fixed_n,
                args.fixed_m,
                args.fixed_c,
            )

    next_encoded = torch.from_numpy(obs["planes"]).to(device=device)
    next_encoded = next_encoded.to(dtype=MODEL_DTYPE if device.type == "cuda" else torch.float32)
    normalizer = _observation_normalizer(model)
    if normalizer is not None:
        next_encoded = normalizer.normalize(next_encoded)
    with torch.inference_mode():
        next_value = _model_forward(model, next_encoded)[1].float().cpu()
    raw_rewards = torch.stack(rewards)
    scaled_rewards = reward_scaler.scale(raw_rewards) if reward_scaler is not None else raw_rewards
    stacked_dones = torch.stack(dones)
    stacked_values = torch.stack(values)
    advantages = torch.zeros_like(scaled_rewards)
    last_gae = torch.zeros(args.num_envs)
    for step in reversed(range(args.rollout_steps)):
        next_value_step = next_value if step == args.rollout_steps - 1 else stacked_values[step + 1]
        nonterminal = 1.0 - stacked_dones[step]
        delta = (
            scaled_rewards[step] + args.gamma * next_value_step * nonterminal - stacked_values[step]
        )
        last_gae = delta + args.gamma * args.gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
    return {
        "obs": torch.stack(observations),
        "actions": torch.stack(actions),
        "logprobs": torch.stack(logprobs),
        "rewards": raw_rewards,
        "scaled_rewards": scaled_rewards,
        "dones": stacked_dones,
        "values": stacked_values,
        "advantages": advantages,
        "returns": advantages + stacked_values,
        "masks": torch.stack(masks),
        "scores": torch.stack(scores),
        "prefix_match_ratios": torch.stack(prefix_match_ratios),
    }, obs


def _last_terminal_values(
    values: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each environment's latest terminal value in a rollout."""
    if values.ndim != 2 or dones.shape != values.shape:
        raise ValueError("values and dones must both have shape (rollout_steps, num_envs)")
    terminal_values: list[torch.Tensor] = []
    for env_id in range(values.shape[1]):
        terminal_steps = torch.nonzero(dones[:, env_id].bool(), as_tuple=False).flatten()
        if terminal_steps.numel():
            terminal_values.append(values[terminal_steps[-1], env_id])
    if not terminal_values:
        return values.new_empty(0), values.new_empty(0, dtype=torch.bool)
    return torch.stack(terminal_values), torch.ones(len(terminal_values), dtype=torch.bool)


def build_rollout_metrics(rollout: dict[str, torch.Tensor]) -> dict[str, float | int]:
    """Build terminal-game metrics for W&B from one PPO rollout."""
    scores = rollout["scores"].float()
    dones = rollout["dones"].float()
    prefix_match_ratios = rollout["prefix_match_ratios"].float()
    terminal_scores, _ = _last_terminal_values(scores, dones)
    terminal_prefix_ratios, _ = _last_terminal_values(prefix_match_ratios, dones)

    def summary(values: torch.Tensor, reducer: str) -> float:
        if values.numel() == 0:
            return float("nan")
        if reducer == "mean":
            return float(values.mean().item())
        if reducer == "max":
            return float(values.max().item())
        return float(values.min().item())

    terminal_count = int(terminal_scores.numel())
    return {
        "rollout/final_prefix_match_ratio_mean": summary(terminal_prefix_ratios, "mean"),
        "rollout/final_prefix_match_ratio_max": summary(terminal_prefix_ratios, "max"),
        "rollout/final_prefix_match_ratio_min": summary(terminal_prefix_ratios, "min"),
        "rollout/final_score_mean": summary(terminal_scores, "mean"),
        "rollout/final_score_max": summary(terminal_scores, "max"),
        "rollout/final_score_min": summary(terminal_scores, "min"),
        "rollout/final_done_count": terminal_count,
        "rollout/final_done_fraction": terminal_count / max(scores.shape[1], 1),
        "rollout/endpoint_score_mean": float(scores[-1].mean().item()),
        "rollout/endpoint_score_max": float(scores[-1].max().item()),
        "rollout/endpoint_score_min": float(scores[-1].min().item()),
    }


def update_model(
    model: nn.Module,
    raw_model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    master_weights: FP32MasterWeights | None = None,
) -> dict[str, float]:
    observations = rollout["obs"].flatten(0, 1).to(device)
    actions = rollout["actions"].flatten().to(device)
    old_logprobs = rollout["logprobs"].flatten().to(device)
    advantages = rollout["advantages"].flatten().to(device)
    returns = rollout["returns"].flatten().to(device)
    masks = rollout["masks"].flatten(0, 1).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = observations.shape[0]
    minibatch_size = min(args.minibatch_size, batch_size)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    count = 0
    grad_norm = 0.0
    for _ in range(args.epochs):
        permutation = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            index = permutation[start : start + minibatch_size]
            logits, value = _model_forward(model, observations[index])
            dist = Categorical(logits=logits.float().masked_fill(~masks[index], -1e9))
            new_logprob = dist.log_prob(actions[index])
            ratio = (new_logprob - old_logprobs[index]).exp()
            surrogate = torch.min(
                ratio * advantages[index],
                ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * advantages[index],
            )
            policy_loss = -surrogate.mean()
            value_loss = 0.5 * (value.float() - returns[index]).square().mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "non-finite PPO loss before backward: "
                    f"policy_loss={policy_loss.item()} value_loss={value_loss.item()} "
                    f"entropy={entropy.item()}"
                )
            raw_model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if master_weights is None:
                try:
                    grad_norm = float(
                        nn.utils.clip_grad_norm_(
                            raw_model.parameters(),
                            args.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                    )
                except RuntimeError as error:
                    bad_gradient = _first_nonfinite_gradient(raw_model) or "unknown"
                    raise FloatingPointError(
                        f"non-finite model gradient before optimizer.step: {bad_gradient}"
                    ) from error
            else:
                master_weights.copy_gradients_from_model()
                try:
                    grad_norm = float(
                        nn.utils.clip_grad_norm_(
                            master_weights.parameters,
                            args.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                    )
                except RuntimeError as error:
                    bad_gradient = master_weights.first_nonfinite_gradient()
                    model_gradient = _first_nonfinite_gradient(raw_model)
                    raise FloatingPointError(
                        "non-finite gradient before optimizer.step: "
                        f"master_parameter_index={bad_gradient} model_parameter={model_gradient}"
                    ) from error
            optimizer.step()
            if master_weights is not None:
                master_weights.copy_master_to_model()
                master_weights.copy_model_to_master()
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy.item())
            totals["clip_frac"] += float((ratio.sub(1.0).abs() > args.clip).float().mean().item())
            count += 1
    return {key: value / max(count, 1) for key, value in totals.items()} | {"grad_norm": grad_norm}


def save_checkpoint(
    args: argparse.Namespace,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    scaler: RunningRewardScaler | None,
    master_weights: FP32MasterWeights,
    *,
    global_step: int,
    update: int,
) -> Path:
    path = args.run_dir / "checkpoints" / f"step_{global_step}.pt"
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "master_weights": master_weights.state_dict(),
        "config": config_for_save(args),
        "global_step": global_step,
        "update": update,
        "reward_scaler": None if scaler is None else scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    torch.save(checkpoint, path)
    torch.save(checkpoint, args.run_dir / "checkpoints" / CHECKPOINT_NAME)
    return path


def load_checkpoint(
    run_dir: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    scaler: RunningRewardScaler | None,
    master_weights: FP32MasterWeights,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint_path = run_dir / "checkpoints" / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if "master_weights" in state:
        master_weights.load_state_dict(state["master_weights"])
    else:
        master_weights.copy_model_to_master()
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("reward_scaler") is not None:
        scaler.load_state_dict(state["reward_scaler"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    return int(state["global_step"]), int(state["update"])


def main() -> None:
    args = parse_args()
    args.run_dir = prepare_run_dir(args)
    args.run_dir.joinpath("config.json").write_text(
        json.dumps(config_for_save(args), indent=2, sort_keys=True) + "\n"
    )
    torch.manual_seed(args.seed_start)
    np.random.seed(args.seed_start)
    device = torch.device(args.device)
    raw_model = create_model(args, device)
    print(f"model parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
    master_weights = FP32MasterWeights(raw_model)
    optimizer = torch.optim.AdamW(master_weights.parameters, lr=args.lr)
    scaler = RunningRewardScaler() if args.reward_scale else None
    global_step = update = 0
    if args.resume_dir is not None:
        global_step, update = load_checkpoint(
            args.resume_dir,
            raw_model,
            optimizer,
            scaler,
            master_weights,
            device,
        )
        master_weights.copy_master_to_model()
        if args.total_steps <= global_step:
            raise ValueError(
                f"total_steps ({args.total_steps}) must be greater than resumed "
                f"global_step ({global_step})"
            )
    env = OuroborosVecEnv(
        args.num_envs,
        args.seed_start,
        args.seed_stride,
        fixed_n=args.fixed_n,
        fixed_m=args.fixed_m,
        fixed_c=args.fixed_c,
        max_steps=args.max_episode_steps,
    )
    model: nn.Module = torch.compile(raw_model) if args.compile else raw_model
    obs = env.obs
    started = time.time()
    wandb_run = None
    if args.wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config_for_save(args),
        )
    try:
        while global_step < args.total_steps:
            rollout, obs = collect_rollout(model, env, obs, args, device, scaler)
            stats = update_model(model, raw_model, optimizer, rollout, args, device, master_weights)
            global_step += args.num_envs * args.rollout_steps
            update += 1
            checkpoint = None
            if update % args.checkpoint_interval_updates == 0 or global_step >= args.total_steps:
                checkpoint = save_checkpoint(
                    args,
                    raw_model,
                    optimizer,
                    scaler,
                    master_weights,
                    global_step=global_step,
                    update=update,
                )
            scores = rollout["scores"].float()
            rollout_metrics = build_rollout_metrics(rollout)
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"update={update} step={global_step} fps={global_step / elapsed:.1f} "
                f"mean_score={scores[-1].mean().item():.1f} "
                f"mean_reward={rollout['rewards'].mean().item():.5f} "
                f"policy_loss={stats['policy_loss']:.5f} value_loss={stats['value_loss']:.5f} "
                f"entropy={stats['entropy']:.5f} checkpoint={checkpoint}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/mean_score": float(scores[-1].mean().item()),
                        "train/mean_reward": float(rollout["rewards"].mean().item()),
                        **rollout_metrics,
                        **{f"train/{key}": value for key, value in stats.items()},
                    },
                    step=global_step,
                )
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    for key, default in DEFAULT_CONFIG.items():
        option = "--" + key.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, dest=key, action="store_true", default=argparse.SUPPRESS)
            parser.add_argument(
                "--no-" + key.replace("_", "-"),
                dest=key,
                action="store_false",
                default=argparse.SUPPRESS,
            )
        elif default is None:
            parser.add_argument(option, dest=key, default=argparse.SUPPRESS)
        elif isinstance(default, int):
            parser.add_argument(option, dest=key, type=int, default=argparse.SUPPRESS)
        elif isinstance(default, float):
            parser.add_argument(option, dest=key, type=float, default=argparse.SUPPRESS)
        elif isinstance(default, Path):
            parser.add_argument(option, dest=key, type=Path, default=argparse.SUPPRESS)
        else:
            parser.add_argument(option, dest=key, default=argparse.SUPPRESS)
    cli = vars(parser.parse_args(argv))
    config_path = cli.pop("config", None)
    resume_dir = cli.get("resume_dir")
    config = DEFAULT_CONFIG.copy()
    if resume_dir is not None:
        config.update(json.loads((Path(resume_dir) / "config.json").read_text()))
    if config_path is not None:
        with Path(config_path).open("rb") as file:
            raw = tomllib.load(file)
        raw = raw.get("train", raw)
        unknown = sorted(set(raw) - set(DEFAULT_CONFIG))
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        config.update(raw)
    config.update(cli)
    for key in ("artifact_dir", "resume_dir", "init_checkpoint"):
        if config.get(key) is not None:
            config[key] = Path(config[key])
    if config.get("resume_dir") is not None and config.get("init_checkpoint") is not None:
        raise ValueError("resume_dir and init_checkpoint are mutually exclusive")
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if config["model_channels"] % 4:
        raise ValueError("model_channels must be divisible by four")
    return argparse.Namespace(**config)


def config_for_save(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in RUNTIME_KEYS
    }


def prepare_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_dir is not None:
        return args.resume_dir
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    base = args.artifact_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix:02d}")
        suffix += 1
    (run_dir / "checkpoints").mkdir(parents=True)
    return run_dir


if __name__ == "__main__":
    main()
