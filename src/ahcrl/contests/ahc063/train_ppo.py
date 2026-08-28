"""PPO trainer for AHC063 using a SphericalAttentionSimba policy."""

import argparse
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ahcrl.envs import RustVecEnv, cargo_server_command
from ahcrl.training import (
    TrainingProgress,
    WandbConfig,
    build_standard_ppo_metrics,
    config_for_save,
    finish_wandb,
    get_wandb_run_id,
    init_wandb,
    load_initial_model,
    load_latest_training_checkpoint,
    prepare_run_dir,
    resolve_config,
    save_training_checkpoint,
    update_run_state,
    write_config,
)

from .encoder import NUM_PLANES
from .model import ActorCritic, RunningObservationNormalizer

ROOT = Path(__file__).resolve().parents[4]
RL_TOOLS_MANIFEST = ROOT / "contests" / "ahc-063" / "rl-tools" / "Cargo.toml"
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
    "wandb_entity": None,
    "wandb_name": None,
    "wandb_mode": "online",
    "wandb_tags": [],
}
RUNTIME_KEYS = {"run_dir", "resume_dir", "init_checkpoint"}
RESUME_ALLOWED_OVERRIDE_KEYS = {"total_steps", "epochs", "lr", "num_envs", "wandb_name"}
WANDB_CONFIG_KEYS = {
    "wandb_enabled",
    "wandb_project",
    "wandb_entity",
    "wandb_name",
    "wandb_mode",
    "wandb_tags",
}


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


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_rollout(
    model: nn.Module,
    env: RustVecEnv,
    obs: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    reward_scaler: RunningRewardScaler | None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray], dict[str, float]]:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    forward_seconds = 0.0
    env_step_seconds = 0.0
    for step in range(args.rollout_steps):
        encoded = torch.from_numpy(obs["planes"]).to(device=device)
        if device.type == "cpu":
            encoded = encoded.clone()
        encoded = encoded.to(dtype=MODEL_DTYPE if device.type == "cuda" else torch.float32)
        normalizer = _observation_normalizer(model)
        if normalizer is not None:
            encoded = normalizer.update_and_normalize(encoded)
        mask = torch.from_numpy(obs["mask"]).to(device=device)
        if device.type == "cpu":
            mask = mask.clone()
        _synchronize_device(device)
        forward_started = time.perf_counter()
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
        _synchronize_device(device)
        forward_seconds += time.perf_counter() - forward_started
        env_step_started = time.perf_counter()
        result = env.step(action.cpu().numpy())
        env_step_seconds += time.perf_counter() - env_step_started
        observations.append(encoded.cpu())
        actions.append(action.cpu())
        logprobs.append(logprob.cpu())
        rewards.append(torch.from_numpy(result.reward.copy()))
        dones.append(torch.from_numpy(result.done.astype(np.float32)))
        scores.append(torch.from_numpy(result.score.copy()))
        values.append(value.float().cpu())
        masks.append(mask.cpu())
        obs = result.obs
        if result.done.any():
            obs = env.reset_done(
                result.done,
                args.seed_start + step + 1,
                args.seed_stride,
            )

    next_encoded = torch.from_numpy(obs["planes"]).to(device=device)
    next_encoded = next_encoded.to(dtype=MODEL_DTYPE if device.type == "cuda" else torch.float32)
    normalizer = _observation_normalizer(model)
    if normalizer is not None:
        next_encoded = normalizer.normalize(next_encoded)
    _synchronize_device(device)
    forward_started = time.perf_counter()
    with torch.inference_mode():
        next_value = _model_forward(model, next_encoded)[1].float().cpu()
    _synchronize_device(device)
    forward_seconds += time.perf_counter() - forward_started
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
    return (
        {
            "obs": torch.stack(observations),
            "actions": torch.stack(actions),
            "logprobs": torch.stack(logprobs),
            "rewards": raw_rewards,
            "scaled_rewards": scaled_rewards,
            "dones": stacked_dones,
            "scores": torch.stack(scores),
            "values": stacked_values,
            "advantages": advantages,
            "returns": advantages + stacked_values,
            "masks": torch.stack(masks),
        },
        obs,
        {"forward_seconds": forward_seconds, "env_step_seconds": env_step_seconds},
    )


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
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "clip_frac": 0.0,
        "weighted_policy_loss": 0.0,
        "weighted_value_loss": 0.0,
        "entropy_loss": 0.0,
        "total_loss": 0.0,
    }
    count = 0
    grad_norm = 0.0
    forward_seconds = 0.0
    backward_seconds = 0.0
    for _ in range(args.epochs):
        permutation = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            index = permutation[start : start + minibatch_size]
            _synchronize_device(device)
            forward_started = time.perf_counter()
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
            weighted_policy_loss = policy_loss
            weighted_value_loss = args.value_coef * value_loss
            entropy_loss = -args.entropy_coef * entropy
            loss = weighted_policy_loss + weighted_value_loss + entropy_loss
            _synchronize_device(device)
            forward_seconds += time.perf_counter() - forward_started
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "non-finite PPO loss before backward: "
                    f"policy_loss={policy_loss.item()} value_loss={value_loss.item()} "
                    f"entropy={entropy.item()}"
                )
            raw_model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            _synchronize_device(device)
            backward_started = time.perf_counter()
            loss.backward()
            _synchronize_device(device)
            backward_seconds += time.perf_counter() - backward_started
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
            totals["weighted_policy_loss"] += float(weighted_policy_loss.item())
            totals["weighted_value_loss"] += float(weighted_value_loss.item())
            totals["entropy_loss"] += float(entropy_loss.item())
            totals["total_loss"] += float(loss.item())
            count += 1
    return {key: value / max(count, 1) for key, value in totals.items()} | {
        "grad_norm": grad_norm,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
    }


def main() -> None:
    args = parse_args()
    args.run_dir = prepare_run_dir(
        artifact_dir=args.artifact_dir,
        resume_dir=args.resume_dir,
    )
    saved_config = config_for_save(vars(args), runtime_keys=RUNTIME_KEYS)
    torch.manual_seed(args.seed_start)
    np.random.seed(args.seed_start)
    device = torch.device(args.device)
    raw_model = create_model(args, device)
    print(f"model parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
    if args.init_checkpoint is not None:
        load_initial_model(args.init_checkpoint, model=raw_model, device=device)
    master_weights = FP32MasterWeights(raw_model)
    optimizer = torch.optim.AdamW(master_weights.parameters, lr=args.lr)
    scaler = RunningRewardScaler() if args.reward_scale else None
    global_step = update = 0
    if args.resume_dir is not None:
        checkpoint = load_latest_training_checkpoint(
            args.resume_dir,
            model=raw_model,
            optimizer=optimizer,
            device=device,
        )
        global_step = checkpoint.progress.global_step
        update = checkpoint.progress.update
        master_state = checkpoint.extras.get("master_weights")
        if not isinstance(master_state, list):
            raise ValueError("checkpoint extras missing master_weights")
        master_weights.load_state_dict(master_state)
        scaler_state = checkpoint.extras.get("reward_scaler")
        if scaler is not None and scaler_state is not None:
            if not isinstance(scaler_state, dict):
                raise ValueError("checkpoint reward_scaler state must be an object")
            scaler.load_state_dict(scaler_state)
        master_weights.copy_master_to_model()
        if args.total_steps <= global_step:
            raise ValueError(
                f"total_steps ({args.total_steps}) must be greater than resumed "
                f"global_step ({global_step})"
            )
    env = RustVecEnv(
        cargo_server_command(RL_TOOLS_MANIFEST),
        args.num_envs,
        config={
            "fixed_n": args.fixed_n,
            "fixed_m": args.fixed_m,
            "fixed_c": args.fixed_c,
            "max_steps": args.max_episode_steps,
        },
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        cwd=ROOT,
    )
    model: nn.Module = cast(nn.Module, torch.compile(raw_model) if args.compile else raw_model)
    obs = env.obs
    started = time.time()
    timing_totals = {"forward_seconds": 0.0, "backward_seconds": 0.0, "env_step_seconds": 0.0}
    wandb_run = None
    try:
        wandb_run = init_wandb(
            WandbConfig(
                enabled=args.wandb_enabled,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                mode=args.wandb_mode,
                tags=args.wandb_tags,
            ),
            resolved_config=saved_config,
            run_id=get_wandb_run_id(args.run_dir),
        )
        write_config(args.run_dir, saved_config)
        update_run_state(
            args.run_dir,
            global_step=global_step,
            update=update,
            wandb_run_id=None if wandb_run is None else wandb_run.id,
        )
        while global_step < args.total_steps:
            rollout, obs, rollout_timing = collect_rollout(model, env, obs, args, device, scaler)
            stats = update_model(model, raw_model, optimizer, rollout, args, device, master_weights)
            timing_totals["forward_seconds"] += (
                rollout_timing["forward_seconds"] + stats["forward_seconds"]
            )
            timing_totals["backward_seconds"] += stats["backward_seconds"]
            timing_totals["env_step_seconds"] += rollout_timing["env_step_seconds"]
            global_step += args.num_envs * args.rollout_steps
            update += 1
            checkpoint = None
            if update % args.checkpoint_interval_updates == 0 or global_step >= args.total_steps:
                checkpoint = save_training_checkpoint(
                    args.run_dir,
                    model=raw_model,
                    optimizer=optimizer,
                    config=saved_config,
                    progress=TrainingProgress(global_step=global_step, update=update),
                    extras={
                        "reward_scaler": None if scaler is None else scaler.state_dict(),
                        "master_weights": master_weights.state_dict(),
                    },
                )
            elapsed = max(time.time() - started, 1e-6)
            metrics = build_standard_ppo_metrics(
                update=update,
                global_step=global_step,
                elapsed=elapsed,
                rollout=rollout,
                update_stats=stats,
            )
            metrics |= {
                "timing/forward_seconds_total": timing_totals["forward_seconds"],
                "timing/backward_seconds_total": timing_totals["backward_seconds"],
                "timing/env_step_seconds_total": timing_totals["env_step_seconds"],
            }
            update_run_state(
                args.run_dir,
                global_step=global_step,
                update=update,
                wandb_run_id=None if wandb_run is None else wandb_run.id,
            )
            print(
                f"update={update} step={global_step} fps={metrics['summary/fps']:.1f} "
                f"mean_reward={metrics['train/mean_reward']:.5f} "
                f"policy_loss={stats['policy_loss']:.5f} value_loss={stats['value_loss']:.5f} "
                f"entropy={stats['entropy']:.5f} checkpoint={checkpoint}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(metrics, step=global_step)
    finally:
        env.close()
        finish_wandb(wandb_run)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resume-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--init-checkpoint", type=Path, default=argparse.SUPPRESS)
    for key, default in DEFAULT_CONFIG.items():
        if key in WANDB_CONFIG_KEYS:
            continue
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
    parser.add_argument(
        "--wandb",
        "--wandb-enabled",
        dest="wandb_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-wandb",
        "--no-wandb-enabled",
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

    cli = vars(parser.parse_args(argv))
    config_path = cli.pop("config", None)
    resume_dir = cli.pop("resume_dir", None)
    init_checkpoint = cli.pop("init_checkpoint", None)
    if resume_dir is not None and init_checkpoint is not None:
        raise ValueError("resume_dir and init_checkpoint are mutually exclusive")
    config = resolve_config(
        DEFAULT_CONFIG,
        cli_values=cli,
        config_path=config_path,
        resume_dir=resume_dir,
        allowed_resume_override_keys=RESUME_ALLOWED_OVERRIDE_KEYS,
        path_keys=("artifact_dir",),
    )
    config["resume_dir"] = resume_dir
    config["init_checkpoint"] = init_checkpoint
    if config["model_channels"] % 4:
        raise ValueError("model_channels must be divisible by four")
    if config["checkpoint_interval_updates"] <= 0:
        raise ValueError("checkpoint_interval_updates must be positive")
    return argparse.Namespace(**config)


if __name__ == "__main__":
    main()
