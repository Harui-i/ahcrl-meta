"""PPO trainer for AHC063 using a SphericalAttentionSimba policy."""

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from ahcrl.envs import RustVecEnv, cargo_server_command
from ahcrl.training import (
    ExplainedVariancePolicyWarmup,
    FixedSeedEvaluation,
    FP32MasterWeights,
    HybridModularOptimizer,
    OptimizerLike,
    TrainingProgress,
    WandbConfig,
    append_evaluation_record,
    build_evaluation_metrics,
    build_optimizer,
    build_standard_ppo_metrics,
    config_for_save,
    evaluate_fixed_seeds,
    finish_wandb,
    get_wandb_run_id,
    init_wandb,
    load_initial_model,
    load_latest_training_checkpoint,
    optimizer_metrics,
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
    "env_workers": 0,
    "total_steps": 2_000_000,
    "rollout_steps": 128,
    "seed_start": 0,
    "seed_stride": 1,
    "fixed_n": None,
    "fixed_m": None,
    "fixed_c": None,
    "max_steps_per_cell": 4,
    "device": "auto",
    "compile": True,
    "lr": 3e-4,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "modula_momentum": 0.95,
    "modula_nesterov": True,
    "modula_diagnostics_interval": 100,
    "modula_initialize": True,
    "modula_project": True,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "epochs": 4,
    "minibatch_size": 1024,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "policy_unfreeze_explained_variance": 0.75,
    "policy_freeze_scope": "all_except_value",
    "policy_warmup_epochs_multiplier": 4,
    "max_grad_norm": 0.5,
    "proximal_ewma": False,
    "proximal_ewma_com": 1024.0,
    "reward_scale": True,
    "obs_norm": True,
    "obs_norm_epsilon": 1e-8,
    "artifact_dir": ROOT / "contests/ahc-063/artifacts/ppo",
    "checkpoint_interval_updates": 20,
    "eval_enabled": False,
    "eval_seed_start": 0,
    "eval_seed_num": 100,
    "eval_seed_stride": 1,
    "eval_temperature": 0.0,
    "eval_fixed_n": None,
    "eval_fixed_m": None,
    "eval_fixed_c": None,
    "eval_max_steps_per_cell": 4,
    "model_channels": 128,
    "model_blocks": 3,
    "wandb_enabled": False,
    "wandb_project": "ahcrl-meta",
    "wandb_entity": None,
    "wandb_name": None,
    "wandb_mode": "online",
    "wandb_tags": [],
}
RUNTIME_KEYS = {"run_dir", "resume_dir", "init_checkpoint"}
EVALUATION_CONFIG_KEYS = {
    "eval_enabled",
    "eval_seed_start",
    "eval_seed_num",
    "eval_seed_stride",
    "eval_temperature",
    "eval_fixed_n",
    "eval_fixed_m",
    "eval_fixed_c",
    "eval_max_steps_per_cell",
}
RESUME_ALLOWED_OVERRIDE_KEYS = {
    "total_steps",
    "epochs",
    "lr",
    "num_envs",
    "env_workers",
    "policy_unfreeze_explained_variance",
    "policy_warmup_epochs_multiplier",
    "wandb_name",
} | EVALUATION_CONFIG_KEYS
WANDB_CONFIG_KEYS = {
    "wandb_enabled",
    "wandb_project",
    "wandb_entity",
    "wandb_name",
    "wandb_mode",
    "wandb_tags",
}
OPTIONAL_INTEGER_CONFIG_KEYS = {
    "fixed_n",
    "fixed_m",
    "fixed_c",
    "eval_fixed_n",
    "eval_fixed_m",
    "eval_fixed_c",
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


class ProximalPolicyEWMA:
    """FP32で平均を保持し、推論用モデルへ同期するproximal policy。"""

    def __init__(
        self,
        model: ActorCritic,
        source_parameters: list[nn.Parameter],
        center_of_mass: float,
    ) -> None:
        if not math.isfinite(center_of_mass) or center_of_mass <= 0.0:
            raise ValueError("proximal_ewma_com must be finite and positive")
        self.center_of_mass = center_of_mass
        self.decay = center_of_mass / (center_of_mass + 1.0)
        self.model = copy.deepcopy(model)
        self.model.eval()
        source_parameter_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if len(source_parameter_names) != len(source_parameters):
            raise ValueError("proximal model parameter count mismatch")
        self.parameter_indices = [
            index
            for index, name in enumerate(source_parameter_names)
            if not name.startswith("value.")
        ]
        proximal_parameters = dict(self.model.named_parameters())
        self.model_parameters = [
            proximal_parameters[source_parameter_names[index]] for index in self.parameter_indices
        ]
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.master_parameters = [
            source_parameters[index].detach().float().clone() for index in self.parameter_indices
        ]
        self.total_weight = 1.0
        self.weighted_age = 0.0
        self._copy_master_to_model()

    @torch.no_grad()
    def _copy_master_to_model(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.master_parameters, strict=True
        ):
            model_parameter.copy_(master_parameter.to(dtype=model_parameter.dtype))

    @torch.no_grad()
    def update(self, source_parameters: list[nn.Parameter]) -> None:
        if len(source_parameters) <= max(self.parameter_indices, default=-1):
            raise ValueError("proximal source parameter count mismatch")
        previous_weight = self.total_weight
        new_weight = 1.0 + self.decay * previous_weight
        previous_coefficient = self.decay * previous_weight / new_weight
        current_coefficient = 1.0 / new_weight
        for proximal, index in zip(self.master_parameters, self.parameter_indices, strict=True):
            current = source_parameters[index]
            proximal.mul_(previous_coefficient).add_(
                current.detach().float(), alpha=current_coefficient
            )
        self.weighted_age = self.decay * (self.weighted_age + previous_weight)
        self.total_weight = new_weight
        self._copy_master_to_model()

    @property
    def effective_center_of_mass(self) -> float:
        return self.weighted_age / self.total_weight

    @torch.no_grad()
    def parameter_rms(self, source_parameters: list[nn.Parameter]) -> float:
        squared_sum = torch.zeros((), device=self.master_parameters[0].device)
        count = 0
        if len(source_parameters) <= max(self.parameter_indices, default=-1):
            raise ValueError("proximal source parameter count mismatch")
        for proximal, index in zip(self.master_parameters, self.parameter_indices, strict=True):
            current = source_parameters[index]
            difference = current.detach().float() - proximal
            squared_sum.add_(difference.square().sum())
            count += difference.numel()
        return math.sqrt(float(squared_sum.item()) / max(count, 1))

    def state_dict(self) -> dict[str, Any]:
        return {
            "center_of_mass": self.center_of_mass,
            "total_weight": self.total_weight,
            "weighted_age": self.weighted_age,
            "master_parameters": [
                parameter.detach().cpu().clone() for parameter in self.master_parameters
            ],
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_center_of_mass = float(state["center_of_mass"])
        if not math.isclose(saved_center_of_mass, self.center_of_mass):
            raise ValueError(
                "proximal EWMA center of mass mismatch: "
                f"checkpoint has {saved_center_of_mass}, configured {self.center_of_mass}"
            )
        saved_parameters = state["master_parameters"]
        if not isinstance(saved_parameters, list) or len(saved_parameters) != len(
            self.master_parameters
        ):
            raise ValueError("proximal EWMA parameter count mismatch")
        self.total_weight = float(state["total_weight"])
        self.weighted_age = float(state["weighted_age"])
        for parameter, saved in zip(self.master_parameters, saved_parameters, strict=True):
            if not isinstance(saved, torch.Tensor):
                raise ValueError("proximal EWMA parameters must be tensors")
            parameter.copy_(saved.to(device=parameter.device, dtype=torch.float32))
        self._copy_master_to_model()


def create_model(args: argparse.Namespace, device: torch.device) -> ActorCritic:
    model = ActorCritic(
        channels=args.model_channels,
        blocks=args.model_blocks,
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


def _to_model_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array).to(device=device)
    if device.type == "cpu":
        tensor = tensor.clone()
    return tensor.to(dtype=MODEL_DTYPE if device.type == "cuda" else torch.float32)


def _evaluation_generator_seed(seed: int) -> int:
    """環境 seed と独立した、再現可能な推論 RNG seed を返す。"""
    value = (seed + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 64) - 1)


def evaluate_policy(
    model: ActorCritic,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float | int], FixedSeedEvaluation]:
    """現在の方策を固定 seed 集合で評価し、W&B メトリクスも返す。"""
    generators: dict[int, torch.Generator] = {}

    def make_env(num_envs: int, seed_start: int, seed_stride: int) -> RustVecEnv:
        return RustVecEnv(
            cargo_server_command(RL_TOOLS_MANIFEST),
            num_envs,
            config={
                "fixed_n": args.eval_fixed_n,
                "fixed_m": args.eval_fixed_m,
                "fixed_c": args.eval_fixed_c,
                "max_steps_per_cell": args.eval_max_steps_per_cell,
            },
            workers=args.env_workers,
            seed_start=seed_start,
            seed_stride=seed_stride,
            cwd=ROOT,
        )

    def select_actions(
        obs: dict[str, np.ndarray], active: np.ndarray, seeds: np.ndarray
    ) -> np.ndarray:
        encoded = _to_model_tensor(obs["planes"], device)
        normalizer = _observation_normalizer(model)
        if normalizer is not None:
            encoded = normalizer.normalize(encoded)
        mask = torch.from_numpy(obs["mask"]).to(device=device)
        if device.type == "cpu":
            mask = mask.clone()
        with torch.inference_mode():
            logits, value = _model_forward(model, encoded)
            nonfinite_output = _first_nonfinite_model_output(logits, value)
            if nonfinite_output is not None:
                raise FloatingPointError(f"non-finite {nonfinite_output} during evaluation")
            masked_logits = logits.float().masked_fill(~mask, -1e9)
            if args.eval_temperature == 0.0:
                actions = masked_logits.argmax(dim=1)
            else:
                probabilities = torch.softmax(masked_logits / args.eval_temperature, dim=1)
                actions = torch.zeros(len(seeds), dtype=torch.long, device=device)
                for index in np.flatnonzero(active):
                    seed = int(seeds[index])
                    generator = generators.get(seed)
                    if generator is None:
                        generator = torch.Generator(device=device).manual_seed(
                            _evaluation_generator_seed(seed)
                        )
                        generators[seed] = generator
                    actions[index] = torch.multinomial(
                        probabilities[index], 1, generator=generator
                    ).squeeze(0)
        return actions.cpu().numpy()

    was_training = model.training
    model.eval()
    try:
        result = evaluate_fixed_seeds(
            make_env=make_env,
            action_selector=select_actions,
            seed_start=args.eval_seed_start,
            seed_num=args.eval_seed_num,
            seed_stride=args.eval_seed_stride,
            num_envs=args.num_envs,
            collect_visualizer_data=True,
        )
    finally:
        model.train(was_training)
    return build_evaluation_metrics(result, temperature=args.eval_temperature), result


def collect_rollout(
    model: nn.Module,
    env: RustVecEnv,
    obs: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    reward_scaler: RunningRewardScaler | None,
    *,
    update_observation_normalizer: bool = True,
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
            encoded = (
                normalizer.update_and_normalize(encoded)
                if update_observation_normalizer
                else normalizer.normalize(encoded)
            )
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


def _policy_surrogate(
    *,
    new_logprob: torch.Tensor,
    behavior_logprob: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
    proximal_logprob: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """通常PPOまたはbehavior/proximalを分離したPPO-EWMA objectiveを返す。"""
    current_behavior_log_ratio = new_logprob - behavior_logprob
    current_behavior_ratio = current_behavior_log_ratio.exp()
    if not bool(torch.isfinite(current_behavior_ratio).all().item()):
        raise FloatingPointError("non-finite current/behavior policy ratio")

    if proximal_logprob is None:
        clipping_ratio = current_behavior_ratio
        importance_weight = torch.ones_like(clipping_ratio)
    else:
        current_proximal_log_ratio = new_logprob - proximal_logprob
        clipping_ratio = current_proximal_log_ratio.exp()
        importance_weight = (proximal_logprob - behavior_logprob).exp()
        if not bool(torch.isfinite(clipping_ratio).all().item()):
            raise FloatingPointError("non-finite current/proximal policy ratio")
        if not bool(torch.isfinite(importance_weight).all().item()):
            raise FloatingPointError("non-finite proximal/behavior importance ratio")

    surrogate = importance_weight * torch.min(
        clipping_ratio * advantages,
        clipping_ratio.clamp(1.0 - clip, 1.0 + clip) * advantages,
    )
    return -surrogate.mean(), {
        "clipping_ratio": clipping_ratio.detach(),
        "current_behavior_log_ratio": current_behavior_log_ratio.detach(),
        "current_behavior_ratio": current_behavior_ratio.detach(),
        "proximal_behavior_ratio": importance_weight.detach(),
    }


def update_model(
    model: nn.Module,
    raw_model: ActorCritic,
    optimizer: OptimizerLike,
    rollout: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    master_weights: FP32MasterWeights | None = None,
    proximal_ewma: ProximalPolicyEWMA | None = None,
    proximal_model: nn.Module | None = None,
    *,
    policy_updates_enabled: bool = True,
    training_epochs: int | None = None,
) -> dict[str, float]:
    if (proximal_ewma is None) != (proximal_model is None):
        raise ValueError("proximal EWMA state and model must be provided together")
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
    proximal_forward_seconds = 0.0
    current_behavior_approx_kl_total = 0.0
    behavior_proximal_approx_kl_total = 0.0
    proximal_current_kl_total = 0.0
    proximal_behavior_ratio_sum = 0.0
    proximal_behavior_ratio_square_sum = 0.0
    proximal_behavior_ratio_min = float("inf")
    proximal_behavior_ratio_max = float("-inf")
    proximal_behavior_ratio_count = 0
    effective_training_epochs = args.epochs if training_epochs is None else training_epochs
    if effective_training_epochs <= 0:
        raise ValueError("training_epochs must be positive")
    for _ in range(effective_training_epochs):
        permutation = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            index = permutation[start : start + minibatch_size]
            _synchronize_device(device)
            forward_started = time.perf_counter()
            logits, value = _model_forward(model, observations[index])
            dist = Categorical(logits=logits.float().masked_fill(~masks[index], -1e9))
            new_logprob = dist.log_prob(actions[index])
            proximal_logprob = None
            proximal_dist = None
            if proximal_model is not None:
                _synchronize_device(device)
                proximal_forward_started = time.perf_counter()
                with torch.inference_mode():
                    proximal_logits, _ = _model_forward(proximal_model, observations[index])
                    proximal_dist = Categorical(
                        logits=proximal_logits.float().masked_fill(~masks[index], -1e9)
                    )
                    proximal_logprob = proximal_dist.log_prob(actions[index])
                _synchronize_device(device)
                proximal_forward_seconds += time.perf_counter() - proximal_forward_started
            policy_loss, policy_stats = _policy_surrogate(
                new_logprob=new_logprob,
                behavior_logprob=old_logprobs[index],
                advantages=advantages[index],
                clip=args.clip,
                proximal_logprob=proximal_logprob,
            )
            value_loss = 0.5 * (value.float() - returns[index]).square().mean()
            entropy = dist.entropy().mean()
            weighted_policy_loss = policy_loss if policy_updates_enabled else policy_loss * 0.0
            weighted_value_loss = args.value_coef * value_loss
            entropy_loss = -args.entropy_coef * entropy if policy_updates_enabled else entropy * 0.0
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
            if not policy_updates_enabled:
                for name, parameter in raw_model.named_parameters():
                    if args.policy_freeze_scope == "all_except_value":
                        keep_gradient = name.startswith("value.")
                    else:
                        keep_gradient = not name.startswith("policy.")
                    if not keep_gradient:
                        parameter.grad = None
            _synchronize_device(device)
            backward_seconds += time.perf_counter() - backward_started
            if isinstance(optimizer, HybridModularOptimizer):
                if master_weights is None:
                    raise ValueError("Modula optimizer requires fp32 master weights")
                master_weights.copy_gradients_from_model()
                grad_norm = optimizer.validate_gradients()
            elif master_weights is None:
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
                if proximal_ewma is not None:
                    proximal_ewma.update(master_weights.parameters)
                master_weights.copy_model_to_master()
            elif proximal_ewma is not None:
                proximal_ewma.update(
                    [parameter for parameter in raw_model.parameters() if parameter.requires_grad]
                )
            clipping_ratio = policy_stats["clipping_ratio"]
            current_behavior_log_ratio = policy_stats["current_behavior_log_ratio"]
            current_behavior_ratio = policy_stats["current_behavior_ratio"]
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy.item())
            totals["clip_frac"] += float(
                (clipping_ratio.sub(1.0).abs() > args.clip).float().mean().item()
            )
            totals["weighted_policy_loss"] += float(weighted_policy_loss.item())
            totals["weighted_value_loss"] += float(weighted_value_loss.item())
            totals["entropy_loss"] += float(entropy_loss.item())
            totals["total_loss"] += float(loss.item())
            current_behavior_approx_kl_total += float(
                (current_behavior_ratio - 1.0 - current_behavior_log_ratio).mean().item()
            )
            if proximal_logprob is not None and proximal_dist is not None:
                behavior_proximal_approx_kl_total += float(
                    (old_logprobs[index] - proximal_logprob).mean().item()
                )
                with torch.no_grad():
                    proximal_current_kl_total += float(
                        kl_divergence(proximal_dist, dist).mean().item()
                    )
                importance_weight = policy_stats["proximal_behavior_ratio"]
                proximal_behavior_ratio_sum += float(importance_weight.sum().item())
                proximal_behavior_ratio_square_sum += float(importance_weight.square().sum().item())
                proximal_behavior_ratio_min = min(
                    proximal_behavior_ratio_min, float(importance_weight.min().item())
                )
                proximal_behavior_ratio_max = max(
                    proximal_behavior_ratio_max, float(importance_weight.max().item())
                )
                proximal_behavior_ratio_count += importance_weight.numel()
            count += 1
    result = (
        {key: value / max(count, 1) for key, value in totals.items()}
        | {
            "grad_norm": grad_norm,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "proximal_forward_seconds": proximal_forward_seconds,
            "current_behavior_approx_kl": current_behavior_approx_kl_total / max(count, 1),
            "policy_updates_enabled": float(policy_updates_enabled),
            "training_epochs": float(effective_training_epochs),
        }
        | optimizer_metrics(optimizer)
    )
    if proximal_ewma is not None:
        ratio_count = max(proximal_behavior_ratio_count, 1)
        ratio_mean = proximal_behavior_ratio_sum / ratio_count
        ratio_variance = max(
            proximal_behavior_ratio_square_sum / ratio_count - ratio_mean * ratio_mean,
            0.0,
        )
        result |= {
            "behavior_proximal_approx_kl": behavior_proximal_approx_kl_total / max(count, 1),
            "proximal_current_kl": proximal_current_kl_total / max(count, 1),
            "proximal_behavior_ratio_mean": ratio_mean,
            "proximal_behavior_ratio_std": math.sqrt(ratio_variance),
            "proximal_behavior_ratio_min": proximal_behavior_ratio_min,
            "proximal_behavior_ratio_max": proximal_behavior_ratio_max,
            "proximal_behavior_ratio_ess_fraction": proximal_behavior_ratio_sum**2
            / max(
                ratio_count * proximal_behavior_ratio_square_sum,
                torch.finfo(torch.float64).tiny,
            ),
            "current_proximal_parameter_rms": proximal_ewma.parameter_rms(
                master_weights.parameters
                if master_weights is not None
                else [parameter for parameter in raw_model.parameters() if parameter.requires_grad]
            ),
            "proximal_ewma_effective_com": proximal_ewma.effective_center_of_mass,
        }
    return result


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
    optimizer = build_optimizer(model=raw_model, master_weights=master_weights, config=vars(args))
    if (
        isinstance(optimizer, HybridModularOptimizer)
        and args.modula_initialize
        and args.resume_dir is None
        and args.init_checkpoint is None
    ):
        optimizer.initialize_modular_parameters()
        master_weights.copy_master_to_model()
    proximal_ewma = (
        ProximalPolicyEWMA(raw_model, master_weights.parameters, args.proximal_ewma_com)
        if args.proximal_ewma
        else None
    )
    scaler = RunningRewardScaler() if args.reward_scale else None
    policy_warmup = ExplainedVariancePolicyWarmup(args.policy_unfreeze_explained_variance)
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
        if not isinstance(master_state, (dict, list)):
            raise ValueError("checkpoint extras missing master_weights")
        master_weights.load_state_dict(master_state)
        if proximal_ewma is not None:
            proximal_state = checkpoint.extras.get("proximal_policy_ewma")
            if not isinstance(proximal_state, dict):
                raise ValueError("checkpoint extras missing proximal_policy_ewma")
            proximal_ewma.load_state_dict(proximal_state)
        policy_warmup_state = checkpoint.extras.get("policy_warmup")
        if policy_warmup_state is not None:
            if not isinstance(policy_warmup_state, dict):
                raise ValueError("checkpoint policy_warmup state must be an object")
            policy_warmup.load_state_dict(policy_warmup_state)
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
            "max_steps_per_cell": args.max_steps_per_cell,
        },
        workers=args.env_workers,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        cwd=ROOT,
    )
    model: nn.Module = cast(nn.Module, torch.compile(raw_model) if args.compile else raw_model)
    proximal_model: nn.Module | None = None
    if proximal_ewma is not None:
        proximal_model = cast(
            nn.Module,
            torch.compile(proximal_ewma.model) if args.compile else proximal_ewma.model,
        )
    obs = env.obs
    started = time.time()
    timing_totals = {
        "forward_seconds": 0.0,
        "backward_seconds": 0.0,
        "proximal_forward_seconds": 0.0,
        "env_step_seconds": 0.0,
    }
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
            rollout, obs, rollout_timing = collect_rollout(
                model,
                env,
                obs,
                args,
                device,
                scaler,
                update_observation_normalizer=policy_warmup.policy_updates_enabled,
            )
            policy_warmup.observe(rollout["values"], rollout["returns"])
            training_epochs = policy_warmup.training_epochs(
                args.epochs, args.policy_warmup_epochs_multiplier
            )
            stats = update_model(
                model,
                raw_model,
                optimizer,
                rollout,
                args,
                device,
                master_weights,
                proximal_ewma,
                proximal_model,
                policy_updates_enabled=policy_warmup.policy_updates_enabled,
                training_epochs=training_epochs,
            )
            timing_totals["forward_seconds"] += (
                rollout_timing["forward_seconds"] + stats["forward_seconds"]
            )
            timing_totals["backward_seconds"] += stats["backward_seconds"]
            timing_totals["proximal_forward_seconds"] += stats["proximal_forward_seconds"]
            timing_totals["env_step_seconds"] += rollout_timing["env_step_seconds"]
            global_step += args.num_envs * args.rollout_steps
            update += 1
            checkpoint = None
            evaluation_metrics: dict[str, float | int] = {}
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
                        "proximal_policy_ewma": (
                            None if proximal_ewma is None else proximal_ewma.state_dict()
                        ),
                        "policy_warmup": policy_warmup.state_dict(),
                    },
                )
                if args.eval_enabled:
                    evaluation_metrics, evaluation = evaluate_policy(raw_model, args, device)
                    append_evaluation_record(
                        args.run_dir,
                        global_step=global_step,
                        update=update,
                        checkpoint_path=checkpoint,
                        evaluation_config={
                            key: getattr(args, key) for key in sorted(EVALUATION_CONFIG_KEYS)
                        },
                        result=evaluation,
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
                "timing/proximal_forward_seconds_total": timing_totals["proximal_forward_seconds"],
                "timing/env_step_seconds_total": timing_totals["env_step_seconds"],
                "train/current_behavior_approx_kl": stats["current_behavior_approx_kl"],
                "train/policy_updates_enabled": stats["policy_updates_enabled"],
                "train/ppo_epochs": stats["training_epochs"],
            }
            ewma_metric_names = {
                "behavior_proximal_approx_kl": "train/behavior_proximal_approx_kl",
                "proximal_current_kl": "train/proximal_current_kl",
                "proximal_behavior_ratio_mean": "train/proximal_behavior_ratio_mean",
                "proximal_behavior_ratio_std": "train/proximal_behavior_ratio_std",
                "proximal_behavior_ratio_min": "train/proximal_behavior_ratio_min",
                "proximal_behavior_ratio_max": "train/proximal_behavior_ratio_max",
                "proximal_behavior_ratio_ess_fraction": (
                    "train/proximal_behavior_ratio_ess_fraction"
                ),
                "current_proximal_parameter_rms": "model/current_proximal_parameter_rms",
                "proximal_ewma_effective_com": "model/proximal_ewma_effective_com",
            }
            metrics |= {
                metric_name: stats[stat_name]
                for stat_name, metric_name in ewma_metric_names.items()
                if stat_name in stats
            }
            metrics |= evaluation_metrics
            update_run_state(
                args.run_dir,
                global_step=global_step,
                update=update,
                wandb_run_id=None if wandb_run is None else wandb_run.id,
            )
            print(
                f"update={update} step={global_step} fps={metrics['summary/fps']:.1f} "
                f"mean_reward={metrics['train/mean_reward']:.5f} "
                f"explained_variance={metrics['train/explained_variance']:.5f} "
                f"policy_updates={bool(stats['policy_updates_enabled'])} "
                f"epochs={int(stats['training_epochs'])} "
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
            if key in OPTIONAL_INTEGER_CONFIG_KEYS:
                parser.add_argument(option, dest=key, type=int, default=argparse.SUPPRESS)
            else:
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
    if not isinstance(config["env_workers"], int) or config["env_workers"] < 0:
        raise ValueError("env_workers must be a non-negative integer")
    if config["model_channels"] % 4:
        raise ValueError("model_channels must be divisible by four")
    if config["checkpoint_interval_updates"] <= 0:
        raise ValueError("checkpoint_interval_updates must be positive")
    if not math.isfinite(config["proximal_ewma_com"]) or config["proximal_ewma_com"] <= 0.0:
        raise ValueError("proximal_ewma_com must be finite and positive")
    threshold = config["policy_unfreeze_explained_variance"]
    if not math.isfinite(threshold) or threshold >= 1.0:
        raise ValueError("policy_unfreeze_explained_variance must be finite and < 1")
    if config["policy_freeze_scope"] not in {"all_except_value", "policy_objective"}:
        raise ValueError("policy_freeze_scope must be 'all_except_value' or 'policy_objective'")
    if config["policy_warmup_epochs_multiplier"] <= 0:
        raise ValueError("policy_warmup_epochs_multiplier must be positive")
    if config["eval_temperature"] < 0.0:
        raise ValueError("eval_temperature must be non-negative")
    if config["eval_seed_num"] <= 0:
        raise ValueError("eval_seed_num must be positive")
    if config["eval_seed_stride"] <= 0:
        raise ValueError("eval_seed_stride must be positive")
    if config["max_steps_per_cell"] <= 0:
        raise ValueError("max_steps_per_cell must be positive")
    if config["eval_max_steps_per_cell"] <= 0:
        raise ValueError("eval_max_steps_per_cell must be positive")
    uint64_max = np.iinfo(np.uint64).max
    if not 0 <= config["eval_seed_start"] <= uint64_max:
        raise ValueError("eval_seed_start must fit in uint64")
    if (
        config["eval_seed_start"] + (config["eval_seed_num"] - 1) * config["eval_seed_stride"]
        > uint64_max
    ):
        raise ValueError("evaluation seed range must fit in uint64")
    return argparse.Namespace(**config)


if __name__ == "__main__":
    main()
