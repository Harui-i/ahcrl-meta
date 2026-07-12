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

from .encoder import BOARD_SIZE, MAX_LEVEL, MAX_PLAYERS, NUM_PLANES, PLANE_M, PLANE_U
from .model import ActorCritic, KeyedObservationNormalizer
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
    "reward_scale_mode": "simbav2",
    "reward_scale_epsilon": 1e-8,
    "reward_scale_g_max": 5.0,
    "obs_norm_mode": "none",
    "obs_norm_epsilon": 1e-8,
    "normalization_grouping": "none",
    "weight_projection": False,
    "symmetry_augmentation": "none",
    "critic_feature_mode": "oracle",
    "artifact_dir": ROOT / "contests/ahc-061/artifacts/ppo",
    "checkpoint_interval_updates": 1,
    "model_channels": 64,
    "model_blocks": 4,
    "model_block_type": "convnext",
    "distillation_teacher_checkpoint": None,
    "distillation_kl_coef": 0.0,
    "wandb_enabled": False,
    "wandb_project": "ahcrl-meta",
    "wandb_entity": None,
    "wandb_name": None,
    "wandb_mode": "online",
    "wandb_tags": [],
}
RESUME_ALLOWED_OVERRIDE_KEYS = {
    "total_steps",
    "epochs",
    "lr",
    "num_envs",
    "wandb_name",
}
RUNTIME_CONFIG_KEYS = {"config", "resume_dir", "run_dir", "init_checkpoint"}
LATEST_CHECKPOINT_NAME = "checkpoint_latest.pt"
STATE_FILE_NAME = "state.json"
CONFIG_FILE_NAME = "config.json"


class ImmediateRewardScaler:
    def __init__(self, *, epsilon: float) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update_and_scale(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor | None = None,
        *,
        m_values: torch.Tensor | None = None,
        u_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
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


class RunningRewardScaler:
    """SimbaV2-style reward scaler based on discounted-return running variance."""

    def __init__(self, *, gamma: float, g_max: float, epsilon: float) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if g_max <= 0.0:
            raise ValueError(f"g_max must be positive, got {g_max}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.g_return: torch.Tensor | None = None
        self.g_return_abs_max = 0.0

    def update_and_scale(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor | None = None,
        *,
        m_values: torch.Tensor | None = None,
        u_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if dones is None:
            raise ValueError("dones are required for SimbaV2 reward scaling")
        if rewards.shape != dones.shape:
            raise ValueError(
                f"rewards and dones must have the same shape: {rewards.shape} != {dones.shape}"
            )
        if rewards.ndim != 2:
            raise ValueError(
                f"expected rewards with shape (steps, envs), got {rewards.ndim} dimensions"
            )

        reward_values = rewards.detach().float()
        done_values = dones.detach().float()
        if reward_values.numel() == 0:
            return rewards, self.stats()

        returns = self._current_returns(
            num_envs=reward_values.shape[1],
            device=reward_values.device,
        )
        return_steps = []
        for reward_t, done_t in zip(reward_values, done_values, strict=True):
            returns = self.gamma * (1.0 - done_t) * returns + reward_t
            return_steps.append(returns.clone())

        discounted_returns = torch.stack(return_steps)
        values = discounted_returns.flatten()
        batch_count = int(values.numel())
        batch_mean = float(values.mean().item())
        batch_m2 = float(values.sub(batch_mean).pow(2).sum().item())
        self._merge(batch_count=batch_count, batch_mean=batch_mean, batch_m2=batch_m2)
        self.g_return = (returns * (1.0 - done_values[-1])).detach().cpu()
        self.g_return_abs_max = max(self.g_return_abs_max, float(values.abs().max().item()))

        scale = self.scale
        return rewards / scale, self.stats()

    @property
    def variance(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.m2 / self.count

    @property
    def min_required_denominator(self) -> float:
        return self.g_return_abs_max / self.g_max

    @property
    def scale(self) -> float:
        return max((self.variance + self.epsilon) ** 0.5, self.min_required_denominator)

    def stats(self) -> dict[str, float]:
        return {
            "reward_scale": self.scale,
            "reward_running_mean": self.mean,
            "reward_running_std": self.variance**0.5,
            "reward_discounted_return_abs_max": self.g_return_abs_max,
            "reward_scale_min_denominator": self.min_required_denominator,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "g_max": self.g_max,
            "g_return": None if self.g_return is None else self.g_return.tolist(),
            "g_return_abs_max": self.g_return_abs_max,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.count = int(state["count"])
        self.mean = float(state["mean"])
        self.m2 = float(state["m2"])
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.gamma = float(state.get("gamma", self.gamma))
        self.g_max = float(state.get("g_max", self.g_max))
        g_return = state.get("g_return")
        self.g_return = (
            None if g_return is None else torch.as_tensor(g_return, dtype=torch.float32).cpu()
        )
        self.g_return_abs_max = float(state.get("g_return_abs_max", 0.0))

    def _current_returns(self, *, num_envs: int, device: torch.device) -> torch.Tensor:
        if self.g_return is None:
            return torch.zeros(num_envs, dtype=torch.float32, device=device)
        if self.g_return.numel() != num_envs:
            if bool(torch.all(self.g_return == 0.0).item()):
                return torch.zeros(num_envs, dtype=torch.float32, device=device)
            raise ValueError(
                f"reward scaler state has {self.g_return.numel()} env returns, "
                f"but rollout has {num_envs} envs"
            )
        return self.g_return.to(device=device, dtype=torch.float32)

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


SingleRewardScaler = RunningRewardScaler | ImmediateRewardScaler


class GroupedRewardScaler:
    def __init__(self, factory: Any) -> None:
        self.global_scaler: SingleRewardScaler = factory()
        self.group_scalers: dict[tuple[int, int], SingleRewardScaler] = {}
        self._factory = factory

    def update_and_scale(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor | None = None,
        *,
        m_values: torch.Tensor | None = None,
        u_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if m_values is None or u_values is None:
            raise ValueError("m_values and u_values are required for grouped reward scaling")
        if m_values.shape != rewards.shape or u_values.shape != rewards.shape:
            raise ValueError(
                "m_values and u_values must have the same shape as rewards for grouped scaling"
            )
        scaled, _ = self.global_scaler.update_and_scale(rewards, dones)
        output = scaled.clone()
        first_m = m_values[0].long()
        first_u = u_values[0].long()
        for key_tensor in torch.unique(torch.stack([first_m, first_u], dim=1), dim=0):
            m_value = int(key_tensor[0].item())
            u_value = int(key_tensor[1].item())
            selector = (first_m == m_value) & (first_u == u_value)
            if not bool(selector.any().item()):
                continue
            scaler = self.group_scalers.setdefault((m_value, u_value), self._factory())
            group_rewards = rewards[:, selector]
            group_dones = None if dones is None else dones[:, selector]
            group_scaled, _ = scaler.update_and_scale(group_rewards, group_dones)
            output[:, selector] = group_scaled
        stats = self.stats()
        return output, stats

    def stats(self) -> dict[str, float]:
        stats = self.global_scaler.stats()
        for (m_value, u_value), scaler in sorted(self.group_scalers.items()):
            group_stats = scaler.stats()
            if "reward_scale" in group_stats:
                stats[f"reward_scale_by_m_u/m_{m_value}_u_{u_value}"] = group_stats["reward_scale"]
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "grouping": "m_u",
            "global": self.global_scaler.state_dict(),
            "groups": {
                f"{m_value}_{u_value}": scaler.state_dict()
                for (m_value, u_value), scaler in self.group_scalers.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("grouping") != "m_u":
            self.global_scaler.load_state_dict(state)
            return
        self.global_scaler.load_state_dict(state["global"])
        self.group_scalers = {}
        for key, scaler_state in state.get("groups", {}).items():
            m_raw, u_raw = key.split("_", maxsplit=1)
            scaler = self._factory()
            scaler.load_state_dict(scaler_state)
            self.group_scalers[(int(m_raw), int(u_raw))] = scaler


RewardScaler = SingleRewardScaler | GroupedRewardScaler


class RunningObservationNormalizer:
    """Channel-wise running standardization for NCHW observation planes."""

    def __init__(self, channels: int, *, epsilon: float) -> None:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.count = 0
        self.mean = torch.zeros((1, channels, 1, 1), dtype=torch.float32)
        self.m2 = torch.zeros((1, channels, 1, 1), dtype=torch.float32)

    def update_and_normalize(self, observations: torch.Tensor) -> torch.Tensor:
        self.update(observations)
        return self.normalize(observations)

    def update(self, observations: torch.Tensor) -> None:
        if observations.ndim != 4:
            raise ValueError(f"expected NCHW observations, got {observations.ndim} dimensions")
        if observations.shape[1] != self.mean.shape[1]:
            raise ValueError(f"expected {self.mean.shape[1]} channels, got {observations.shape[1]}")

        values = observations.detach().float()
        batch_count = int(values.shape[0] * values.shape[2] * values.shape[3])
        if batch_count == 0:
            return
        batch_mean = values.mean(dim=(0, 2, 3), keepdim=True).cpu()
        batch_m2 = (
            values.sub(batch_mean.to(device=values.device))
            .pow(2)
            .sum(
                dim=(0, 2, 3),
                keepdim=True,
            )
            .cpu()
        )
        self._merge(batch_count=batch_count, batch_mean=batch_mean, batch_m2=batch_m2)

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=observations.device)
        variance = self.variance.to(device=observations.device)
        normalized = (observations.float() - mean) / torch.sqrt(variance + self.epsilon)
        return normalized.to(dtype=observations.dtype)

    @property
    def variance(self) -> torch.Tensor:
        if self.count <= 0:
            return torch.ones_like(self.mean)
        return self.m2 / self.count

    def stats(self) -> dict[str, float]:
        variance = self.variance
        return {
            "obs_norm_count": float(self.count),
            "obs_norm_mean_abs_max": float(self.mean.abs().max().item()),
            "obs_norm_std_min": float(torch.sqrt(variance).min().item()),
            "obs_norm_std_max": float(torch.sqrt(variance).max().item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.count = int(state["count"])
        self.mean = state["mean"].detach().float().cpu().clone()
        self.m2 = state["m2"].detach().float().cpu().clone()
        self.epsilon = float(state.get("epsilon", self.epsilon))

    def _merge(
        self,
        *,
        batch_count: int,
        batch_mean: torch.Tensor,
        batch_m2: torch.Tensor,
    ) -> None:
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total_count
        self.m2 += batch_m2 + delta.pow(2) * self.count * batch_count / total_count
        self.count = total_count


class GroupedObservationNormalizer:
    def __init__(self, channels: int, *, epsilon: float) -> None:
        self.channels = channels
        self.epsilon = epsilon
        self.global_normalizer = RunningObservationNormalizer(channels, epsilon=epsilon)
        self.group_normalizers: dict[tuple[int, int], RunningObservationNormalizer] = {}

    def update_and_normalize(
        self,
        observations: torch.Tensor,
        *,
        m_values: torch.Tensor,
        u_values: torch.Tensor,
    ) -> torch.Tensor:
        self.update(observations, m_values=m_values, u_values=u_values)
        return self.normalize(observations, m_values=m_values, u_values=u_values)

    def update(
        self,
        observations: torch.Tensor,
        *,
        m_values: torch.Tensor,
        u_values: torch.Tensor,
    ) -> None:
        self._validate_group_values(observations, m_values, u_values)
        self.global_normalizer.update(observations)
        m_cpu = m_values.detach().long().cpu()
        u_cpu = u_values.detach().long().cpu()
        for key_tensor in torch.unique(torch.stack([m_cpu, u_cpu], dim=1), dim=0):
            m_value = int(key_tensor[0].item())
            u_value = int(key_tensor[1].item())
            selector = (m_cpu == m_value) & (u_cpu == u_value)
            if not bool(selector.any().item()):
                continue
            normalizer = self.group_normalizers.setdefault(
                (m_value, u_value),
                RunningObservationNormalizer(self.channels, epsilon=self.epsilon),
            )
            normalizer.update(observations[selector.to(device=observations.device)])

    def normalize(
        self,
        observations: torch.Tensor,
        *,
        m_values: torch.Tensor,
        u_values: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_group_values(observations, m_values, u_values)
        normalized = self.global_normalizer.normalize(observations)
        m_cpu = m_values.detach().long().cpu()
        u_cpu = u_values.detach().long().cpu()
        for key_tensor in torch.unique(torch.stack([m_cpu, u_cpu], dim=1), dim=0):
            m_value = int(key_tensor[0].item())
            u_value = int(key_tensor[1].item())
            normalizer = self.group_normalizers.get((m_value, u_value))
            if normalizer is None or normalizer.count <= 0:
                continue
            selector = (m_cpu == m_value) & (u_cpu == u_value)
            normalized[selector.to(device=observations.device)] = normalizer.normalize(
                observations[selector.to(device=observations.device)]
            )
        normalized[:, PLANE_M] = observations[:, PLANE_M]
        normalized[:, PLANE_U] = observations[:, PLANE_U]
        return normalized

    def stats(self) -> dict[str, float]:
        stats = self.global_normalizer.stats()
        for (m_value, u_value), normalizer in sorted(self.group_normalizers.items()):
            stats[f"obs_norm_count_by_m_u/m_{m_value}_u_{u_value}"] = float(normalizer.count)
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "grouping": "m_u",
            "global": self.global_normalizer.state_dict(),
            "groups": {
                f"{m_value}_{u_value}": normalizer.state_dict()
                for (m_value, u_value), normalizer in self.group_normalizers.items()
            },
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("grouping") != "m_u":
            self.global_normalizer.load_state_dict(state)
            self.group_normalizers = {}
            return
        self.global_normalizer.load_state_dict(state["global"])
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.group_normalizers = {}
        for key, normalizer_state in state.get("groups", {}).items():
            m_raw, u_raw = key.split("_", maxsplit=1)
            normalizer = RunningObservationNormalizer(self.channels, epsilon=self.epsilon)
            normalizer.load_state_dict(normalizer_state)
            self.group_normalizers[(int(m_raw), int(u_raw))] = normalizer

    def _validate_group_values(
        self,
        observations: torch.Tensor,
        m_values: torch.Tensor,
        u_values: torch.Tensor,
    ) -> None:
        if observations.ndim != 4:
            raise ValueError(f"expected NCHW observations, got {observations.ndim} dimensions")
        if m_values.shape != (observations.shape[0],) or u_values.shape != (observations.shape[0],):
            raise ValueError("m_values and u_values must have shape (batch,)")


ObservationNormalizer = RunningObservationNormalizer | GroupedObservationNormalizer


def create_reward_scaler(args: argparse.Namespace) -> RewardScaler | None:
    if args.reward_scale_mode == "none":
        return None

    def factory() -> SingleRewardScaler:
        if args.reward_scale_mode == "running_std":
            return ImmediateRewardScaler(epsilon=args.reward_scale_epsilon)
        if args.reward_scale_mode == "simbav2":
            return RunningRewardScaler(
                gamma=args.gamma,
                g_max=args.reward_scale_g_max,
                epsilon=args.reward_scale_epsilon,
            )
        raise ValueError(f"unsupported reward_scale_mode: {args.reward_scale_mode}")

    if args.normalization_grouping == "m_u":
        return GroupedRewardScaler(factory)
    if args.reward_scale_mode == "running_std":
        return ImmediateRewardScaler(epsilon=args.reward_scale_epsilon)
    if args.reward_scale_mode == "simbav2":
        return RunningRewardScaler(
            gamma=args.gamma,
            g_max=args.reward_scale_g_max,
            epsilon=args.reward_scale_epsilon,
        )
    raise ValueError(f"unsupported reward_scale_mode: {args.reward_scale_mode}")


def create_obs_normalizer(args: argparse.Namespace) -> ObservationNormalizer | None:
    if args.obs_norm_mode == "none":
        return None
    if args.obs_norm_mode != "running_channel":
        raise ValueError(f"unsupported obs_norm_mode: {args.obs_norm_mode}")
    if args.normalization_grouping == "m_u":
        return GroupedObservationNormalizer(NUM_PLANES, epsilon=args.obs_norm_epsilon)
    return RunningObservationNormalizer(NUM_PLANES, epsilon=args.obs_norm_epsilon)


def create_model(args: argparse.Namespace, device: torch.device) -> ActorCritic:
    model = ActorCritic(
        channels=args.model_channels,
        blocks=args.model_blocks,
        block_type=args.model_block_type,
        critic_feature_mode=args.critic_feature_mode,
    ).to(device=device, dtype=MODEL_DTYPE)
    if args.obs_norm_mode != "none":
        model.observation_normalizer = KeyedObservationNormalizer(
            NUM_PLANES,
            epsilon=args.obs_norm_epsilon,
            grouping=args.normalization_grouping,
        ).to(device=device)
    return model


def main() -> None:
    args = parse_args()
    args.run_dir = prepare_run_dir(args)
    print_resolved_config(args)

    torch.manual_seed(args.seed_start)
    np.random.seed(args.seed_start)
    device = torch.device(args.device)
    raw_model = create_model(args, device)
    total_parameters, trainable_parameters = _parameter_counts(raw_model)
    print(
        f"model_parameters total={total_parameters:,} trainable={trainable_parameters:,}",
        flush=True,
    )
    if args.init_checkpoint is not None:
        load_initial_model(args.init_checkpoint, raw_model, device)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)
    reward_scaler = create_reward_scaler(args)
    obs_normalizer = raw_model.observation_normalizer
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
    distillation_teacher_model: ActorCritic | None = None
    distillation_teacher_normalizer: KeyedObservationNormalizer | None = None
    if args.distillation_teacher_checkpoint is not None:
        distillation_teacher_model, distillation_teacher_normalizer = load_distillation_teacher(
            args.distillation_teacher_checkpoint,
            device,
        )
    if args.resume_dir is not None:
        resume_state = load_training_state(args.resume_dir, raw_model, optimizer, device)
        global_step = resume_state["global_step"]
        update = resume_state["update"]
        if reward_scaler is not None and resume_state["reward_scaler_state"] is not None:
            reward_scaler.load_state_dict(resume_state["reward_scaler_state"])
        if resume_state["obs_normalizer_state"] is not None:
            raise ValueError("legacy top-level obs_normalizer state is not supported")
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
                obs_normalizer=obs_normalizer,
                distillation_teacher_normalizer=distillation_teacher_normalizer,
                collect_teacher_observations=distillation_teacher_model is not None,
            )
            obs = rollout.pop("last_obs")
            next_seed_start = rollout.pop("next_seed_start")
            global_step += args.num_envs * args.rollout_steps
            update += 1
            stats = update_model(
                model,
                raw_model,
                optimizer,
                rollout,
                args,
                device,
                distillation_teacher_model=distillation_teacher_model,
                distillation_kl_coef=args.distillation_kl_coef,
            )

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
                    obs_normalizer=obs_normalizer,
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
                        f"weighted_policy_loss={stats['weighted_policy_loss']:.5f}",
                        f"weighted_value_loss={stats['weighted_value_loss']:.5f}",
                        f"entropy_loss={stats['entropy_loss']:.5f}",
                        f"total_loss={stats['total_loss']:.5f}",
                        f"normalized_entropy={stats['normalized_entropy']:.5f}",
                        f"approx_kl={stats['approx_kl']:.5f}",
                        f"distillation_forward_kl={stats['distillation_forward_kl']:.5f}",
                        f"distillation_kl_coef={stats['distillation_kl_coef']:.5f}",
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
    reward_scaler: RewardScaler | None = None,
    obs_normalizer: KeyedObservationNormalizer | ObservationNormalizer | None = None,
    distillation_teacher_normalizer: KeyedObservationNormalizer | None = None,
    collect_teacher_observations: bool = False,
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
    critic_feature_buf = []
    teacher_obs_buf = []

    for _ in range(args.rollout_steps):
        m_values, u_values = _extract_m_u_from_obs(obs)
        encoded = torch.from_numpy(obs["planes"]).to(device=device, dtype=MODEL_DTYPE)
        teacher_encoded = encoded
        if distillation_teacher_normalizer is not None:
            teacher_encoded = _normalize(
                distillation_teacher_normalizer,
                teacher_encoded,
                m_values=torch.from_numpy(m_values).to(device=device),
                u_values=torch.from_numpy(u_values).to(device=device),
            )
        if obs_normalizer is not None:
            encoded = _update_and_normalize(
                obs_normalizer,
                encoded,
                m_values=torch.from_numpy(m_values).to(device=device),
                u_values=torch.from_numpy(u_values).to(device=device),
            )
        mask = torch.from_numpy(obs["mask"]).to(device)
        critic_features = _critic_features_from_obs(obs, args.critic_feature_mode, device)
        with torch.inference_mode():
            if critic_features is None:
                logits, value = model(encoded, None, False)
            else:
                logits, value = model(encoded, critic_features, False)
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
        if collect_teacher_observations:
            teacher_obs_buf.append(teacher_encoded.cpu())
        if critic_features is not None:
            critic_feature_buf.append(critic_features.cpu())

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
        if obs_normalizer is not None:
            next_m_values, next_u_values = _extract_m_u_from_obs(obs)
            next_encoded = _normalize(
                obs_normalizer,
                next_encoded,
                m_values=torch.from_numpy(next_m_values).to(device=device),
                u_values=torch.from_numpy(next_u_values).to(device=device),
            )
        next_critic_features = _critic_features_from_obs(obs, args.critic_feature_mode, device)
        if next_critic_features is None:
            next_value = model(next_encoded, None, False)[1].float().cpu()
        else:
            next_value = model(next_encoded, next_critic_features, False)[1].float().cpu()

    rewards = torch.stack(reward_buf)
    dones = torch.stack(done_buf)
    if reward_scaler is None:
        scaled_rewards = rewards
        reward_scale_stats = {
            "reward_scale": 1.0,
            "reward_running_mean": float(rewards.float().mean().item()),
            "reward_running_std": float(rewards.float().std(unbiased=False).item()),
        }
    else:
        scaled_rewards, reward_scale_stats = reward_scaler.update_and_scale(
            rewards,
            dones,
            m_values=torch.stack(m_buf),
            u_values=torch.stack(u_buf),
        )
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
        **({} if not teacher_obs_buf else {"teacher_obs": torch.stack(teacher_obs_buf)}),
        **({} if not critic_feature_buf else {"critic_features": torch.stack(critic_feature_buf)}),
        "last_obs": obs,
        "next_seed_start": next_seed_start,
        **reward_scale_stats,
        **({} if obs_normalizer is None else obs_normalizer.stats()),
    }


def _update_and_normalize(
    normalizer: KeyedObservationNormalizer | ObservationNormalizer,
    observations: torch.Tensor,
    *,
    m_values: torch.Tensor,
    u_values: torch.Tensor,
) -> torch.Tensor:
    if isinstance(normalizer, KeyedObservationNormalizer):
        return normalizer.update_and_normalize(
            observations,
            m_values=m_values,
            u_values=u_values,
        )
    if isinstance(normalizer, GroupedObservationNormalizer):
        return normalizer.update_and_normalize(
            observations,
            m_values=m_values,
            u_values=u_values,
        )
    return normalizer.update_and_normalize(observations)


def _normalize(
    normalizer: KeyedObservationNormalizer | ObservationNormalizer,
    observations: torch.Tensor,
    *,
    m_values: torch.Tensor,
    u_values: torch.Tensor,
) -> torch.Tensor:
    if isinstance(normalizer, KeyedObservationNormalizer):
        return normalizer.normalize(observations, m_values=m_values, u_values=u_values)
    if isinstance(normalizer, GroupedObservationNormalizer):
        return normalizer.normalize(observations, m_values=m_values, u_values=u_values)
    return normalizer.normalize(observations)


def _forward_normalized(
    model: nn.Module,
    observations: torch.Tensor,
    critic_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, ActorCritic) or hasattr(model, "_orig_mod"):
        return model(observations, critic_features, False)  # type: ignore[call-arg]
    if critic_features is None:
        return model(observations)  # type: ignore[call-arg]
    return model(observations, critic_features)  # type: ignore[call-arg]


def _extract_m_u_from_obs(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    m_values = np.rint(obs["planes"][:, PLANE_M, 0, 0] * MAX_PLAYERS).astype(np.int64)
    u_values = np.rint(obs["planes"][:, PLANE_U, 0, 0] * MAX_LEVEL).astype(np.int64)
    return m_values, u_values


def _critic_features_from_obs(
    obs: dict[str, np.ndarray],
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode not in ("posterior", "oracle"):
        raise ValueError(f"unknown critic feature mode: {mode}")
    return torch.from_numpy(obs[f"critic_{mode}"]).to(device=device, dtype=MODEL_DTYPE)


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
    reward_scaler: RewardScaler | None = None,
    obs_normalizer: KeyedObservationNormalizer | None = None,
    checkpoint_name: str | None = None,
    update_latest: bool = True,
) -> Path:
    checkpoints_dir = args.run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / (
        f"step_{global_step}.pt" if checkpoint_name is None else checkpoint_name
    )
    if obs_normalizer is not model.observation_normalizer:
        raise AssertionError("observation normalizer must be owned by the model")
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
    if update_latest:
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
    if checkpoint.get("obs_normalizer") is not None:
        raise ValueError("legacy top-level obs_normalizer state is not supported")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return {
        "global_step": int(checkpoint["global_step"]),
        "update": int(checkpoint["update"]),
        "next_seed_start": int(checkpoint["next_seed_start"]),
        "torch_rng_state": checkpoint["torch_rng_state"],
        "numpy_rng_state": checkpoint["numpy_rng_state"],
        "reward_scaler_state": checkpoint.get("reward_scaler"),
        "obs_normalizer_state": None,
    }


def load_initial_model(
    checkpoint_path: Path,
    model: ActorCritic,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("checkpoint must contain a model state")
    if checkpoint.get("obs_normalizer") is not None:
        raise ValueError("legacy top-level obs_normalizer state is not supported")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)


def load_distillation_teacher(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ActorCritic, KeyedObservationNormalizer | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("distillation teacher checkpoint must contain model and config")
    teacher_args = argparse.Namespace(**checkpoint["config"])
    teacher_model = create_model(teacher_args, device)
    if checkpoint.get("obs_normalizer") is not None:
        raise ValueError("legacy top-level obs_normalizer state is not supported")
    teacher_model.load_state_dict(checkpoint["model"])
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    teacher_normalizer = teacher_model.observation_normalizer
    return teacher_model, teacher_normalizer


def update_model(
    model: nn.Module,
    grad_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    *,
    distillation_teacher_model: nn.Module | None = None,
    distillation_kl_coef: float = 0.0,
) -> dict[str, float]:
    obs = rollout["obs"].flatten(0, 1).to(device)  # type: ignore[union-attr]
    actions = rollout["actions"].flatten().to(device)  # type: ignore[union-attr]
    old_logprobs = rollout["logprobs"].flatten().to(device)  # type: ignore[union-attr]
    advantages = rollout["advantages"].flatten().to(device)  # type: ignore[union-attr]
    returns = rollout["returns"].flatten().to(device)  # type: ignore[union-attr]
    masks = rollout["masks"].flatten(0, 1).to(device)  # type: ignore[union-attr]
    critic_features = rollout.get("critic_features")
    if critic_features is not None:
        critic_features = critic_features.flatten(0, 1).to(device)
    teacher_obs = rollout.get("teacher_obs")
    if teacher_obs is not None:
        teacher_obs = teacher_obs.flatten(0, 1).to(device)
    if distillation_teacher_model is not None and teacher_obs is None:
        raise ValueError("distillation teacher observations are missing from rollout")
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    batch_size = obs.shape[0]
    indices = torch.arange(batch_size)
    stat_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "weighted_policy_loss": 0.0,
        "weighted_value_loss": 0.0,
        "entropy_loss": 0.0,
        "total_loss": 0.0,
        "normalized_entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "grad_norm": 0.0,
        "distillation_forward_kl": 0.0,
        "distillation_kl_loss": 0.0,
        "distillation_kl_coef": 0.0,
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
            mb_critic_features = None if critic_features is None else critic_features[mb]
            mb_teacher_obs = None if teacher_obs is None else teacher_obs[mb]
            transform_ids = range(8) if args.symmetry_augmentation == "full_d4" else range(1)
            for transform_id in transform_ids:
                aug_obs = _transform_board_d4(mb_obs, transform_id)
                aug_masks = _transform_flat_board_d4(mb_masks, transform_id)
                aug_actions = _transform_actions_d4(mb_actions, transform_id)
                logits, value = _forward_normalized(model, aug_obs, mb_critic_features)
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
                distillation_forward_kl = torch.zeros((), device=device)
                if distillation_teacher_model is not None and distillation_kl_coef != 0.0:
                    if mb_teacher_obs is None:
                        raise AssertionError("teacher observations must be available")
                    with torch.no_grad():
                        teacher_logits, _ = _forward_normalized(
                            distillation_teacher_model,
                            _transform_board_d4(mb_teacher_obs, transform_id),
                        )
                        teacher_logits = teacher_logits.float().masked_fill(~aug_masks, -1e9)
                        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
                        teacher_probs = teacher_log_probs.exp()
                    current_log_probs = F.log_softmax(logits, dim=-1)
                    distillation_forward_kl = (
                        (teacher_probs * (teacher_log_probs - current_log_probs)).sum(dim=-1).mean()
                    )
                distillation_kl_loss = distillation_kl_coef * distillation_forward_kl
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy
                    + distillation_kl_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    grad_model.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()
                if args.weight_projection or getattr(args, "model_block_type", "") in (
                    "simbav2_block",
                    "spherical_attention_simba",
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
                stat_sums["weighted_policy_loss"] += float(policy_loss.item()) * weight
                stat_sums["weighted_value_loss"] += (
                    float((args.value_coef * value_loss).item()) * weight
                )
                stat_sums["entropy_loss"] += float((-args.entropy_coef * entropy).item()) * weight
                stat_sums["total_loss"] += float(loss.item()) * weight
                stat_sums["normalized_entropy"] += float(normalized_entropy.item()) * weight
                stat_sums["approx_kl"] += float(approx_kl.item()) * weight
                stat_sums["clip_frac"] += float(clip_frac.item()) * weight
                stat_sums["grad_norm"] += float(grad_norm.item()) * weight
                stat_sums["distillation_forward_kl"] += (
                    float(distillation_forward_kl.item()) * weight
                )
                stat_sums["distillation_kl_loss"] += float(distillation_kl_loss.item()) * weight
                stat_sums["distillation_kl_coef"] += distillation_kl_coef * weight
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


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


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
        "train/reward_discounted_return_abs_max": float(
            rollout.get("reward_discounted_return_abs_max", 0.0)
        ),
        "train/reward_scale_min_denominator": float(
            rollout.get("reward_scale_min_denominator", 0.0)
        ),
        "train/obs_norm_count": float(rollout.get("obs_norm_count", 0.0)),
        "train/obs_norm_mean_abs_max": float(rollout.get("obs_norm_mean_abs_max", 0.0)),
        "train/obs_norm_std_min": float(rollout.get("obs_norm_std_min", 1.0)),
        "train/obs_norm_std_max": float(rollout.get("obs_norm_std_max", 1.0)),
        "train/done_count": int(dones.sum().item()),
        "train/mean_value": float(values.mean().item()),
        "train/mean_return": float(returns.mean().item()),
        "train/explained_variance": float(explained_variance.item()),
        "train/mean_advantage": float(advantages.mean().item()),
        "train/std_advantage": float(advantages.std(unbiased=False).item()),
        "train/valid_action_fraction": float(masks.mean().item()),
        "loss/policy": stats["policy_loss"],
        "loss/value": stats["value_loss"],
        "loss/weighted_policy": stats["weighted_policy_loss"],
        "loss/weighted_value": stats["weighted_value_loss"],
        "loss/entropy": stats["entropy_loss"],
        "loss/total": stats["total_loss"],
        "train/entropy": stats["entropy"],
        "train/normalized_entropy": stats["normalized_entropy"],
        "train/approx_kl": stats["approx_kl"],
        "train/clip_fraction": stats["clip_frac"],
        "train/distillation_forward_kl": stats.get("distillation_forward_kl", 0.0),
        "train/distillation_kl_coef": stats.get("distillation_kl_coef", 0.0),
        "loss/distillation_kl": stats.get("distillation_kl_loss", 0.0),
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
    for key, value in rollout.items():
        if key.startswith("reward_scale_by_m_u/"):
            metrics[f"train/{key}"] = float(value)
        if key.startswith("obs_norm_count_by_m_u/"):
            metrics[f"train/{key}"] = float(value)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Path to a TOML config file.")
    parser.add_argument("--resume-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--init-checkpoint", type=Path, default=argparse.SUPPRESS)
    parser.add_argument(
        "--distillation-teacher-checkpoint",
        type=Path,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--distillation-kl-coef", type=float, default=argparse.SUPPRESS)
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
        choices=("none", "running_std", "simbav2"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--reward-scale-epsilon", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--reward-scale-g-max", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--obs-norm-mode",
        choices=("none", "running_channel"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--obs-norm-epsilon", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--normalization-grouping",
        choices=("none", "m_u"),
        default=argparse.SUPPRESS,
    )
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
    parser.add_argument(
        "--critic-feature-mode",
        choices=("none", "posterior", "oracle"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--artifact-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--model-channels", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--model-blocks", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--model-block-type",
        choices=(
            "convnext",
            "per_cell_mlp",
            "residual",
            "simbav2_block",
            "spherical_attention_simba",
        ),
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
    config["distillation_teacher_checkpoint"] = (
        None
        if config.get("distillation_teacher_checkpoint") is None
        else Path(config["distillation_teacher_checkpoint"])
    )
    if config["checkpoint_interval_updates"] <= 0:
        raise ValueError("checkpoint_interval_updates must be positive")
    if config["distillation_kl_coef"] < 0.0:
        raise ValueError("distillation_kl_coef must be non-negative")
    if config["distillation_kl_coef"] > 0.0 and config["distillation_teacher_checkpoint"] is None:
        raise ValueError(
            "distillation_teacher_checkpoint is required when distillation_kl_coef is positive"
        )
    if config["pf_particles"] <= 0:
        raise ValueError("pf_particles must be positive")
    if config["reward_scale_mode"] not in ("none", "running_std", "simbav2"):
        raise ValueError("reward_scale_mode must be one of: none, running_std, simbav2")
    if config["reward_scale_epsilon"] <= 0.0:
        raise ValueError("reward_scale_epsilon must be positive")
    if config["reward_scale_g_max"] <= 0.0:
        raise ValueError("reward_scale_g_max must be positive")
    if config["obs_norm_mode"] not in ("none", "running_channel"):
        raise ValueError("obs_norm_mode must be one of: none, running_channel")
    if config["obs_norm_epsilon"] <= 0.0:
        raise ValueError("obs_norm_epsilon must be positive")
    if config["normalization_grouping"] not in ("none", "m_u"):
        raise ValueError("normalization_grouping must be one of: none, m_u")
    if config["symmetry_augmentation"] not in ("none", "full_d4"):
        raise ValueError("symmetry_augmentation must be one of: none, full_d4")
    if config["critic_feature_mode"] not in ("none", "posterior", "oracle"):
        raise ValueError("critic_feature_mode must be one of: none, posterior, oracle")
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return argparse.Namespace(**config)


def validate_resume_overrides(overrides: dict[str, Any]) -> None:
    disallowed = sorted(set(overrides) - RESUME_ALLOWED_OVERRIDE_KEYS)
    if disallowed:
        raise ValueError(
            "resume only allows approved training overrides; disallowed keys: "
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
    printable = {key: _jsonable(value) for key, value in sorted(vars(args).items())}
    print("config=" + json.dumps(printable, sort_keys=True), flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
