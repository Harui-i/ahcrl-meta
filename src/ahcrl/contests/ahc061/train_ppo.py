"""PPO trainer for AHC061 using the shared Rust vector-environment protocol."""

import argparse
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ahcrl.envs import RustVecEnv, cargo_server_command
from ahcrl.nn.modula import validate_modula_graph
from ahcrl.training import (
    FP32MasterWeights,
    HybridModularOptimizer,
    OptimizerLike,
    RolloutBuffer,
    RolloutFieldSpec,
    TrainingProgress,
    WandbConfig,
    build_optimizer,
    build_standard_ppo_metrics,
    config_for_save,
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
from ahcrl.training.ppo import policy_surrogate, tensor_range

from .encoder import NUM_PLANES
from .model import ActorCritic, RunningObservationNormalizer

ROOT = Path(__file__).resolve().parents[4]
RL_TOOLS_MANIFEST = ROOT / "contests" / "ahc-061" / "rl-tools" / "Cargo.toml"
MODEL_DTYPE = torch.bfloat16
DEFAULT_CONFIG: dict[str, Any] = {
    "num_envs": 64,
    "env_workers": 0,
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
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "modula_momentum": 0.95,
    "modula_nesterov": True,
    "modula_diagnostics_interval": 100,
    "modula_initialize": True,
    "modula_project": True,
    "modula_target_kl": 0.03,
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
    "artifact_dir": ROOT / "contests/ahc-061/artifacts/ppo",
    "checkpoint_interval_updates": 20,
    "model_channels": 64,
    "model_blocks": 4,
    "wandb_enabled": False,
    "wandb_project": "ahcrl-meta",
    "wandb_entity": None,
    "wandb_name": None,
    "wandb_mode": "online",
    "wandb_tags": [],
}
RUNTIME_KEYS = {"run_dir", "resume_dir", "init_checkpoint"}
RESUME_ALLOWED_OVERRIDE_KEYS = {
    "total_steps",
    "epochs",
    "lr",
    "num_envs",
    "env_workers",
    "wandb_name",
}
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
        return rewards / max((self.m2 / max(self.count, 1)) ** 0.5, self.epsilon)

    def state_dict(self) -> dict[str, float | int]:
        return {"epsilon": self.epsilon, "count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epsilon = float(state["epsilon"])
        self.count = int(state["count"])
        self.mean = float(state["mean"])
        self.m2 = float(state["m2"])


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
    model: nn.Module, observations: torch.Tensor, critic_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(observations, critic_features, False)  # type: ignore[call-arg]


def _observation_normalizer(model: nn.Module) -> RunningObservationNormalizer | None:
    original = getattr(model, "_orig_mod", model)
    normalizer = getattr(original, "observation_normalizer", None)
    return normalizer if isinstance(normalizer, RunningObservationNormalizer) else None


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_model_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array).to(device=device)
    if device.type == "cpu":
        tensor = tensor.clone()
    return tensor.to(dtype=MODEL_DTYPE if device.type == "cuda" else torch.float32)


def _accumulate_timing(
    target: dict[str, float], source: dict[str, float], prefix: str = ""
) -> None:
    """Temporary detailed profiler used only on the ahc061-measure branch."""
    for key, value in source.items():
        target[f"{prefix}{key}"] = target.get(f"{prefix}{key}", 0.0) + value


def _make_rollout_buffer(
    args: argparse.Namespace, device: torch.device, obs: dict[str, np.ndarray]
) -> RolloutBuffer:
    model_device = device if device.type == "cuda" else torch.device("cpu")
    model_dtype = MODEL_DTYPE if device.type == "cuda" else torch.float32
    return RolloutBuffer(
        args.rollout_steps,
        args.num_envs,
        {
            "obs": RolloutFieldSpec(tuple(obs["planes"].shape[1:]), model_dtype, model_device),
            "critic_features": RolloutFieldSpec(
                tuple(obs["critic_oracle"].shape[1:]), model_dtype, model_device
            ),
            "actions": RolloutFieldSpec((), torch.int64, torch.device("cpu")),
            "logprobs": RolloutFieldSpec((), torch.float32, torch.device("cpu")),
            "rewards": RolloutFieldSpec((), torch.float32, torch.device("cpu")),
            "dones": RolloutFieldSpec((), torch.float32, torch.device("cpu")),
            "scores": RolloutFieldSpec((), torch.int64, torch.device("cpu")),
            "values": RolloutFieldSpec((), torch.float32, torch.device("cpu")),
            "masks": RolloutFieldSpec(
                tuple(obs["mask"].shape[1:]), torch.bool, torch.device("cpu")
            ),
        },
    )


def collect_rollout(
    model: nn.Module,
    env: RustVecEnv,
    obs: dict[str, np.ndarray],
    next_seed_start: int,
    args: argparse.Namespace,
    device: torch.device,
    reward_scaler: RunningRewardScaler | None,
    rollout_buffer: RolloutBuffer | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray], int, dict[str, float]]:
    collect_started = time.perf_counter()
    if rollout_buffer is None:
        rollout_buffer = _make_rollout_buffer(args, device, obs)
    rollout_buffer.reset()
    forward_seconds = 0.0
    env_step_seconds = 0.0
    detailed: dict[str, float] = {}
    for step in range(args.rollout_steps):
        started = time.perf_counter()
        encoded = _to_model_tensor(obs["planes"], device)
        normalizer = _observation_normalizer(model)
        if normalizer is not None:
            encoded = normalizer.update_and_normalize(encoded)
        oracle = _to_model_tensor(obs["critic_oracle"], device)
        mask = torch.from_numpy(obs["mask"]).to(device=device)
        if device.type == "cpu":
            mask = mask.clone()
        _synchronize_device(device)
        detailed["rollout_input_h2d_normalize_seconds"] = detailed.get(
            "rollout_input_h2d_normalize_seconds", 0.0
        ) + (time.perf_counter() - started)
        started = time.perf_counter()
        with torch.inference_mode():
            logits, value = _model_forward(model, encoded, oracle)
            if not bool(torch.isfinite(logits).all().item() and torch.isfinite(value).all().item()):
                raise FloatingPointError("non-finite model output during rollout")
            dist = Categorical(logits=logits.float().masked_fill(~mask, -1e9))
            action = dist.sample()
            logprob = dist.log_prob(action)
        _synchronize_device(device)
        forward_seconds += time.perf_counter() - started
        started = time.perf_counter()
        action_numpy = action.cpu().numpy()
        detailed["rollout_action_d2h_seconds"] = detailed.get("rollout_action_d2h_seconds", 0.0) + (
            time.perf_counter() - started
        )
        started = time.perf_counter()
        result = env.step(action_numpy)
        env_step_seconds += time.perf_counter() - started
        _accumulate_timing(detailed, result.timings, "env_")
        rust_until_header = sum(
            result.timings.get(key, 0.0)
            for key in (
                "rust_input_seconds",
                "rust_validate_seconds",
                "rust_step_seconds",
                "rust_prepare_seconds",
                "rust_encode_seconds",
            )
        )
        detailed["env_header_residual_seconds"] = detailed.get(
            "env_header_residual_seconds", 0.0
        ) + max(result.timings.get("client_header_wait_seconds", 0.0) - rust_until_header, 0.0)
        started = time.perf_counter()
        rollout_buffer.store(
            step,
            obs=encoded,
            critic_features=oracle,
            actions=action.cpu(),
            logprobs=logprob.float().cpu(),
            rewards=torch.from_numpy(result.reward.copy()),
            dones=torch.from_numpy(result.done.astype(np.float32)),
            scores=torch.from_numpy(result.score.copy()),
            values=value.float().cpu(),
            masks=mask.cpu(),
        )
        _synchronize_device(device)
        detailed["rollout_store_d2h_seconds"] = detailed.get("rollout_store_d2h_seconds", 0.0) + (
            time.perf_counter() - started
        )
        obs = result.obs
        if result.done.any():
            started = time.perf_counter()
            obs = env.reset_done(result.done, next_seed_start, args.seed_stride)
            detailed["rollout_reset_seconds"] = detailed.get("rollout_reset_seconds", 0.0) + (
                time.perf_counter() - started
            )
            _accumulate_timing(detailed, env.last_timings, "reset_")
            next_seed_start += args.num_envs * args.seed_stride

    bootstrap_started = time.perf_counter()
    next_encoded = _to_model_tensor(obs["planes"], device)
    normalizer = _observation_normalizer(model)
    if normalizer is not None:
        next_encoded = normalizer.normalize(next_encoded)
    next_oracle = _to_model_tensor(obs["critic_oracle"], device)
    _synchronize_device(device)
    detailed["rollout_bootstrap_input_seconds"] = time.perf_counter() - bootstrap_started
    bootstrap_forward_started = time.perf_counter()
    with torch.inference_mode():
        next_value = _model_forward(model, next_encoded, next_oracle)[1].float().cpu()
    _synchronize_device(device)
    detailed["rollout_bootstrap_forward_seconds"] = time.perf_counter() - bootstrap_forward_started
    finalize_started = time.perf_counter()
    stored = rollout_buffer.as_dict()
    raw_rewards = stored["rewards"]
    scaled_rewards = reward_scaler.scale(raw_rewards) if reward_scaler is not None else raw_rewards
    stacked_dones = stored["dones"]
    stacked_values = stored["values"]
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
    detailed["rollout_stack_gae_seconds"] = time.perf_counter() - finalize_started
    detailed["rollout_forward_seconds"] = forward_seconds
    detailed["rollout_total_seconds"] = time.perf_counter() - collect_started
    directly_accounted = (
        detailed.get("rollout_input_h2d_normalize_seconds", 0.0)
        + forward_seconds
        + detailed.get("rollout_action_d2h_seconds", 0.0)
        + env_step_seconds
        + detailed.get("rollout_store_d2h_seconds", 0.0)
        + detailed.get("rollout_reset_seconds", 0.0)
        + detailed.get("rollout_bootstrap_input_seconds", 0.0)
        + detailed.get("rollout_bootstrap_forward_seconds", 0.0)
        + detailed.get("rollout_stack_gae_seconds", 0.0)
    )
    detailed["rollout_unclassified_seconds"] = max(
        detailed["rollout_total_seconds"] - directly_accounted, 0.0
    )
    return (
        {
            **stored,
            "rewards": raw_rewards,
            "scaled_rewards": scaled_rewards,
            "dones": stacked_dones,
            "values": stacked_values,
            "advantages": advantages,
            "returns": advantages + stacked_values,
        },
        obs,
        next_seed_start,
        {
            "forward_seconds": forward_seconds,
            "env_step_seconds": env_step_seconds,
            **detailed,
        },
    )


def update_model(
    model: nn.Module,
    raw_model: ActorCritic,
    optimizer: OptimizerLike,
    rollout: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    master_weights: FP32MasterWeights,
) -> dict[str, float]:
    update_started = time.perf_counter()
    prepare_started = time.perf_counter()
    observations = rollout["obs"].flatten(0, 1).to(device)
    critic_features = rollout["critic_features"].flatten(0, 1).to(device)
    actions = rollout["actions"].flatten().to(device)
    old_logprobs = rollout["logprobs"].flatten().to(device)
    advantages = rollout["advantages"].flatten().to(device)
    returns = rollout["returns"].flatten().to(device)
    masks = rollout["masks"].flatten(0, 1).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    _synchronize_device(device)
    update_prepare_seconds = time.perf_counter() - prepare_started
    batch_size = observations.shape[0]
    minibatch_size = min(args.minibatch_size, batch_size)
    totals = {
        key: 0.0
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "clip_frac",
            "weighted_policy_loss",
            "weighted_value_loss",
            "entropy_loss",
            "total_loss",
        )
    }
    count = 0
    grad_norm = 0.0
    forward_seconds = 0.0
    backward_seconds = 0.0
    permutation_seconds = 0.0
    zero_grad_seconds = 0.0
    gradient_processing_seconds = 0.0
    optimizer_step_seconds = 0.0
    master_to_model_seconds = 0.0
    stats_seconds = 0.0
    current_behavior_approx_kl_total = 0.0
    current_behavior_approx_kl_count = 0
    max_abs_log_ratio = 0.0
    logits_max_abs = 0.0
    value_max_abs = 0.0
    trust_region_stop_count = 0
    trust_region_stopped = False
    modula_trust_region = isinstance(optimizer, HybridModularOptimizer)
    for _ in range(args.epochs):
        _synchronize_device(device)
        started = time.perf_counter()
        permutation = torch.randperm(batch_size, device=device)
        _synchronize_device(device)
        permutation_seconds += time.perf_counter() - started
        for start in range(0, batch_size, minibatch_size):
            index = permutation[start : start + minibatch_size]
            _synchronize_device(device)
            started = time.perf_counter()
            logits, value = _model_forward(model, observations[index], critic_features[index])
            logits_max_abs = max(logits_max_abs, float(logits.float().abs().max().item()))
            value_max_abs = max(value_max_abs, float(value.float().abs().max().item()))
            if not bool(torch.isfinite(logits).all().item()) or not bool(
                torch.isfinite(value).all().item()
            ):
                raise FloatingPointError(
                    "non-finite PPO model output before distribution: "
                    f"logits=({tensor_range(logits)}) value=({tensor_range(value)})"
                )
            dist = Categorical(logits=logits.float().masked_fill(~masks[index], -1e9))
            new_logprob = dist.log_prob(actions[index])
            policy = policy_surrogate(
                new_logprob=new_logprob,
                behavior_logprob=old_logprobs[index],
                advantages=advantages[index],
                clip=args.clip,
                max_abs_log_ratio=20.0 if modula_trust_region else None,
                target_kl=args.modula_target_kl if modula_trust_region else None,
            )
            max_abs_log_ratio = max(max_abs_log_ratio, float(policy.max_abs_log_ratio.item()))
            if policy.current_behavior_approx_kl is not None:
                current_behavior_approx_kl_total += float(policy.current_behavior_approx_kl.item())
                current_behavior_approx_kl_count += 1
            if policy.stop_reason is not None:
                trust_region_stop_count += 1
                trust_region_stopped = True
                break
            assert policy.loss is not None
            assert policy.clipping_ratio is not None
            policy_loss = policy.loss
            ratio = policy.clipping_ratio
            value_loss = 0.5 * (value.float() - returns[index]).square().mean()
            entropy = dist.entropy().mean()
            weighted_policy_loss = policy_loss
            weighted_value_loss = args.value_coef * value_loss
            entropy_loss = -args.entropy_coef * entropy
            loss = weighted_policy_loss + weighted_value_loss + entropy_loss
            _synchronize_device(device)
            forward_seconds += time.perf_counter() - started
            if not bool(torch.isfinite(loss).item()):
                log_ratio = policy.current_behavior_log_ratio
                raise FloatingPointError(
                    "non-finite PPO loss before backward: "
                    f"policy_loss={policy_loss.item()} value_loss={value_loss.item()} "
                    f"entropy={entropy.item()} log_ratio_min={log_ratio.min().item()} "
                    f"log_ratio_max={log_ratio.max().item()} "
                    f"logits=({tensor_range(logits)}) value=({tensor_range(value)})"
                )
            _synchronize_device(device)
            started = time.perf_counter()
            raw_model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            _synchronize_device(device)
            zero_grad_seconds += time.perf_counter() - started
            started = time.perf_counter()
            loss.backward()
            _synchronize_device(device)
            backward_seconds += time.perf_counter() - started
            started = time.perf_counter()
            master_weights.copy_gradients_from_model()
            if isinstance(optimizer, HybridModularOptimizer):
                grad_norm = optimizer.validate_gradients()
            else:
                try:
                    grad_norm = float(
                        nn.utils.clip_grad_norm_(
                            master_weights.parameters, args.max_grad_norm, error_if_nonfinite=True
                        )
                    )
                except RuntimeError as error:
                    raise FloatingPointError(
                        "non-finite gradient at master parameter "
                        f"{master_weights.first_nonfinite_gradient()}"
                    ) from error
            _synchronize_device(device)
            gradient_processing_seconds += time.perf_counter() - started
            started = time.perf_counter()
            optimizer.step()
            _synchronize_device(device)
            optimizer_step_seconds += time.perf_counter() - started
            started = time.perf_counter()
            master_weights.copy_master_to_model()
            _synchronize_device(device)
            master_to_model_seconds += time.perf_counter() - started
            started = time.perf_counter()
            for key, value_ in {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "clip_frac": (ratio.sub(1.0).abs() > args.clip).float().mean(),
                "weighted_policy_loss": weighted_policy_loss,
                "weighted_value_loss": weighted_value_loss,
                "entropy_loss": entropy_loss,
                "total_loss": loss,
            }.items():
                totals[key] += float(value_.item())
            count += 1
            stats_seconds += time.perf_counter() - started
        if trust_region_stopped:
            break
    update_total_seconds = time.perf_counter() - update_started
    update_accounted_seconds = (
        update_prepare_seconds
        + permutation_seconds
        + forward_seconds
        + zero_grad_seconds
        + backward_seconds
        + gradient_processing_seconds
        + optimizer_step_seconds
        + master_to_model_seconds
        + stats_seconds
    )
    return (
        {key: value / max(count, 1) for key, value in totals.items()}
        | {
            "grad_norm": grad_norm,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "update_forward_seconds": forward_seconds,
            "update_backward_seconds": backward_seconds,
            "update_prepare_h2d_seconds": update_prepare_seconds,
            "update_permutation_seconds": permutation_seconds,
            "update_zero_grad_seconds": zero_grad_seconds,
            "update_gradient_processing_seconds": gradient_processing_seconds,
            "update_optimizer_step_seconds": optimizer_step_seconds,
            "update_master_to_model_seconds": master_to_model_seconds,
            "update_stats_seconds": stats_seconds,
            "update_total_seconds": update_total_seconds,
            "update_unclassified_seconds": max(
                update_total_seconds - update_accounted_seconds, 0.0
            ),
            "current_behavior_approx_kl": current_behavior_approx_kl_total
            / max(current_behavior_approx_kl_count, 1),
            "max_abs_log_ratio": max_abs_log_ratio,
            "trust_region_stop_count": float(trust_region_stop_count),
            "logits_max_abs": logits_max_abs,
            "value_max_abs": value_max_abs,
        }
        | optimizer_metrics(optimizer)
    )


def main() -> None:
    args = parse_args()
    args.run_dir = prepare_run_dir(artifact_dir=args.artifact_dir, resume_dir=args.resume_dir)
    saved_config = config_for_save(vars(args), runtime_keys=RUNTIME_KEYS)
    torch.manual_seed(args.seed_start)
    np.random.seed(args.seed_start)
    device = torch.device(args.device)
    raw_model = create_model(args, device)
    print(f"model parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
    if args.optimizer == "modula":
        print(f"modula input sensitivity: {validate_modula_graph(raw_model.modula_graph()):g}")
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
    scaler = RunningRewardScaler() if args.reward_scale else None
    global_step = update = 0
    next_seed_start = args.seed_start + args.num_envs * args.seed_stride
    if args.resume_dir is not None:
        checkpoint = load_latest_training_checkpoint(
            args.resume_dir, model=raw_model, optimizer=optimizer, device=device
        )
        global_step, update = checkpoint.progress.global_step, checkpoint.progress.update
        master_state = checkpoint.extras.get("master_weights")
        if not isinstance(master_state, (dict, list)):
            raise ValueError("checkpoint extras missing master_weights")
        master_weights.load_state_dict(master_state)
        master_weights.copy_master_to_model()
        scaler_state = checkpoint.extras.get("reward_scaler")
        if scaler is not None and scaler_state is not None:
            if not isinstance(scaler_state, dict):
                raise ValueError("checkpoint reward_scaler state must be an object")
            scaler.load_state_dict(scaler_state)
        next_seed_start = args.seed_start + (update + 1) * args.num_envs * args.seed_stride
        if args.total_steps <= global_step:
            raise ValueError("total_steps must exceed resumed global_step")
    env = RustVecEnv(
        cargo_server_command(RL_TOOLS_MANIFEST),
        args.num_envs,
        config={
            "fixed_m": args.fixed_m,
            "fixed_u": args.fixed_u,
            "pf_particles": args.pf_particles,
        },
        workers=args.env_workers,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        cwd=ROOT,
    )
    model: nn.Module = cast(nn.Module, torch.compile(raw_model) if args.compile else raw_model)
    obs = env.obs
    rollout_buffer = _make_rollout_buffer(args, device, obs)
    started = time.time()
    timing_totals: dict[str, float] = {"checkpoint_seconds": 0.0}
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
            rollout, obs, next_seed_start, rollout_timing = collect_rollout(
                model, env, obs, next_seed_start, args, device, scaler, rollout_buffer
            )
            stats = update_model(model, raw_model, optimizer, rollout, args, device, master_weights)
            _accumulate_timing(timing_totals, rollout_timing)
            _accumulate_timing(
                timing_totals,
                {key: value for key, value in stats.items() if key.endswith("_seconds")},
            )
            global_step += args.num_envs * args.rollout_steps
            update += 1
            checkpoint_path = None
            checkpoint_started = time.perf_counter()
            if update % args.checkpoint_interval_updates == 0 or global_step >= args.total_steps:
                checkpoint_path = save_training_checkpoint(
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
            timing_totals["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
            metrics = build_standard_ppo_metrics(
                update=update,
                global_step=global_step,
                elapsed=max(time.time() - started, 1e-6),
                rollout=rollout,
                update_stats=stats,
            )
            metrics |= {f"timing/{key}_total": value for key, value in timing_totals.items()}
            measured_wall = max(time.time() - started, 0.0)
            timing_totals["top_level_unclassified_seconds"] = max(
                measured_wall
                - timing_totals.get("rollout_total_seconds", 0.0)
                - timing_totals.get("update_total_seconds", 0.0)
                - timing_totals["checkpoint_seconds"],
                0.0,
            )
            metrics["timing/top_level_unclassified_seconds_total"] = timing_totals[
                "top_level_unclassified_seconds"
            ]
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
                f"entropy={stats['entropy']:.5f} checkpoint={checkpoint_path}",
                flush=True,
            )
            print(
                "timing_total "
                f"elapsed={measured_wall:.3f} "
                f"rollout={timing_totals.get('rollout_total_seconds', 0.0):.3f} "
                f"update={timing_totals.get('update_total_seconds', 0.0):.3f} "
                f"checkpoint={timing_totals['checkpoint_seconds']:.3f} "
                f"top_other={timing_totals['top_level_unclassified_seconds']:.3f}",
                flush=True,
            )
            print(
                "timing_detail "
                f"env={timing_totals.get('env_step_seconds', 0.0):.3f} "
                f"rust_validate={timing_totals.get('env_rust_validate_seconds', 0.0):.3f} "
                f"rust_step={timing_totals.get('env_rust_step_seconds', 0.0):.3f} "
                f"rust_prepare={timing_totals.get('env_rust_prepare_seconds', 0.0):.3f} "
                f"rust_encode={timing_totals.get('env_rust_encode_seconds', 0.0):.3f} "
                f"shared_batch_wait={timing_totals.get('env_client_header_wait_seconds', 0.0):.3f} "
                f"h2d_norm={timing_totals.get('rollout_input_h2d_normalize_seconds', 0.0):.3f} "
                f"rollout_store={timing_totals.get('rollout_store_d2h_seconds', 0.0):.3f} "
                f"stack_gae={timing_totals.get('rollout_stack_gae_seconds', 0.0):.3f} "
                f"update_h2d={timing_totals.get('update_prepare_h2d_seconds', 0.0):.3f} "
                f"update_forward={timing_totals.get('update_forward_seconds', 0.0):.3f} "
                f"backward={timing_totals.get('update_backward_seconds', 0.0):.3f} "
                f"grad={timing_totals.get('update_gradient_processing_seconds', 0.0):.3f} "
                f"optimizer={timing_totals.get('update_optimizer_step_seconds', 0.0):.3f} "
                f"master_copy={timing_totals.get('update_master_to_model_seconds', 0.0):.3f}",
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
            parser.add_argument(
                option,
                dest=key,
                type=int if key in {"fixed_m", "fixed_u"} else str,
                default=argparse.SUPPRESS,
            )
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
        "--wandb-mode", choices=("online", "offline", "disabled"), default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--wandb-tag", dest="wandb_tags", action="append", default=argparse.SUPPRESS
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
    if config["pf_particles"] <= 0:
        raise ValueError("pf_particles must be positive")
    return argparse.Namespace(**config)


if __name__ == "__main__":
    main()
