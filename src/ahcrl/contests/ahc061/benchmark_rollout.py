import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .model import ActorCritic
from .rust_vec_env import RustVecEnv
from .train_ppo import (
    MODEL_DTYPE,
    _advance_seed_start,
    _initial_next_seed_start,
    load_initial_model,
    parse_args,
)


@dataclass
class PhaseTimer:
    device: torch.device
    totals: dict[str, float]

    def measure(self, phase: str, fn: Callable[[], Any]) -> Any:
        self._sync()
        started = time.perf_counter()
        result = fn()
        self._sync()
        self.totals[phase] = self.totals.get(phase, 0.0) + time.perf_counter() - started
        return result

    def add(self, phase: str, elapsed: float) -> None:
        self.totals[phase] = self.totals.get(phase, 0.0) + elapsed

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def main(argv: list[str] | None = None) -> None:
    bench_args, train_argv = parse_benchmark_args(argv)
    args = parse_args(train_argv)
    if bench_args.rollout_steps is not None:
        args.rollout_steps = bench_args.rollout_steps
    if bench_args.num_envs is not None:
        args.num_envs = bench_args.num_envs
    if bench_args.device is not None:
        args.device = bench_args.device
    if bench_args.compile is not None:
        args.compile = bench_args.compile

    torch.set_num_threads(bench_args.torch_threads)
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
    raw_model.eval()
    model = cast(nn.Module, torch.compile(raw_model)) if args.compile else raw_model

    env = RustVecEnv(
        num_envs=args.num_envs,
        seed_start=args.seed_start,
        seed_stride=args.seed_stride,
        fixed_m=args.fixed_m,
        fixed_u=args.fixed_u,
    )
    try:
        next_seed_start = _initial_next_seed_start(args)
        warmup_started = time.perf_counter()
        for _ in range(bench_args.warmup_updates):
            _, next_seed_start = timed_rollout(
                model,
                env,
                env.obs,
                next_seed_start,
                args,
                device,
                measure=False,
            )
        warmup_sec = time.perf_counter() - warmup_started

        totals: dict[str, float] = {}
        timer = PhaseTimer(device=device, totals=totals)
        measured_started = time.perf_counter()
        obs = env.obs
        for _ in range(bench_args.updates):
            obs, next_seed_start = timed_rollout(
                model,
                env,
                obs,
                next_seed_start,
                args,
                device,
                measure=True,
                timer=timer,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        measured_sec = time.perf_counter() - measured_started

        env_only = None
        if bench_args.env_only_steps > 0:
            env_only = benchmark_env_io_breakdown(env, args, bench_args.env_only_steps)

        result = build_result(args, bench_args, totals, warmup_sec, measured_sec, env_only)
        print_result(result)
        if bench_args.profile_json is not None:
            bench_args.profile_json.parent.mkdir(parents=True, exist_ok=True)
            bench_args.profile_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"profile_json={bench_args.profile_json}", flush=True)
    finally:
        env.close()


def timed_rollout(
    model: nn.Module,
    env: RustVecEnv,
    obs: dict[str, np.ndarray],
    next_seed_start: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    measure: bool,
    timer: PhaseTimer | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    totals: dict[str, float] = {}
    local_timer = timer if timer is not None else PhaseTimer(device=device, totals=totals)
    obs_buf = []
    action_buf = []
    logprob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    mask_buf = []
    score_buf = []

    def timed(phase: str, fn: Callable[[], Any]) -> Any:
        if not measure:
            return fn()
        return local_timer.measure(phase, fn)

    for _ in range(args.rollout_steps):
        encoded = timed(
            "planes_to_device",
            lambda current_obs=obs: torch.from_numpy(current_obs["planes"]).to(
                device=device,
                dtype=MODEL_DTYPE,
            ),
        )
        mask = timed(
            "mask_to_device",
            lambda current_obs=obs: torch.from_numpy(current_obs["mask"]).to(device),
        )

        def infer(
            current_encoded: torch.Tensor = encoded,
            current_mask: torch.Tensor = mask,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                logits, value = model(current_encoded)
                logits = logits.float()
                value = value.float()
                logits = logits.masked_fill(~current_mask, -1e9)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
            return action, logprob, value, current_mask

        action, logprob, value, mask = timed("torch_inference_sample", infer)
        action_np = timed(
            "action_to_numpy",
            lambda current_action=action: current_action.cpu().numpy().astype(
                np.int64,
                copy=False,
            ),
        )
        step = timed("env_step", lambda current_action_np=action_np: env.step(current_action_np))

        def store_buffers(
            current_encoded: torch.Tensor = encoded,
            current_action: torch.Tensor = action,
            current_logprob: torch.Tensor = logprob,
            current_value: torch.Tensor = value,
            current_mask: torch.Tensor = mask,
            current_step: Any = step,
        ) -> None:
            obs_buf.append(current_encoded.cpu())
            action_buf.append(current_action.cpu())
            logprob_buf.append(current_logprob.cpu())
            reward_buf.append(torch.from_numpy(current_step.reward.copy()))
            done_buf.append(torch.from_numpy(current_step.done.astype(np.float32)))
            value_buf.append(current_value.cpu())
            mask_buf.append(current_mask.cpu())
            score_buf.append(torch.from_numpy(current_step.score.copy()))

        timed("rollout_store_cpu", store_buffers)

        obs = step.obs
        if step.done.any():
            obs = timed(
                "env_reset",
                lambda current_seed_start=next_seed_start: env.reset(
                    current_seed_start,
                    args.seed_stride,
                    args.fixed_m,
                    args.fixed_u,
                ),
            )
            next_seed_start = _advance_seed_start(next_seed_start, args)

    next_encoded = timed(
        "bootstrap_planes_to_device",
        lambda current_obs=obs: torch.from_numpy(current_obs["planes"]).to(
            device=device,
            dtype=MODEL_DTYPE,
        ),
    )
    next_value = timed(
        "bootstrap_inference",
        lambda: model(next_encoded)[1].float().cpu(),
    )

    def build_gae() -> None:
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
        _ = {
            "obs": torch.stack(obs_buf),
            "actions": torch.stack(action_buf),
            "logprobs": torch.stack(logprob_buf),
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "advantages": advantages,
            "returns": advantages + values,
            "masks": torch.stack(mask_buf),
            "scores": torch.stack(score_buf),
        }

    timed("gae_and_stack", build_gae)
    return obs, next_seed_start


def benchmark_env_io_breakdown(
    env: RustVecEnv,
    args: argparse.Namespace,
    steps: int,
) -> dict[str, dict[str, float]]:
    encoded_roundtrip = benchmark_env_encoded_roundtrip(env, args, steps)
    noobs_roundtrip = benchmark_env_noobs_roundtrip(env, args, steps)
    rust_internal = benchmark_env_rust_internal(env, args, steps)
    return {
        "encoded_roundtrip": encoded_roundtrip,
        "noobs_roundtrip": noobs_roundtrip,
        "rust_internal": rust_internal,
    }


def benchmark_env_encoded_roundtrip(
    env: RustVecEnv,
    args: argparse.Namespace,
    steps: int,
) -> dict[str, float]:
    obs = env.reset(args.seed_start, args.seed_stride, args.fixed_m, args.fixed_u)
    next_seed_start = _initial_next_seed_start(args)
    started = time.perf_counter()
    reset_sec = 0.0
    for _ in range(steps):
        actions = first_legal_actions(obs["mask"])
        step = env.step(actions)
        obs = step.obs
        if step.done.any():
            reset_started = time.perf_counter()
            obs = env.reset(next_seed_start, args.seed_stride, args.fixed_m, args.fixed_u)
            reset_sec += time.perf_counter() - reset_started
            next_seed_start = _advance_seed_start(next_seed_start, args)
    elapsed = time.perf_counter() - started
    env_steps = args.num_envs * steps
    return {
        "steps": float(steps),
        "env_steps": float(env_steps),
        "elapsed_sec": elapsed,
        "fps": env_steps / max(elapsed, 1e-9),
        "reset_sec": reset_sec,
    }


def benchmark_env_noobs_roundtrip(
    env: RustVecEnv,
    args: argparse.Namespace,
    steps: int,
) -> dict[str, float]:
    env.reset(args.seed_start, args.seed_stride, args.fixed_m, args.fixed_u)
    started = time.perf_counter()
    for _ in range(steps):
        env.step_first_legal_noobs()
    elapsed = time.perf_counter() - started
    env_steps = args.num_envs * steps
    return {
        "steps": float(steps),
        "env_steps": float(env_steps),
        "elapsed_sec": elapsed,
        "fps": env_steps / max(elapsed, 1e-9),
    }


def benchmark_env_rust_internal(
    env: RustVecEnv,
    args: argparse.Namespace,
    steps: int,
) -> dict[str, float]:
    env.reset(args.seed_start, args.seed_stride, args.fixed_m, args.fixed_u)
    wall_started = time.perf_counter()
    env_steps, rust_elapsed = env.bench_first_legal_internal(steps)
    wall_elapsed = time.perf_counter() - wall_started
    return {
        "steps": float(steps),
        "env_steps": float(env_steps),
        "rust_elapsed_sec": rust_elapsed,
        "wall_elapsed_sec": wall_elapsed,
        "rust_fps": env_steps / max(rust_elapsed, 1e-9),
        "wall_fps": env_steps / max(wall_elapsed, 1e-9),
    }


def first_legal_actions(mask: np.ndarray) -> np.ndarray:
    actions = np.zeros(mask.shape[0], dtype=np.int64)
    for i in range(mask.shape[0]):
        legal = np.flatnonzero(mask[i])
        if legal.size == 0:
            actions[i] = 0
        else:
            actions[i] = int(legal[0])
    return actions


def build_result(
    args: argparse.Namespace,
    bench_args: argparse.Namespace,
    totals: dict[str, float],
    warmup_sec: float,
    measured_sec: float,
    env_only: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    env_steps = args.num_envs * args.rollout_steps * bench_args.updates
    accounted = sum(totals.values())
    phases = {
        phase: {
            "sec": sec,
            "pct_measured": sec / max(measured_sec, 1e-9) * 100.0,
            "us_per_env_step": sec / max(env_steps, 1) * 1_000_000.0,
        }
        for phase, sec in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    }
    return {
        "config": {
            "num_envs": args.num_envs,
            "rollout_steps": args.rollout_steps,
            "updates": bench_args.updates,
            "warmup_updates": bench_args.warmup_updates,
            "device": str(args.device),
            "compile": bool(args.compile),
            "model_channels": args.model_channels,
            "model_blocks": args.model_blocks,
            "model_block_type": args.model_block_type,
            "torch_threads": bench_args.torch_threads,
        },
        "summary": {
            "env_steps": env_steps,
            "warmup_sec": warmup_sec,
            "measured_sec": measured_sec,
            "accounted_sec": accounted,
            "unaccounted_sec": measured_sec - accounted,
            "fps": env_steps / max(measured_sec, 1e-9),
        },
        "phases": phases,
        "env_only": env_only,
    }


def print_result(result: dict[str, Any]) -> None:
    config = result["config"]
    summary = result["summary"]
    print(
        "benchmark_config="
        + json.dumps(config, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print(
        "summary "
        f"env_steps={summary['env_steps']} "
        f"measured_sec={summary['measured_sec']:.6f} "
        f"fps={summary['fps']:.1f} "
        f"warmup_sec={summary['warmup_sec']:.6f} "
        f"unaccounted_sec={summary['unaccounted_sec']:.6f}",
        flush=True,
    )
    print("phase                         sec        pct   us/env_step", flush=True)
    for phase, stats in result["phases"].items():
        print(
            f"{phase:<26} {stats['sec']:>9.6f} {stats['pct_measured']:>7.2f}"
            f" {stats['us_per_env_step']:>11.3f}",
            flush=True,
        )
    if result["env_only"] is not None:
        env_only = result["env_only"]["encoded_roundtrip"]
        print(
            "env_encoded_roundtrip "
            f"steps={int(env_only['steps'])} "
            f"env_steps={int(env_only['env_steps'])} "
            f"elapsed_sec={env_only['elapsed_sec']:.6f} "
            f"fps={env_only['fps']:.1f} "
            f"reset_sec={env_only['reset_sec']:.6f}",
            flush=True,
        )
        noobs = result["env_only"]["noobs_roundtrip"]
        print(
            "env_noobs_roundtrip "
            f"steps={int(noobs['steps'])} "
            f"env_steps={int(noobs['env_steps'])} "
            f"elapsed_sec={noobs['elapsed_sec']:.6f} "
            f"fps={noobs['fps']:.1f}",
            flush=True,
        )
        internal = result["env_only"]["rust_internal"]
        print(
            "env_rust_internal "
            f"steps={int(internal['steps'])} "
            f"env_steps={int(internal['env_steps'])} "
            f"rust_elapsed_sec={internal['rust_elapsed_sec']:.6f} "
            f"rust_fps={internal['rust_fps']:.1f} "
            f"wall_elapsed_sec={internal['wall_elapsed_sec']:.6f} "
            f"wall_fps={internal['wall_fps']:.1f}",
            flush=True,
        )


def parse_benchmark_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--env-only-steps", type=int, default=0)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--compile", dest="compile", action="store_true", default=None)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    bench_args, train_argv = parser.parse_known_args(argv)
    if bench_args.updates <= 0:
        raise ValueError("--updates must be positive")
    if bench_args.warmup_updates < 0:
        raise ValueError("--warmup-updates must be non-negative")
    if bench_args.env_only_steps < 0:
        raise ValueError("--env-only-steps must be non-negative")
    if bench_args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    return bench_args, train_argv


if __name__ == "__main__":
    main()
