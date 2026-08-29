"""固定 seed に対する方策評価の共通処理。"""

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

EVALUATIONS_FILE_NAME = "evaluations.jsonl"
_UINT64_MAX = np.iinfo(np.uint64).max


@runtime_checkable
class EvaluationStepResult(Protocol):
    done: np.ndarray
    score: np.ndarray


@runtime_checkable
class EvaluationEnv(Protocol):
    obs: dict[str, np.ndarray]

    def __enter__(self) -> "EvaluationEnv": ...

    def __exit__(self, *_args: object) -> None: ...

    def step_mask(self, active: np.ndarray, actions: np.ndarray) -> EvaluationStepResult: ...

    def visualizer_data(self) -> list[tuple[str, str]]: ...


EvaluationEnvFactory = Callable[[int, int, int], EvaluationEnv]
EvaluationActionSelector = Callable[[dict[str, np.ndarray], np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FixedSeedEvaluation:
    """固定 seed 評価の最終 score と所要時間。"""

    seed_scores: tuple[tuple[int, int], ...]
    elapsed_seconds: float
    seed_visualizer_data: tuple[tuple[int, str, str], ...] = ()

    def summary(self) -> dict[str, float | int]:
        scores = np.asarray([score for _, score in self.seed_scores], dtype=np.float64)
        if scores.size == 0:
            raise ValueError("evaluation must contain at least one score")
        return {
            "count": int(scores.size),
            "mean": float(scores.mean()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "std": float(scores.std()),
        }


def evaluate_fixed_seeds(
    *,
    make_env: EvaluationEnvFactory,
    action_selector: EvaluationActionSelector,
    seed_start: int,
    seed_num: int,
    seed_stride: int,
    num_envs: int,
    collect_visualizer_data: bool = False,
) -> FixedSeedEvaluation:
    """固定 seed 列の各エピソードを最後まで rollout する。

    ``action_selector`` は各バッチの seed と active mask を受け取り、全
    environment 分の action を返す。inactive slot の action は無視される。
    """
    _validate_evaluation_seeds(seed_start, seed_num, seed_stride)
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    started = time.perf_counter()
    scores: list[tuple[int, int]] = []
    visualizer_data: list[tuple[int, str, str]] = []
    for offset in range(0, seed_num, num_envs):
        batch_size = min(num_envs, seed_num - offset)
        batch_seed_start = seed_start + offset * seed_stride
        seeds = np.asarray(
            [batch_seed_start + index * seed_stride for index in range(batch_size)],
            dtype=np.uint64,
        )
        with make_env(batch_size, batch_seed_start, seed_stride) as env:
            active = np.ones(batch_size, dtype=np.bool_)
            while active.any():
                actions = np.asarray(action_selector(env.obs, active.copy(), seeds.copy()))
                if actions.shape != (batch_size,):
                    raise ValueError(
                        f"action_selector returned shape {actions.shape}, expected ({batch_size},)"
                    )
                result = env.step_mask(active, actions)
                completed = active & result.done
                scores.extend(
                    (int(seeds[index]), int(result.score[index]))
                    for index in np.flatnonzero(completed)
                )
                active &= ~result.done
            if collect_visualizer_data:
                batch_visualizer_data = env.visualizer_data()
                if len(batch_visualizer_data) != batch_size:
                    raise RuntimeError(
                        "visualizer data count does not match the evaluation batch size"
                    )
                visualizer_data.extend(
                    (int(seed), input_text, output_text)
                    for seed, (input_text, output_text) in zip(
                        seeds, batch_visualizer_data, strict=True
                    )
                )
    if len(scores) != seed_num:
        raise RuntimeError(f"evaluation completed {len(scores)} seeds, expected {seed_num}")
    if collect_visualizer_data and len(visualizer_data) != seed_num:
        raise RuntimeError(
            f"evaluation collected {len(visualizer_data)} visualizer outputs, expected {seed_num}"
        )
    return FixedSeedEvaluation(
        tuple(sorted(scores)),
        time.perf_counter() - started,
        tuple(sorted(visualizer_data)),
    )


def build_evaluation_metrics(
    result: FixedSeedEvaluation, *, temperature: float
) -> dict[str, float | int]:
    """W&B 向けの固定 seed 評価メトリクスを返す。"""
    summary = result.summary()
    return {
        "eval/score_mean": summary["mean"],
        "eval/score_min": summary["min"],
        "eval/score_max": summary["max"],
        "eval/score_std": summary["std"],
        "eval/seed_count": summary["count"],
        "eval/temperature": temperature,
        "eval/elapsed_sec": result.elapsed_seconds,
    }


def append_evaluation_record(
    run_dir: Path,
    *,
    global_step: int,
    update: int,
    checkpoint_path: Path,
    evaluation_config: Mapping[str, Any],
    result: FixedSeedEvaluation,
) -> Path:
    """評価結果を run directory の JSONL に 1 行追記する。"""
    try:
        checkpoint = checkpoint_path.relative_to(run_dir)
    except ValueError as error:
        raise ValueError("checkpoint_path must be inside run_dir") from error
    visualizer_dir = write_visualizer_artifacts(run_dir, global_step=global_step, result=result)
    record = {
        "global_step": global_step,
        "update": update,
        "checkpoint": str(checkpoint),
        "evaluation": dict(evaluation_config),
        "elapsed_sec": result.elapsed_seconds,
        "summary": result.summary(),
        "seed_scores": [{"seed": seed, "score": score} for seed, score in result.seed_scores],
    }
    if visualizer_dir is not None:
        record["visualizer_dir"] = str(visualizer_dir.relative_to(run_dir))
    path = run_dir / EVALUATIONS_FILE_NAME
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def write_visualizer_artifacts(
    run_dir: Path, *, global_step: int, result: FixedSeedEvaluation
) -> Path | None:
    """可視化ツールへ渡せる ``in/`` と ``out/`` を評価 run に保存する。"""
    if not result.seed_visualizer_data:
        return None
    directory = run_dir / "evaluations" / f"step_{global_step}"
    input_dir = directory / "in"
    output_dir = directory / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed, input_text, output_text in result.seed_visualizer_data:
        filename = f"{seed:04d}.txt"
        (input_dir / filename).write_text(input_text)
        (output_dir / filename).write_text(output_text)
    return directory


def _validate_evaluation_seeds(seed_start: int, seed_num: int, seed_stride: int) -> None:
    if not isinstance(seed_start, int) or not 0 <= seed_start <= _UINT64_MAX:
        raise ValueError("seed_start must fit in uint64")
    if not isinstance(seed_num, int) or seed_num <= 0:
        raise ValueError("seed_num must be positive")
    if not isinstance(seed_stride, int) or seed_stride <= 0:
        raise ValueError("seed_stride must be positive")
    if seed_start + (seed_num - 1) * seed_stride > _UINT64_MAX:
        raise ValueError("evaluation seed range must fit in uint64")
