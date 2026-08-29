import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ahcrl.training.evaluation import (
    append_evaluation_record,
    build_evaluation_metrics,
    evaluate_fixed_seeds,
)


@dataclass
class FakeStepResult:
    done: np.ndarray
    score: np.ndarray


class FakeEnv:
    def __init__(self, seeds: np.ndarray) -> None:
        self.seeds = seeds
        self.remaining = seeds.astype(np.int64) % 3 + 1
        self.done = np.zeros(len(seeds), dtype=np.bool_)
        self.closed = False
        self.obs = {"seed": seeds.copy()}

    def __enter__(self) -> "FakeEnv":
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def step_mask(self, active: np.ndarray, actions: np.ndarray) -> FakeStepResult:
        assert actions.shape == self.seeds.shape
        assert not np.any(active & self.done)
        self.remaining[active] -= 1
        self.done |= self.remaining == 0
        return FakeStepResult(done=self.done.copy(), score=(self.seeds * 10).astype(np.int64))


def test_evaluate_fixed_seeds_handles_different_episode_lengths_and_writes_jsonl(
    tmp_path: Path,
) -> None:
    def make_env(num_envs: int, seed_start: int, seed_stride: int) -> FakeEnv:
        seeds = np.asarray(
            [seed_start + index * seed_stride for index in range(num_envs)], dtype=np.uint64
        )
        return FakeEnv(seeds)

    result = evaluate_fixed_seeds(
        make_env=make_env,
        action_selector=lambda _obs, _active, seeds: np.zeros(len(seeds), dtype=np.int64),
        seed_start=3,
        seed_num=4,
        seed_stride=2,
        num_envs=3,
    )

    assert result.seed_scores == ((3, 30), (5, 50), (7, 70), (9, 90))
    assert result.summary() == {
        "count": 4,
        "mean": 60.0,
        "min": 30.0,
        "max": 90.0,
        "std": pytest.approx(22.360679774997898),
    }
    assert build_evaluation_metrics(result, temperature=0.0)["eval/score_mean"] == 60.0

    checkpoint = tmp_path / "checkpoints" / "step_12.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    path = append_evaluation_record(
        tmp_path,
        global_step=12,
        update=3,
        checkpoint_path=checkpoint,
        evaluation_config={"eval_seed_num": 4, "eval_temperature": 0.0},
        result=result,
    )
    record = json.loads(path.read_text())
    assert record["checkpoint"] == "checkpoints/step_12.pt"
    assert record["seed_scores"] == [
        {"seed": 3, "score": 30},
        {"seed": 5, "score": 50},
        {"seed": 7, "score": 70},
        {"seed": 9, "score": 90},
    ]
