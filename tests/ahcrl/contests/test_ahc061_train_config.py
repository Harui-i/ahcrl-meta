from pathlib import Path

import pytest
import torch

from ahcrl.contests.ahc061.train_ppo import (
    _advance_seed_start,
    _initial_next_seed_start,
    _normalized_entropy,
    build_log_metrics,
    parse_args,
)


def test_parse_args_loads_toml_config_and_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "\n".join(
            [
                "[train]",
                "num_envs = 8",
                "total_steps = 1024",
                "model_channels = 32",
                "model_blocks = 2",
                'model_block_type = "residual"',
                'device = "cpu"',
                'checkpoint_dir = "tmp/checkpoints"',
                "checkpoint_interval_updates = 5",
                "wandb_enabled = true",
                'wandb_project = "test-project"',
                'wandb_name = "test-run"',
                'wandb_tags = ["ahc061", "test"]',
            ]
        )
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--num-envs",
            "4",
            "--model-blocks",
            "3",
            "--model-block-type",
            "convnext",
        ]
    )

    assert args.num_envs == 4
    assert args.total_steps == 1024
    assert args.model_channels == 32
    assert args.model_blocks == 3
    assert args.model_block_type == "convnext"
    assert args.device == "cpu"
    assert args.checkpoint_dir == Path("tmp/checkpoints")
    assert args.checkpoint_interval_updates == 5
    assert args.wandb_enabled is True
    assert args.wandb_project == "test-project"
    assert args.wandb_name == "test-run"
    assert args.wandb_tags == ["ahc061", "test"]


def test_parse_args_rejects_unknown_toml_key(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\nunknown = 1\n")

    with pytest.raises(ValueError, match="unknown config keys"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_non_positive_checkpoint_interval(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\ncheckpoint_interval_updates = 0\n")

    with pytest.raises(ValueError, match="checkpoint_interval_updates"):
        parse_args(["--config", str(config_path)])


def test_seed_blocks_advance_by_parallel_env_span(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "\n".join(
            [
                "[train]",
                "num_envs = 256",
                "seed_start = 1000",
                "seed_stride = 3",
            ]
        )
    )

    args = parse_args(["--config", str(config_path)])

    assert _initial_next_seed_start(args) == 1768
    assert _advance_seed_start(1768, args) == 2536


def test_build_log_metrics_contains_required_wandb_stats() -> None:
    rollout = {
        "scores": torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        "rewards": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        "dones": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "values": torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
        "advantages": torch.tensor([[1.0, -1.0], [0.5, -0.5]]),
        "returns": torch.tensor([[1.5, 1.6], [1.7, 1.8]]),
        "masks": torch.ones(2, 2, 10),
    }
    stats = {
        "policy_loss": 0.01,
        "value_loss": 0.02,
        "entropy": 0.03,
        "normalized_entropy": 0.5,
        "approx_kl": 0.04,
        "clip_frac": 0.05,
    }

    metrics = build_log_metrics(
        update=3,
        global_step=128,
        elapsed=2.0,
        rollout=rollout,
        stats=stats,
        checkpoint_path=Path("checkpoint.pt"),
    )

    assert metrics["summary/cumulative_env_steps"] == 128
    assert metrics["summary/updates"] == 3
    assert metrics["summary/fps"] == 64.0
    assert metrics["train/mean_score"] == 4.0
    assert metrics["train/final_mean_score"] == 6.0
    assert metrics["train/mean_reward"] == pytest.approx(0.25)
    assert metrics["loss/policy"] == 0.01
    assert metrics["train/normalized_entropy"] == 0.5
    assert metrics["checkpoint/path"] == "checkpoint.pt"


def test_normalized_entropy_scales_by_valid_action_count() -> None:
    entropy = torch.tensor([torch.log(torch.tensor(4.0)), 0.0])
    mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, False, False, False, False],
        ]
    )

    assert _normalized_entropy(entropy, mask).item() == pytest.approx(0.5)
