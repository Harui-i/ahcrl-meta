import json
from pathlib import Path

import pytest
import torch

from ahcrl.contests.ahc061.model import ActorCritic
from ahcrl.contests.ahc061.train_ppo import (
    MODEL_DTYPE,
    _advance_seed_start,
    _initial_next_seed_start,
    _normalized_entropy,
    build_log_metrics,
    config_for_save,
    load_initial_model,
    load_training_state,
    parse_args,
    save_training_state,
)


def test_parse_args_loads_toml_config_and_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "\n".join(
            [
                "[train]",
                "num_envs = 8",
                "total_steps = 1024",
                "pf_particles = 32",
                "model_channels = 32",
                "model_blocks = 2",
                'model_block_type = "residual"',
                'device = "cpu"',
                "compile = false",
                'artifact_dir = "tmp/artifacts"',
                "checkpoint_interval_updates = 5",
                'symmetry_augmentation = "full_d4"',
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
            "--symmetry-augmentation",
            "none",
            "--compile",
        ]
    )

    assert args.num_envs == 4
    assert args.total_steps == 1024
    assert args.pf_particles == 32
    assert args.model_channels == 32
    assert args.model_blocks == 3
    assert args.model_block_type == "convnext"
    assert not hasattr(args, "model_dtype")
    assert args.device == "cpu"
    assert args.compile is True
    assert args.artifact_dir == Path("tmp/artifacts")
    assert args.checkpoint_interval_updates == 5
    assert args.symmetry_augmentation == "none"
    assert args.wandb_enabled is True
    assert args.wandb_project == "test-project"
    assert args.wandb_name == "test-run"
    assert args.wandb_tags == ["ahc061", "test"]


def test_parse_args_compile_defaults_to_true_and_can_be_disabled() -> None:
    assert parse_args(["--device", "cpu"]).compile is True
    assert parse_args(["--device", "cpu", "--no-compile"]).compile is False
    assert parse_args(["--device", "cpu"]).pf_particles == 16


def test_parse_args_symmetry_augmentation_defaults_to_none_and_loads_toml(
    tmp_path: Path,
) -> None:
    assert parse_args(["--device", "cpu"]).symmetry_augmentation == "none"

    config_path = tmp_path / "ppo.toml"
    config_path.write_text('[train]\nsymmetry_augmentation = "full_d4"\n')

    args = parse_args(["--config", str(config_path), "--device", "cpu"])

    assert args.symmetry_augmentation == "full_d4"


def test_parse_args_rejects_unknown_toml_key(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\nunknown = 1\n")

    with pytest.raises(ValueError, match="unknown config keys"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_unknown_symmetry_augmentation(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text('[train]\nsymmetry_augmentation = "d4"\n')

    with pytest.raises(ValueError, match="symmetry_augmentation"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_non_positive_checkpoint_interval(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\ncheckpoint_interval_updates = 0\n")

    with pytest.raises(ValueError, match="checkpoint_interval_updates"):
        parse_args(["--config", str(config_path)])


def test_parse_args_resume_loads_saved_config_and_only_allows_total_steps(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    config_path = run_dir / "config.json"
    args = parse_args(
        [
            "--total-steps",
            "128",
            "--num-envs",
            "4",
            "--device",
            "cpu",
            "--artifact-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    config_path.write_text(json.dumps(config_for_save(args), sort_keys=True) + "\n")

    resumed = parse_args(
        [
            "--resume-dir",
            str(run_dir),
            "--total-steps",
            "256",
        ]
    )

    assert resumed.resume_dir == run_dir
    assert resumed.total_steps == 256
    assert resumed.num_envs == 4
    assert resumed.device == "cpu"

    with pytest.raises(ValueError, match="resume only allows overriding total_steps"):
        parse_args(["--resume-dir", str(run_dir), "--lr", "0.001"])


def test_parse_args_rejects_resume_with_init_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}\n")

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_args(
            [
                "--resume-dir",
                str(run_dir),
                "--init-checkpoint",
                str(tmp_path / "checkpoint.pt"),
            ]
        )


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
        "m_values": torch.tensor([[2, 3], [2, 3]]),
        "u_values": torch.tensor([[1, 2], [1, 1]]),
    }
    stats = {
        "policy_loss": 0.01,
        "value_loss": 0.02,
        "entropy": 0.03,
        "normalized_entropy": 0.5,
        "approx_kl": 0.04,
        "clip_frac": 0.05,
        "grad_norm": 0.06,
        "weight_norm": 0.07,
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
    assert metrics["train/final_min_score"] == 5.0
    assert metrics["train/final_max_score"] == 7.0
    assert metrics["train/final_mean_score_by_m/m_2"] == 5.0
    assert metrics["train/final_mean_score_by_m/m_3"] == 7.0
    assert metrics["train/final_mean_score_by_u/u_1"] == 6.0
    assert metrics["train/mean_reward"] == pytest.approx(0.25)
    assert metrics["train/explained_variance"] == pytest.approx(1.0)
    assert metrics["loss/policy"] == 0.01
    assert metrics["train/normalized_entropy"] == 0.5
    assert metrics["model/grad_norm"] == 0.06
    assert metrics["model/weight_norm"] == 0.07
    assert metrics["checkpoint/path"] == "checkpoint.pt"


def test_save_and_load_training_state_round_trips_resume_state(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--model-channels",
            "8",
            "--model-blocks",
            "1",
            "--artifact-dir",
            str(tmp_path),
        ]
    )
    args.run_dir = tmp_path / "run_1"
    args.run_dir.mkdir()
    model = ActorCritic(channels=8, blocks=1).to(dtype=MODEL_DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    torch.manual_seed(123)

    checkpoint_path = save_training_state(
        args,
        model,
        optimizer,
        global_step=128,
        update=2,
        next_seed_start=64,
        wandb_run_id="abc123",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert {
        parameter.dtype
        for parameter in checkpoint["model"].values()
        if torch.is_floating_point(parameter)
    } == {MODEL_DTYPE}

    reloaded_model = ActorCritic(channels=8, blocks=1).to(dtype=MODEL_DTYPE)
    reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters(), lr=args.lr)
    state = load_training_state(
        args.run_dir,
        reloaded_model,
        reloaded_optimizer,
        torch.device("cpu"),
    )

    assert checkpoint_path == args.run_dir / "checkpoints" / "step_128.pt"
    assert (args.run_dir / "checkpoint_latest.pt").exists()
    assert state["global_step"] == 128
    assert state["update"] == 2
    assert state["next_seed_start"] == 64
    for left, right in zip(model.parameters(), reloaded_model.parameters(), strict=True):
        assert torch.equal(left, right)


def test_load_initial_model_loads_only_model_state(tmp_path: Path) -> None:
    source = ActorCritic(channels=8, blocks=1)
    target = ActorCritic(channels=8, blocks=1).to(dtype=MODEL_DTYPE)
    for parameter in source.parameters():
        parameter.data.fill_(0.5)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": source.state_dict(), "global_step": 999}, checkpoint_path)

    load_initial_model(checkpoint_path, target, torch.device("cpu"))

    for left, right in zip(source.parameters(), target.parameters(), strict=True):
        assert right.dtype == MODEL_DTYPE
        assert torch.equal(left.to(dtype=MODEL_DTYPE), right)


def test_normalized_entropy_scales_by_valid_action_count() -> None:
    entropy = torch.tensor([torch.log(torch.tensor(4.0)), 0.0])
    mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, False, False, False, False],
        ]
    )

    assert _normalized_entropy(entropy, mask).item() == pytest.approx(0.5)
