import json
from pathlib import Path

import pytest
import torch

import ahcrl.contests.ahc063.train_ppo as train_ppo
from ahcrl.contests.ahc063.train_ppo import (
    RUNTIME_KEYS,
    FP32MasterWeights,
    RunningRewardScaler,
    _observation_normalizer,
    create_model,
    evaluate_policy,
    parse_args,
)
from ahcrl.training import (
    TrainingProgress,
    config_for_save,
    load_latest_training_checkpoint,
    save_training_checkpoint,
)


def test_parse_args_loads_wandb_settings_and_aliases(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "\n".join(
            [
                "[training]",
                'device = "cpu"',
                "",
                "[wandb]",
                'entity = "entity"',
                'mode = "offline"',
                'tags = ["base"]',
                "",
                "[evaluation]",
                "enabled = true",
                "seed_num = 3",
                "temperature = 0.5",
            ]
        )
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--wandb",
            "--wandb-name",
            "run-name",
            "--wandb-tag",
            "cli-a",
            "--wandb-tag",
            "cli-b",
        ]
    )

    assert args.wandb_enabled is True
    assert args.wandb_entity == "entity"
    assert args.wandb_mode == "offline"
    assert args.wandb_name == "run-name"
    assert args.wandb_tags == ["cli-a", "cli-b"]
    assert args.device == "cpu"
    assert args.eval_enabled is True
    assert args.eval_seed_num == 3
    assert args.eval_temperature == 0.5


def test_parse_args_resume_rejects_disallowed_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = parse_args(["--device", "cpu", "--artifact-dir", str(tmp_path / "artifacts")])
    (run_dir / "config.json").write_text(
        json.dumps(config_for_save(vars(args), runtime_keys=RUNTIME_KEYS))
    )

    resumed = parse_args(["--resume-dir", str(run_dir), "--total-steps", "99"])
    assert resumed.total_steps == 99
    assert resumed.device == "cpu"
    resumed_eval = parse_args(
        ["--resume-dir", str(run_dir), "--eval-enabled", "--eval-seed-num", "3"]
    )
    assert resumed_eval.eval_enabled is True
    assert resumed_eval.eval_seed_num == 3
    with pytest.raises(ValueError, match="resume only allows approved training overrides"):
        parse_args(["--resume-dir", str(run_dir), "--gamma", "0.9"])


def test_parse_args_rejects_invalid_evaluation_values() -> None:
    with pytest.raises(ValueError, match="eval_temperature"):
        parse_args(["--eval-temperature", "-0.1"])
    with pytest.raises(ValueError, match="eval_seed_num"):
        parse_args(["--eval-seed-num", "0"])


def test_evaluate_policy_rolls_out_fixed_seeds_reproducibly_without_updating_obs_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_command = train_ppo.cargo_server_command
    monkeypatch.setattr(
        train_ppo,
        "cargo_server_command",
        lambda manifest: original_command(manifest, release=False),
    )
    args = parse_args(
        [
            "--device",
            "cpu",
            "--model-channels",
            "4",
            "--model-blocks",
            "1",
            "--num-envs",
            "2",
            "--eval-seed-num",
            "3",
            "--eval-temperature",
            "1.0",
            "--eval-fixed-n",
            "8",
            "--eval-fixed-m",
            "16",
            "--eval-fixed-c",
            "3",
            "--eval-max-episode-steps",
            "1",
        ]
    )
    model = create_model(args, torch.device("cpu"))
    normalizer = _observation_normalizer(model)
    assert normalizer is not None
    count_before = normalizer.count.clone()

    metrics, first = evaluate_policy(model, args, torch.device("cpu"))
    _, second = evaluate_policy(model, args, torch.device("cpu"))

    assert metrics["eval/seed_count"] == 3
    assert first.seed_scores == second.seed_scores
    assert torch.equal(normalizer.count, count_before)
    assert model.training


def test_ahc063_checkpoint_round_trips_reward_scaler_and_master_weights(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--model-channels",
            "4",
            "--model-blocks",
            "1",
        ]
    )
    model = create_model(args, torch.device("cpu"))
    master_weights = FP32MasterWeights(model)
    optimizer = torch.optim.AdamW(master_weights.parameters, lr=args.lr)
    scaler = RunningRewardScaler()
    scaler.scale(torch.tensor([[1.0, 2.0]]))
    config = config_for_save(vars(args), runtime_keys=RUNTIME_KEYS)

    save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        config=config,
        progress=TrainingProgress(global_step=8, update=1),
        extras={
            "reward_scaler": scaler.state_dict(),
            "master_weights": master_weights.state_dict(),
        },
    )

    reloaded_model = create_model(args, torch.device("cpu"))
    reloaded_master_weights = FP32MasterWeights(reloaded_model)
    reloaded_optimizer = torch.optim.AdamW(reloaded_master_weights.parameters, lr=args.lr)
    loaded = load_latest_training_checkpoint(
        tmp_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
    )
    reloaded_master_weights.load_state_dict(loaded.extras["master_weights"])
    reloaded_scaler = RunningRewardScaler()
    reloaded_scaler.load_state_dict(loaded.extras["reward_scaler"])

    assert loaded.progress == TrainingProgress(global_step=8, update=1)
    assert reloaded_scaler.state_dict() == scaler.state_dict()
    for left, right in zip(
        master_weights.parameters, reloaded_master_weights.parameters, strict=True
    ):
        assert torch.equal(left, right)
