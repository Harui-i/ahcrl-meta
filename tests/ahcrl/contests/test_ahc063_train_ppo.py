import json
from pathlib import Path

import pytest
import torch

from ahcrl.contests.ahc063.train_ppo import (
    RUNTIME_KEYS,
    FP32MasterWeights,
    RunningRewardScaler,
    create_model,
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
    with pytest.raises(ValueError, match="resume only allows approved training overrides"):
        parse_args(["--resume-dir", str(run_dir), "--gamma", "0.9"])


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
