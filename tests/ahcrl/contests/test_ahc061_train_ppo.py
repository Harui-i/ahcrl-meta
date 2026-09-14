from pathlib import Path

import pytest
import torch

from ahcrl.contests.ahc061.train_ppo import (
    RUNTIME_KEYS,
    FP32MasterWeights,
    RunningRewardScaler,
    _clip_epsilons,
    _surrogate_clip_mask,
    create_model,
    parse_args,
)
from ahcrl.training import (
    TrainingProgress,
    config_for_save,
    load_latest_training_checkpoint,
    save_training_checkpoint,
)


def test_parse_args_uses_shared_toml_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "[training]\nnum_envs = 4\ndevice = 'cpu'\n\n"
        "[model]\nchannels = 8\nblocks = 1\n\n"
        "[contest]\npf_particles = 2\n"
    )
    args = parse_args(["--config", str(config_path), "--fixed-m", "4", "--fixed-u", "3"])
    assert args.num_envs == 4
    assert args.env_workers == 0
    assert args.model_channels == 8
    assert args.model_blocks == 1
    assert args.pf_particles == 2
    assert args.fixed_m == 4
    assert args.fixed_u == 3


def test_parse_args_accepts_and_rejects_env_workers() -> None:
    assert parse_args(["--env-workers", "3"]).env_workers == 3
    with pytest.raises(ValueError, match="env_workers"):
        parse_args(["--env-workers", "-1"])


def test_adaptive_clip_configuration_and_validation() -> None:
    args = parse_args(
        [
            "--adaptive-clip",
            "--adaptive-clip-kappa",
            "0.15",
            "--adaptive-clip-min",
            "0.05",
            "--adaptive-clip-max",
            "0.4",
        ]
    )
    assert args.adaptive_clip
    assert args.adaptive_clip_kappa == 0.15
    with pytest.raises(ValueError, match="adaptive_clip_kappa"):
        parse_args(["--adaptive-clip-kappa", "-0.1"])
    with pytest.raises(ValueError, match="adaptive_clip_min"):
        parse_args(["--adaptive-clip-min", "-0.1"])
    with pytest.raises(ValueError, match="adaptive_clip_max"):
        parse_args(["--adaptive-clip-min", "0.2", "--adaptive-clip-max", "0.1"])


def test_clip_epsilons_use_fixed_or_adaptive_bounds() -> None:
    fixed_args = parse_args(["--clip", "0.2"])
    advantages = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert torch.equal(_clip_epsilons(advantages, fixed_args), torch.full_like(advantages, 0.2))

    adaptive_args = parse_args(
        [
            "--adaptive-clip",
            "--adaptive-clip-kappa",
            "0.15",
            "--adaptive-clip-min",
            "0.05",
            "--adaptive-clip-max",
            "0.4",
        ]
    )
    assert torch.equal(
        _clip_epsilons(advantages, adaptive_args),
        torch.tensor([0.4, 0.15, 0.05, 0.15, 0.4]),
    )


def test_surrogate_clip_mask_respects_advantage_sign() -> None:
    ratio = torch.tensor([1.3, 0.7, 1.3, 0.7, 1.0])
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0])
    clip_epsilons = torch.full_like(ratio, 0.2)
    assert torch.equal(
        _surrogate_clip_mask(ratio, advantages, clip_epsilons),
        torch.tensor([True, False, False, True, False]),
    )


def test_checkpoint_round_trips_master_weights_and_reward_scaler(tmp_path: Path) -> None:
    args = parse_args(["--device", "cpu", "--model-channels", "8", "--model-blocks", "1"])
    model = create_model(args, torch.device("cpu"))
    master = FP32MasterWeights(model)
    optimizer = torch.optim.AdamW(master.parameters, lr=args.lr)
    scaler = RunningRewardScaler()
    scaler.scale(torch.tensor([[1.0, 2.0]]))
    config = config_for_save(vars(args), runtime_keys=RUNTIME_KEYS)
    save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        config=config,
        progress=TrainingProgress(global_step=8, update=1),
        extras={"master_weights": master.state_dict(), "reward_scaler": scaler.state_dict()},
    )
    loaded_model = create_model(args, torch.device("cpu"))
    loaded_master = FP32MasterWeights(loaded_model)
    loaded_optimizer = torch.optim.AdamW(loaded_master.parameters, lr=args.lr)
    loaded = load_latest_training_checkpoint(
        tmp_path, model=loaded_model, optimizer=loaded_optimizer, device=torch.device("cpu")
    )
    loaded_master.load_state_dict(loaded.extras["master_weights"])
    assert loaded.progress == TrainingProgress(global_step=8, update=1)
    assert loaded.config == config
