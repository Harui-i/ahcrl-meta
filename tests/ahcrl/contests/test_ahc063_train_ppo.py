from pathlib import Path

import torch

from ahcrl.contests.ahc063.train_ppo import (
    DEFAULT_CONFIG,
    RunningRewardScaler,
    create_model,
    parse_args,
)


def test_smoke_config_uses_slot_fusion_schema() -> None:
    args = parse_args(["--config", "contests/ahc-063/configs/ppo_smoke.toml"])
    assert args.model_architecture == "slot_fusion_conv_v1"
    assert args.model_sequence_channels == 16
    assert not hasattr(args, "obs_norm")


def test_default_config_drops_legacy_model_fields() -> None:
    assert "model_channels" not in DEFAULT_CONFIG
    assert "model_blocks" not in DEFAULT_CONFIG
    assert "obs_norm" not in DEFAULT_CONFIG


def test_model_creation_from_config() -> None:
    args = parse_args(["--config", "contests/ahc-063/configs/ppo_smoke.toml"])
    model = create_model(args, torch.device("cpu"))
    assert model.architecture == "slot_fusion_conv_v1"


def test_reward_scaler_round_trip() -> None:
    scaler = RunningRewardScaler()
    scaler.scale(torch.tensor([1.0, 3.0]))
    state = scaler.state_dict()
    restored = RunningRewardScaler()
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_config_paths_are_resolved() -> None:
    args = parse_args(["--artifact-dir", "/tmp/ahc063-test-artifacts"])
    assert isinstance(args.artifact_dir, Path)
