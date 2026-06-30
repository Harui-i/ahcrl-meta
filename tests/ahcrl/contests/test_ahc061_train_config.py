from pathlib import Path

import pytest

from ahcrl.contests.ahc061.train_ppo import parse_args


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
                'device = "cpu"',
                'checkpoint_dir = "tmp/checkpoints"',
            ]
        )
    )

    args = parse_args(["--config", str(config_path), "--num-envs", "4", "--model-blocks", "3"])

    assert args.num_envs == 4
    assert args.total_steps == 1024
    assert args.model_channels == 32
    assert args.model_blocks == 3
    assert args.device == "cpu"
    assert args.checkpoint_dir == Path("tmp/checkpoints")


def test_parse_args_rejects_unknown_toml_key(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\nunknown = 1\n")

    with pytest.raises(ValueError, match="unknown config keys"):
        parse_args(["--config", str(config_path)])
