import json
from pathlib import Path

import pytest

from ahcrl.training.config import config_for_save, load_toml_config, resolve_config

DEFAULTS = {
    "device": "auto",
    "artifact_dir": Path("artifacts"),
    "total_steps": 10,
    "lr": 0.1,
}


def test_resolve_config_merges_saved_toml_and_cli_in_order(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"device": "cpu", "artifact_dir": "saved", "total_steps": 20, "lr": 0.2})
    )
    config_path = tmp_path / "train.toml"
    config_path.write_text("[ppo]\ntotal_steps = 30\nlr = 0.3\n")

    config = resolve_config(
        DEFAULTS,
        cli_values={"total_steps": 40},
        config_path=config_path,
        resume_dir=run_dir,
        allowed_resume_override_keys={"total_steps", "lr"},
        path_keys={"artifact_dir"},
    )

    assert config["total_steps"] == 40
    assert config["lr"] == 0.3
    assert config["device"] == "cpu"
    assert config["artifact_dir"] == Path("saved")


def test_resolve_config_rejects_unknown_and_disallowed_resume_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "unknown.toml"
    config_path.write_text("[contest]\nunknown = 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        resolve_config(
            DEFAULTS,
            cli_values={},
            config_path=config_path,
            resume_dir=None,
            allowed_resume_override_keys=set(),
            path_keys=set(),
        )


def test_resolve_config_rejects_legacy_train_section(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy.toml"
    config_path.write_text("[train]\ntotal_steps = 30\n")

    with pytest.raises(ValueError, match=r"legacy \[train\] section"):
        resolve_config(
            DEFAULTS,
            cli_values={},
            config_path=config_path,
            resume_dir=None,
            allowed_resume_override_keys=set(),
            path_keys=set(),
        )


def test_load_toml_config_flattens_named_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "sections.toml"
    config_path.write_text(
        "\n".join(
            [
                "[training]",
                "num_envs = 4",
                "",
                "[ppo]",
                "lr = 0.2",
                "",
                "[wandb]",
                'project = "project"',
                "",
                "[model]",
                "channels = 8",
                "",
                "[contest]",
                "fixed_size = 3",
            ]
        )
    )

    config = load_toml_config(
        config_path,
        defaults={
            "num_envs": 1,
            "lr": 0.1,
            "wandb_project": "default",
            "model_channels": 4,
            "fixed_size": None,
        },
    )

    assert config == {
        "num_envs": 4,
        "lr": 0.2,
        "wandb_project": "project",
        "model_channels": 8,
        "fixed_size": 3,
    }

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config_for_save(DEFAULTS, runtime_keys=set())))
    with pytest.raises(ValueError, match="resume only allows approved training overrides"):
        resolve_config(
            DEFAULTS,
            cli_values={"lr": 0.2},
            config_path=None,
            resume_dir=run_dir,
            allowed_resume_override_keys={"total_steps"},
            path_keys=set(),
        )


def test_config_for_save_converts_paths_and_removes_runtime_values() -> None:
    config = config_for_save(
        {"artifact_dir": Path("artifacts"), "run_dir": Path("run"), "count": 3},
        runtime_keys={"run_dir"},
    )

    assert config == {"artifact_dir": "artifacts", "count": 3}
