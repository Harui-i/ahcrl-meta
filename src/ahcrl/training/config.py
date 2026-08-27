"""学習設定の読込・保存に共通する処理。"""

import copy
import json
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

TOML_SECTION_PREFIXES = {
    "training": "",
    "ppo": "",
    "wandb": "wandb_",
    "model": "model_",
    "contest": "",
}


def jsonable(value: Any) -> Any:
    """JSON に保存できる値へ変換する。"""
    if isinstance(value, Path):
        return str(value)
    return value


def config_for_save(config: Mapping[str, Any], *, runtime_keys: Iterable[str]) -> dict[str, Any]:
    """実行時だけの値を除いた、保存用の解決済み設定を返す。"""
    excluded = set(runtime_keys)
    return {key: jsonable(value) for key, value in sorted(config.items()) if key not in excluded}


def load_toml_config(path: Path, *, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """名前空間付き TOML を平坦な trainer 設定へ変換して検証する。"""
    if path.suffix != ".toml":
        raise ValueError(f"config must be a .toml file: {path}")
    with path.open("rb") as file:
        raw = tomllib.load(file)
    if "train" in raw:
        raise ValueError("legacy [train] section is not supported; use named config sections")
    unknown_sections = sorted(set(raw) - set(TOML_SECTION_PREFIXES))
    if unknown_sections:
        raise ValueError(f"unknown config sections: {', '.join(unknown_sections)}")

    config: dict[str, Any] = {}
    for section, prefix in TOML_SECTION_PREFIXES.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"config section [{section}] must be a table")
        for key, value in values.items():
            config_key = prefix + key
            if config_key in config:
                raise ValueError(f"duplicate config key across sections: {config_key}")
            config[config_key] = value
    _validate_known_keys(config, defaults, source="config")
    return config


def resolve_config(
    defaults: Mapping[str, Any],
    *,
    cli_values: Mapping[str, Any],
    config_path: Path | None,
    resume_dir: Path | None,
    allowed_resume_override_keys: Iterable[str],
    path_keys: Iterable[str],
    resolve_auto_device: bool = True,
) -> dict[str, Any]:
    """既定値、保存済み設定、TOML、CLI の順に設定を解決する。"""
    config = copy.deepcopy(dict(defaults))
    file_config: dict[str, Any] = {}
    if resume_dir is not None:
        config.update(load_saved_config(resume_dir, defaults=defaults))
    if config_path is not None:
        file_config = load_toml_config(config_path, defaults=defaults)

    if resume_dir is not None:
        allowed = set(allowed_resume_override_keys)
        _validate_resume_overrides(file_config, allowed)
        _validate_resume_overrides(
            {key: value for key, value in cli_values.items() if key != "resume_dir"}, allowed
        )

    config.update(file_config)
    config.update(cli_values)
    for key in path_keys:
        if config.get(key) is not None:
            config[key] = Path(config[key])
    if resolve_auto_device and config.get("device") == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def load_saved_config(run_dir: Path, *, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """新形式の run directory から設定を読み込む。"""
    path = run_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"saved config not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"saved config must be an object: {path}")
    _validate_known_keys(raw, defaults, source="saved config")
    return raw


def _validate_known_keys(
    config: Mapping[str, Any], defaults: Mapping[str, Any], *, source: str
) -> None:
    unknown = sorted(set(config) - set(defaults))
    if unknown:
        raise ValueError(f"unknown {source} keys: {', '.join(unknown)}")


def _validate_resume_overrides(overrides: Mapping[str, Any], allowed: set[str]) -> None:
    disallowed = sorted(set(overrides) - allowed)
    if disallowed:
        raise ValueError(
            "resume only allows approved training overrides; disallowed keys: "
            + ", ".join(disallowed)
        )
