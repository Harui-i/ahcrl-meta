"""モデル非依存の学習 checkpoint 形式。"""

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

FORMAT_VERSION = 1
LATEST_CHECKPOINT_NAME = "checkpoint_latest.pt"


@dataclass(frozen=True)
class TrainingProgress:
    global_step: int
    update: int


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    progress: TrainingProgress
    config: dict[str, Any]
    extras: dict[str, Any]


def save_training_checkpoint(
    run_dir: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    progress: TrainingProgress,
    extras: Mapping[str, Any],
) -> Path:
    """checkpoint を保存し、run root の latest を更新する。"""
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoints_dir / f"step_{progress.global_step}.pt"
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": dict(config),
        "progress": {
            "global_step": progress.global_step,
            "update": progress.update,
        },
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        },
        "extras": dict(extras),
    }
    torch.save(payload, path)
    shutil.copy2(path, run_dir / LATEST_CHECKPOINT_NAME)
    return path


def load_latest_training_checkpoint(
    run_dir: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> LoadedTrainingCheckpoint:
    path = run_dir / LATEST_CHECKPOINT_NAME
    if not path.exists():
        raise FileNotFoundError(f"latest checkpoint not found: {path}")
    return load_training_checkpoint(path, model=model, optimizer=optimizer, device=device)


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> LoadedTrainingCheckpoint:
    payload = _load_payload(path, device=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    np.random.set_state(payload["rng"]["numpy"])
    progress = payload["progress"]
    return LoadedTrainingCheckpoint(
        progress=TrainingProgress(
            global_step=int(progress["global_step"]),
            update=int(progress["update"]),
        ),
        config=dict(payload["config"]),
        extras=dict(payload["extras"]),
    )


def load_initial_model(path: Path, *, model: nn.Module, device: torch.device) -> None:
    """新形式 checkpoint の model state のみを読み込む。"""
    payload = _load_payload(path, device=device)
    model.load_state_dict(payload["model"])


def _load_payload(path: Path, *, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {path}; expected format_version={FORMAT_VERSION}"
        )
    required = {"model", "optimizer", "config", "progress", "rng", "extras"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint missing required fields: {', '.join(missing)}")
    if not isinstance(payload["progress"], dict) or not isinstance(payload["rng"], dict):
        raise ValueError("checkpoint progress and rng must be objects")
    for key in ("global_step", "update"):
        if key not in payload["progress"]:
            raise ValueError(f"checkpoint progress missing {key}")
    for key in ("torch", "numpy"):
        if key not in payload["rng"]:
            raise ValueError(f"checkpoint rng missing {key}")
    return payload
