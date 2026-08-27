"""学習 run directory と軽量な実行状態を管理する。"""

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

CONFIG_FILE_NAME = "config.json"
STATE_FILE_NAME = "state.json"


def prepare_run_dir(*, artifact_dir: Path, resume_dir: Path | None) -> Path:
    if resume_dir is not None:
        return resume_dir

    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = artifact_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix:02d}")
        suffix += 1
    (run_dir / "checkpoints").mkdir(parents=True)
    return run_dir


def write_config(run_dir: Path, config: Mapping[str, Any]) -> None:
    (run_dir / CONFIG_FILE_NAME).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def update_run_state(
    run_dir: Path,
    *,
    global_step: int,
    update: int,
    wandb_run_id: str | None,
) -> None:
    path = run_dir / STATE_FILE_NAME
    previous: dict[str, Any] = {}
    if path.exists():
        previous = json.loads(path.read_text())
    now = datetime.now().isoformat(timespec="seconds")
    state = {
        "created_at": previous.get("created_at", now),
        "updated_at": now,
        "global_step": global_step,
        "update": update,
        "wandb_run_id": wandb_run_id if wandb_run_id is not None else previous.get("wandb_run_id"),
    }
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def get_wandb_run_id(run_dir: Path) -> str | None:
    path = run_dir / STATE_FILE_NAME
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    run_id = state.get("wandb_run_id")
    return run_id if isinstance(run_id, str) and run_id else None
