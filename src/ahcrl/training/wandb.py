"""Weights & Biases 連携の共通処理。"""

from dataclasses import dataclass
from typing import Any, Literal, cast

WandbMode = Literal["online", "offline", "disabled", "shared"]


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    name: str | None
    mode: WandbMode
    tags: list[str]


def init_wandb(
    settings: WandbConfig,
    *,
    resolved_config: dict[str, Any],
    run_id: str | None,
) -> Any | None:
    if not settings.enabled:
        return None

    import wandb

    run = wandb.init(
        project=settings.project,
        entity=settings.entity,
        name=settings.name,
        mode=cast(Literal["online", "offline", "disabled", "shared"], settings.mode),
        tags=settings.tags,
        config=resolved_config,
        id=run_id,
        resume="must" if run_id is not None else None,
    )
    wandb.define_metric("summary/cumulative_env_steps")
    wandb.define_metric("*", step_metric="summary/cumulative_env_steps")
    return run


def finish_wandb(run: Any | None) -> None:
    if run is not None:
        run.finish()
