import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ahcrl.training.checkpoint import (
    FORMAT_VERSION,
    TrainingProgress,
    load_initial_model,
    load_latest_training_checkpoint,
    save_training_checkpoint,
)
from ahcrl.training.run import get_wandb_run_id, update_run_state, write_config


def test_checkpoint_round_trips_model_optimizer_rng_and_extras(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    torch.manual_seed(123)
    np.random.seed(456)
    path = save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        config={"name": "test"},
        progress=TrainingProgress(global_step=16, update=2),
        extras={"value": {"nested": 1}},
    )
    expected_torch_random = torch.rand(1)
    expected_numpy_random = np.random.rand()
    torch.rand(3)
    np.random.rand(3)

    reloaded_model = nn.Linear(2, 1)
    reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters(), lr=0.1)
    loaded = load_latest_training_checkpoint(
        tmp_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
    )

    assert path == tmp_path / "checkpoints" / "step_16.pt"
    assert (tmp_path / "checkpoint_latest.pt").exists()
    assert loaded.progress == TrainingProgress(global_step=16, update=2)
    assert loaded.config == {"name": "test"}
    assert loaded.extras == {"value": {"nested": 1}}
    for expected, actual in zip(expected_parameters, reloaded_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert reloaded_optimizer.state_dict()["state"]
    assert torch.equal(torch.rand(1), expected_torch_random)
    assert np.random.rand() == pytest.approx(expected_numpy_random)

    initial_model = nn.Linear(2, 1)
    load_initial_model(path, model=initial_model, device=torch.device("cpu"))
    for expected, actual in zip(expected_parameters, initial_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_checkpoint_rejects_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    torch.save({"format_version": FORMAT_VERSION - 1}, path)

    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_initial_model(path, model=nn.Linear(1, 1), device=torch.device("cpu"))


def test_run_state_preserves_wandb_id_and_writes_config(tmp_path: Path) -> None:
    write_config(tmp_path, {"path": "artifact", "steps": 10})
    update_run_state(tmp_path, global_step=10, update=1, wandb_run_id="run-1")
    update_run_state(tmp_path, global_step=20, update=2, wandb_run_id=None)

    assert json.loads((tmp_path / "config.json").read_text()) == {"path": "artifact", "steps": 10}
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["global_step"] == 20
    assert state["update"] == 2
    assert get_wandb_run_id(tmp_path) == "run-1"
