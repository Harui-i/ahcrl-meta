import pytest
import torch

from ahcrl.training.rollout import RolloutBuffer, RolloutFieldSpec


def _buffer() -> RolloutBuffer:
    return RolloutBuffer(
        2,
        3,
        {
            "obs": RolloutFieldSpec((2,), torch.float32, torch.device("cpu")),
            "action": RolloutFieldSpec((), torch.int64, torch.device("cpu")),
        },
    )


def test_rollout_buffer_reuses_storage_and_returns_fixed_layout() -> None:
    buffer = _buffer()
    first = {
        "obs": torch.ones(3, 2),
        "action": torch.arange(3, dtype=torch.int64),
    }
    buffer.store(0, **first)
    buffer.store(1, obs=first["obs"] * 2, action=first["action"] + 1)
    result = buffer.as_dict()
    pointer = result["obs"].data_ptr()
    assert result["obs"].shape == (2, 3, 2)
    assert result["action"].tolist() == [[0, 1, 2], [1, 2, 3]]
    buffer.reset()
    buffer.store(0, **first)
    buffer.store(1, obs=first["obs"] * 2, action=first["action"] + 1)
    assert buffer.as_dict()["obs"].data_ptr() == pointer


def test_rollout_buffer_rejects_incomplete_duplicate_and_bad_values() -> None:
    buffer = _buffer()
    with pytest.raises(RuntimeError, match="was not written"):
        buffer.as_dict()
    buffer.store(0, obs=torch.ones(3, 2), action=torch.zeros(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="already written"):
        buffer.store(0, obs=torch.ones(3, 2), action=torch.zeros(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="shape"):
        buffer.store(1, obs=torch.ones(3, 3), action=torch.zeros(3, dtype=torch.int64))
