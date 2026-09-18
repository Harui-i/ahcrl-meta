from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ahcrl.contests.ahc063.encoder import ACTION_COUNT, NUM_PLANES
from ahcrl.contests.ahc063.model import PolicyNetwork, PPOModel
from ahcrl.nn.observation import RunningObservationNormalizer


def _load_exporter() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "contests"
        / "ahc-063"
        / "scripts"
        / "export_torchscript_submit.py"
    )
    spec = spec_from_file_location("ahc063_exporter", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load exporter spec from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exporter_loads_and_traces_actor_only_model(tmp_path: Path) -> None:
    exporter = _load_exporter()
    model = PPOModel(channels=8, blocks=1)
    model.observation_normalizer = RunningObservationNormalizer(NUM_PLANES)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"format_version": 1, "model": model.state_dict()}, checkpoint_path)
    config: dict[str, Any] = {
        "model_channels": 8,
        "model_blocks": 1,
        "obs_norm": True,
    }

    policy = exporter.load_policy(checkpoint_path, config)
    assert isinstance(policy.model, PolicyNetwork)
    output = policy(torch.zeros(1, NUM_PLANES, 16, 16))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, ACTION_COUNT)

    model_bytes = exporter.export_torchscript(checkpoint_path, config)
    traced_path = tmp_path / "export.pt"
    traced_path.write_bytes(model_bytes)
    traced = torch.jit.load(str(traced_path))
    traced_output = traced(torch.zeros(1, NUM_PLANES, 16, 16))
    assert isinstance(traced_output, torch.Tensor)
    assert traced_output.shape == (1, ACTION_COUNT)


def test_exporter_cpp_consumes_logits_tensor_directly() -> None:
    exporter = _load_exporter()

    assert "toTensor().contiguous()" in exporter.CPP_TEMPLATE
    assert "toTuple()" not in exporter.CPP_TEMPLATE
    assert "output->elements()" not in exporter.ARGMAX_ACTION_SELECTION
    assert "output->elements()" not in exporter.SOFTMAX_ACTION_SELECTION
