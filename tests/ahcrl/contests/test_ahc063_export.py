from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch

from ahcrl.contests.ahc063.model import PPOModel


def _load_exporter() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "contests"
        / "ahc-063"
        / "scripts"
        / "export_torchscript_submit.py"
    )
    spec = spec_from_file_location("ahc063_exporter", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exported_actor_has_nine_inputs(tmp_path: Path) -> None:
    model = PPOModel(
        sequence_channels=4,
        sequence_blocks=1,
        spatial_channels=4,
        spatial_blocks=1,
        head_channels=4,
        head_blocks=1,
    ).float()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    config = {
        "model_architecture": "slot_fusion_conv_v1",
        "model_sequence_channels": 4,
        "model_sequence_blocks": 1,
        "model_spatial_channels": 4,
        "model_spatial_blocks": 1,
        "model_head_channels": 4,
        "model_head_blocks": 1,
    }
    payload = _load_exporter().export_torchscript(checkpoint, config)
    exported = torch.jit.load("/tmp/ahc063_export.pt")
    assert payload
    assert len(list(exported.graph.inputs())) == 10  # self + nine tensors


def test_old_checkpoint_config_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": {}}, checkpoint)
    load_policy = _load_exporter().load_policy

    try:
        load_policy(checkpoint, {"model_channels": 16, "model_blocks": 1})
    except ValueError as error:
        assert "旧AHC063" in str(error)
    else:
        raise AssertionError("legacy config was accepted")
