from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ahcrl.contests.ahc061.encoder import NUM_PLANES
from ahcrl.contests.ahc061.model import ActorCritic
from ahcrl.nn.observation import RunningObservationNormalizer


def _load_exporter() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "contests"
        / "ahc-061"
        / "scripts"
        / "export_torchscript_submit.py"
    )
    spec = spec_from_file_location("ahc061_exporter", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load exporter spec from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q4_tensor_layout_contains_cell_encoder_before_trunk() -> None:
    exporter = _load_exporter()
    model = ActorCritic(channels=8, blocks=2)

    names = exporter._q4_tensor_names(model)  # type: ignore[attr-defined]

    assert len(names) == 15 + 9 * 2
    assert names[3:7] == [
        "cell_encoder.0.weight",
        "cell_encoder.0.bias",
        "cell_encoder.2.weight",
        "cell_encoder.2.bias",
    ]
    assert names[7] == "trunk.0.weight"
    assert all(name in model.state_dict() for name in names)


def test_q4_pack_and_render_use_version_three_cell_encoder_layout(tmp_path: Path) -> None:
    exporter = _load_exporter()
    model = ActorCritic(channels=8, blocks=1).to(dtype=torch.bfloat16)
    model.observation_normalizer = RunningObservationNormalizer(NUM_PLANES)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"format_version": 1, "model": model.state_dict()}, checkpoint_path)
    config: dict[str, Any] = {
        "model_channels": 8,
        "model_blocks": 1,
        "obs_norm": True,
    }

    packed = exporter.pack_q4_policy(checkpoint_path, config)
    rendered = exporter.render_q4_cpp(
        "encoded",
        checkpoint_name="test",
        packed_size=len(packed),
        pf_particles=1,
        temperature=1.0,
        channels=8,
        blocks=1,
    )

    assert packed[:9] == b"AHC061Q4\x03"
    assert int.from_bytes(packed[9:11], "little") == 15 + 9
    assert "cell_encoder1_weight" in rendered
    assert "expected_tensor_count = 15 + 9 * MODEL_BLOCKS" in rendered
    assert "cell_encoder1_weight = reader.take({MODEL_CHANNELS * 4, TYPED_NUM_PLANES})" in rendered
