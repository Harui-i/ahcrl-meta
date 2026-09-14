from pathlib import Path

import pytest
import torch
from torch import nn

from ahcrl.nn.modula import (
    BoundedDiagonalGeometry,
    BoundedRMSVectorGeometry,
    ModularLinear,
    ModularSequential,
)
from ahcrl.training.checkpoint import (
    TrainingProgress,
    load_initial_model,
    load_training_checkpoint,
    save_training_checkpoint,
)
from ahcrl.training.optimizer import (
    FP32MasterWeights,
    HybridModularOptimizer,
    build_optimizer,
)


def _config(*, project: bool = False) -> dict[str, object]:
    return {
        "optimizer": "modula",
        "lr": 0.01,
        "weight_decay": 0.01,
        "modula_momentum": 0.95,
        "modula_nesterov": True,
        "modula_diagnostics_interval": 100,
        "modula_project": project,
        "max_grad_norm": 0.5,
    }


def _model() -> nn.Module:
    return ModularSequential(ModularLinear(3, 4), nn.LayerNorm(4), ModularLinear(4, 2))


def _take_step(
    model: nn.Module, master: FP32MasterWeights, optimizer: HybridModularOptimizer
) -> None:
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 3)).square().mean().backward()
    master.copy_gradients_from_model()
    optimizer.step()
    master.copy_master_to_model()


def test_master_weights_retain_fp32_updates_after_model_sync() -> None:
    model = _model().to(dtype=torch.bfloat16)
    master = FP32MasterWeights(model)
    before = master.parameters[0].detach().clone()
    update = torch.full_like(before, 1e-5)

    with torch.no_grad():
        master.parameters[0].add_(update)
    master.copy_master_to_model()

    assert torch.equal(master.parameters[0], before + update)
    model_weight = dict(model.named_parameters())["0.weight"]
    assert not torch.equal(master.parameters[0], model_weight.float())


def test_hybrid_optimizer_updates_modular_and_bounded_parameters() -> None:
    model = _model()
    master = FP32MasterWeights(model)
    optimizer = build_optimizer(model=model, master_weights=master, config=_config())
    assert isinstance(optimizer, HybridModularOptimizer)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    _take_step(model, master, optimizer)

    assert optimizer.momentum_buffers
    assert optimizer.bounded_first_moments
    assert optimizer.bounded_second_moments
    assert "optimizer/linear/update_spectral_norm" in optimizer.last_metrics
    assert "optimizer/bounded_rms_vector/update_natural_norm" in optimizer.last_metrics
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
    )


def test_bounded_parameter_uses_global_lr_and_allocated_rms_budget() -> None:
    model = ModularSequential(ModularLinear(3, 2))
    master = FP32MasterWeights(model)
    config = _config()
    config["lr"] = 0.2
    config["weight_decay"] = 0.0
    optimizer = build_optimizer(model=model, master_weights=master, config=config)
    assert isinstance(optimizer, HybridModularOptimizer)
    bias_spec = next(spec for spec in optimizer.specs if spec.name == "0.bias")
    bias = dict(master.named_parameters())["0.bias"]
    before = bias.detach().clone()
    for parameter in master.parameters:
        parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    update_rms = float((bias - before).square().mean().sqrt().item())
    assert update_rms == pytest.approx(0.2 * bias_spec.target_norm, rel=1e-5)


def test_bounded_parameters_remain_within_constraints_under_repeated_gradients() -> None:
    model = _model()
    master = FP32MasterWeights(model)
    config = _config()
    config["lr"] = 0.5
    optimizer = build_optimizer(model=model, master_weights=master, config=config)
    assert isinstance(optimizer, HybridModularOptimizer)
    specs = {spec.name: spec for spec in optimizer.specs}

    for _ in range(2000):
        for name, parameter in master.named_parameters():
            parameter.grad = torch.ones_like(parameter) if specs[name].role == "bounded" else None
        optimizer.step()

    parameters = dict(master.named_parameters())
    for name, parameter in parameters.items():
        geometry = specs[name].geometry
        assert geometry is not None
        if isinstance(geometry, BoundedRMSVectorGeometry):
            assert float(parameter.square().mean().sqrt().item()) <= geometry.radius + 1e-6
        elif isinstance(geometry, BoundedDiagonalGeometry):
            assert float(parameter.min().item()) >= geometry.center - geometry.radius
            assert float(parameter.max().item()) <= geometry.center + geometry.radius


def test_modular_optimizer_checkpoint_resume_reproduces_next_step(tmp_path: Path) -> None:
    model = _model()
    master = FP32MasterWeights(model)
    optimizer = build_optimizer(model=model, master_weights=master, config=_config())
    assert isinstance(optimizer, HybridModularOptimizer)
    _take_step(model, master, optimizer)
    path = save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        config={"optimizer": "modula"},
        progress=TrainingProgress(global_step=1, update=1),
        extras={"master_weights": master.state_dict()},
    )

    resumed_model = _model()
    resumed_master = FP32MasterWeights(resumed_model)
    resumed_optimizer = build_optimizer(
        model=resumed_model, master_weights=resumed_master, config=_config()
    )
    assert isinstance(resumed_optimizer, HybridModularOptimizer)
    loaded = load_training_checkpoint(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        device=torch.device("cpu"),
    )
    resumed_master.load_state_dict(loaded.extras["master_weights"])
    resumed_master.copy_master_to_model()

    _take_step(model, master, optimizer)
    _take_step(resumed_model, resumed_master, resumed_optimizer)

    for expected, actual in zip(model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_checkpoint_rejects_optimizer_type_mismatch(tmp_path: Path) -> None:
    model = _model()
    adam = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=adam,
        config={"optimizer": "adamw"},
        progress=TrainingProgress(global_step=0, update=0),
        extras={},
    )
    resumed_model = _model()
    resumed_master = FP32MasterWeights(resumed_model)
    modula = build_optimizer(model=resumed_model, master_weights=resumed_master, config=_config())
    assert isinstance(modula, HybridModularOptimizer)

    try:
        load_training_checkpoint(
            path, model=resumed_model, optimizer=modula, device=torch.device("cpu")
        )
    except ValueError as error:
        assert "optimizer mismatch" in str(error)
    else:
        raise AssertionError("optimizer mismatch must be rejected")


def test_modular_optimizer_rejects_v3_full_resume_but_allows_model_only(
    tmp_path: Path,
) -> None:
    model = _model()
    master = FP32MasterWeights(model)
    optimizer = build_optimizer(model=model, master_weights=master, config=_config())
    assert isinstance(optimizer, HybridModularOptimizer)
    path = save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        config={"optimizer": "modula"},
        progress=TrainingProgress(global_step=0, update=0),
        extras={"master_weights": master.state_dict()},
    )
    payload = torch.load(path, weights_only=False)
    payload["optimizer"]["format_version"] = 3
    torch.save(payload, path)

    initialized_model = _model()
    load_initial_model(path, model=initialized_model, device=torch.device("cpu"))
    for expected, actual in zip(model.parameters(), initialized_model.parameters(), strict=True):
        assert torch.equal(expected, actual)

    with pytest.raises(ValueError, match="unsupported Modula optimizer state format"):
        load_training_checkpoint(
            path,
            model=initialized_model,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )


def test_modular_optimizer_rejects_nonfinite_gradient() -> None:
    model = _model()
    master = FP32MasterWeights(model)
    optimizer = build_optimizer(model=model, master_weights=master, config=_config())
    assert isinstance(optimizer, HybridModularOptimizer)
    master.parameters[0].grad = torch.full_like(master.parameters[0], float("inf"))

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        optimizer.validate_gradients()
