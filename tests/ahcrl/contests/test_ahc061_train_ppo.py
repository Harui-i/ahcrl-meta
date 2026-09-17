from pathlib import Path

import pytest
import torch

from ahcrl.contests.ahc061.encoder import CRITIC_FEATURE_SHAPE, NUM_PLANES
from ahcrl.contests.ahc061.train_ppo import (
    RUNTIME_KEYS,
    FP32MasterWeights,
    ProximalPolicyEWMA,
    RunningRewardScaler,
    create_model,
    parse_args,
    update_model,
)
from ahcrl.training import (
    TrainingProgress,
    config_for_save,
    load_latest_training_checkpoint,
    save_training_checkpoint,
)


def test_parse_args_uses_shared_toml_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "[training]\nnum_envs = 4\ndevice = 'cpu'\n\n"
        "[model]\nchannels = 8\nblocks = 1\n\n"
        "[contest]\npf_particles = 2\n"
    )
    args = parse_args(["--config", str(config_path), "--fixed-m", "4", "--fixed-u", "3"])
    assert args.num_envs == 4
    assert args.env_workers == 0
    assert args.model_channels == 8
    assert args.model_blocks == 1
    assert args.pf_particles == 2
    assert args.fixed_m == 4
    assert args.fixed_u == 3


def test_parse_args_accepts_and_rejects_env_workers() -> None:
    assert parse_args(["--env-workers", "3"]).env_workers == 3
    with pytest.raises(ValueError, match="env_workers"):
        parse_args(["--env-workers", "-1"])


def test_parse_args_supports_proximal_ewma_and_rejects_invalid_com() -> None:
    args = parse_args(["--proximal-ewma", "--proximal-ewma-com", "8"])

    assert args.proximal_ewma is True
    assert args.proximal_ewma_com == 8.0
    with pytest.raises(ValueError, match="proximal_ewma_com"):
        parse_args(["--proximal-ewma-com", "0"])


def test_proximal_policy_ewma_uses_bias_corrected_fp32_weights_and_round_trips() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--model-channels",
            "4",
            "--model-blocks",
            "1",
            "--no-obs-norm",
        ]
    )
    model = create_model(args, torch.device("cpu"))
    source = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in source:
        parameter.data.zero_()
    ewma = ProximalPolicyEWMA(model, source, center_of_mass=1.0)
    assert all(not name.startswith("value.") for name, _ in ewma.model.named_parameters())
    assert sum(parameter.numel() for parameter in ewma.model.parameters()) == sum(
        parameter.numel() for parameter in model.policy.parameters()
    )
    for parameter in source:
        parameter.data.fill_(2.0)

    ewma.update(source)

    assert ewma.decay == pytest.approx(0.5)
    assert ewma.effective_center_of_mass == pytest.approx(1.0 / 3.0)
    for parameter in ewma.master_parameters:
        assert torch.allclose(parameter, torch.full_like(parameter, 4.0 / 3.0))

    restored = ProximalPolicyEWMA(model, source, center_of_mass=1.0)
    restored.load_state_dict(ewma.state_dict())
    assert restored.total_weight == pytest.approx(ewma.total_weight)
    assert restored.weighted_age == pytest.approx(ewma.weighted_age)
    for left, right in zip(restored.master_parameters, ewma.master_parameters, strict=True):
        assert torch.equal(left, right)


def test_checkpoint_round_trips_master_weights_and_reward_scaler(tmp_path: Path) -> None:
    args = parse_args(["--device", "cpu", "--model-channels", "8", "--model-blocks", "1"])
    model = create_model(args, torch.device("cpu"))
    master = FP32MasterWeights(model)
    proximal = ProximalPolicyEWMA(model, master.parameters, center_of_mass=8.0)
    proximal.update(master.parameters)
    optimizer = torch.optim.AdamW(master.parameters, lr=args.lr)
    scaler = RunningRewardScaler()
    scaler.scale(torch.tensor([[1.0, 2.0]]))
    config = config_for_save(vars(args), runtime_keys=RUNTIME_KEYS)
    save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        config=config,
        progress=TrainingProgress(global_step=8, update=1),
        extras={
            "master_weights": master.state_dict(),
            "proximal_policy_ewma": proximal.state_dict(),
            "reward_scaler": scaler.state_dict(),
        },
    )
    loaded_model = create_model(args, torch.device("cpu"))
    loaded_master = FP32MasterWeights(loaded_model)
    loaded_proximal = ProximalPolicyEWMA(loaded_model, loaded_master.parameters, center_of_mass=8.0)
    loaded_optimizer = torch.optim.AdamW(loaded_master.parameters, lr=args.lr)
    loaded = load_latest_training_checkpoint(
        tmp_path, model=loaded_model, optimizer=loaded_optimizer, device=torch.device("cpu")
    )
    loaded_master.load_state_dict(loaded.extras["master_weights"])
    loaded_proximal.load_state_dict(loaded.extras["proximal_policy_ewma"])
    assert loaded.progress == TrainingProgress(global_step=8, update=1)
    assert loaded.config == config
    assert loaded_proximal.total_weight == pytest.approx(proximal.total_weight)


def test_update_model_records_proximal_diagnostics_for_identical_policies() -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--model-channels",
            "4",
            "--model-blocks",
            "1",
            "--no-obs-norm",
            "--epochs",
            "1",
            "--minibatch-size",
            "2",
            "--lr",
            "0",
        ]
    )
    model = create_model(args, torch.device("cpu"))
    master_weights = FP32MasterWeights(model)
    optimizer = torch.optim.AdamW(master_weights.parameters, lr=0.0)
    proximal = ProximalPolicyEWMA(model, master_weights.parameters, center_of_mass=2.0)
    observations = torch.randn(1, 2, NUM_PLANES, 8, 8)
    critic_features = torch.zeros(1, 2, *CRITIC_FEATURE_SHAPE)
    masks = torch.ones(1, 2, 64, dtype=torch.bool)
    with torch.inference_mode():
        logits, values = model(
            observations.flatten(0, 1), critic_features.flatten(0, 1), normalize_input=False
        )
        distribution = torch.distributions.Categorical(logits=logits.float())
        actions = torch.tensor([0, 1])
        behavior_logprobs = distribution.log_prob(actions)
    rollout = {
        "obs": observations,
        "critic_features": critic_features,
        "actions": actions.reshape(1, 2),
        "logprobs": behavior_logprobs.reshape(1, 2),
        "advantages": torch.tensor([[1.0, -1.0]]),
        "returns": values.detach().reshape(1, 2),
        "masks": masks,
    }

    stats = update_model(
        model,
        model,
        optimizer,
        rollout,
        args,
        torch.device("cpu"),
        master_weights,
        proximal,
        proximal.model,
    )

    assert stats["clip_frac"] == pytest.approx(0.0)
    assert stats["current_behavior_approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert stats["behavior_proximal_approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert stats["proximal_current_kl"] == pytest.approx(0.0, abs=1e-7)
    assert stats["proximal_behavior_ratio_mean"] == pytest.approx(1.0)
    assert stats["proximal_behavior_ratio_std"] == pytest.approx(0.0)
    assert stats["proximal_behavior_ratio_ess_fraction"] == pytest.approx(1.0)
