import json
import math
from pathlib import Path

import pytest
import torch

import ahcrl.contests.ahc063.train_ppo as train_ppo
from ahcrl.contests.ahc063.train_ppo import (
    RUNTIME_KEYS,
    FP32MasterWeights,
    ProximalPolicyEWMA,
    RunningRewardScaler,
    _observation_normalizer,
    create_model,
    evaluate_policy,
    parse_args,
    update_model,
)
from ahcrl.training import (
    TrainingProgress,
    config_for_save,
    load_latest_training_checkpoint,
    save_training_checkpoint,
)
from ahcrl.training.ppo import policy_surrogate


def test_parse_args_loads_wandb_settings_and_aliases(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "\n".join(
            [
                "[training]",
                'device = "cpu"',
                "",
                "[wandb]",
                'entity = "entity"',
                'mode = "offline"',
                'tags = ["base"]',
                "",
                "[evaluation]",
                "enabled = true",
                "seed_num = 3",
                "temperature = 0.5",
            ]
        )
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--wandb",
            "--wandb-name",
            "run-name",
            "--wandb-tag",
            "cli-a",
            "--wandb-tag",
            "cli-b",
        ]
    )

    assert args.wandb_enabled is True
    assert args.wandb_entity == "entity"
    assert args.wandb_mode == "offline"
    assert args.wandb_name == "run-name"
    assert args.wandb_tags == ["cli-a", "cli-b"]
    assert args.device == "cpu"
    assert args.eval_enabled is True
    assert args.eval_seed_num == 3
    assert args.eval_temperature == 0.5


def test_parse_args_resume_rejects_disallowed_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = parse_args(["--device", "cpu", "--artifact-dir", str(tmp_path / "artifacts")])
    (run_dir / "config.json").write_text(
        json.dumps(config_for_save(vars(args), runtime_keys=RUNTIME_KEYS))
    )

    resumed = parse_args(["--resume-dir", str(run_dir), "--total-steps", "99"])
    assert resumed.total_steps == 99
    assert resumed.device == "cpu"
    resumed_eval = parse_args(
        ["--resume-dir", str(run_dir), "--eval-enabled", "--eval-seed-num", "3"]
    )
    assert resumed_eval.eval_enabled is True
    assert resumed_eval.eval_seed_num == 3
    with pytest.raises(ValueError, match="resume only allows approved training overrides"):
        parse_args(["--resume-dir", str(run_dir), "--gamma", "0.9"])


def test_parse_args_rejects_invalid_evaluation_values() -> None:
    with pytest.raises(ValueError, match="eval_temperature"):
        parse_args(["--eval-temperature", "-0.1"])
    with pytest.raises(ValueError, match="eval_seed_num"):
        parse_args(["--eval-seed-num", "0"])
    with pytest.raises(ValueError, match="max_steps_per_cell"):
        parse_args(["--max-steps-per-cell", "0"])
    with pytest.raises(ValueError, match="eval_max_steps_per_cell"):
        parse_args(["--eval-max-steps-per-cell", "0"])
    with pytest.raises(ValueError, match="env_workers"):
        parse_args(["--env-workers", "-1"])


def test_parse_args_rejects_removed_fixed_step_options(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy.toml"
    config_path.write_text("[contest]\nmax_episode_steps = 256\n")

    with pytest.raises(ValueError, match="unknown config keys: max_episode_steps"):
        parse_args(["--config", str(config_path)])


def test_parse_args_supports_proximal_ewma_and_rejects_invalid_com() -> None:
    args = parse_args(["--proximal-ewma", "--proximal-ewma-com", "8"])

    assert args.proximal_ewma is True
    assert args.proximal_ewma_com == 8.0
    with pytest.raises(ValueError, match="proximal_ewma_com"):
        parse_args(["--proximal-ewma-com", "0"])


def test_parse_args_supports_policy_warmup_and_rejects_invalid_values() -> None:
    args = parse_args(
        [
            "--policy-unfreeze-explained-variance",
            "0.8",
            "--policy-freeze-scope",
            "policy_objective",
            "--policy-warmup-epochs-multiplier",
            "3",
        ]
    )

    assert args.policy_unfreeze_explained_variance == 0.8
    assert args.policy_freeze_scope == "policy_objective"
    assert args.policy_warmup_epochs_multiplier == 3
    with pytest.raises(ValueError, match="policy_unfreeze_explained_variance"):
        parse_args(["--policy-unfreeze-explained-variance", "1"])
    with pytest.raises(ValueError, match="policy_freeze_scope"):
        parse_args(["--policy-freeze-scope", "invalid"])
    with pytest.raises(ValueError, match="policy_warmup_epochs_multiplier"):
        parse_args(["--policy-warmup-epochs-multiplier", "0"])


def test_evaluate_policy_rolls_out_fixed_seeds_reproducibly_without_updating_obs_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_command = train_ppo.cargo_server_command
    monkeypatch.setattr(
        train_ppo,
        "cargo_server_command",
        lambda manifest: original_command(manifest, release=False),
    )
    args = parse_args(
        [
            "--device",
            "cpu",
            "--model-channels",
            "4",
            "--model-blocks",
            "1",
            "--num-envs",
            "2",
            "--eval-seed-num",
            "3",
            "--eval-temperature",
            "1.0",
            "--eval-fixed-n",
            "8",
            "--eval-fixed-m",
            "16",
            "--eval-fixed-c",
            "3",
            "--eval-max-steps-per-cell",
            "1",
        ]
    )
    model = create_model(args, torch.device("cpu"))
    normalizer = _observation_normalizer(model)
    assert normalizer is not None
    count_before = normalizer.count.clone()

    metrics, first = evaluate_policy(model, args, torch.device("cpu"))
    _, second = evaluate_policy(model, args, torch.device("cpu"))

    assert metrics["eval/seed_count"] == 3
    assert first.seed_scores == second.seed_scores
    assert len(first.seed_visualizer_data) == 3
    assert first.seed_visualizer_data[0][2]
    assert torch.equal(normalizer.count, count_before)
    assert model.training


def test_ahc063_checkpoint_round_trips_reward_scaler_and_master_weights(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--model-channels",
            "4",
            "--model-blocks",
            "1",
        ]
    )
    model = create_model(args, torch.device("cpu"))
    master_weights = FP32MasterWeights(model)
    proximal = ProximalPolicyEWMA(model, master_weights.parameters, center_of_mass=8.0)
    proximal.update(master_weights.parameters)
    optimizer = torch.optim.AdamW(master_weights.parameters, lr=args.lr)
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
            "reward_scaler": scaler.state_dict(),
            "master_weights": master_weights.state_dict(),
            "proximal_policy_ewma": proximal.state_dict(),
        },
    )

    reloaded_model = create_model(args, torch.device("cpu"))
    reloaded_master_weights = FP32MasterWeights(reloaded_model)
    reloaded_proximal = ProximalPolicyEWMA(
        reloaded_model, reloaded_master_weights.parameters, center_of_mass=8.0
    )
    reloaded_optimizer = torch.optim.AdamW(reloaded_master_weights.parameters, lr=args.lr)
    loaded = load_latest_training_checkpoint(
        tmp_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
    )
    reloaded_master_weights.load_state_dict(loaded.extras["master_weights"])
    reloaded_proximal.load_state_dict(loaded.extras["proximal_policy_ewma"])
    reloaded_scaler = RunningRewardScaler()
    reloaded_scaler.load_state_dict(loaded.extras["reward_scaler"])

    assert loaded.progress == TrainingProgress(global_step=8, update=1)
    assert reloaded_scaler.state_dict() == scaler.state_dict()
    for left, right in zip(
        master_weights.parameters, reloaded_master_weights.parameters, strict=True
    ):
        assert torch.equal(left, right)
    assert reloaded_proximal.total_weight == pytest.approx(proximal.total_weight)
    for left, right in zip(
        proximal.master_parameters, reloaded_proximal.master_parameters, strict=True
    ):
        assert torch.equal(left, right)


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
    source = [nn_parameter for nn_parameter in model.parameters() if nn_parameter.requires_grad]
    for parameter in source:
        parameter.data.zero_()
    ewma = ProximalPolicyEWMA(model, source, center_of_mass=1.0)
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


def test_decoupled_policy_surrogate_matches_ppo_when_proximal_is_behavior() -> None:
    new_logprob = torch.tensor([-0.2, -1.4])
    behavior_logprob = torch.tensor([-0.3, -1.0])
    advantages = torch.tensor([1.5, -0.5])
    ordinary = policy_surrogate(
        new_logprob=new_logprob,
        behavior_logprob=behavior_logprob,
        advantages=advantages,
        clip=0.2,
        proximal_logprob=None,
    )
    decoupled = policy_surrogate(
        new_logprob=new_logprob,
        behavior_logprob=behavior_logprob,
        advantages=advantages,
        clip=0.2,
        proximal_logprob=behavior_logprob,
    )

    assert decoupled.loss is not None and ordinary.loss is not None
    assert decoupled.clipping_ratio is not None and ordinary.clipping_ratio is not None
    assert decoupled.proximal_behavior_ratio is not None
    assert torch.equal(decoupled.loss, ordinary.loss)
    assert torch.equal(decoupled.clipping_ratio, ordinary.clipping_ratio)
    assert torch.equal(
        decoupled.proximal_behavior_ratio,
        torch.ones_like(behavior_logprob),
    )


def test_decoupled_policy_surrogate_uses_proximal_clip_and_behavior_weight() -> None:
    result = policy_surrogate(
        new_logprob=torch.tensor([math.log(0.6)]),
        behavior_logprob=torch.tensor([math.log(0.25)]),
        advantages=torch.tensor([2.0]),
        clip=0.1,
        proximal_logprob=torch.tensor([math.log(0.5)]),
    )

    assert result.loss is not None
    assert result.clipping_ratio is not None
    assert result.proximal_behavior_ratio is not None
    assert float(result.loss.item()) == pytest.approx(-4.4)
    assert float(result.clipping_ratio.item()) == pytest.approx(1.2)
    assert float(result.proximal_behavior_ratio.item()) == pytest.approx(2.0)


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
    observations = torch.randn(1, 2, model.NUM_PLANES, 8, 8)
    masks = torch.ones(1, 2, model.ACTION_COUNT, dtype=torch.bool)
    with torch.inference_mode():
        logits, values = model(observations.flatten(0, 1))
        distribution = torch.distributions.Categorical(logits=logits.float())
        actions = torch.tensor([0, 1])
        behavior_logprobs = distribution.log_prob(actions)
    rollout = {
        "obs": observations,
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


def test_update_model_warmup_updates_only_value_head() -> None:
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
            "0.01",
        ]
    )
    model = create_model(args, torch.device("cpu"))
    master_weights = FP32MasterWeights(model)
    optimizer = torch.optim.AdamW(master_weights.parameters, lr=args.lr)
    observations = torch.randn(1, 2, model.NUM_PLANES, 8, 8)
    masks = torch.ones(1, 2, model.ACTION_COUNT, dtype=torch.bool)
    with torch.inference_mode():
        logits, values = model(observations.flatten(0, 1))
        distribution = torch.distributions.Categorical(logits=logits.float())
        actions = torch.tensor([0, 1])
    rollout = {
        "obs": observations,
        "actions": actions.reshape(1, 2),
        "logprobs": distribution.log_prob(actions).reshape(1, 2),
        "advantages": torch.tensor([[1.0, -1.0]]),
        "returns": values.detach().reshape(1, 2) + 1.0,
        "masks": masks,
    }
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    stats = update_model(
        model,
        model,
        optimizer,
        rollout,
        args,
        torch.device("cpu"),
        master_weights,
        policy_updates_enabled=False,
    )

    assert stats["policy_updates_enabled"] == 0.0
    assert stats["training_epochs"] == 1.0
    assert stats["weighted_policy_loss"] == 0.0
    assert stats["entropy_loss"] == 0.0
    for name, parameter in model.named_parameters():
        if name.startswith("value."):
            continue
        assert torch.equal(parameter, before[name]), name
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
        if name.startswith("value.")
    )
