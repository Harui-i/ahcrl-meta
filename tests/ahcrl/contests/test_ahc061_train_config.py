import json
from pathlib import Path

import pytest
import torch

from ahcrl.contests.ahc061.encoder import NUM_PLANES, PLANE_M, PLANE_U
from ahcrl.contests.ahc061.model import ActorCritic
from ahcrl.contests.ahc061.train_ppo import (
    MODEL_DTYPE,
    GroupedObservationNormalizer,
    GroupedRewardScaler,
    ImmediateRewardScaler,
    RunningObservationNormalizer,
    RunningRewardScaler,
    _advance_seed_start,
    _initial_next_seed_start,
    _normalized_entropy,
    _parameter_norm_stats,
    build_log_metrics,
    config_for_save,
    load_initial_model,
    load_training_state,
    parse_args,
    save_training_state,
)


def test_parse_args_loads_toml_config_and_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "\n".join(
            [
                "[train]",
                "num_envs = 8",
                "total_steps = 1024",
                "pf_particles = 32",
                "model_channels = 32",
                "model_blocks = 2",
                'model_block_type = "residual"',
                'device = "cpu"',
                "compile = false",
                'artifact_dir = "tmp/artifacts"',
                "checkpoint_interval_updates = 5",
                'symmetry_augmentation = "full_d4"',
                'reward_scale_mode = "running_std"',
                "reward_scale_epsilon = 0.001",
                "reward_scale_g_max = 7.0",
                'obs_norm_mode = "running_channel"',
                "obs_norm_epsilon = 0.0001",
                'normalization_grouping = "m_u"',
                "weight_projection = true",
                "wandb_enabled = true",
                'wandb_project = "test-project"',
                'wandb_name = "test-run"',
                'wandb_tags = ["ahc061", "test"]',
            ]
        )
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--num-envs",
            "4",
            "--model-blocks",
            "3",
            "--model-block-type",
            "spherical_depthwise_simba",
            "--symmetry-augmentation",
            "none",
            "--no-weight-projection",
            "--compile",
        ]
    )

    assert args.num_envs == 4
    assert args.total_steps == 1024
    assert args.pf_particles == 32
    assert args.model_channels == 32
    assert args.model_blocks == 3
    assert args.model_block_type == "spherical_depthwise_simba"
    assert not hasattr(args, "model_dtype")
    assert args.device == "cpu"
    assert args.compile is True
    assert args.artifact_dir == Path("tmp/artifacts")
    assert args.checkpoint_interval_updates == 5
    assert args.symmetry_augmentation == "none"
    assert args.reward_scale_mode == "running_std"
    assert args.reward_scale_epsilon == 0.001
    assert args.reward_scale_g_max == 7.0
    assert args.obs_norm_mode == "running_channel"
    assert args.obs_norm_epsilon == 0.0001
    assert args.normalization_grouping == "m_u"
    assert args.weight_projection is False
    assert args.wandb_enabled is True
    assert args.wandb_project == "test-project"
    assert args.wandb_name == "test-run"
    assert args.wandb_tags == ["ahc061", "test"]


def test_parse_args_compile_defaults_to_true_and_can_be_disabled() -> None:
    assert parse_args(["--device", "cpu"]).compile is True
    assert parse_args(["--device", "cpu", "--no-compile"]).compile is False
    assert parse_args(["--device", "cpu"]).pf_particles == 16
    assert parse_args(["--device", "cpu"]).weight_projection is False
    assert parse_args(["--device", "cpu"]).obs_norm_mode == "none"
    assert parse_args(["--device", "cpu"]).normalization_grouping == "none"
    assert parse_args(["--device", "cpu", "--weight-projection"]).weight_projection is True


def test_parse_args_symmetry_augmentation_defaults_to_none_and_loads_toml(
    tmp_path: Path,
) -> None:
    assert parse_args(["--device", "cpu"]).symmetry_augmentation == "none"

    config_path = tmp_path / "ppo.toml"
    config_path.write_text('[train]\nsymmetry_augmentation = "full_d4"\n')

    args = parse_args(["--config", str(config_path), "--device", "cpu"])

    assert args.symmetry_augmentation == "full_d4"


def test_parse_args_rejects_unknown_toml_key(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\nunknown = 1\n")

    with pytest.raises(ValueError, match="unknown config keys"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_unknown_symmetry_augmentation(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text('[train]\nsymmetry_augmentation = "d4"\n')

    with pytest.raises(ValueError, match="symmetry_augmentation"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_unknown_reward_scale_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text('[train]\nreward_scale_mode = "centered"\n')

    with pytest.raises(ValueError, match="reward_scale_mode"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_unknown_normalization_grouping(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text('[train]\nnormalization_grouping = "by_m"\n')

    with pytest.raises(ValueError, match="normalization_grouping"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_non_positive_reward_scale_epsilon(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\nreward_scale_epsilon = 0.0\n")

    with pytest.raises(ValueError, match="reward_scale_epsilon"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_non_positive_reward_scale_g_max(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\nreward_scale_g_max = 0.0\n")

    with pytest.raises(ValueError, match="reward_scale_g_max"):
        parse_args(["--config", str(config_path)])


def test_parse_args_rejects_non_positive_checkpoint_interval(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text("[train]\ncheckpoint_interval_updates = 0\n")

    with pytest.raises(ValueError, match="checkpoint_interval_updates"):
        parse_args(["--config", str(config_path)])


def test_parse_args_resume_loads_saved_config_and_only_allows_total_steps(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    config_path = run_dir / "config.json"
    args = parse_args(
        [
            "--total-steps",
            "128",
            "--num-envs",
            "4",
            "--device",
            "cpu",
            "--artifact-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    config_path.write_text(json.dumps(config_for_save(args), sort_keys=True) + "\n")

    resumed = parse_args(
        [
            "--resume-dir",
            str(run_dir),
            "--total-steps",
            "256",
        ]
    )

    assert resumed.resume_dir == run_dir
    assert resumed.total_steps == 256
    assert resumed.num_envs == 4
    assert resumed.device == "cpu"

    with pytest.raises(ValueError, match="resume only allows overriding total_steps"):
        parse_args(["--resume-dir", str(run_dir), "--lr", "0.001"])


def test_parse_args_rejects_resume_with_init_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}\n")

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_args(
            [
                "--resume-dir",
                str(run_dir),
                "--init-checkpoint",
                str(tmp_path / "checkpoint.pt"),
            ]
        )


def test_seed_blocks_advance_by_parallel_env_span(tmp_path: Path) -> None:
    config_path = tmp_path / "ppo.toml"
    config_path.write_text(
        "\n".join(
            [
                "[train]",
                "num_envs = 256",
                "seed_start = 1000",
                "seed_stride = 3",
            ]
        )
    )

    args = parse_args(["--config", str(config_path)])

    assert _initial_next_seed_start(args) == 1768
    assert _advance_seed_start(1768, args) == 2536


def test_build_log_metrics_contains_required_wandb_stats() -> None:
    rollout = {
        "scores": torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        "rewards": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        "scaled_rewards": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "reward_scale": 0.1,
        "reward_running_mean": 0.25,
        "reward_running_std": 0.1118,
        "reward_discounted_return_abs_max": 0.5,
        "reward_scale_min_denominator": 0.2,
        "dones": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        "values": torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
        "advantages": torch.tensor([[1.0, -1.0], [0.5, -0.5]]),
        "returns": torch.tensor([[1.5, 1.6], [1.7, 1.8]]),
        "masks": torch.ones(2, 2, 10),
        "m_values": torch.tensor([[2, 3], [2, 3]]),
        "u_values": torch.tensor([[1, 2], [1, 1]]),
    }
    stats = {
        "policy_loss": 0.01,
        "value_loss": 0.02,
        "entropy": 0.03,
        "normalized_entropy": 0.5,
        "approx_kl": 0.04,
        "clip_frac": 0.05,
        "grad_norm": 0.06,
        "weight_norm": 0.07,
        "linear_conv_weight_norm": 0.071,
        "norm_affine_norm": 0.072,
        "hyperspherical_scale_norm": 0.073,
        "param_rms": 0.074,
        "trunk_feature_norm_mean": 0.08,
        "trunk_feature_norm_std": 0.09,
        "trunk_feature_norm_max": 0.10,
    }

    metrics = build_log_metrics(
        update=3,
        global_step=128,
        elapsed=2.0,
        rollout=rollout,
        stats=stats,
        checkpoint_path=Path("checkpoint.pt"),
    )

    assert metrics["summary/cumulative_env_steps"] == 128
    assert metrics["summary/updates"] == 3
    assert metrics["summary/fps"] == 64.0
    assert metrics["train/mean_score"] == 4.0
    assert metrics["train/final_mean_score"] == 6.0
    assert metrics["train/final_min_score"] == 5.0
    assert metrics["train/final_max_score"] == 7.0
    assert metrics["train/final_mean_score_by_m/m_2"] == 5.0
    assert metrics["train/final_mean_score_by_m/m_3"] == 7.0
    assert metrics["train/final_mean_score_by_u/u_1"] == 6.0
    assert metrics["train/mean_reward"] == pytest.approx(0.25)
    assert metrics["train/mean_scaled_reward"] == pytest.approx(2.5)
    assert metrics["train/reward_scale"] == pytest.approx(0.1)
    assert metrics["train/reward_running_mean"] == pytest.approx(0.25)
    assert metrics["train/reward_running_std"] == pytest.approx(0.1118)
    assert metrics["train/reward_discounted_return_abs_max"] == pytest.approx(0.5)
    assert metrics["train/reward_scale_min_denominator"] == pytest.approx(0.2)
    assert metrics["train/explained_variance"] == pytest.approx(1.0)
    assert metrics["loss/policy"] == 0.01
    assert metrics["train/normalized_entropy"] == 0.5
    assert metrics["model/grad_norm"] == 0.06
    assert metrics["model/weight_norm"] == 0.07
    assert metrics["model/linear_conv_weight_norm"] == 0.071
    assert metrics["model/norm_affine_norm"] == 0.072
    assert metrics["model/hyperspherical_scale_norm"] == 0.073
    assert metrics["model/param_rms"] == 0.074
    assert metrics["model/trunk_feature_norm_mean"] == 0.08
    assert metrics["model/trunk_feature_norm_std"] == 0.09
    assert metrics["model/trunk_feature_norm_max"] == 0.10
    assert metrics["checkpoint/path"] == "checkpoint.pt"


def test_parameter_norm_stats_separates_spherical_scale_from_total_norm() -> None:
    convnext = ActorCritic(channels=8, blocks=1, block_type="convnext")
    spherical = ActorCritic(channels=8, blocks=1, block_type="simbav2_block")

    convnext_stats = _parameter_norm_stats(convnext)
    spherical_stats = _parameter_norm_stats(spherical)

    assert convnext_stats["hyperspherical_scale_norm"] == 0.0
    assert spherical_stats["hyperspherical_scale_norm"] > 0.0
    assert spherical_stats["weight_norm"] > spherical_stats["hyperspherical_scale_norm"]
    assert spherical_stats["param_rms"] > 0.0
    assert spherical_stats["linear_conv_weight_norm"] > 0.0
    assert convnext_stats["norm_affine_norm"] > 0.0


def test_save_and_load_training_state_round_trips_resume_state(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--device",
            "cpu",
            "--model-channels",
            "8",
            "--model-blocks",
            "1",
            "--artifact-dir",
            str(tmp_path),
        ]
    )
    args.run_dir = tmp_path / "run_1"
    args.run_dir.mkdir()
    model = ActorCritic(channels=8, blocks=1).to(dtype=MODEL_DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    reward_scaler = RunningRewardScaler(gamma=0.5, g_max=5.0, epsilon=1e-8)
    reward_scaler.update_and_scale(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    )
    obs_normalizer = RunningObservationNormalizer(channels=NUM_PLANES, epsilon=1e-8)
    obs_normalizer.update(torch.ones(2, NUM_PLANES, 10, 10))
    torch.manual_seed(123)

    checkpoint_path = save_training_state(
        args,
        model,
        optimizer,
        global_step=128,
        update=2,
        next_seed_start=64,
        wandb_run_id="abc123",
        reward_scaler=reward_scaler,
        obs_normalizer=obs_normalizer,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert {
        parameter.dtype
        for parameter in checkpoint["model"].values()
        if torch.is_floating_point(parameter)
    } == {MODEL_DTYPE}

    reloaded_model = ActorCritic(channels=8, blocks=1).to(dtype=MODEL_DTYPE)
    reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters(), lr=args.lr)
    state = load_training_state(
        args.run_dir,
        reloaded_model,
        reloaded_optimizer,
        torch.device("cpu"),
    )

    assert checkpoint_path == args.run_dir / "checkpoints" / "step_128.pt"
    assert (args.run_dir / "checkpoint_latest.pt").exists()
    assert state["global_step"] == 128
    assert state["update"] == 2
    assert state["next_seed_start"] == 64
    assert state["reward_scaler_state"] == reward_scaler.state_dict()
    assert state["obs_normalizer_state"]["count"] == obs_normalizer.state_dict()["count"]
    assert torch.equal(state["obs_normalizer_state"]["mean"], obs_normalizer.state_dict()["mean"])
    for left, right in zip(model.parameters(), reloaded_model.parameters(), strict=True):
        assert torch.equal(left, right)


def test_running_reward_scaler_uses_discounted_return_variance_and_g_max_floor() -> None:
    scaler = RunningRewardScaler(gamma=0.5, g_max=5.0, epsilon=1e-8)
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dones = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    scaled_rewards, stats = scaler.update_and_scale(rewards, dones)

    discounted_returns = torch.tensor([[1.0, 2.0], [3.5, 5.0]])
    expected_std = discounted_returns.std(unbiased=False)
    expected_scale = max(float(torch.sqrt(expected_std.square() + torch.tensor(1e-8)).item()), 1.0)
    assert stats["reward_running_mean"] == pytest.approx(float(discounted_returns.mean().item()))
    assert stats["reward_running_std"] == pytest.approx(float(expected_std.item()))
    assert stats["reward_discounted_return_abs_max"] == pytest.approx(5.0)
    assert stats["reward_scale_min_denominator"] == pytest.approx(1.0)
    assert stats["reward_scale"] == pytest.approx(expected_scale)
    assert torch.allclose(scaled_rewards, rewards / expected_scale)


def test_running_reward_scaler_state_round_trips() -> None:
    scaler = RunningRewardScaler(gamma=0.5, g_max=5.0, epsilon=1e-8)
    scaler.update_and_scale(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    )

    reloaded = RunningRewardScaler(gamma=0.9, g_max=1.0, epsilon=1.0)
    reloaded.load_state_dict(scaler.state_dict())

    assert reloaded.state_dict() == scaler.state_dict()


def test_running_reward_scaler_resets_state_after_terminal_step() -> None:
    scaler = RunningRewardScaler(gamma=0.5, g_max=5.0, epsilon=1e-8)

    scaler.update_and_scale(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[1.0, 0.0]]),
    )

    assert scaler.g_return is not None
    assert scaler.g_return.tolist() == pytest.approx([0.0, 2.0])


def test_running_reward_scaler_allows_zero_state_resize_after_terminal_rollout() -> None:
    scaler = RunningRewardScaler(gamma=0.5, g_max=5.0, epsilon=1e-8)
    scaler.update_and_scale(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[1.0, 1.0]]),
    )

    scaler.update_and_scale(
        torch.tensor([[3.0, 4.0, 5.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
    )

    assert scaler.g_return is not None
    assert scaler.g_return.tolist() == pytest.approx([3.0, 4.0, 5.0])


def test_grouped_reward_scaler_uses_separate_m_u_scales() -> None:
    scaler = GroupedRewardScaler(lambda: ImmediateRewardScaler(epsilon=1e-8))
    rewards = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
    m_values = torch.tensor([[2, 3], [2, 3]])
    u_values = torch.tensor([[1, 2], [1, 2]])

    scaled, stats = scaler.update_and_scale(rewards, m_values=m_values, u_values=u_values)

    assert scaled[:, 0].tolist() == pytest.approx([1.0, 3.0])
    assert scaled[:, 1].tolist() == pytest.approx([1.0, 3.0])
    assert stats["reward_scale_by_m_u/m_2_u_1"] == pytest.approx(1.0)
    assert stats["reward_scale_by_m_u/m_3_u_2"] == pytest.approx(10.0)


def test_immediate_reward_scaler_divides_by_running_std_without_centering() -> None:
    scaler = ImmediateRewardScaler(epsilon=1e-8)
    rewards = torch.tensor([1.0, 2.0, 3.0])

    scaled_rewards, stats = scaler.update_and_scale(rewards)

    expected_std = rewards.std(unbiased=False)
    assert stats["reward_running_mean"] == pytest.approx(2.0)
    assert stats["reward_running_std"] == pytest.approx(float(expected_std.item()))
    assert stats["reward_scale"] == pytest.approx(float(expected_std.item()))
    assert torch.allclose(scaled_rewards, rewards / expected_std)


def test_running_observation_normalizer_standardizes_per_channel() -> None:
    normalizer = RunningObservationNormalizer(channels=2, epsilon=1e-8)
    observations = torch.tensor(
        [
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        ]
    )

    normalized = normalizer.update_and_normalize(observations)

    assert normalizer.count == 4
    assert normalizer.mean.flatten().tolist() == pytest.approx([4.0, 2.0])
    assert torch.allclose(normalized[:, 0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized[:, 0].std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(normalized[:, 1], torch.zeros_like(normalized[:, 1]))


def test_running_observation_normalizer_state_round_trips() -> None:
    normalizer = RunningObservationNormalizer(channels=2, epsilon=1e-8)
    normalizer.update(torch.randn(3, 2, 4, 4))

    reloaded = RunningObservationNormalizer(channels=2, epsilon=1.0)
    reloaded.load_state_dict(normalizer.state_dict())

    assert reloaded.count == normalizer.count
    assert reloaded.epsilon == normalizer.epsilon
    assert torch.equal(reloaded.mean, normalizer.mean)
    assert torch.equal(reloaded.m2, normalizer.m2)


def test_grouped_observation_normalizer_keeps_m_u_planes_raw() -> None:
    normalizer = GroupedObservationNormalizer(channels=NUM_PLANES, epsilon=1e-8)
    observations = torch.zeros((2, NUM_PLANES, 2, 2))
    observations[0, 0] = 1.0
    observations[1, 0] = 10.0
    observations[0, PLANE_M] = 2.0 / 8.0
    observations[1, PLANE_M] = 3.0 / 8.0
    observations[0, PLANE_U] = 1.0 / 5.0
    observations[1, PLANE_U] = 2.0 / 5.0
    m_values = torch.tensor([2, 3])
    u_values = torch.tensor([1, 2])

    normalized = normalizer.update_and_normalize(
        observations,
        m_values=m_values,
        u_values=u_values,
    )

    assert torch.equal(normalized[:, PLANE_M], observations[:, PLANE_M])
    assert torch.equal(normalized[:, PLANE_U], observations[:, PLANE_U])
    assert normalizer.group_normalizers[(2, 1)].count == 4
    assert normalizer.group_normalizers[(3, 2)].count == 4


def test_load_initial_model_loads_only_model_state(tmp_path: Path) -> None:
    source = ActorCritic(channels=8, blocks=1)
    target = ActorCritic(channels=8, blocks=1).to(dtype=MODEL_DTYPE)
    for parameter in source.parameters():
        parameter.data.fill_(0.5)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": source.state_dict(), "global_step": 999}, checkpoint_path)

    load_initial_model(checkpoint_path, target, torch.device("cpu"))

    for left, right in zip(source.parameters(), target.parameters(), strict=True):
        assert right.dtype == MODEL_DTYPE
        assert torch.equal(left.to(dtype=MODEL_DTYPE), right)


def test_normalized_entropy_scales_by_valid_action_count() -> None:
    entropy = torch.tensor([torch.log(torch.tensor(4.0)), 0.0])
    mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, False, False, False, False],
        ]
    )

    assert _normalized_entropy(entropy, mask).item() == pytest.approx(0.5)
