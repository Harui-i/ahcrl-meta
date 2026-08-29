from pathlib import Path

import numpy as np
import pytest

from ahcrl.envs import RustVecEnv, cargo_server_command

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "contests" / "ahc-063" / "rl-tools" / "Cargo.toml"
GOLDEN = Path(__file__).with_name("data") / "ahc063_seed0_observation_golden.npz"


def create_env(
    num_envs: int = 1,
    *,
    seed_start: int = 0,
    seed_stride: int = 1,
) -> RustVecEnv:
    return RustVecEnv(
        cargo_server_command(MANIFEST, release=False),
        num_envs,
        config={
            "fixed_n": None,
            "fixed_m": None,
            "fixed_c": None,
            "max_steps": 100_000,
        },
        seed_start=seed_start,
        seed_stride=seed_stride,
        cwd=ROOT,
    )


def test_schema_and_observations_match_numpy_golden() -> None:
    golden = np.load(GOLDEN)
    with create_env() as env:
        assert env.obs["planes"].shape == (1, 43, 16, 16)
        assert env.obs["planes"].dtype == np.float32
        assert env.obs["mask"].shape == (1, 4)
        assert env.obs["mask"].dtype == np.bool_
        np.testing.assert_array_equal(env.obs["planes"][0], golden["planes"][0])
        np.testing.assert_array_equal(env.obs["mask"][0], golden["mask"][0])

        for frame, action in enumerate(golden["actions"], start=1):
            result = env.step(np.asarray([action], dtype=np.uint32))
            np.testing.assert_array_equal(result.obs["planes"][0], golden["planes"][frame])
            np.testing.assert_array_equal(result.obs["mask"][0], golden["mask"][frame])
            assert result.reward[0] == golden["reward"][frame]
            assert result.done[0] == golden["done"][frame]
            assert result.score[0] == golden["score"][frame]
            assert result.metrics["prefix_match_ratio"][0] == golden["prefix_match_ratio"][frame]


def test_seed_reproducibility_and_partial_reset() -> None:
    with create_env(3, seed_start=0, seed_stride=1) as env:
        initial = env.obs["planes"].copy()
        first_legal = np.argmax(env.obs["mask"], axis=1).astype(np.uint32)
        stepped = env.step(first_legal).obs["planes"].copy()
        assert not np.array_equal(stepped, initial)

        reset = env.reset_done(
            np.asarray([False, True, False]),
            seed_start=100,
            seed_stride=7,
        )["planes"].copy()
        np.testing.assert_array_equal(reset[0], stepped[0])
        np.testing.assert_array_equal(reset[2], stepped[2])

        with create_env(seed_start=107) as expected:
            np.testing.assert_array_equal(reset[1], expected.obs["planes"][0])

        repeated = env.reset(seed_start=0, seed_stride=1)["planes"].copy()
        np.testing.assert_array_equal(repeated, initial)


def test_invalid_batch_is_rejected_without_mutation() -> None:
    golden = np.load(GOLDEN)
    with create_env() as env:
        with pytest.raises(RuntimeError, match="invalid action 0"):
            env.step(np.asarray([0], dtype=np.uint32))

        result = env.step(np.asarray([1], dtype=np.uint32))
        assert result.score[0] == golden["score"][1]
        np.testing.assert_array_equal(result.obs["planes"][0], golden["planes"][1])


def test_step_mask_preserves_inactive_environment_without_validating_its_action() -> None:
    with create_env() as env:
        before = env.obs["planes"].copy()
        result = env.step_mask(np.asarray([False]), np.asarray([999], dtype=np.int64))
        np.testing.assert_array_equal(result.obs["planes"], before)
        assert result.done.tolist() == [False]


def test_client_validation_and_process_failure() -> None:
    env = create_env()
    with pytest.raises(TypeError, match="integer dtype"):
        env.step(np.asarray([1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        env.step(np.asarray([1, 1], dtype=np.uint32))

    env._proc.terminate()
    env._proc.wait(timeout=5)
    with pytest.raises(RuntimeError, match="exit code"):
        env.step(np.asarray([1], dtype=np.uint32))
    env.close()
    env.close()
