from pathlib import Path

import numpy as np

from ahcrl.envs import RustVecEnv, cargo_server_command

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "contests" / "ahc-063" / "rl-tools" / "Cargo.toml"
GOLDEN = (
    ROOT / "tests" / "ahcrl" / "contests" / "data" / "ahc063_seed0_typed_observation_golden.npz"
)


def create_env(num_envs: int = 1, seed_start: int = 0) -> RustVecEnv:
    return RustVecEnv(
        cargo_server_command(MANIFEST),
        num_envs,
        config={"fixed_n": 8, "fixed_m": 16, "fixed_c": 3, "max_steps_per_cell": 4},
        workers=0,
        seed_start=seed_start,
        seed_stride=1,
        cwd=ROOT,
    )


def test_typed_observation_schema() -> None:
    golden = np.load(GOLDEN)
    with create_env() as env:
        assert env.obs["board_food"].shape == (1, 16, 16)
        assert env.obs["board_food"].dtype == np.uint8
        assert env.obs["board_features"].shape == (1, 8, 16, 16)
        assert env.obs["slot_colors"].shape == (1, 192, 2)
        assert env.obs["slot_positions"].shape == (1, 192, 2)
        assert env.obs["global_features"].shape == (1, 10)
        assert env.obs["previous_action"].shape == (1, 1)
        assert env.obs["action_colors"].shape == (1, 4)
        assert env.obs["action_features"].shape == (1, 4, 7)
        assert env.obs["mask"].shape == (1, 4)
        assert np.all(env.obs["slot_positions"][0, 5:] == 255)
        for key in golden.files:
            np.testing.assert_array_equal(env.obs[key][0], golden[key][0])


def test_seed_reproducibility() -> None:
    with create_env(seed_start=0) as first, create_env(seed_start=0) as second:
        for key in first.obs:
            np.testing.assert_array_equal(first.obs[key], second.obs[key])


def test_action_preview_changes_after_step() -> None:
    golden = np.load(GOLDEN)
    with create_env() as env:
        before = {key: value.copy() for key, value in env.obs.items()}
        action = int(np.flatnonzero(before["mask"][0])[0])
        result = env.step(np.asarray([action], dtype=np.uint32))
        assert result.obs["previous_action"][0, 0] == action + 1
        assert not np.array_equal(result.obs["global_features"], before["global_features"])
        for key in golden.files:
            np.testing.assert_array_equal(result.obs[key][0], golden[key][1])
