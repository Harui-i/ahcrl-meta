from pathlib import Path

import numpy as np

from ahcrl.contests.ahc061.encoder import CRITIC_FEATURE_SHAPE, NUM_PLANES
from ahcrl.envs import RustVecEnv, cargo_server_command

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "contests" / "ahc-061" / "rl-tools" / "Cargo.toml"


def create_env(num_envs: int = 1, *, seed_start: int = 0) -> RustVecEnv:
    return RustVecEnv(
        cargo_server_command(MANIFEST, release=False),
        num_envs,
        config={"fixed_m": 4, "fixed_u": 3, "pf_particles": 2},
        seed_start=seed_start,
        cwd=ROOT,
    )


def test_schema_step_and_partial_reset() -> None:
    with create_env(2) as env:
        initial = env.obs["planes"].copy()
        assert initial.shape == (2, NUM_PLANES, 10, 10)
        assert initial.dtype == np.float32
        assert env.obs["mask"].shape == (2, 100)
        assert env.obs["mask"].dtype == np.bool_
        assert env.obs["critic_oracle"].shape == (2, *CRITIC_FEATURE_SHAPE)
        assert env.obs["critic_oracle"].dtype == np.float32
        actions = np.argmax(env.obs["mask"], axis=1).astype(np.uint32)
        stepped = env.step(actions).obs["planes"].copy()
        reset = env.reset_done(np.asarray([False, True]), seed_start=100)["planes"].copy()
        np.testing.assert_array_equal(reset[0], stepped[0])
        with create_env(seed_start=101) as expected:
            np.testing.assert_array_equal(reset[1], expected.obs["planes"][0])
