import numpy as np

from ahcrl.contests.ahc061.encoder import NUM_PLANES, PLANE_LEGAL_MASK, PLANE_M, PLANE_U
from ahcrl.contests.ahc061.rust_vec_env import RustVecEnv


def test_rust_vec_env_reset_and_step() -> None:
    env = RustVecEnv(num_envs=1, seed_start=0, fixed_m=4, fixed_u=3, release=False)
    try:
        obs = env.obs
        assert obs["planes"].shape == (1, NUM_PLANES, 10, 10)
        assert obs["planes"].dtype == np.float32
        assert obs["mask"].shape == (1, 100)
        assert obs["reward"].shape == (1,)
        assert obs["done"].shape == (1,)
        assert obs["score"].shape == (1,)
        assert obs["planes"][0, PLANE_M, 0, 0] == np.float32(4 / 8)
        assert obs["planes"][0, PLANE_U, 0, 0] == np.float32(3 / 5)
        np.testing.assert_array_equal(
            obs["planes"][0, PLANE_LEGAL_MASK].reshape(100).astype(bool),
            obs["mask"][0],
        )
        assert obs["mask"][0].any()
        action = int(np.flatnonzero(obs["mask"][0])[0])
        step = env.step(np.array([action], dtype=np.int64))
        assert step.obs["planes"].shape == (1, NUM_PLANES, 10, 10)
        assert step.done.shape == (1,)
        assert np.isfinite(step.reward[0])
    finally:
        env.close()
