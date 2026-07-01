import numpy as np

from ahcrl.contests.ahc061.rust_vec_env import RustVecEnv


def test_rust_vec_env_reset_and_step() -> None:
    env = RustVecEnv(num_envs=1, seed_start=0, fixed_m=4, fixed_u=3, release=False)
    try:
        obs = env.obs
        assert obs["mask"].shape == (1, 100)
        assert obs["enemy_params"].shape == (1, 40)
        assert np.all((0.0 <= obs["enemy_params"]) & (obs["enemy_params"] <= 1.0))
        assert obs["mask"][0].any()
        action = int(np.flatnonzero(obs["mask"][0])[0])
        step = env.step(np.array([action], dtype=np.int64))
        assert step.obs["turn"][0] == 1
        assert np.isfinite(step.reward[0])
    finally:
        env.close()
