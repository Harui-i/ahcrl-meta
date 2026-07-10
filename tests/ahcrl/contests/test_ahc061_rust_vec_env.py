import numpy as np

from ahcrl.contests.ahc061.encoder import (
    CRITIC_FEATURE_SHAPE,
    NUM_PLANES,
    PLANE_COMP_START,
    PLANE_DIST_CENTER,
    PLANE_LEGAL_MASK,
    PLANE_M,
    PLANE_NEXT_GREEDY_START,
    PLANE_ORACLE_PARAM_START,
    PLANE_PLAYER_AGG_START,
    PLANE_POS0_X_NORM,
    PLANE_POS0_Y_NORM,
    PLANE_REACH_START,
    PLANE_U,
    PLANE_X_NORM,
    PLANE_Y_NORM,
    PLAYER_AGG_COMP_LEVEL_SUM,
    PLAYER_AGG_COMP_LEVEL_VALUE_SUM,
    PLAYER_AGG_FEATURES,
    PLAYER_AGG_OWNER_LEVEL_SUM,
    PLAYER_AGG_OWNER_LEVEL_VALUE_SUM,
)
from ahcrl.contests.ahc061.rust_vec_env import RustVecEnv


def test_rust_vec_env_reset_and_step() -> None:
    env = RustVecEnv(num_envs=1, seed_start=0, fixed_m=4, fixed_u=3, release=False)
    try:
        obs = env.obs
        assert obs["planes"].shape == (1, NUM_PLANES, 10, 10)
        assert obs["planes"].dtype == np.float16
        assert obs["mask"].shape == (1, 100)
        assert obs["reward"].shape == (1,)
        assert obs["done"].shape == (1,)
        assert obs["score"].shape == (1,)
        assert obs["critic_posterior"].shape == (1, *CRITIC_FEATURE_SHAPE)
        assert obs["critic_oracle"].shape == (1, *CRITIC_FEATURE_SHAPE)
        assert obs["critic_posterior"].dtype == np.float16
        assert obs["critic_oracle"].dtype == np.float16
        np.testing.assert_allclose(obs["planes"][0, PLANE_M, 0, 0], 4 / 8, rtol=1e-3)
        np.testing.assert_allclose(obs["planes"][0, PLANE_U, 0, 0], 3 / 5, rtol=1e-3)
        np.testing.assert_array_equal(
            obs["planes"][0, PLANE_LEGAL_MASK].reshape(100).astype(bool),
            obs["mask"][0],
        )
        np.testing.assert_array_equal(
            obs["planes"][0, PLANE_REACH_START].reshape(100).astype(bool),
            obs["mask"][0],
        )
        np.testing.assert_array_equal(
            obs["planes"][0, PLANE_NEXT_GREEDY_START].reshape(100).astype(bool),
            obs["mask"][0],
        )
        player0_param_planes = obs["planes"][
            0, PLANE_ORACLE_PARAM_START : PLANE_ORACLE_PARAM_START + 5
        ]
        assert np.all(player0_param_planes == 0.0)
        ai1_param_planes = obs["planes"][
            0, PLANE_ORACLE_PARAM_START + 5 : PLANE_ORACLE_PARAM_START + 10
        ]
        assert np.all(np.isfinite(ai1_param_planes))
        assert np.all((0.3 <= ai1_param_planes[:4]) & (ai1_param_planes[:4] <= 1.0))
        assert np.all((0.1 <= ai1_param_planes[4]) & (ai1_param_planes[4] <= 0.5))
        expected_posterior = np.empty(CRITIC_FEATURE_SHAPE, dtype=np.float16)
        for player in range(CRITIC_FEATURE_SHAPE[0]):
            for param in range(5):
                expected_posterior[player, param] = obs["planes"][
                    0,
                    PLANE_ORACLE_PARAM_START + player * 5 + param,
                    0,
                    0,
                ]
        np.testing.assert_array_equal(obs["critic_posterior"][0], expected_posterior)
        assert np.all(obs["critic_oracle"][0, 0] == 0.0)
        oracle_weights = obs["critic_oracle"][0, 1, :4]
        assert np.all((0.3 <= oracle_weights) & (oracle_weights <= 1.0))
        assert 0.1 <= obs["critic_oracle"][0, 1, 4] <= 0.5
        ai1_next = obs["planes"][0, PLANE_NEXT_GREEDY_START + 1].reshape(100)
        assert np.all(np.isfinite(ai1_next))
        assert np.all(ai1_next >= 0.0)
        np.testing.assert_allclose(ai1_next.sum(), 1.0, rtol=1e-3, atol=1e-3)
        assert np.all(obs["planes"][0, PLANE_COMP_START] <= obs["planes"][0, PLANE_REACH_START])
        np.testing.assert_allclose(obs["planes"][0, PLANE_DIST_CENTER, 0, 0], 1.0, rtol=1e-3)
        np.testing.assert_allclose(obs["planes"][0, PLANE_DIST_CENTER, 4, 4], 1 / 9, rtol=1e-3)
        np.testing.assert_allclose(obs["planes"][0, PLANE_X_NORM, 0, 0], 0.0, atol=1e-3)
        np.testing.assert_allclose(obs["planes"][0, PLANE_X_NORM, 9, 0], 1.0, rtol=1e-3)
        np.testing.assert_allclose(obs["planes"][0, PLANE_Y_NORM, 0, 0], 0.0, atol=1e-3)
        np.testing.assert_allclose(obs["planes"][0, PLANE_Y_NORM, 0, 9], 1.0, rtol=1e-3)

        pos0_idx = int(np.flatnonzero(obs["planes"][0, 15].reshape(100))[0])
        pos0_x, pos0_y = divmod(pos0_idx, 10)
        np.testing.assert_allclose(
            obs["planes"][0, PLANE_POS0_X_NORM, 0, 0], pos0_x / 9, rtol=1e-3
        )
        np.testing.assert_allclose(
            obs["planes"][0, PLANE_POS0_Y_NORM, 0, 0], pos0_y / 9, rtol=1e-3
        )

        player0_agg = PLANE_PLAYER_AGG_START
        np.testing.assert_allclose(
            obs["planes"][0, player0_agg + PLAYER_AGG_OWNER_LEVEL_SUM, 0, 0],
            1 / 300,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            obs["planes"][0, player0_agg + PLAYER_AGG_COMP_LEVEL_SUM, 0, 0],
            obs["planes"][0, player0_agg + PLAYER_AGG_OWNER_LEVEL_SUM, 0, 0],
            rtol=1e-3,
        )
        assert obs["planes"][0, player0_agg + PLAYER_AGG_OWNER_LEVEL_VALUE_SUM, 0, 0] > 0
        np.testing.assert_allclose(
            obs["planes"][0, player0_agg + PLAYER_AGG_COMP_LEVEL_VALUE_SUM, 0, 0],
            obs["planes"][0, player0_agg + PLAYER_AGG_OWNER_LEVEL_VALUE_SUM, 0, 0],
            rtol=1e-3,
        )
        assert PLAYER_AGG_FEATURES == 4
        assert obs["mask"][0].any()
        action = int(np.flatnonzero(obs["mask"][0])[0])
        step = env.step(np.array([action], dtype=np.int64))
        assert step.obs["planes"].shape == (1, NUM_PLANES, 10, 10)
        assert step.done.shape == (1,)
        assert np.isfinite(step.reward[0])
    finally:
        env.close()
