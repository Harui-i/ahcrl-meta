import numpy as np

from ahcrl.contests.ahc061.encoder import (
    NUM_PLANES,
    PLANE_LEGAL_MASK,
    PLANE_M,
    PLANE_PLAYER_SCORE_START,
    PLANE_ORACLE_PARAM_START,
    PLANE_SCORE_DIFF,
    PLANE_SCORE_RATIO,
    PLANE_U,
    ORACLE_PARAMS_PER_PLAYER,
    encode_batch,
)


def test_encode_batch_shape() -> None:
    obs = {
        "m": np.array([2], dtype=np.int64),
        "u": np.array([3], dtype=np.int64),
        "turn": np.array([0], dtype=np.int64),
        "values": np.ones((1, 100), dtype=np.float32),
        "owner": np.full((1, 100), -1, dtype=np.int64),
        "level": np.zeros((1, 100), dtype=np.int64),
        "pos": np.full((1, 16), -1, dtype=np.int64),
        "enemy_params": np.zeros((1, 40), dtype=np.float32),
        "mask": np.zeros((1, 100), dtype=bool),
    }
    obs["owner"][0, 0] = 0
    obs["owner"][0, 11] = 1
    obs["level"][0, 0] = 1
    obs["level"][0, 11] = 1
    obs["pos"][0, 0:4] = np.array([0, 0, 1, 1])
    obs["enemy_params"][0, 5:10] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    obs["mask"][0, [0, 1]] = True

    encoded = encode_batch(obs)

    assert encoded.shape == (1, NUM_PLANES, 10, 10)
    assert encoded.dtype == np.float32
    assert encoded[0, 2, 0, 0] == 1.0
    assert encoded[0, 15, 0, 0] == 1.0
    assert encoded[0, PLANE_M, 0, 0] == 2 / 8
    assert encoded[0, PLANE_U, 0, 0] == 3 / 5
    assert encoded[0, PLANE_SCORE_RATIO, 0, 0] == 1.0
    assert encoded[0, PLANE_SCORE_DIFF, 0, 0] == 0.0
    assert encoded[0, PLANE_LEGAL_MASK, 0, 0] == 1.0
    assert encoded[0, PLANE_LEGAL_MASK, 0, 2] == 0.0
    assert encoded[0, PLANE_PLAYER_SCORE_START, 0, 0] == 1 / 300
    enemy_param_start = PLANE_ORACLE_PARAM_START + ORACLE_PARAMS_PER_PLAYER
    assert encoded[0, enemy_param_start, 0, 0] == np.float32(0.1)
    assert encoded[0, enemy_param_start + 4, 0, 0] == np.float32(0.5)
