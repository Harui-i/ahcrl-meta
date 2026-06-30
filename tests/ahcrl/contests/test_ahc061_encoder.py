import numpy as np

from ahcrl.contests.ahc061.encoder import encode_batch


def test_encode_batch_shape() -> None:
    obs = {
        "m": np.array([2], dtype=np.int64),
        "u": np.array([3], dtype=np.int64),
        "turn": np.array([0], dtype=np.int64),
        "values": np.ones((1, 100), dtype=np.float32),
        "owner": np.full((1, 100), -1, dtype=np.int64),
        "level": np.zeros((1, 100), dtype=np.int64),
        "pos": np.full((1, 16), -1, dtype=np.int64),
    }
    obs["owner"][0, 0] = 0
    obs["owner"][0, 11] = 1
    obs["level"][0, 0] = 1
    obs["level"][0, 11] = 1
    obs["pos"][0, 0:4] = np.array([0, 0, 1, 1])

    encoded = encode_batch(obs)

    assert encoded.shape == (1, 24, 10, 10)
    assert encoded.dtype == np.float32
    assert encoded[0, 2, 0, 0] == 1.0
    assert encoded[0, 15, 0, 0] == 1.0
