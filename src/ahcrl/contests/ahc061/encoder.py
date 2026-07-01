import numpy as np

BOARD_SIZE = 10
MAX_PLAYERS = 8
MAX_LEVEL = 5
BASE_PLANES = 24
NUM_PLANES = BASE_PLANES + 2 + 2 + 1 + MAX_PLAYERS
PLANE_M = 24
PLANE_U = 25
PLANE_SCORE_RATIO = 26
PLANE_SCORE_DIFF = 27
PLANE_LEGAL_MASK = 28
PLANE_PLAYER_SCORE_START = 29


def encode_batch(obs: dict[str, np.ndarray]) -> np.ndarray:
    batch = int(obs["owner"].shape[0])
    values = obs["values"].reshape(batch, BOARD_SIZE, BOARD_SIZE)
    owners = obs["owner"].reshape(batch, BOARD_SIZE, BOARD_SIZE)
    levels = obs["level"].reshape(batch, BOARD_SIZE, BOARD_SIZE)
    positions = obs["pos"].reshape(batch, MAX_PLAYERS, 2)
    turns = obs["turn"].astype(np.float32)
    legal_masks = obs["mask"].reshape(batch, BOARD_SIZE, BOARD_SIZE)

    planes = np.zeros((batch, NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    mean_values = values.mean(axis=(1, 2), keepdims=True)
    planes[:, 0] = values / np.maximum(mean_values, 1.0)

    for b in range(batch):
        m = int(obs["m"][b])
        u = int(obs["u"][b])
        scores = _player_scores(values[b], owners[b], levels[b], m)
        player_map = _player_id_map(scores, m)
        mapped_owners = np.full((BOARD_SIZE, BOARD_SIZE), -1, dtype=np.int64)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                owner = int(owners[b, i, j])
                mapped_owners[i, j] = -1 if owner < 0 else player_map[owner]

        for owner_id in range(-1, MAX_PLAYERS):
            planes[b, owner_id + 2][mapped_owners == owner_id] = 1.0

        for level in range(1, MAX_LEVEL + 1):
            planes[b, 9 + level][levels[b] == level] = 1.0

        for player in range(m):
            x, y = positions[b, player]
            mapped_player = player_map[player]
            if 0 <= mapped_player < MAX_PLAYERS and x >= 0 and y >= 0:
                planes[b, 15 + mapped_player, x, y] = 1.0

        planes[b, 23, :, :] = (100.0 - turns[b]) / 100.0
        planes[b, PLANE_M, :, :] = m / MAX_PLAYERS
        planes[b, PLANE_U, :, :] = u / MAX_LEVEL

        player0_score = scores[0]
        max_ai_score = max(scores[1:], default=0.0)
        total_capacity = max(float(values[b].sum() * max(u, 1)), 1.0)
        planes[b, PLANE_SCORE_RATIO, :, :] = player0_score / max(max_ai_score, 1.0)
        planes[b, PLANE_SCORE_DIFF, :, :] = (player0_score - max_ai_score) / total_capacity
        planes[b, PLANE_LEGAL_MASK] = legal_masks[b].astype(np.float32)

        for player in range(m):
            mapped_player = player_map[player]
            if 0 <= mapped_player < MAX_PLAYERS:
                planes[b, PLANE_PLAYER_SCORE_START + mapped_player, :, :] = (
                    scores[player] / total_capacity
                )

    return planes


def _player_scores(
    values: np.ndarray, owner: np.ndarray, level: np.ndarray, m: int
) -> list[float]:
    scores = [0.0] * m
    for player in range(m):
        scores[player] = float((values[owner == player] * level[owner == player]).sum())
    return scores


def _player_id_map(scores: list[float], m: int) -> dict[int, int]:
    enemy_order = sorted(range(1, m), key=lambda player: (-scores[player], player))
    mapped = {0: 0}
    for rank, player in enumerate(enemy_order, start=1):
        mapped[player] = rank
    return mapped
