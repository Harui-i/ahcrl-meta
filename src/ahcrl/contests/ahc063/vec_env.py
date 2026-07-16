"""A vectorized NumPy simulator for AHC063 training."""

from dataclasses import dataclass

import numpy as np

from .encoder import (
    ACTION_COUNT,
    INITIAL_SNAKE_LENGTH,
    MAX_BOARD_SIZE,
    MAX_COLORS,
    NUM_PLANES,
    action_delta,
)


@dataclass
class StepResult:
    obs: dict[str, np.ndarray]
    reward: np.ndarray
    done: np.ndarray
    score: np.ndarray
    prefix_match_ratio: np.ndarray


class OuroborosVecEnv:
    """Independent random AHC063 instances sharing a padded observation shape."""

    def __init__(
        self,
        num_envs: int,
        seed_start: int = 0,
        seed_stride: int = 1,
        *,
        fixed_n: int | None = None,
        fixed_m: int | None = None,
        fixed_c: int | None = None,
        max_steps: int = 100_000,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.seed_stride = seed_stride
        self.max_steps = max_steps
        self.fixed_n, self.fixed_m, self.fixed_c = fixed_n, fixed_m, fixed_c
        self.n = np.zeros(num_envs, dtype=np.int16)
        self.m = np.zeros(num_envs, dtype=np.int16)
        self.c = np.zeros(num_envs, dtype=np.int16)
        self.desired = np.zeros((num_envs, MAX_BOARD_SIZE * MAX_BOARD_SIZE), dtype=np.int8)
        self.food = np.zeros((num_envs, MAX_BOARD_SIZE, MAX_BOARD_SIZE), dtype=np.int8)
        self.positions = np.zeros((num_envs, MAX_BOARD_SIZE * MAX_BOARD_SIZE, 2), dtype=np.int16)
        self.colors = np.zeros((num_envs, MAX_BOARD_SIZE * MAX_BOARD_SIZE), dtype=np.int8)
        self.length = np.zeros(num_envs, dtype=np.int16)
        self.previous_action = np.full(num_envs, -1, dtype=np.int8)
        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.obs = self.reset(seed_start, seed_stride)

    def reset(
        self,
        seed_start: int = 0,
        seed_stride: int = 1,
        fixed_n: int | None = None,
        fixed_m: int | None = None,
        fixed_c: int | None = None,
    ) -> dict[str, np.ndarray]:
        fixed_n = self.fixed_n if fixed_n is None else fixed_n
        fixed_m = self.fixed_m if fixed_m is None else fixed_m
        fixed_c = self.fixed_c if fixed_c is None else fixed_c
        self.food.fill(0)
        self.positions.fill(0)
        self.colors.fill(0)
        self.desired.fill(0)
        self.length.fill(INITIAL_SNAKE_LENGTH)
        self.previous_action.fill(-1)
        self.steps.fill(0)
        for env_id in range(self.num_envs):
            self._reset_one(
                env_id,
                seed_start + env_id * seed_stride,
                fixed_n,
                fixed_m,
                fixed_c,
            )
        self.obs = self._encode()
        return self.obs

    def reset_done(
        self,
        done: np.ndarray,
        seed_start: int = 0,
        seed_stride: int = 1,
        fixed_n: int | None = None,
        fixed_m: int | None = None,
        fixed_c: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset only finished environments and preserve all other episodes."""
        done = np.asarray(done, dtype=bool)
        if done.shape != (self.num_envs,):
            raise ValueError(f"done must have shape ({self.num_envs},), got {done.shape}")
        fixed_n = self.fixed_n if fixed_n is None else fixed_n
        fixed_m = self.fixed_m if fixed_m is None else fixed_m
        fixed_c = self.fixed_c if fixed_c is None else fixed_c
        for env_id in np.flatnonzero(done):
            self._reset_one(
                int(env_id),
                seed_start + int(env_id) * seed_stride,
                fixed_n,
                fixed_m,
                fixed_c,
            )
        self.obs = self._encode()
        return self.obs

    def legal_actions(self) -> np.ndarray:
        mask = np.ones((self.num_envs, ACTION_COUNT), dtype=bool)
        for env_id in range(self.num_envs):
            row, col = self.positions[env_id, 0]
            for action in range(ACTION_COUNT):
                dr, dc = action_delta(action)
                new_row, new_col = int(row) + dr, int(col) + dc
                mask[env_id, action] = (
                    0 <= new_row < self.n[env_id]
                    and 0 <= new_col < self.n[env_id]
                    and not self._is_u_turn(env_id, new_row, new_col)
                )
        return mask

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},), got {actions.shape}")
        old_score = self.score().copy()
        legal_actions = self.legal_actions()
        for env_id, action_value in enumerate(actions):
            action = int(action_value)
            legal = legal_actions[env_id]
            if action < 0 or action >= ACTION_COUNT or not legal[action]:
                action = int(np.flatnonzero(legal)[0])
            self._step_one(env_id, action)
        new_score = self.score()
        done = (self._food_count() == 0) | (self.steps >= self.max_steps)
        reward = (old_score - new_score).astype(np.float32) / 10_000.0
        prefix_match_ratio = self._prefix_match_ratio()
        self.obs = self._encode()
        return StepResult(self.obs, reward, done, new_score, prefix_match_ratio)

    def score(self) -> np.ndarray:
        mismatch = np.zeros(self.num_envs, dtype=np.int32)
        for env_id in range(self.num_envs):
            k = int(self.length[env_id])
            mismatch[env_id] = np.count_nonzero(self.colors[env_id, :k] != self.desired[env_id, :k])
        return self.steps.astype(np.int64) + 10_000 * (mismatch + 2 * (self.m - self.length))

    def close(self) -> None:
        return

    def _reset_one(
        self,
        env_id: int,
        seed: int,
        fixed_n: int | None,
        fixed_m: int | None,
        fixed_c: int | None,
    ) -> None:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(8, MAX_BOARD_SIZE + 1) if fixed_n is None else fixed_n)
        c = int(rng.integers(3, MAX_COLORS + 1) if fixed_c is None else fixed_c)
        min_m = (n * n + 3) // 4
        max_m = 3 * n * n // 4
        m = int(rng.integers(min_m, max_m + 1) if fixed_m is None else fixed_m)
        if not 8 <= n <= MAX_BOARD_SIZE or not 3 <= c <= MAX_COLORS:
            raise ValueError("fixed_n/fixed_c are outside the AHC063 limits")
        if not min_m <= m <= max_m or m <= INITIAL_SNAKE_LENGTH:
            raise ValueError("fixed_m is outside the AHC063 limits")
        self.food[env_id].fill(0)
        self.positions[env_id].fill(0)
        self.colors[env_id].fill(0)
        self.desired[env_id].fill(0)
        self.length[env_id] = INITIAL_SNAKE_LENGTH
        self.previous_action[env_id] = -1
        self.steps[env_id] = 0
        self.n[env_id], self.m[env_id], self.c[env_id] = n, m, c
        self.desired[env_id, :INITIAL_SNAKE_LENGTH] = 1
        self.desired[env_id, INITIAL_SNAKE_LENGTH:m] = rng.integers(
            1, c + 1, m - INITIAL_SNAKE_LENGTH
        )
        cells = np.arange(n * n)
        cells = cells[~np.isin(cells, np.array([0, n, 2 * n, 3 * n, 4 * n]))]
        chosen = rng.choice(cells, size=m - INITIAL_SNAKE_LENGTH, replace=False)
        shuffled_colors = self.desired[env_id, INITIAL_SNAKE_LENGTH:m].copy()
        rng.shuffle(shuffled_colors)
        self.food[env_id, chosen // n, chosen % n] = shuffled_colors
        for index in range(INITIAL_SNAKE_LENGTH):
            self.positions[env_id, index] = (4 - index, 0)
            self.colors[env_id, index] = 1

    def _step_one(self, env_id: int, action: int) -> None:
        self.steps[env_id] += 1
        self.previous_action[env_id] = action
        dr, dc = action_delta(action)
        old_length = int(self.length[env_id])
        old_positions = self.positions[env_id, :old_length].copy()
        old_colors = self.colors[env_id, :old_length].copy()
        head_row, head_col = old_positions[0]
        new_head = (int(head_row) + dr, int(head_col) + dc)
        moved_positions = np.empty_like(old_positions)
        moved_positions[0] = new_head
        moved_positions[1:] = old_positions[:-1]
        collision = np.flatnonzero(
            (moved_positions[1:, 0] == new_head[0]) & (moved_positions[1:, 1] == new_head[1])
        )
        if collision.size:
            h = int(collision[0]) + 1
            for index in range(h + 1, old_length):
                row, col = moved_positions[index]
                self.food[env_id, row, col] = old_colors[index]
            self.length[env_id] = h + 1
            self.positions[env_id, : h + 1] = moved_positions[: h + 1]
            self.colors[env_id, : h + 1] = old_colors[: h + 1]
            return
        food_color = int(self.food[env_id, new_head[0], new_head[1]])
        if food_color:
            self.food[env_id, new_head[0], new_head[1]] = 0
            self.length[env_id] = old_length + 1
            self.positions[env_id, : old_length + 1] = np.concatenate(
                [moved_positions, old_positions[-1:]], axis=0
            )
            self.colors[env_id, : old_length + 1] = np.concatenate(
                [old_colors, np.array([food_color], dtype=np.int8)]
            )
        else:
            self.length[env_id] = old_length
            self.positions[env_id, :old_length] = moved_positions
            self.colors[env_id, :old_length] = old_colors

    def _is_u_turn(self, env_id: int, row: int, col: int) -> bool:
        return (
            self.length[env_id] > 1
            and row == self.positions[env_id, 1, 0]
            and col == self.positions[env_id, 1, 1]
        )

    def _food_count(self) -> np.ndarray:
        return np.count_nonzero(self.food, axis=(1, 2))

    def _prefix_match_ratio(self) -> np.ndarray:
        """Return the matching leading color-prefix length divided by target M."""
        ratios = np.zeros(self.num_envs, dtype=np.float32)
        for env_id in range(self.num_envs):
            target_length = int(self.m[env_id])
            snake_length = int(self.length[env_id])
            prefix_length = 0
            while (
                prefix_length < snake_length
                and prefix_length < target_length
                and self.colors[env_id, prefix_length] == self.desired[env_id, prefix_length]
            ):
                prefix_length += 1
            ratios[env_id] = prefix_length / max(target_length, 1)
        return ratios

    def _encode(self) -> dict[str, np.ndarray]:
        planes = np.zeros(
            (self.num_envs, NUM_PLANES, MAX_BOARD_SIZE, MAX_BOARD_SIZE), dtype=np.float32
        )
        mask = self.legal_actions()
        food_counts = self._food_count()
        for env_id in range(self.num_envs):
            n, m, c = int(self.n[env_id]), int(self.m[env_id]), int(self.c[env_id])
            food = self.food[env_id]
            for color in range(1, c + 1):
                planes[env_id, color - 1, :n, :n] = food[:n, :n] == color
            k = int(self.length[env_id])
            for index in range(k):
                row, col = self.positions[env_id, index]
                color = int(self.colors[env_id, index])
                planes[env_id, 7 + color - 1, row, col] = 1.0
                planes[env_id, 14, row, col] = float(index == 0)
                planes[env_id, 15, row, col] = float(0 < index < k - 1)
                planes[env_id, 16, row, col] = float(index == k - 1)
            target = int(self.desired[env_id, k]) if k < m else 0
            if target:
                planes[env_id, 16 + target, :n, :n] = 1.0
            remaining = self.desired[env_id, k:m]
            for color in range(1, c + 1):
                planes[env_id, 23 + color, :n, :n] = np.count_nonzero(remaining == color) / max(
                    m, 1
                )
            row, col = self.positions[env_id, 0]
            scalar_values = (
                n / MAX_BOARD_SIZE,
                c / MAX_COLORS,
                k / max(m, 1),
                min(k, m) / max(m, 1),
                row / max(n - 1, 1),
                col / max(n - 1, 1),
                food_counts[env_id] / max(m - INITIAL_SNAKE_LENGTH, 1),
                self.steps[env_id] / max(self.max_steps, 1),
            )
            for plane, value in enumerate(scalar_values, start=31):
                planes[env_id, plane, :n, :n] = value
            if self.previous_action[env_id] >= 0:
                planes[env_id, 39 + self.previous_action[env_id], :n, :n] = 1.0
        return {"planes": planes, "mask": mask}
