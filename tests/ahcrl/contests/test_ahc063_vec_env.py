import numpy as np

from ahcrl.contests.ahc063.vec_env import OuroborosVecEnv


def _full_length_env() -> OuroborosVecEnv:
    env = OuroborosVecEnv(1, fixed_n=8, fixed_m=16, fixed_c=3)
    env.food.fill(0)
    env.length[0] = env.m[0]
    env.colors[0, : env.m[0]] = env.desired[0, : env.m[0]]
    return env


def test_done_requires_target_sequence_after_all_food_is_eaten() -> None:
    env = _full_length_env()

    result = env.step(np.array([1]))

    assert result.done[0]


def test_done_stays_false_when_full_length_sequence_is_wrong() -> None:
    env = _full_length_env()
    env.colors[0, 0] = 2

    result = env.step(np.array([1]))

    assert not result.done[0]
