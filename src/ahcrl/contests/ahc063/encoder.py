"""Observation encoding for AHC063 (Colorful Ouroboros)."""

MAX_BOARD_SIZE = 16
MAX_COLORS = 7
ACTION_COUNT = 4

FOOD_COLOR_START = 0
SNAKE_COLOR_START = FOOD_COLOR_START + MAX_COLORS
PLANE_HEAD = SNAKE_COLOR_START + MAX_COLORS
PLANE_BODY = PLANE_HEAD + 1
PLANE_TAIL = PLANE_BODY + 1
TARGET_COLOR_START = PLANE_TAIL + 1
TARGET_REMAIN_COUNT_START = TARGET_COLOR_START + MAX_COLORS
PLANE_N = TARGET_REMAIN_COUNT_START + MAX_COLORS
PLANE_C = PLANE_N + 1
PLANE_LENGTH = PLANE_C + 1
PLANE_TARGET_INDEX = PLANE_LENGTH + 1
PLANE_HEAD_ROW = PLANE_TARGET_INDEX + 1
PLANE_HEAD_COL = PLANE_HEAD_ROW + 1
PLANE_REMAINING_FOOD = PLANE_HEAD_COL + 1
PLANE_STEP = PLANE_REMAINING_FOOD + 1
PLANE_PREV_ACTION_START = PLANE_STEP + 1
NUM_PLANES = PLANE_PREV_ACTION_START + ACTION_COUNT

# The initial snake occupies (4, 0), ..., (0, 0).
INITIAL_SNAKE_LENGTH = 5


def action_delta(action: int) -> tuple[int, int]:
    """Return (row, column) displacement for U, D, L, R."""
    if action == 0:
        return -1, 0
    if action == 1:
        return 1, 0
    if action == 2:
        return 0, -1
    if action == 3:
        return 0, 1
    raise ValueError(f"unknown action: {action}")
