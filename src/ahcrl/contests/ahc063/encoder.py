"""Typed observation schema for AHC063 (Colorful Ouroboros)."""

MAX_BOARD_SIZE = 16
MAX_COLORS = 7
ACTION_COUNT = 4
MAX_SEQUENCE_LENGTH = 192
BOARD_FEATURE_COUNT = 8
GLOBAL_FEATURE_COUNT = 10
ACTION_FEATURE_COUNT = 7
POSITION_SENTINEL = 255

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


__all__ = [
    "ACTION_COUNT",
    "ACTION_FEATURE_COUNT",
    "BOARD_FEATURE_COUNT",
    "GLOBAL_FEATURE_COUNT",
    "INITIAL_SNAKE_LENGTH",
    "MAX_BOARD_SIZE",
    "MAX_COLORS",
    "MAX_SEQUENCE_LENGTH",
    "POSITION_SENTINEL",
    "action_delta",
]
