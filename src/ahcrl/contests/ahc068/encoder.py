"""Contest-specific observation encoder."""

# TODO: Define the exact observation layout shared by Python and C++.

# 問題文で決まるやつら
BOARD_SIZE = (
    20  # N,盤面サイズ. すべてのテストケースにおいて、盤面サイズ `N` は `20` に固定されている。
)

# 特徴量設計などで決まるやつら

NUM_PLANES = 10 # 後々決まっていく
