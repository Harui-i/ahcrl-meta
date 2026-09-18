import torch
from torch import nn

from ahcrl.nn.modula import (
    ModulaGraphNode,
    ModularLinear,
    ModularReadoutLinear,
    ModularSequential,
    module_to_modula_graph,
    validate_modula_graph,
)
from ahcrl.nn.observation import (
    CategoricalPlaneAdapter,
    CategoricalPlaneGroup,
    RunningObservationNormalizer,
)
from ahcrl.nn.trunk import make_trunk

from .encoder import (
    ACTION_COUNT,
    FOOD_COLOR_COUNT,
    FOOD_COLOR_START,
    NUM_PLANES,
    PLANE_PREV_ACTION_START,
    PREV_ACTION_COUNT,
    SEGMENT_COUNT,
    SEGMENT_START,
    SNAKE_COLOR_COUNT,
    SNAKE_COLOR_START,
    TARGET_COLOR_COUNT,
    TARGET_COLOR_START,
    TYPED_NUM_PLANES,
)

AHC063_CATEGORICAL_GROUPS = (
    CategoricalPlaneGroup(
        "food_color",
        FOOD_COLOR_START,
        FOOD_COLOR_COUNT,
        implicit_zero=True,
        embedding_dim=FOOD_COLOR_COUNT + 1,
    ),
    CategoricalPlaneGroup(
        "snake_color",
        SNAKE_COLOR_START,
        SNAKE_COLOR_COUNT,
        implicit_zero=True,
        embedding_dim=SNAKE_COLOR_COUNT + 1,
    ),
    CategoricalPlaneGroup(
        "segment",
        SEGMENT_START,
        SEGMENT_COUNT,
        implicit_zero=True,
        embedding_dim=SEGMENT_COUNT + 1,
    ),
    CategoricalPlaneGroup(
        "target_color",
        TARGET_COLOR_START,
        TARGET_COLOR_COUNT,
        implicit_zero=True,
        embedding_dim=TARGET_COLOR_COUNT + 1,
    ),
    CategoricalPlaneGroup(
        "previous_action",
        PLANE_PREV_ACTION_START,
        PREV_ACTION_COUNT,
        implicit_zero=True,
        embedding_dim=PREV_ACTION_COUNT + 1,
    ),
)


class _FeatureNetwork(nn.Module):
    """Independent observation-to-trunk feature network for actor or critic."""

    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        if in_channels != NUM_PLANES:
            raise ValueError(f"AHC063 expects {NUM_PLANES} input planes, got {in_channels}")
        if channels <= 0 or blocks <= 0:
            raise ValueError("channels and blocks must be positive")
        self.input_adapter = CategoricalPlaneAdapter(in_channels, AHC063_CATEGORICAL_GROUPS)
        if self.input_adapter.output_channels != TYPED_NUM_PLANES:
            raise AssertionError("AHC063 categorical adapter width is inconsistent")
        self.cell_encoder = ModularSequential(
            ModularLinear(TYPED_NUM_PLANES, channels * 4),
            nn.GELU(),
            ModularLinear(channels * 4, channels),
        )
        self.trunk = make_trunk(
            in_channels=channels,
            channels=channels,
            blocks=blocks,
        )

    def _trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        adapted = self.input_adapter(x)
        cells = adapted.permute(0, 2, 3, 1)
        cells = self.cell_encoder(cells)
        return self.trunk(cells.permute(0, 3, 1, 2))

    @torch.no_grad()
    def feature_norm_stats(self, x: torch.Tensor) -> dict[str, float]:
        h = self._trunk_features(x)
        norm = h.float().pow(2).sum(dim=1).sqrt()
        return {
            "trunk_feature_norm_mean": float(norm.mean().item()),
            "trunk_feature_norm_std": float(norm.std(unbiased=False).item()),
            "trunk_feature_norm_max": float(norm.max().item()),
        }


class PolicyNetwork(_FeatureNetwork):
    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__(in_channels=in_channels, channels=channels, blocks=blocks)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.policy = ModularSequential(
            ModularLinear(channels * 2, channels),
            nn.ReLU(inplace=True),
            ModularReadoutLinear(channels, ACTION_COUNT, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._trunk_features(x)
        pooled = torch.cat([h.mean(dim=(-2, -1)), self.max_pool(h).flatten(1)], dim=1)
        return self.policy(pooled)


class ValueNetwork(_FeatureNetwork):
    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__(in_channels=in_channels, channels=channels, blocks=blocks)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.value = ModularSequential(
            ModularLinear(channels * 2, channels),
            nn.ReLU(inplace=True),
            ModularReadoutLinear(channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._trunk_features(x)
        pooled = torch.cat([h.mean(dim=(-2, -1)), self.max_pool(h).flatten(1)], dim=1)
        return self.value(pooled)


class PPOModel(nn.Module):
    """独立した policy/value network を PPO 学習用に束ねるコンテナ。"""

    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        self.NUM_PLANES = NUM_PLANES
        self.ACTION_COUNT = ACTION_COUNT
        self.observation_normalizer: RunningObservationNormalizer | None = None
        self.policy = PolicyNetwork(in_channels=in_channels, channels=channels, blocks=blocks)
        self.value = ValueNetwork(in_channels=in_channels, channels=channels, blocks=blocks)
        validate_modula_graph(self.modula_graph())

    def modula_graph(self) -> ModulaGraphNode:
        return ModulaGraphNode(
            "parallel",
            (
                module_to_modula_graph(self.policy, "policy"),
                module_to_modula_graph(self.value, "value"),
            ),
        )

    def _normalize(self, x: torch.Tensor, normalize_input: bool) -> torch.Tensor:
        if normalize_input and self.observation_normalizer is not None:
            return self.observation_normalizer(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        normalize_input: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._normalize(x, normalize_input)
        return self.policy(x), self.value(x).squeeze(-1)

    def policy_logits(self, x: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
        """独立した critic を評価せずに actor の logits を計算する。"""
        return self.policy(self._normalize(x, normalize_input))

    def value_predictions(self, x: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
        """独立した actor を評価せずに critic の value を計算する。"""
        return self.value(self._normalize(x, normalize_input)).squeeze(-1)

    @torch.no_grad()
    def feature_norm_stats(self, x: torch.Tensor) -> dict[str, float]:
        return self.policy.feature_norm_stats(x)
