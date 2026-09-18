from typing import cast

import torch
from jaxtyping import Float
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


class ActorCritic(nn.Module):
    """A shared trunk with four directional actions."""

    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        self.NUM_PLANES = NUM_PLANES
        self.ACTION_COUNT = ACTION_COUNT

        if in_channels != NUM_PLANES:
            raise ValueError(f"AHC063 expects {NUM_PLANES} input planes, got {in_channels}")
        if channels <= 0 or blocks <= 0:
            raise ValueError("channels and blocks must be positive")
        self.observation_normalizer: nn.Module | None = None
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
        # Keep max pooling as an explicit op.  With a bf16 hyperspherical trunk,
        # Inductor can fuse the final fp32-to-bf16 cast into Tensor.amax and make
        # its tie-counting backward divide by zero.
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.policy = ModularSequential(
            ModularLinear(channels * 2, channels),
            nn.ReLU(inplace=True),
            ModularReadoutLinear(channels, ACTION_COUNT, bias=False),
        )
        self.value = ModularSequential(
            ModularLinear(channels * 2, channels),
            nn.ReLU(inplace=True),
            ModularReadoutLinear(channels, 1, bias=False),
        )
        validate_modula_graph(self.modula_graph())

    def modula_graph(self) -> ModulaGraphNode:
        return ModulaGraphNode(
            "sequence",
            (
                module_to_modula_graph(self.input_adapter, "input_adapter"),
                module_to_modula_graph(self.cell_encoder, "cell_encoder"),
                module_to_modula_graph(self.trunk, "trunk"),
                ModulaGraphNode(
                    "parallel",
                    (
                        module_to_modula_graph(self.policy, "policy"),
                        module_to_modula_graph(self.value, "value"),
                    ),
                ),
            ),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch {self.NUM_PLANES} H W"]
    ) -> tuple[Float[torch.Tensor, "batch {self.ACTION_COUNT}"], Float[torch.Tensor, "batch"]]:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got {x.ndim} dimensions")
        h = self._trunk_features(x)
        pooled = torch.cat(
            [h.mean(dim=(-2, -1)), self.max_pool(h).flatten(1)],
            dim=1,
        )
        return self.policy(pooled), self.value(pooled).squeeze(-1)

    def _trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        cells = self.input_adapter(x).permute(0, 2, 3, 1)
        cells = self.cell_encoder(cells)
        return self.trunk(cells.permute(0, 3, 1, 2))

    @torch.no_grad()
    def feature_norm_stats(
        self, x: Float[torch.Tensor, "batch {self.NUM_PLANES} H W"]
    ) -> dict[str, float]:
        h = cast(torch.Tensor, self._trunk_features(x))
        norm = h.float().pow(2).sum(dim=1).sqrt()
        return {
            "trunk_feature_norm_mean": float(norm.mean().item()),
            "trunk_feature_norm_std": float(norm.std(unbiased=False).item()),
            "trunk_feature_norm_max": float(norm.max().item()),
        }
