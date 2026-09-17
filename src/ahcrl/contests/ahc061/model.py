import torch
from torch import nn

from ahcrl.contests.ahc061.encoder import (
    CRITIC_FEATURE_SHAPE,
    LEVEL_PLANE_COUNT,
    LEVEL_PLANE_START,
    MAX_PLAYERS,
    NUM_PLANES,
    OWNER_PLANE_COUNT,
    OWNER_PLANE_START,
    POSITION_PLANE_COUNT,
    POSITION_PLANE_START,
    TYPED_NUM_PLANES,
)
from ahcrl.nn.blocks import ConvNeXtBlock
from ahcrl.nn.components import make_group_norm
from ahcrl.nn.modula import (
    ModulaGraphNode,
    ModularConv2d,
    ModularLinear,
    ModularReadoutConv2d,
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

AHC061_CATEGORICAL_GROUPS = (
    CategoricalPlaneGroup(
        "owner",
        OWNER_PLANE_START,
        OWNER_PLANE_COUNT,
        embedding_dim=OWNER_PLANE_COUNT,
    ),
    CategoricalPlaneGroup(
        "level",
        LEVEL_PLANE_START,
        LEVEL_PLANE_COUNT,
        implicit_zero=True,
        embedding_dim=LEVEL_PLANE_COUNT + 1,
    ),
    CategoricalPlaneGroup(
        "position",
        POSITION_PLANE_START,
        POSITION_PLANE_COUNT,
        implicit_zero=True,
        embedding_dim=POSITION_PLANE_COUNT + 1,
    ),
)


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = NUM_PLANES,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        if in_channels != NUM_PLANES:
            raise ValueError(f"AHC061 expects {NUM_PLANES} input planes, got {in_channels}")
        self.observation_normalizer: RunningObservationNormalizer | None = None
        self.input_adapter = CategoricalPlaneAdapter(in_channels, AHC061_CATEGORICAL_GROUPS)
        if self.input_adapter.output_channels != TYPED_NUM_PLANES:
            raise AssertionError("AHC061 categorical adapter width is inconsistent")
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
        self.policy = ModularSequential(
            ModularConv2d(channels, channels, kernel_size=1, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            ModularReadoutConv2d(channels, 1, kernel_size=1, bias=False),
            nn.Flatten(),
        )
        # Keep the actor module names stable: the submission exporter consumes
        # these tensors directly.  The critic owns a second copy of the full
        # observation representation instead of consuming actor trunk features.
        self.value = IndependentValueNetwork(
            in_channels=in_channels,
            channels=channels,
            blocks=blocks,
        )
        validate_modula_graph(self.modula_graph())

    def modula_graph(self) -> ModulaGraphNode:
        return ModulaGraphNode(
            "parallel",
            (
                ModulaGraphNode(
                    "sequence",
                    (
                        module_to_modula_graph(self.input_adapter, "input_adapter"),
                        module_to_modula_graph(self.cell_encoder, "cell_encoder"),
                        module_to_modula_graph(self.trunk, "trunk"),
                        module_to_modula_graph(self.policy, "policy"),
                    ),
                ),
                module_to_modula_graph(self.value, "value"),
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        critic_features: torch.Tensor | None = None,
        normalize_input: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normalize_input and self.observation_normalizer is not None:
            x = self.observation_normalizer(x)
        return self.policy_logits(x, normalize_input=False), self.value_predictions(
            x, critic_features, normalize_input=False
        )

    def policy_logits(self, x: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
        """Compute actor logits without evaluating the independent critic."""
        if normalize_input and self.observation_normalizer is not None:
            x = self.observation_normalizer(x)
        return self.policy(self._trunk_features(x))

    def value_predictions(
        self,
        x: torch.Tensor,
        critic_features: torch.Tensor | None = None,
        normalize_input: bool = True,
    ) -> torch.Tensor:
        """Compute critic values without evaluating the independent actor."""
        if normalize_input and self.observation_normalizer is not None:
            x = self.observation_normalizer(x)
        return self.value(x, critic_features).squeeze(-1)

    def _trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.input_adapter(x)
        cells = adapted.permute(0, 2, 3, 1)
        cells = self.cell_encoder(cells)
        encoded = cells.permute(0, 3, 1, 2)
        return self.trunk(encoded)

    @torch.no_grad()
    def trunk_feature_norm_stats(self, x: torch.Tensor) -> dict[str, float]:
        h = self._trunk_features(x)
        feature_norm = h.float().pow(2).sum(dim=1).sqrt()
        return {
            "trunk_feature_norm_mean": float(feature_norm.mean().item()),
            "trunk_feature_norm_std": float(feature_norm.std(unbiased=False).item()),
            "trunk_feature_norm_max": float(feature_norm.max().item()),
        }


class IndependentValueNetwork(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        channels: int,
        blocks: int,
    ) -> None:
        super().__init__()
        self.input_adapter = CategoricalPlaneAdapter(in_channels, AHC061_CATEGORICAL_GROUPS)
        if self.input_adapter.output_channels != TYPED_NUM_PLANES:
            raise AssertionError("AHC061 categorical adapter width is inconsistent")
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
        self.blocks = ModularSequential(
            ConvNeXtBlock(channels, residual_branch_scale=0.5),
            ConvNeXtBlock(channels, residual_branch_scale=0.5),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        stats_channels = in_channels * 2
        pooled_channels = channels * 2
        hidden_channels = channels * 2
        player_embedding_channels = max(8, channels // MAX_PLAYERS)
        self.critic_player_encoder = ModularSequential(
            ModularLinear(CRITIC_FEATURE_SHAPE[1], player_embedding_channels),
            nn.ReLU(inplace=True),
        )
        self.critic_encoder = ModularSequential(
            ModularLinear(MAX_PLAYERS * player_embedding_channels, channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = ModularSequential(
            ModularLinear(pooled_channels + stats_channels + channels, hidden_channels),
            nn.ReLU(inplace=True),
            ModularLinear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            ModularReadoutLinear(hidden_channels, 1, bias=False),
        )

    def _trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.input_adapter(x)
        cells = adapted.permute(0, 2, 3, 1)
        cells = self.cell_encoder(cells)
        encoded = cells.permute(0, 3, 1, 2)
        return self.trunk(encoded)

    def forward(
        self,
        raw_planes: torch.Tensor,
        critic_features: torch.Tensor | None,
    ) -> torch.Tensor:
        trunk_features = self._trunk_features(raw_planes)
        h = self.blocks(trunk_features)
        avg_features = self.avg_pool(h).flatten(1)
        max_features = self.max_pool(h).flatten(1)
        plane_mean = raw_planes.mean(dim=(-2, -1))
        plane_max = raw_planes.amax(dim=(-2, -1))
        features = [avg_features, max_features, plane_mean, plane_max]
        if critic_features is None:
            critic_features = torch.zeros(
                (trunk_features.shape[0], *CRITIC_FEATURE_SHAPE),
                device=trunk_features.device,
                dtype=trunk_features.dtype,
            )
        critic_embedding = self.critic_player_encoder(critic_features).flatten(1)
        features.append(self.critic_encoder(critic_embedding))
        features = torch.cat(features, dim=1)
        return self.mlp(features)
