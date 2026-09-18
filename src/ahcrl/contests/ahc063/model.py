"""Slot-fusion actor/critic model for AHC063."""

from __future__ import annotations

import torch
from torch import nn

from ahcrl.nn.fusion_blocks import (
    ConvexFusion,
    ResidualMLPStack,
    SequenceStack,
    SpatialStack,
    UnitGELU,
)
from ahcrl.nn.modula import (
    ModulaGraphNode,
    ModularEmbedding,
    ModularLinear,
    ModularReadoutLinear,
    ModularSequential,
    module_to_modula_graph,
    validate_modula_graph,
)

from .encoder import (
    ACTION_COUNT,
    ACTION_FEATURE_COUNT,
    BOARD_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT,
    MAX_BOARD_SIZE,
    MAX_COLORS,
    MAX_SEQUENCE_LENGTH,
)


def _parameter_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


class _TokenEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.target_embedding = ModularEmbedding(MAX_COLORS + 1, 8)
        self.actual_embedding = ModularEmbedding(MAX_COLORS + 1, 8)
        numeric_dim = 14
        input_dim = 16 + numeric_dim
        self.branches = ConvexFusion(
            ModularSequential(
                ModularLinear(input_dim, channels * 2),
                UnitGELU(),
                ModularLinear(channels * 2, channels),
            ),
            ModularSequential(
                ModularLinear(input_dim, channels * 2),
                UnitGELU(),
                ModularLinear(channels * 2, channels),
            ),
            ModularSequential(
                ModularLinear(input_dim, channels * 2),
                UnitGELU(),
                ModularLinear(channels * 2, channels),
            ),
        )

    def forward(
        self,
        slot_colors: torch.Tensor,
        slot_positions: torch.Tensor,
        length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = _parameter_dtype(self)
        colors = slot_colors.long()
        positions = slot_positions.long()
        target, actual = colors.unbind(dim=-1)
        target_valid = target > 0
        actual_valid = actual > 0
        valid = target_valid | actual_valid
        occupied = actual_valid & (positions[..., 0] < MAX_BOARD_SIZE)
        match = target_valid & (target == actual)
        row = torch.where(
            occupied, positions[..., 0].float() / 15.0, torch.zeros_like(target.float())
        )
        col = torch.where(
            occupied, positions[..., 1].float() / 15.0, torch.zeros_like(target.float())
        )
        p = torch.arange(MAX_SEQUENCE_LENGTH, device=colors.device, dtype=torch.float32)[None, :]
        p = p.expand(colors.shape[0], -1)
        length = length.unsqueeze(1)
        relative = (p - length.float()) / 192.0
        head = actual_valid & (p == 0)
        tail = actual_valid & (p == length.float() - 1)
        next_slot = target_valid & (p == length.float())
        safe_pos = torch.where(occupied.unsqueeze(-1), positions, torch.zeros_like(positions))
        predecessor = torch.cat((safe_pos[:, :1], safe_pos[:, :-1]), dim=1)
        successor = torch.cat((safe_pos[:, 1:], safe_pos[:, -1:]), dim=1)
        has_predecessor = occupied & (p > 0)
        has_successor = occupied & (p + 1 < length)
        pred_row = torch.where(
            has_predecessor,
            (safe_pos[..., 0] - predecessor[..., 0]).float() / 15.0,
            torch.zeros_like(row),
        )
        pred_col = torch.where(
            has_predecessor,
            (safe_pos[..., 1] - predecessor[..., 1]).float() / 15.0,
            torch.zeros_like(col),
        )
        succ_row = torch.where(
            has_successor,
            (successor[..., 0] - safe_pos[..., 0]).float() / 15.0,
            torch.zeros_like(row),
        )
        succ_col = torch.where(
            has_successor,
            (successor[..., 1] - safe_pos[..., 1]).float() / 15.0,
            torch.zeros_like(col),
        )
        numeric = torch.stack(
            (
                valid.float(),
                occupied.float(),
                match.float(),
                row,
                col,
                p / 191.0,
                relative,
                head.float(),
                tail.float(),
                next_slot.float(),
                pred_row,
                pred_col,
                succ_row,
                succ_col,
            ),
            dim=-1,
        ).to(dtype=dtype)
        embedded = torch.cat((self.target_embedding(target), self.actual_embedding(actual)), dim=-1)
        tokens = self.branches(torch.cat((embedded, numeric), dim=-1))
        return tokens * valid.unsqueeze(-1).to(dtype=tokens.dtype), occupied


class _SpatialEncoder(nn.Module):
    def __init__(self, sequence_channels: int, spatial_channels: int, spatial_blocks: int) -> None:
        super().__init__()
        self.food_embedding = ModularEmbedding(MAX_COLORS + 1, 32)
        self.food_branch = ModularSequential(
            ModularLinear(32 + BOARD_FEATURE_COUNT, spatial_channels * 2),
            UnitGELU(),
            ModularLinear(spatial_channels * 2, spatial_channels),
        )
        self.sequence_projection = ModularLinear(sequence_channels, spatial_channels)
        self.global_embedding = ModularEmbedding(ACTION_COUNT + 1, 16)
        self.global_branch = ModularSequential(
            ModularLinear(GLOBAL_FEATURE_COUNT + 16, spatial_channels),
            UnitGELU(),
            ModularLinear(spatial_channels, spatial_channels),
        )
        self.stack = SpatialStack(spatial_channels, spatial_blocks)

    def _scatter_sequence(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        occupied: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, _ = tokens.shape
        channels = self.sequence_projection.out_features
        cells = MAX_BOARD_SIZE * MAX_BOARD_SIZE
        flat_pos = (positions[..., 0].long() * MAX_BOARD_SIZE + positions[..., 1].long()).clamp(
            0, cells - 1
        )
        values = self.sequence_projection(tokens) * occupied.unsqueeze(-1).to(tokens.dtype)
        output = torch.zeros(batch, cells, channels, device=tokens.device, dtype=tokens.dtype)
        count = torch.zeros(batch, cells, 1, device=tokens.device, dtype=tokens.dtype)
        output.scatter_add_(1, flat_pos.unsqueeze(-1).expand(-1, -1, channels), values)
        count.scatter_add_(1, flat_pos.unsqueeze(-1), occupied.unsqueeze(-1).to(tokens.dtype))
        return (
            (output / count.clamp_min(1.0))
            .transpose(1, 2)
            .reshape(batch, channels, MAX_BOARD_SIZE, MAX_BOARD_SIZE)
        )

    def forward(
        self,
        board_food: torch.Tensor,
        board_features: torch.Tensor,
        tokens: torch.Tensor,
        slot_positions: torch.Tensor,
        occupied: torch.Tensor,
        global_features: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = _parameter_dtype(self)
        board_features = board_features.to(dtype=dtype)
        food = self.food_embedding(board_food.long())
        food_input = torch.cat((food, board_features.permute(0, 2, 3, 1)), dim=-1)
        food_map = self.food_branch(food_input).permute(0, 3, 1, 2)
        sequence_map = self._scatter_sequence(tokens, slot_positions, occupied)
        previous = self.global_embedding(previous_action.long().flatten(1)[:, 0])
        global_input = torch.cat((global_features.to(dtype=dtype), previous), dim=-1)
        global_map = self.global_branch(global_input)[:, :, None, None].expand_as(food_map)
        valid = board_features[:, 0].clamp(0.0, 1.0)
        spatial = (food_map + sequence_map + global_map) / 3.0
        return self.stack(spatial, valid), valid


class _FusionEncoder(nn.Module):
    def __init__(
        self,
        sequence_channels: int,
        sequence_blocks: int,
        spatial_channels: int,
        spatial_blocks: int,
    ) -> None:
        super().__init__()
        self.token_encoder = _TokenEncoder(sequence_channels)
        self.sequence = SequenceStack(sequence_channels, sequence_blocks)
        self.spatial = _SpatialEncoder(sequence_channels, spatial_channels, spatial_blocks)

    def forward(
        self,
        board_food: torch.Tensor,
        board_features: torch.Tensor,
        slot_colors: torch.Tensor,
        slot_positions: torch.Tensor,
        global_features: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = (slot_colors[..., 1] > 0).sum(dim=1).to(dtype=global_features.dtype)
        tokens, occupied = self.token_encoder(slot_colors, slot_positions, length)
        tokens = self.sequence(tokens, (slot_colors[..., 0] > 0) | (slot_colors[..., 1] > 0))
        return self.spatial(
            board_food,
            board_features,
            tokens,
            slot_positions,
            occupied,
            global_features,
            previous_action,
        )


class _ActionHead(nn.Module):
    def __init__(self, spatial_channels: int, head_channels: int, head_blocks: int) -> None:
        super().__init__()
        self.direction_embedding = ModularEmbedding(ACTION_COUNT, 8)
        self.color_embedding = ModularEmbedding(MAX_COLORS + 1, 8)
        action_dim = 8 + 8 + ACTION_FEATURE_COUNT + 1
        self.head_branch = ModularLinear(spatial_channels, head_channels)
        self.destination_branch = ModularLinear(spatial_channels, head_channels)
        self.pool_branch = ModularLinear(spatial_channels * 2, head_channels)
        self.action_branch = ModularLinear(action_dim, head_channels)
        self.mlp = ResidualMLPStack(head_channels, head_blocks)
        self.readout = ModularReadoutLinear(head_channels, 1, bias=False)

    def forward(
        self,
        spatial: torch.Tensor,
        board_features: torch.Tensor,
        action_colors: torch.Tensor,
        action_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = spatial.shape[0]
        flat = spatial.flatten(2)
        valid = board_features[:, 0].flatten(1).to(dtype=spatial.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (flat * valid[:, None]).sum(dim=-1) / denom
        maximum = flat.masked_fill(valid[:, None] <= 0, torch.finfo(spatial.dtype).min).amax(dim=-1)
        head_weight = board_features[:, 2].flatten(1)
        head_index = head_weight.argmax(dim=1)
        head = flat.transpose(1, 2)[torch.arange(batch, device=spatial.device), head_index]
        destination = []
        for action in range(ACTION_COUNT):
            row = (head_index // MAX_BOARD_SIZE) + (
                0 if action not in (0, 1) else (-1 if action == 0 else 1)
            )
            col = (head_index % MAX_BOARD_SIZE) + (
                0 if action not in (2, 3) else (-1 if action == 2 else 1)
            )
            legal = (row >= 0) & (row < MAX_BOARD_SIZE) & (col >= 0) & (col < MAX_BOARD_SIZE)
            index = row.clamp(0, MAX_BOARD_SIZE - 1) * MAX_BOARD_SIZE + col.clamp(
                0, MAX_BOARD_SIZE - 1
            )
            value = flat.transpose(1, 2)[torch.arange(batch, device=spatial.device), index]
            destination.append(value * legal.unsqueeze(-1).to(value.dtype))
        destination_tensor = torch.stack(destination, dim=1)
        head_tensor = head[:, None].expand(-1, ACTION_COUNT, -1)
        pooled = torch.cat((mean, maximum), dim=-1)[:, None].expand(-1, ACTION_COUNT, -1)
        directions = self.direction_embedding(torch.arange(ACTION_COUNT, device=spatial.device))[
            None
        ].expand(batch, -1, -1)
        colors = self.color_embedding(action_colors.long())
        action_input = torch.cat(
            (
                directions,
                colors,
                action_features.to(spatial.dtype),
                mask[:, :, None].to(spatial.dtype),
            ),
            dim=-1,
        )
        fused = (
            self.head_branch(head_tensor)
            + self.destination_branch(destination_tensor)
            + self.pool_branch(pooled)
            + self.action_branch(action_input)
        ) / 4.0
        return self.readout(self.mlp(fused.reshape(batch * ACTION_COUNT, -1))).reshape(
            batch, ACTION_COUNT
        )


class _Actor(nn.Module):
    def __init__(
        self,
        sequence_channels: int,
        sequence_blocks: int,
        spatial_channels: int,
        spatial_blocks: int,
        head_channels: int,
        head_blocks: int,
    ) -> None:
        super().__init__()
        self.encoder = _FusionEncoder(
            sequence_channels, sequence_blocks, spatial_channels, spatial_blocks
        )
        self.head = _ActionHead(spatial_channels, head_channels, head_blocks)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        spatial, valid = self.encoder(*inputs[:6])
        return self.head(spatial, inputs[1], inputs[6], inputs[7], inputs[8])


class _Critic(nn.Module):
    def __init__(
        self,
        sequence_channels: int,
        sequence_blocks: int,
        spatial_channels: int,
        spatial_blocks: int,
        head_channels: int,
        head_blocks: int,
    ) -> None:
        super().__init__()
        self.encoder = _FusionEncoder(
            sequence_channels, sequence_blocks, spatial_channels, spatial_blocks
        )
        self.pool = ModularLinear(spatial_channels * 2, head_channels)
        self.mlp = ResidualMLPStack(head_channels, head_blocks)
        self.readout = ModularReadoutLinear(head_channels, 1, bias=False)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        spatial, valid = self.encoder(*inputs[:6])
        flat = spatial.flatten(2)
        valid = valid.flatten(1).to(dtype=spatial.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (flat * valid[:, None]).sum(dim=-1) / denom
        maximum = flat.masked_fill(valid[:, None] <= 0, torch.finfo(spatial.dtype).min).amax(dim=-1)
        return self.readout(self.mlp(self.pool(torch.cat((mean, maximum), dim=-1)))).squeeze(-1)


class PPOModel(nn.Module):
    """Independent actor and critic with explicit typed observation inputs."""

    def __init__(
        self,
        *,
        architecture: str = "slot_fusion_conv_v1",
        sequence_channels: int = 64,
        sequence_blocks: int = 8,
        spatial_channels: int = 128,
        spatial_blocks: int = 4,
        head_channels: int = 128,
        head_blocks: int = 2,
        channels: int | None = None,
        blocks: int | None = None,
    ) -> None:
        super().__init__()
        if architecture != "slot_fusion_conv_v1":
            raise ValueError(f"unsupported AHC063 architecture: {architecture}")
        if channels is not None:
            sequence_channels = spatial_channels = head_channels = channels
        if blocks is not None:
            sequence_blocks = spatial_blocks = head_blocks = blocks
        self.architecture = architecture
        self.ACTION_COUNT = ACTION_COUNT
        self.policy = _Actor(
            sequence_channels,
            sequence_blocks,
            spatial_channels,
            spatial_blocks,
            head_channels,
            head_blocks,
        )
        self.value = _Critic(
            sequence_channels,
            sequence_blocks,
            spatial_channels,
            spatial_blocks,
            head_channels,
            head_blocks,
        )
        validate_modula_graph(self.modula_graph())

    def modula_graph(self) -> ModulaGraphNode:
        return ModulaGraphNode(
            "parallel",
            (
                module_to_modula_graph(self.policy, "policy"),
                module_to_modula_graph(self.value, "value"),
            ),
        )

    def forward(
        self,
        board_food: torch.Tensor,
        board_features: torch.Tensor,
        slot_colors: torch.Tensor,
        slot_positions: torch.Tensor,
        global_features: torch.Tensor,
        previous_action: torch.Tensor,
        action_colors: torch.Tensor,
        action_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = (
            board_food,
            board_features,
            slot_colors,
            slot_positions,
            global_features,
            previous_action,
            action_colors,
            action_features,
            mask,
        )
        return self.policy(*inputs), self.value(*inputs[:6])

    def policy_logits(
        self,
        board_food: torch.Tensor,
        board_features: torch.Tensor,
        slot_colors: torch.Tensor,
        slot_positions: torch.Tensor,
        global_features: torch.Tensor,
        previous_action: torch.Tensor,
        action_colors: torch.Tensor,
        action_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy(
            board_food,
            board_features,
            slot_colors,
            slot_positions,
            global_features,
            previous_action,
            action_colors,
            action_features,
            mask,
        )

    def value_predictions(
        self,
        board_food: torch.Tensor,
        board_features: torch.Tensor,
        slot_colors: torch.Tensor,
        slot_positions: torch.Tensor,
        global_features: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        return self.value(
            board_food,
            board_features,
            slot_colors,
            slot_positions,
            global_features,
            previous_action,
        )

    @torch.no_grad()
    def feature_norm_stats(self, *inputs: torch.Tensor) -> dict[str, float]:
        spatial, _ = self.policy.encoder(*inputs[:6])
        norm = spatial.float().pow(2).sum(dim=1).sqrt()
        return {
            "spatial_feature_norm_mean": float(norm.mean().item()),
            "spatial_feature_norm_std": float(norm.std(unbiased=False).item()),
            "spatial_feature_norm_max": float(norm.max().item()),
        }


__all__ = ["PPOModel"]
