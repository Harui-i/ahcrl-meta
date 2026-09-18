import torch

from ahcrl.contests.ahc063.encoder import (
    ACTION_FEATURE_COUNT,
    BOARD_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT,
    MAX_SEQUENCE_LENGTH,
)
from ahcrl.contests.ahc063.model import PPOModel
from ahcrl.nn.fusion_blocks import SequenceStack, SpatialStack
from ahcrl.nn.modula import module_to_modula_graph


def observations(batch: int = 2) -> tuple[torch.Tensor, ...]:
    board_food = torch.zeros(batch, 16, 16, dtype=torch.uint8)
    board_features = torch.zeros(batch, BOARD_FEATURE_COUNT, 16, 16)
    board_features[:, 0] = 1
    board_features[:, 1, 4, 0] = 1
    board_features[:, 2, 4, 0] = 1
    slot_colors = torch.zeros(batch, MAX_SEQUENCE_LENGTH, 2, dtype=torch.uint8)
    slot_colors[:, :5, 0] = 1
    slot_colors[:, :5, 1] = 1
    slot_positions = torch.full((batch, MAX_SEQUENCE_LENGTH, 2), 255, dtype=torch.uint8)
    slot_positions[:, :5] = torch.tensor([[4, 0], [3, 0], [2, 0], [1, 0], [0, 0]])
    global_features = torch.zeros(batch, GLOBAL_FEATURE_COUNT)
    global_features[:, :4] = torch.tensor([0.5, 0.1, 0.4, 0.2])
    previous_action = torch.zeros(batch, 1, dtype=torch.uint8)
    action_colors = torch.zeros(batch, 4, dtype=torch.uint8)
    action_features = torch.zeros(batch, 4, ACTION_FEATURE_COUNT)
    mask = torch.ones(batch, 4, dtype=torch.uint8)
    return (
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


def small_model() -> PPOModel:
    return PPOModel(
        sequence_channels=8,
        sequence_blocks=2,
        spatial_channels=8,
        spatial_blocks=2,
        head_channels=8,
        head_blocks=2,
    )


def test_typed_forward_and_core_value_io() -> None:
    model = small_model()
    inputs = observations()
    logits, value = model(*inputs)
    assert logits.shape == (2, 4)
    assert value.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.allclose(value, model.value_predictions(*inputs[:6]))


def test_actor_and_critic_do_not_share_parameters() -> None:
    model = small_model()
    actor = {id(parameter) for parameter in model.policy.parameters()}
    critic = {id(parameter) for parameter in model.value.parameters()}
    assert actor.isdisjoint(critic)


def test_modula_stack_tare_preserves_sensitivity_and_mass() -> None:
    for module in (SequenceStack(8, 2), SpatialStack(8, 2)):
        graph = module_to_modula_graph(module)
        assert graph.mass == 5
        assert graph.sensitivity == 1


def test_slot_order_changes_logits() -> None:
    model = small_model().eval()
    inputs = list(observations(1))
    with torch.no_grad():
        baseline = model.policy_logits(*inputs)
        inputs[2] = inputs[2].clone()
        inputs[2][:, 0, 0] = 2
        changed = model.policy_logits(*inputs)
    assert not torch.allclose(baseline, changed)


def test_bfloat16_and_compile_forward() -> None:
    model = small_model().to(dtype=torch.bfloat16).eval()
    inputs = list(observations(1))
    inputs[1] = inputs[1].to(torch.float16)
    inputs[4] = inputs[4].to(torch.float16)
    inputs[7] = inputs[7].to(torch.float16)
    logits, value = model(*inputs)
    assert logits.dtype == torch.bfloat16
    assert value.dtype == torch.bfloat16
    compiled = torch.compile(model, backend="eager")
    compiled(*inputs)
