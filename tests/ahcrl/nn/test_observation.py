from typing import cast

import pytest
import torch

from ahcrl.nn.modula import ModularEmbedding
from ahcrl.nn.observation import CategoricalPlaneAdapter, CategoricalPlaneGroup


def test_categorical_adapter_without_groups_is_identity() -> None:
    adapter = CategoricalPlaneAdapter(3, ())
    x = torch.randn(2, 3, 4, 4)

    assert adapter.output_channels == 3
    assert torch.equal(adapter(x), x)


def test_categorical_adapter_replaces_groups_and_preserves_pass_through_order() -> None:
    groups = (
        CategoricalPlaneGroup("first", 1, 2, embedding_dim=3),
        CategoricalPlaneGroup("second", 4, 2, implicit_zero=True, embedding_dim=3),
    )
    adapter = CategoricalPlaneAdapter(6, groups)
    with torch.no_grad():
        cast(ModularEmbedding, adapter.embeddings[0]).weight.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        cast(ModularEmbedding, adapter.embeddings[1]).weight.copy_(
            torch.arange(9, dtype=torch.float32).reshape(3, 3)
        )

    x = torch.zeros(1, 6, 1, 1)
    x[0, 0, 0, 0] = 10.0
    x[0, 2, 0, 0] = 1.0
    x[0, 3, 0, 0] = 20.0
    x[0, 5, 0, 0] = 1.0

    output = adapter(x)

    expected = torch.tensor([10.0, 3.0, 4.0, 5.0, 20.0, 6.0, 7.0, 8.0]).reshape(1, 8, 1, 1)
    assert adapter.output_channels == 8
    assert torch.equal(output, expected)


def test_categorical_adapter_implicit_zero_is_leading_class() -> None:
    group = CategoricalPlaneGroup("level", 0, 2, implicit_zero=True, embedding_dim=3)
    adapter = CategoricalPlaneAdapter(2, (group,))
    with torch.no_grad():
        cast(ModularEmbedding, adapter.embeddings[0]).weight.copy_(torch.eye(3))

    zeros = adapter(torch.zeros(1, 2, 2, 2))
    assert torch.equal(zeros[:, :, 0, 0], torch.tensor([[1.0, 0.0, 0.0]]))

    one_hot = torch.zeros(1, 2, 2, 2)
    one_hot[:, 1] = 1.0
    encoded = adapter(one_hot)
    assert torch.equal(encoded[:, :, 0, 0], torch.tensor([[0.0, 0.0, 1.0]]))


@pytest.mark.parametrize(
    "groups",
    [
        (CategoricalPlaneGroup("late", 3, 1), CategoricalPlaneGroup("early", 1, 1)),
        (CategoricalPlaneGroup("left", 1, 2), CategoricalPlaneGroup("overlap", 2, 1)),
        (CategoricalPlaneGroup("outside", 3, 2),),
    ],
)
def test_categorical_adapter_rejects_invalid_group_layout(
    groups: tuple[CategoricalPlaneGroup, ...],
) -> None:
    with pytest.raises(ValueError):
        CategoricalPlaneAdapter(4, groups)
