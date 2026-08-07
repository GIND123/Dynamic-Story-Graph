"""Grounded training objectives for C4-C5."""

from __future__ import annotations

from dataclasses import dataclass

from gnsm.exceptions import OptionalDependencyError

try:
    import torch
    from torch import Tensor
    from torch.nn import functional as functional
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    functional = None


@dataclass(frozen=True, slots=True)
class LossWeights:
    graph_reconstruction: float = 1.0
    attributes: float = 0.5
    transition_delta: float = 1.0
    latent_regression: float = 0.1
    supervised_contrastive: float = 0.2
    emotion: float = 0.3


def grounded_state_loss(
    *,
    edge_logits: Tensor,
    edge_labels: Tensor,
    attribute_logits: Tensor,
    attribute_labels: Tensor,
    delta_logits: Tensor,
    delta_labels: Tensor,
    predicted_state: Tensor,
    encoded_next_state: Tensor,
    emotion_logits: Tensor | None = None,
    emotion_labels: Tensor | None = None,
    weights: LossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if functional is None:
        raise OptionalDependencyError("Training losses require the `training` extra.")
    selected_weights = weights or LossWeights()
    components = {
        "graph_reconstruction": functional.cross_entropy(edge_logits, edge_labels),
        "attributes": functional.cross_entropy(attribute_logits, attribute_labels),
        "transition_delta": functional.cross_entropy(delta_logits, delta_labels),
        "latent_regression": functional.mse_loss(predicted_state, encoded_next_state.detach()),
    }
    total = (
        selected_weights.graph_reconstruction * components["graph_reconstruction"]
        + selected_weights.attributes * components["attributes"]
        + selected_weights.transition_delta * components["transition_delta"]
        + selected_weights.latent_regression * components["latent_regression"]
    )
    if emotion_logits is not None and emotion_labels is not None:
        components["emotion"] = functional.cross_entropy(emotion_logits, emotion_labels)
        total = total + selected_weights.emotion * components["emotion"]
    return total, components
