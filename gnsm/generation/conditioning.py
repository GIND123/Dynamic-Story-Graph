"""Convert structured/latent state into adapter-ready conditioning packets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gnsm.schemas import NarrativeState


@dataclass(frozen=True, slots=True)
class ConditioningPacket:
    global_state: NDArray[np.float32]
    node_states: NDArray[np.float32]
    node_ids: tuple[str, ...]
    symbolic_constraints: tuple[str, ...]


def build_conditioning_packet(
    state: NarrativeState,
    relevant_entity_ids: tuple[str, ...] = (),
) -> ConditioningPacket:
    selected_ids = relevant_entity_ids or tuple(state.node_vectors)
    selected_ids = tuple(entity_id for entity_id in selected_ids if entity_id in state.node_vectors)
    if selected_ids:
        node_states = np.stack([state.node_vectors[entity_id] for entity_id in selected_ids])
    else:
        node_states = np.empty((0, state.dimension), dtype=np.float32)
    attributes = tuple(
        f"{attribute.entity_id}.{attribute.key}={attribute.value}"
        for attribute in state.graph.attributes
    )
    edges = tuple(f"{edge.source} {edge.type.value} {edge.target}" for edge in state.graph.edges)
    return ConditioningPacket(
        global_state=state.global_vector,
        node_states=node_states,
        node_ids=selected_ids,
        symbolic_constraints=attributes + edges,
    )
