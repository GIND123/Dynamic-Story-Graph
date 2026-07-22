"""State encoder interface and deterministic baseline encoder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from gnsm.schemas import Entity, NarrativeState, RelationEdge, SceneGraph
from gnsm.state.diagnostics import collapse_diagnostics


class StateEncoder(Protocol):
    def encode(self, graph: SceneGraph) -> NarrativeState: ...


@dataclass(slots=True)
class HashingStateEncoder:
    """Stable feature-hashing baseline with graph-aware global pooling.

    It is useful for Stage 0 plumbing, tests, and ablations. It is not intended
    to replace the ModernBERT + Graph Transformer research model.
    """

    dimension: int = 128

    def encode(self, graph: SceneGraph) -> NarrativeState:
        node_vectors = {
            entity_id: self._entity_vector(entity, graph)
            for entity_id, entity in graph.entities.items()
        }
        if node_vectors:
            matrix = np.stack(list(node_vectors.values()))
            global_vector = matrix.mean(axis=0, dtype=np.float32)
            edge_context = self._edge_context(graph.edges)
            global_vector = self._normalize(global_vector + edge_context)
            diagnostics = collapse_diagnostics(matrix)
        else:
            global_vector = np.zeros(self.dimension, dtype=np.float32)
            diagnostics = {"effective_rank": 0.0, "mean_variance": 0.0, "min_variance": 0.0}
        return NarrativeState(
            scene_id=graph.scene_id,
            global_vector=global_vector,
            node_vectors=node_vectors,
            graph=graph,
            diagnostics=diagnostics,
        )

    def _entity_vector(self, entity: Entity, graph: SceneGraph) -> NDArray[np.float32]:
        tokens = [entity.id, entity.name, entity.type.value, *entity.aliases]
        tokens.extend(
            f"attr:{attribute.key}:{attribute.value}"
            for attribute in graph.attributes
            if attribute.entity_id == entity.id
        )
        tokens.extend(
            f"edge:{edge.type.value}:{edge.target}:{edge.polarity:.2f}"
            for edge in graph.edges
            if edge.source == entity.id
        )
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            index, sign = self._hash(token)
            vector[index] += sign
        return self._normalize(vector)

    def _edge_context(self, edges: list[RelationEdge]) -> NDArray[np.float32]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for edge in edges:
            index, sign = self._hash(f"{edge.source}|{edge.type}|{edge.target}")
            vector[index] += sign * edge.confidence
        return self._normalize(vector)

    def _hash(self, value: str) -> tuple[int, float]:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        integer = int.from_bytes(digest, byteorder="big", signed=False)
        return integer % self.dimension, 1.0 if integer & 1 else -1.0

    @staticmethod
    def _normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)
