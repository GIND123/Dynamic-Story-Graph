"""Supervised state-transition interface and executable rule baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from gnsm.schemas import GraphDelta, NarrativeState, PlotAction, TransitionPrediction
from gnsm.state.graph_ops import apply_delta


class StateTransitionModel(Protocol):
    def predict(self, state: NarrativeState, action: PlotAction) -> TransitionPrediction: ...


@dataclass(slots=True)
class RuleBasedTransitionModel:
    """Apply an outline-provided structured delta and perturb the latent state."""

    action_scale: float = 0.05

    def predict(self, state: NarrativeState, action: PlotAction) -> TransitionPrediction:
        delta = action.proposed_delta or GraphDelta()
        action_signal = self._action_signal(action.intent, state.dimension)
        next_vector = state.global_vector + self.action_scale * action_signal
        norm = float(np.linalg.norm(next_vector))
        if norm:
            next_vector = next_vector / norm
        confidence = 0.9 if action.proposed_delta is not None else 0.5
        return TransitionPrediction(
            next_vector=next_vector.astype(np.float32),
            predicted_delta=delta,
            confidence=confidence,
        )

    def predicted_graph(self, state: NarrativeState, action: PlotAction) -> object:
        prediction = self.predict(state, action)
        return apply_delta(state.graph, prediction.predicted_delta)

    @staticmethod
    def _action_signal(intent: str, dimension: int) -> np.ndarray:
        vector = np.zeros(dimension, dtype=np.float32)
        encoded = intent.encode("utf-8")
        for index, byte in enumerate(encoded):
            vector[(index * 31 + byte) % dimension] += 1.0 if byte & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector
