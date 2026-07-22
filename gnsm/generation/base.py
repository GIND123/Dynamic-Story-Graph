"""Frozen generator contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from gnsm.schemas import NarrativeState, PlotAction, TransitionPrediction


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    previous_scene: str
    rolling_summary: str
    action: PlotAction
    state: NarrativeState
    transition: TransitionPrediction
    corrective_constraint: str = ""


class SceneGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> str: ...
