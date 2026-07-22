"""Generate, re-extract, verify, and apply targeted corrections."""

from __future__ import annotations

from dataclasses import dataclass

from gnsm.extraction.pipeline import ExtractionPipeline
from gnsm.generation.base import GenerationRequest, SceneGenerator
from gnsm.schemas import (
    Entity,
    NarrativeState,
    PlotAction,
    SceneGraph,
    TransitionPrediction,
    VerificationReport,
)
from gnsm.state.graph_ops import apply_delta
from gnsm.verifier.consistency import ConsistencyVerifier


@dataclass(slots=True)
class ControlledDraft:
    text: str
    graph: SceneGraph
    report: VerificationReport
    attempts: int


@dataclass(slots=True)
class GenerationController:
    generator: SceneGenerator
    extractor: ExtractionPipeline
    verifier: ConsistencyVerifier
    max_regenerations: int = 2

    def run(
        self,
        *,
        previous_scene: str,
        rolling_summary: str,
        action: PlotAction,
        state: NarrativeState,
        transition: TransitionPrediction,
        next_scene_id: str,
    ) -> ControlledDraft:
        correction = ""
        registry = self._predicted_registry(state.graph, transition, next_scene_id)
        last_text = ""
        last_graph = state.graph
        last_report = VerificationReport(accepted=False)
        for attempt in range(1, self.max_regenerations + 2):
            request = GenerationRequest(
                previous_scene=previous_scene,
                rolling_summary=rolling_summary,
                action=action,
                state=state,
                transition=transition,
                corrective_constraint=correction,
            )
            last_text = self.generator.generate(request)
            last_graph = self.extractor.extract(
                last_text,
                scene_id=next_scene_id,
                registry=registry,
            )
            last_report = self.verifier.verify(state.graph, last_graph, transition.predicted_delta)
            if last_report.accepted:
                return ControlledDraft(last_text, last_graph, last_report, attempt)
            correction = last_report.corrective_constraint()
        return ControlledDraft(last_text, last_graph, last_report, self.max_regenerations + 1)

    @staticmethod
    def _predicted_registry(
        graph: SceneGraph,
        transition: TransitionPrediction,
        next_scene_id: str,
    ) -> dict[str, Entity]:
        predicted = apply_delta(graph, transition.predicted_delta, scene_id=next_scene_id)
        return predicted.entities
