"""End-to-end GNSM facade spanning extraction, state, generation, and verification."""

from __future__ import annotations

from dataclasses import dataclass

from gnsm.extraction.pipeline import ExtractionPipeline
from gnsm.generation.base import SceneGenerator
from gnsm.generation.reference import TemplateSceneGenerator
from gnsm.schemas import NarrativeState, PlotAction, SceneGraph, VerificationReport
from gnsm.state.encoder import HashingStateEncoder, StateEncoder
from gnsm.state.graph_ops import apply_delta
from gnsm.state.transition import RuleBasedTransitionModel, StateTransitionModel
from gnsm.verifier.consistency import ConsistencyVerifier
from gnsm.verifier.controller import GenerationController


@dataclass(slots=True)
class SceneResult:
    text: str
    graph: SceneGraph
    state: NarrativeState
    verification: VerificationReport
    attempts: int


@dataclass(slots=True)
class GNSMSystem:
    extractor: ExtractionPipeline
    encoder: StateEncoder
    transition_model: StateTransitionModel
    controller: GenerationController

    @classmethod
    def reference(
        cls,
        *,
        dimension: int = 128,
        max_regenerations: int = 2,
        generator: SceneGenerator | None = None,
    ) -> GNSMSystem:
        extractor = ExtractionPipeline.reference()
        selected_generator = generator or TemplateSceneGenerator()
        return cls(
            extractor=extractor,
            encoder=HashingStateEncoder(dimension),
            transition_model=RuleBasedTransitionModel(),
            controller=GenerationController(
                generator=selected_generator,
                extractor=extractor,
                verifier=ConsistencyVerifier(),
                max_regenerations=max_regenerations,
            ),
        )

    def initialize(self, scene_text: str, scene_id: str = "scene-0") -> NarrativeState:
        graph = self.extractor.extract(scene_text, scene_id=scene_id)
        return self.encoder.encode(graph)

    def generate_next(
        self,
        *,
        previous_scene: str,
        state: NarrativeState,
        action: PlotAction,
        next_scene_id: str,
        rolling_summary: str = "",
    ) -> SceneResult:
        transition = self.transition_model.predict(state, action)
        draft = self.controller.run(
            previous_scene=previous_scene,
            rolling_summary=rolling_summary,
            action=action,
            state=state,
            transition=transition,
            next_scene_id=next_scene_id,
        )
        if draft.report.accepted:
            next_graph = apply_delta(
                state.graph, transition.predicted_delta, scene_id=next_scene_id
            )
            next_graph.entities.update(draft.graph.entities)
            next_graph.mentions = draft.graph.mentions
            next_graph.quotes = draft.graph.quotes
            next_graph.summary = draft.graph.summary
            next_state = self.encoder.encode(next_graph)
        else:
            next_graph = draft.graph
            next_state = self.encoder.encode(next_graph)
        return SceneResult(
            text=draft.text,
            graph=next_graph,
            state=next_state,
            verification=draft.report,
            attempts=draft.attempts,
        )
