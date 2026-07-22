"""Dependency-free generator that makes controller behavior testable."""

from __future__ import annotations

from dataclasses import dataclass

from gnsm.generation.base import GenerationRequest


@dataclass(slots=True)
class TemplateSceneGenerator:
    """Emit a transparent draft from the plot action and current canon."""

    def generate(self, request: GenerationRequest) -> str:
        names = [
            request.state.graph.entities[entity_id].name
            for entity_id in request.action.participants
            if entity_id in request.state.graph.entities
        ]
        subject = " and ".join(names) if names else "The characters"
        lines = [f"{subject} moved into the next moment.", request.action.intent.strip()]
        if request.corrective_constraint:
            lines.append(request.corrective_constraint)
        return " ".join(line for line in lines if line)
