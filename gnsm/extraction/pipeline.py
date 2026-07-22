"""C1-C3 orchestration into one validated scene graph."""

from __future__ import annotations

from dataclasses import dataclass

from gnsm.extraction.base import (
    EntityCoreferenceExtractor,
    QuoteSpeakerAttributor,
    RelationStateExtractor,
)
from gnsm.extraction.reference import (
    RuleBasedEntityExtractor,
    RuleBasedQuoteAttributor,
    RuleBasedRelationExtractor,
)
from gnsm.schemas import EdgeType, Entity, RelationEdge, SceneGraph


@dataclass(slots=True)
class ExtractionPipeline:
    entity_extractor: EntityCoreferenceExtractor
    quote_attributor: QuoteSpeakerAttributor
    relation_extractor: RelationStateExtractor

    @classmethod
    def reference(cls) -> ExtractionPipeline:
        return cls(
            entity_extractor=RuleBasedEntityExtractor(),
            quote_attributor=RuleBasedQuoteAttributor(),
            relation_extractor=RuleBasedRelationExtractor(),
        )

    def extract(
        self,
        text: str,
        scene_id: str,
        registry: dict[str, Entity] | None = None,
        summary: str = "",
    ) -> SceneGraph:
        entity_result = self.entity_extractor.extract(text, registry)
        relation_result = self.relation_extractor.extract(text, entity_result.entities)
        quotes = self.quote_attributor.attribute(text, relation_result.entities)
        edges = list(relation_result.edges)
        edges.extend(
            RelationEdge(
                source=quote.speaker_id,
                target=quote.addressee_id,
                type=EdgeType.SPEAKS_TO,
                confidence=quote.confidence,
            )
            for quote in quotes
            if quote.speaker_id and quote.addressee_id
        )
        graph = SceneGraph(
            scene_id=scene_id,
            entities=relation_result.entities,
            mentions=entity_result.mentions,
            edges=_deduplicate_edges(edges),
            attributes=relation_result.attributes,
            quotes=quotes,
            summary=summary,
        )
        errors = graph.validate()
        if errors:
            raise ValueError("invalid extracted graph: " + "; ".join(errors))
        return graph


def _deduplicate_edges(edges: list[RelationEdge]) -> list[RelationEdge]:
    by_key: dict[tuple[str, str, EdgeType], RelationEdge] = {}
    for edge in edges:
        previous = by_key.get(edge.key)
        if previous is None or edge.confidence > previous.confidence:
            by_key[edge.key] = edge
    return list(by_key.values())
