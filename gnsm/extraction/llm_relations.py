"""Schema-constrained LLM relation extraction port."""

from __future__ import annotations

from typing import Protocol

from gnsm.extraction.base import RelationExtraction
from gnsm.schemas import Entity


class StructuredRelationClient(Protocol):
    def extract_relations(self, text: str, entities: list[Entity]) -> RelationExtraction: ...


class LLMRelationStateExtractor:
    def __init__(self, client: StructuredRelationClient) -> None:
        self.client = client

    def extract(self, text: str, entities: dict[str, Entity]) -> RelationExtraction:
        return self.client.extract_relations(text, list(entities.values()))
