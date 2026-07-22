"""Ports implemented by BookNLP/Maverick, PDNC, and relation extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from gnsm.schemas import (
    AttributeState,
    Entity,
    Mention,
    QuoteAttribution,
    RelationEdge,
)


@dataclass(slots=True)
class EntityExtraction:
    entities: dict[str, Entity] = field(default_factory=dict)
    mentions: list[Mention] = field(default_factory=list)


@dataclass(slots=True)
class RelationExtraction:
    entities: dict[str, Entity] = field(default_factory=dict)
    edges: list[RelationEdge] = field(default_factory=list)
    attributes: list[AttributeState] = field(default_factory=list)


class EntityCoreferenceExtractor(Protocol):
    def extract(self, text: str, registry: dict[str, Entity] | None = None) -> EntityExtraction: ...


class QuoteSpeakerAttributor(Protocol):
    def attribute(self, text: str, entities: dict[str, Entity]) -> list[QuoteAttribution]: ...


class RelationStateExtractor(Protocol):
    def extract(self, text: str, entities: dict[str, Entity]) -> RelationExtraction: ...
