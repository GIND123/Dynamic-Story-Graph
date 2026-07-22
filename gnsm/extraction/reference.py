"""Deterministic reference extractors for tests, demos, and bootstrap labeling.

These deliberately conservative rules are not research-quality replacements
for Maverick/BookNLP/PDNC. They make every downstream interface executable
before model weights and datasets are available.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from gnsm.extraction.base import EntityExtraction, RelationExtraction
from gnsm.schemas import (
    AttributeState,
    EdgeType,
    Entity,
    EntityType,
    Mention,
    QuoteAttribution,
    RelationEdge,
)

_NAME = r"[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)?"
_PLACE = r"[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)?"
_ENTITY_PATTERN = re.compile(rf"\b({_NAME})\b")
_QUOTE_PATTERN = re.compile(r"[\"“](.+?)[\"”]", re.DOTALL)
_LOCATION_MENTION_PATTERN = re.compile(rf"\b(?:in|at|inside)\s+(?:the\s+)?(?P<location>{_PLACE})\b")
_SPEECH_PATTERN = re.compile(
    rf"\b(?P<speaker>{_NAME})\s+(?:said|asked|replied|whispered|shouted)\b",
    re.IGNORECASE,
)
_LOCATION_PATTERN = re.compile(
    rf"\b(?P<name>{_NAME})\s+(?:is|was|stood|waited|remained)\s+(?:in|at|inside)\s+"
    rf"(?:the\s+)?(?P<location>{_PLACE})\b",
)
_STATUS_PATTERN = re.compile(
    rf"\b(?P<name>{_NAME})\s+(?:is|was)\s+(?P<status>alive|dead|missing|injured)\b",
    re.IGNORECASE,
)
_POSSESSION_PATTERN = re.compile(
    rf"\b(?P<name>{_NAME})\s+(?:has|held|carried|possessed)\s+(?:the\s+|an?\s+)?"
    r"(?P<object>[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?)\b",
)
_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "But",
    "For",
    "He",
    "Her",
    "His",
    "I",
    "If",
    "In",
    "It",
    "Its",
    "No",
    "She",
    "The",
    "Their",
    "They",
    "This",
    "We",
    "When",
    "You",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def entity_id(name: str, entity_type: EntityType = EntityType.CHARACTER) -> str:
    return f"{entity_type.value}:{slug(name)}"


class RuleBasedEntityExtractor:
    """Find capitalized proper names and preserve canonical IDs across scenes."""

    def extract(self, text: str, registry: dict[str, Entity] | None = None) -> EntityExtraction:
        known = dict(registry or {})
        by_alias = self._alias_index(known.values())
        mentions: list[Mention] = []
        quote_spans = [(match.start(1), match.end(1)) for match in _QUOTE_PATTERN.finditer(text)]
        location_spans = {
            (match.start("location"), match.end("location"))
            for match in _LOCATION_MENTION_PATTERN.finditer(text)
        }

        for match in _ENTITY_PATTERN.finditer(text):
            name = match.group(1).strip()
            if name in _STOPWORDS:
                continue
            if any(start <= match.start(1) < end for start, end in quote_spans):
                continue
            inferred_type = (
                EntityType.LOCATION
                if (match.start(1), match.end(1)) in location_spans
                else EntityType.CHARACTER
            )
            canonical_id = by_alias.get(name.casefold())
            if canonical_id is None:
                canonical_id = entity_id(name, inferred_type)
                known[canonical_id] = Entity(
                    id=canonical_id,
                    name=name,
                    type=inferred_type,
                )
                by_alias[name.casefold()] = canonical_id
            mentions.append(
                Mention(
                    text=name,
                    start=match.start(1),
                    end=match.end(1),
                    entity_id=canonical_id,
                )
            )
        return EntityExtraction(entities=known, mentions=mentions)

    @staticmethod
    def _alias_index(entities: Iterable[Entity]) -> dict[str, str]:
        result: dict[str, str] = {}
        for entity in entities:
            result[entity.name.casefold()] = entity.id
            result.update({alias.casefold(): entity.id for alias in entity.aliases})
        return result


class RuleBasedQuoteAttributor:
    """Attach quotes to the nearest explicit speech attribution."""

    def attribute(self, text: str, entities: dict[str, Entity]) -> list[QuoteAttribution]:
        name_to_id = {entity.name.casefold(): entity.id for entity in entities.values()}
        result: list[QuoteAttribution] = []
        speech_mentions = list(_SPEECH_PATTERN.finditer(text))
        for quote in _QUOTE_PATTERN.finditer(text):
            candidates = [
                mention
                for mention in speech_mentions
                if abs(mention.start() - quote.start()) <= 160
            ]
            nearest = min(
                candidates,
                key=lambda mention: abs(mention.start() - quote.start()),
                default=None,
            )
            speaker_id = None
            confidence = 0.0
            if nearest is not None:
                speaker_id = name_to_id.get(nearest.group("speaker").casefold())
                confidence = 0.85 if speaker_id else 0.25
            result.append(
                QuoteAttribution(
                    quote=quote.group(1),
                    start=quote.start(1),
                    end=quote.end(1),
                    speaker_id=speaker_id,
                    confidence=confidence,
                )
            )
        return result


class RuleBasedRelationExtractor:
    """Extract a minimal typed graph from unambiguous surface patterns."""

    def extract(self, text: str, entities: dict[str, Entity]) -> RelationExtraction:
        expanded = dict(entities)
        by_name = {entity.name.casefold(): entity.id for entity in expanded.values()}
        edges: list[RelationEdge] = []
        attributes: list[AttributeState] = []

        for match in _STATUS_PATTERN.finditer(text):
            subject = by_name.get(match.group("name").casefold())
            if subject:
                attributes.append(
                    AttributeState(subject, "status", match.group("status").casefold(), 0.95)
                )

        for match in _LOCATION_PATTERN.finditer(text):
            subject = by_name.get(match.group("name").casefold())
            if not subject:
                continue
            location_name = match.group("location").strip().title()
            location_id = entity_id(location_name, EntityType.LOCATION)
            expanded.setdefault(
                location_id,
                Entity(location_id, location_name, EntityType.LOCATION),
            )
            edges.append(RelationEdge(subject, location_id, EdgeType.LOCATED_AT, confidence=0.9))

        for match in _POSSESSION_PATTERN.finditer(text):
            subject = by_name.get(match.group("name").casefold())
            if not subject:
                continue
            object_name = match.group("object").strip()
            object_id = entity_id(object_name, EntityType.OBJECT)
            expanded.setdefault(object_id, Entity(object_id, object_name, EntityType.OBJECT))
            edges.append(RelationEdge(subject, object_id, EdgeType.POSSESSES, confidence=0.8))

        return RelationExtraction(entities=expanded, edges=edges, attributes=attributes)
