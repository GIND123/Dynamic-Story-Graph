"""Canonical, serializable contracts shared by all three GNSM planes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class EntityType(StrEnum):
    CHARACTER = "character"
    LOCATION = "location"
    OBJECT = "object"
    ORGANIZATION = "organization"
    CONCEPT = "concept"


class EdgeType(StrEnum):
    LOCATED_AT = "located_at"
    POSSESSES = "possesses"
    RELATES_TO = "relates_to"
    SPEAKS_TO = "speaks_to"
    KNOWS_ABOUT = "knows_about"
    MEMBER_OF = "member_of"
    CONTROLS = "controls"
    FAMILY_OF = "family_of"


class DeltaOperation(StrEnum):
    ADD = "add"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class Mention:
    text: str
    start: int
    end: int
    entity_id: str


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    name: str
    type: EntityType
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RelationEdge:
    source: str
    target: str
    type: EdgeType
    polarity: float = 0.0
    confidence: float = 1.0
    properties: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> tuple[str, str, EdgeType]:
        return self.source, self.target, self.type


@dataclass(frozen=True, slots=True)
class AttributeState:
    entity_id: str
    key: str
    value: str
    confidence: float = 1.0

    @property
    def key_tuple(self) -> tuple[str, str]:
        return self.entity_id, self.key


@dataclass(frozen=True, slots=True)
class QuoteAttribution:
    quote: str
    start: int
    end: int
    speaker_id: str | None = None
    addressee_id: str | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class SceneGraph:
    scene_id: str
    entities: dict[str, Entity] = field(default_factory=dict)
    mentions: list[Mention] = field(default_factory=list)
    edges: list[RelationEdge] = field(default_factory=list)
    attributes: list[AttributeState] = field(default_factory=list)
    quotes: list[QuoteAttribution] = field(default_factory=list)
    summary: str = ""

    def edge_index(self) -> dict[tuple[str, str, EdgeType], RelationEdge]:
        return {edge.key: edge for edge in self.edges}

    def attribute_index(self) -> dict[tuple[str, str], AttributeState]:
        return {attribute.key_tuple: attribute for attribute in self.attributes}

    def validate(self) -> list[str]:
        errors: list[str] = []
        for edge in self.edges:
            if edge.source not in self.entities:
                errors.append(f"edge source {edge.source!r} is not registered")
            if edge.target not in self.entities:
                errors.append(f"edge target {edge.target!r} is not registered")
        for attribute in self.attributes:
            if attribute.entity_id not in self.entities:
                errors.append(f"attribute entity {attribute.entity_id!r} is not registered")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EdgeDelta:
    operation: DeltaOperation
    edge: RelationEdge


@dataclass(frozen=True, slots=True)
class AttributeDelta:
    entity_id: str
    key: str
    old_value: str | None
    new_value: str | None


@dataclass(slots=True)
class GraphDelta:
    edge_changes: list[EdgeDelta] = field(default_factory=list)
    attribute_changes: list[AttributeDelta] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.edge_changes and not self.attribute_changes


@dataclass(frozen=True, slots=True)
class PlotAction:
    intent: str
    participants: tuple[str, ...] = ()
    proposed_delta: GraphDelta | None = None


@dataclass(slots=True)
class NarrativeState:
    scene_id: str
    global_vector: NDArray[np.float32]
    node_vectors: dict[str, NDArray[np.float32]]
    graph: SceneGraph
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return int(self.global_vector.shape[0])


@dataclass(slots=True)
class TransitionPrediction:
    next_vector: NDArray[np.float32]
    predicted_delta: GraphDelta
    confidence: float


class IssueKind(StrEnum):
    DEAD_CHARACTER_ACTS = "dead_character_acts"
    WRONG_LOCATION = "wrong_location"
    OBJECT_TELEPORT = "object_teleport"
    RELATION_VIOLATION = "relation_violation"
    MISSING_EXPECTED_CHANGE = "missing_expected_change"
    UNEXPECTED_CHANGE = "unexpected_change"


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    kind: IssueKind
    message: str
    entity_ids: tuple[str, ...] = ()
    severity: float = 1.0


@dataclass(slots=True)
class VerificationReport:
    accepted: bool
    issues: list[VerificationIssue] = field(default_factory=list)
    precision_denominator: int = 0
    recall_denominator: int = 0

    def corrective_constraint(self) -> str:
        if self.accepted:
            return ""
        return "Revise the scene to satisfy canon:\n- " + "\n- ".join(
            issue.message for issue in self.issues
        )
