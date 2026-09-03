"""Core contracts for revision-aware, prefix-causal narrative state.

Every object here is *reader-time indexed*: it records not only what is
asserted about the story world but **when in the discourse** a reader could
first have known it, and whether that belief still stands. This is what lets
the store distinguish a story-world change from a correction of the reader's
model, and what makes the state auditable after the fact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any


class CommitLevel(StrEnum):
    """How firmly the state is committed to a node or assertion.

    ``PROVISIONAL`` structure is *deliberately underspecified*: it may be
    refined or absorbed later by a monotone ``ELABORATE`` step. ``COMMITTED``
    structure can only be changed non-monotonically, which is the expensive
    case the calculus exists to avoid.
    """

    PROVISIONAL = "provisional"
    COMMITTED = "committed"


class Status(StrEnum):
    BELIEVED = "believed"
    SUPERSEDED = "superseded"      # was true, story world moved on
    RETRACTED = "retracted"        # never was true; reader was misled
    MERGED_AWAY = "merged_away"    # node turned out to be another node


class Op(StrEnum):
    """The revision calculus.

    ``ASSERT``/``ELABORATE``/``MERGE_PROVISIONAL`` are monotone: earlier state
    stays a valid substructure of later state. ``SUPERSEDE`` preserves history
    by closing a validity interval rather than deleting. ``REVISE``,
    ``MERGE_COMMITTED`` and ``SPLIT`` are non-monotone rollbacks -- the
    operations a system with no revision layer simply cannot express.
    """

    ASSERT = "assert"
    ELABORATE = "elaborate"
    MERGE_PROVISIONAL = "merge_provisional"
    SUPERSEDE = "supersede"
    REVISE = "revise"
    MERGE_COMMITTED = "merge_committed"
    SPLIT = "split"
    NOOP = "noop"


MONOTONE_OPS: frozenset[Op] = frozenset(
    {Op.ASSERT, Op.ELABORATE, Op.MERGE_PROVISIONAL, Op.NOOP}
)
ROLLBACK_OPS: frozenset[Op] = frozenset({Op.REVISE, Op.MERGE_COMMITTED, Op.SPLIT})


class Certainty(StrEnum):
    """Evidential source of an assertion, which drives revision classification.

    A fact the narrator states outranks one a character merely claims; when the
    two collide, the reader's earlier belief was wrong (``REVISE``) rather than
    the world having changed (``SUPERSEDE``).
    """

    NARRATED = "narrated"      # the narration asserts it
    REPORTED = "reported"      # a character asserts it (may be wrong or lying)
    IMPLIED = "implied"        # inferred, not stated


@dataclass(frozen=True, slots=True)
class Span:
    """A character span in the source text -- the provenance of every belief."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"malformed span ({self.start}, {self.end})")

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(slots=True)
class EntityNode:
    """A character (or other entity) as the reader currently understands them."""

    id: str
    canonical: str
    kind: str = "character"
    commit: CommitLevel = CommitLevel.PROVISIONAL
    status: Status = Status.BELIEVED
    surfaces: dict[str, int] = field(default_factory=dict)
    proper_names: set[str] = field(default_factory=set)
    first_seen: int = 0
    last_seen: int = 0
    merged_into: str | None = None
    provenance: list[Span] = field(default_factory=list)

    @property
    def live(self) -> bool:
        return self.status is Status.BELIEVED

    def observe(self, surface: str, index: int, span: Span | None, proper: bool) -> None:
        self.surfaces[surface] = self.surfaces.get(surface, 0) + 1
        if proper:
            self.proper_names.add(surface)
        self.last_seen = max(self.last_seen, index)
        if span is not None and len(self.provenance) < 64:
            self.provenance.append(span)

    def mention_count(self) -> int:
        return sum(self.surfaces.values())


@dataclass(slots=True)
class Assertion:
    """One time-indexed belief about the story world.

    Two clocks are tracked deliberately. ``d_open``/``d_close`` are *discourse*
    time -- when the reader came to and stopped holding the belief. ``valid_from``
    /``valid_to`` are *story* time -- when the fact held in the world. A reveal
    moves the first without moving the second; a plot event moves the second.
    """

    id: str
    subject: str
    predicate: str
    object: str
    polarity: bool = True
    certainty: Certainty = Certainty.NARRATED
    commit: CommitLevel = CommitLevel.PROVISIONAL
    status: Status = Status.BELIEVED
    confidence: float = 0.5
    d_open: int = 0
    d_close: int | None = None
    valid_from: int | None = None
    valid_to: int | None = None
    provenance: Span | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    retracted_by: str | None = None
    derived_from: tuple[str, ...] = ()

    @property
    def live(self) -> bool:
        return self.status is Status.BELIEVED

    @property
    def slot(self) -> tuple[str, str]:
        return self.subject, self.predicate

    def key(self) -> tuple[str, str, str, bool]:
        return self.subject, self.predicate, self.object, self.polarity


@dataclass(frozen=True, slots=True)
class UpdateRecord:
    """One entry in the revision log -- the auditable trace of how state moved."""

    index: int
    op: Op
    target: str
    detail: str = ""
    payload: str | None = None

    @property
    def monotone(self) -> bool:
        return self.op in MONOTONE_OPS

    @property
    def rollback(self) -> bool:
        return self.op in ROLLBACK_OPS


@dataclass(frozen=True, slots=True)
class Violation:
    """A structural invariant that the state failed to satisfy."""

    index: int
    code: str
    detail: str


@dataclass(slots=True)
class Window:
    """One prefix-causal reading step."""

    index: int
    start: int
    end: int
    text: str

    @property
    def span(self) -> Span:
        return Span(self.start, self.end)


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("␟".join(str(p) for p in parts).encode()).hexdigest()
    return f"{prefix}{digest[:12]}"


def evolve(obj: Any, **changes: Any) -> Any:
    return replace(obj, **changes)
