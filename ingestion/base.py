"""Normalized intermediate representation (IR) and the DatasetLoader contract.

Every concrete loader (PDNC, LitBank, ...) parses its own on-disk format and
returns a `DatasetIR`. `project.project_to_graph` consumes only the IR, so the
projection logic is dataset-agnostic and each loader is independently testable
against tiny fixtures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

# Entity types map 1:1 to graph node labels (Character/Location/Organization/Object).
EntityType = Literal["CHARACTER", "LOCATION", "ORGANIZATION", "OBJECT"]

# PDNC quotation types, normalized to lowercase.
QuoteType = Literal["explicit", "implicit", "anaphoric"]


@dataclass
class Entity:
    """A resolved entity: one real-world entity = one record (= one node).

    `aliases` are the alternative surface forms collapsed into this entity by
    the loader's entity resolver (PDNC character aliases, LitBank coref chains).
    """

    name: str
    type: EntityType
    aliases: list[str] = field(default_factory=list)


@dataclass
class Event:
    """A narrative event with its participants and a global ordering index."""

    summary: str
    participant_names: list[str] = field(default_factory=list)
    order: int = 0


@dataclass
class Quotation:
    """A dialogue act: who spoke, to whom, and the quote text.

    speaker_name and addressee_names are already resolved to canonical entity
    names by the loader. `ref` is the dataset's stable id for the quote.
    """

    speaker_name: str
    addressee_names: list[str]
    quote_type: QuoteType
    text: str
    ref: str


@dataclass
class Segment:
    """An ordered passage-like text chunk; `index` is its position in the work."""

    index: int
    text: str


@dataclass
class DatasetIR:
    """The full normalized payload a loader returns."""

    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    quotations: list[Quotation] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)


class DatasetLoader(ABC):
    """Abstract loader: parse a dataset on disk into a normalized `DatasetIR`."""

    @abstractmethod
    def load(self) -> DatasetIR:
        """Parse the dataset and return its normalized intermediate representation."""
        raise NotImplementedError

    @abstractmethod
    def namespace(self) -> str:
        """Return the manuscript_id namespace for this source.

        e.g. 'pdnc:PrideAndPrejudice' or 'litbank:1023_bleak_house'. Used by
        project_to_graph so eval data is isolated from generation manuscripts.
        """
        raise NotImplementedError


def chunk_text(text: str, max_chars: int = 1200) -> list[Segment]:
    """Split text into ordered passage-like Segments at paragraph boundaries.

    Paragraphs (split on blank lines) are packed greedily up to `max_chars`.
    A single oversized paragraph becomes its own segment (no mid-word cut).
    Pure function — unit-testable without any dataset.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    segments: list[Segment] = []
    buffer = ""
    index = 0
    for para in paragraphs:
        if buffer and len(buffer) + len(para) + 2 > max_chars:
            segments.append(Segment(index=index, text=buffer.strip()))
            index += 1
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer.strip():
        segments.append(Segment(index=index, text=buffer.strip()))
    return segments
