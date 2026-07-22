"""Power-aware human-study cell definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class NarrativeType(StrEnum):
    KINETIC = "kinetic"
    INTROSPECTIVE = "introspective"


@dataclass(frozen=True, slots=True)
class StudyCell:
    system: str
    narrative_type: NarrativeType
    stories: int


def preregistered_cells(
    systems: tuple[str, ...] = ("base_llm", "symbolic_kg", "gnsm"),
    stories_per_cell: int = 20,
) -> list[StudyCell]:
    if stories_per_cell < 20:
        raise ValueError("the system design requires at least 20 stories per cell")
    return [
        StudyCell(system, narrative_type, stories_per_cell)
        for system in systems
        for narrative_type in NarrativeType
    ]
