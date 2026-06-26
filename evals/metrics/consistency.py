"""METRIC 3 — Consistency / contradiction detection over the Tier-1 gate.

The ingested eval datasets contain no planted contradictions, so — following the
existing evals/cases.json pattern (planted contradictions + clean controls) — we
build a SYNTHETIC labeled set of (canon, extraction) pairs and measure
graph.tier1_check's precision / recall / F1 with the same conventions as the
judge eval (TP/FP/FN/TN). A per-relation-type breakdown (stance / kinship /
trait / identity / location) shows which Change-2 contradiction classes the
deterministic gate catches. This evaluates the gate itself, offline and
deterministically — no graph, no datasets, no LLM.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from app.models import (
    ChapterExtraction,
    ExtractedCharacter,
    ExtractedRelation,
    RelationType,
)

Checker = Callable[[ChapterExtraction, dict], list[str]]


@dataclass
class ConsistencyCase:
    name: str
    cls: str  # stance | kinship | trait | identity | location
    canon: dict
    extraction: ChapterExtraction
    expected_contradiction: bool


def _canon(
    relations: dict | None = None, characters: list[tuple[str, str]] = ()
) -> dict:
    return {
        "characters": [
            {
                "name": n,
                "status": s,
                "traits": "",
                "location": None,
                "recent_events": [],
            }
            for n, s in characters
        ],
        "rules": [],
        "events": [],
        "relations": relations or {},
    }


def _ext(
    *relations: ExtractedRelation, characters: list[tuple[str, str]] = ()
) -> ChapterExtraction:
    return ChapterExtraction(
        characters=[
            ExtractedCharacter(name=n, status=s, note="") for n, s in characters
        ],
        events=[],
        relations=list(relations),
    )


def default_cases() -> list[ConsistencyCase]:
    """A small labeled set: one tripping + one clean case per relation class.

    Each case is scoped to a single class so the per-class breakdown is clean.
    """
    R = RelationType
    return [
        # --- stance ---
        ConsistencyCase(
            "stance_trip",
            "stance",
            _canon(
                {
                    "relates_to": [
                        {"source": "Mara", "target": "Brann", "stance": "enemy"}
                    ]
                }
            ),
            _ext(
                ExtractedRelation(
                    type=R.RELATES_TO, source="Brann", target="Mara", stance="ally"
                )
            ),
            True,
        ),
        ConsistencyCase(
            "stance_clean",
            "stance",
            _canon(
                {
                    "relates_to": [
                        {"source": "Mara", "target": "Brann", "stance": "ally"}
                    ]
                }
            ),
            _ext(
                ExtractedRelation(
                    type=R.RELATES_TO, source="Mara", target="Brann", stance="ally"
                )
            ),
            False,
        ),
        # --- kinship ---
        ConsistencyCase(
            "kinship_trip",
            "kinship",
            _canon(),
            _ext(
                ExtractedRelation(
                    type=R.FAMILY_OF, source="Mara", target="Mara", subtype="sibling"
                )
            ),
            True,
        ),
        ConsistencyCase(
            "kinship_clean",
            "kinship",
            _canon(
                {
                    "family_of": [
                        {"source": "Brann", "target": "Mara", "subtype": "parent"}
                    ]
                }
            ),
            _ext(
                ExtractedRelation(
                    type=R.FAMILY_OF, source="Mara", target="Brann", subtype="child"
                )
            ),
            False,
        ),
        # --- trait ---
        ConsistencyCase(
            "trait_trip",
            "trait",
            _canon(
                {
                    "has_trait": [
                        {
                            "character": "Mara",
                            "trait": "blue",
                            "category": "physical",
                            "trait_key": "eye_color",
                            "trait_value": "blue",
                        }
                    ]
                }
            ),
            _ext(
                ExtractedRelation(
                    type=R.HAS_TRAIT,
                    source="Mara",
                    target="green",
                    category="physical",
                    trait_key="eye_color",
                    trait_value="green",
                )
            ),
            True,
        ),
        ConsistencyCase(
            "trait_clean",
            "trait",
            _canon(
                {
                    "has_trait": [
                        {
                            "character": "Mara",
                            "trait": "tall",
                            "category": "physical",
                            "trait_key": "height",
                            "trait_value": "tall",
                        }
                    ]
                }
            ),
            _ext(
                ExtractedRelation(
                    type=R.HAS_TRAIT,
                    source="Mara",
                    target="lean",
                    category="physical",
                    trait_key="build",
                    trait_value="lean",
                )
            ),
            False,
        ),
        # --- identity ---
        ConsistencyCase(
            "identity_trip",
            "identity",
            _canon({"identified_as": [{"character": "Brann", "alias": "The Captain"}]}),
            _ext(
                ExtractedRelation(
                    type=R.IDENTIFIED_AS,
                    source="Mara",
                    target="The Captain",
                    kind="alias",
                )
            ),
            True,
        ),
        ConsistencyCase(
            "identity_clean",
            "identity",
            _canon({"identified_as": [{"character": "Brann", "alias": "The Captain"}]}),
            _ext(
                ExtractedRelation(
                    type=R.IDENTIFIED_AS,
                    source="Brann",
                    target="The Captain",
                    kind="alias",
                )
            ),
            False,
        ),
        # --- location ---
        ConsistencyCase(
            "location_trip",
            "location",
            _canon(),
            _ext(
                ExtractedRelation(type=R.LOCATED_AT, source="Mara", target="Docks"),
                ExtractedRelation(type=R.LOCATED_AT, source="Mara", target="Citadel"),
            ),
            True,
        ),
        ConsistencyCase(
            "location_clean",
            "location",
            _canon(),
            _ext(ExtractedRelation(type=R.LOCATED_AT, source="Mara", target="Docks")),
            False,
        ),
    ]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return precision, recall, f1


def evaluate_consistency(
    cases: list[ConsistencyCase] | None = None,
    check: Checker | None = None,
) -> dict:
    """Run `check` (default tier1_check) over the cases; return P/R/F1 + per class.

    Predicted-positive = the checker returns at least one violation string.
    """
    if cases is None:
        cases = default_cases()
    if check is None:
        from app.graph import tier1_check

        check = tier1_check

    overall = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_class: dict[str, dict] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )

    for case in cases:
        predicted = len(check(case.extraction, case.canon)) > 0
        expected = case.expected_contradiction
        if expected and predicted:
            cell = "tp"
        elif not expected and predicted:
            cell = "fp"
        elif expected and not predicted:
            cell = "fn"
        else:
            cell = "tn"
        overall[cell] += 1
        per_class[case.cls][cell] += 1

    precision, recall, f1 = _prf(overall["tp"], overall["fp"], overall["fn"])
    per_class_report = {}
    for cls, c in per_class.items():
        p, r, f = _prf(c["tp"], c["fp"], c["fn"])
        per_class_report[cls] = {
            "precision": p,
            "recall": r,
            "f1": f,
            **c,
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class_report,
        "n_cases": len(cases),
        **overall,
    }
