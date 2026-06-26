"""Unit tests for the Tier-1 deterministic canon gate.

tier1_check is a pure function: no database, no LLM, no fixtures required.
It works entirely from the canon dict returned by graph.get_canon.
"""

from app.graph import tier1_check
from app.models import (
    ChapterExtraction,
    ExtractedCharacter,
    ExtractedEvent,
    ExtractedRelation,
    RelationType,
)


def _canon(*char_specs: tuple[str, str]) -> dict:
    """Build a minimal canon dict with (name, status) pairs."""
    return {
        "characters": [
            {
                "name": name,
                "status": status,
                "traits": "",
                "location": None,
                "recent_events": [],
            }
            for name, status in char_specs
        ],
        "rules": [],
        "events": [],
    }


def _ext(
    char_specs: list[tuple[str, str]] = (),
    event_involves: list[str] = (),
) -> ChapterExtraction:
    chars = [ExtractedCharacter(name=n, status=s, note="") for n, s in char_specs]
    events = (
        [ExtractedEvent(summary="event", involves=list(event_involves))]
        if event_involves
        else []
    )
    return ChapterExtraction(characters=chars, events=events)


class TestTier1Check:
    def test_dead_in_involves_is_violation(self) -> None:
        """Dead character named in event.involves → violation."""
        canon = _canon(("Mara", "alive"), ("Brann", "dead"))
        ext = _ext(char_specs=[("Mara", "alive")], event_involves=["Mara", "Brann"])

        violations = tier1_check(ext, canon)

        assert len(violations) == 1
        assert "Brann" in violations[0]

    def test_resurrection_is_violation(self) -> None:
        """Dead-in-canon character extracted as alive → violation."""
        canon = _canon(("Mara", "alive"), ("Brann", "dead"))
        # Brann dead in canon, but extraction marks him alive
        ext = _ext(
            char_specs=[("Mara", "alive"), ("Brann", "alive")],
            event_involves=["Mara"],
        )

        violations = tier1_check(ext, canon)

        assert len(violations) == 1
        assert "Brann" in violations[0]

    def test_dead_correctly_extracted_as_dead_is_not_violation(self) -> None:
        """Character who is dead in both canon and extraction is correct — no violation."""
        canon = _canon(("Mara", "alive"), ("Brann", "dead"))
        ext = _ext(
            char_specs=[("Mara", "alive"), ("Brann", "dead")],
            event_involves=["Mara"],
        )

        assert tier1_check(ext, canon) == []

    def test_clean_extraction_passes(self) -> None:
        """Only alive characters, no dead names anywhere → no violations."""
        canon = _canon(("Mara", "alive"), ("Sera", "alive"), ("Brann", "dead"))
        ext = _ext(
            char_specs=[("Mara", "alive"), ("Sera", "alive")],
            event_involves=["Mara", "Sera"],
        )

        assert tier1_check(ext, canon) == []

    def test_new_character_not_in_canon_is_not_violation(self) -> None:
        """New character introduced this chapter has no canon entry — not a violation."""
        canon = _canon(("Mara", "alive"))
        ext = _ext(
            char_specs=[("Mara", "alive"), ("Kira", "alive")],
            event_involves=["Mara", "Kira"],
        )

        assert tier1_check(ext, canon) == []

    def test_empty_extraction_no_violations(self) -> None:
        canon = _canon(("Brann", "dead"))
        assert tier1_check(ChapterExtraction(characters=[], events=[]), canon) == []

    def test_empty_canon_no_violations(self) -> None:
        """No canon at all — nothing to violate."""
        ext = _ext(char_specs=[("Mara", "alive")], event_involves=["Mara"])
        assert tier1_check(ext, {"characters": [], "rules": [], "events": []}) == []

    def test_both_violations_detected_simultaneously(self) -> None:
        """Dead-in-involves and resurrection can both be present at once."""
        canon = _canon(("Mara", "alive"), ("Brann", "dead"), ("Sera", "dead"))
        ext = _ext(
            char_specs=[("Mara", "alive"), ("Brann", "alive")],  # Brann resurrection
            event_involves=["Mara", "Sera"],  # Sera (dead) in involves
        )

        violations = tier1_check(ext, canon)

        assert len(violations) == 2
        names_mentioned = " ".join(violations)
        assert "Brann" in names_mentioned
        assert "Sera" in names_mentioned


def _canon_rel(
    relations: dict | None = None,
    char_specs: list[tuple[str, str]] = (),
) -> dict:
    """Canon dict with a 'relations' block (and optional character statuses)."""
    return {
        "characters": [
            {
                "name": n,
                "status": s,
                "traits": "",
                "location": None,
                "recent_events": [],
            }
            for n, s in char_specs
        ],
        "rules": [],
        "events": [],
        "relations": relations or {},
    }


def _ext_rel(*relations: ExtractedRelation) -> ChapterExtraction:
    """Extraction carrying only relations (no characters/events)."""
    return ChapterExtraction(characters=[], events=[], relations=list(relations))


class TestTier1RelationChecks:
    # --- Check 3: contradictory RELATES_TO stance ---
    def test_stance_contradiction_trips(self) -> None:
        canon = _canon_rel(
            {"relates_to": [{"source": "Mara", "target": "Brann", "stance": "enemy"}]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.RELATES_TO,
                source="Brann",
                target="Mara",
                stance="ally",
            )
        )
        violations = tier1_check(ext, canon)
        assert any("Stance contradiction" in v for v in violations)

    def test_stance_consistent_is_clean(self) -> None:
        canon = _canon_rel(
            {"relates_to": [{"source": "Mara", "target": "Brann", "stance": "ally"}]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.RELATES_TO,
                source="Mara",
                target="Brann",
                stance="ally",
            )
        )
        assert tier1_check(ext, canon) == []

    # --- Check 4: two-places-at-once (within this chapter) ---
    def test_two_places_at_once_trips(self) -> None:
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.LOCATED_AT, source="Mara", target="Docks"
            ),
            ExtractedRelation(
                type=RelationType.LOCATED_AT, source="Mara", target="Citadel"
            ),
        )
        violations = tier1_check(ext, _canon_rel())
        assert any("Two places at once" in v for v in violations)

    def test_single_location_is_clean(self) -> None:
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.LOCATED_AT, source="Mara", target="Docks"
            )
        )
        assert tier1_check(ext, _canon_rel()) == []

    # --- Check 5: kinship impossibility ---
    def test_kinship_self_loop_trips(self) -> None:
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.FAMILY_OF,
                source="Mara",
                target="Mara",
                subtype="sibling",
            )
        )
        violations = tier1_check(ext, _canon_rel())
        assert any("self-loop" in v for v in violations)

    def test_kinship_parent_inversion_trips(self) -> None:
        canon = _canon_rel(
            {"family_of": [{"source": "Brann", "target": "Mara", "subtype": "parent"}]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.FAMILY_OF,
                source="Mara",
                target="Brann",
                subtype="parent",
            )
        )
        violations = tier1_check(ext, canon)
        assert any("Kinship contradiction" in v for v in violations)

    def test_spouse_of_dead_flags_for_review(self) -> None:
        canon = _canon_rel(char_specs=[("Brann", "dead")])
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.FAMILY_OF,
                source="Mara",
                target="Brann",
                subtype="spouse",
            )
        )
        violations = tier1_check(ext, canon)
        assert any("Marriage to dead character" in v for v in violations)

    def test_normal_kinship_is_clean(self) -> None:
        canon = _canon_rel(
            {"family_of": [{"source": "Brann", "target": "Mara", "subtype": "parent"}]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.FAMILY_OF,
                source="Mara",
                target="Brann",
                subtype="child",  # consistent inverse of canon parent
            )
        )
        assert tier1_check(ext, canon) == []

    # --- Check 6: physical-trait contradiction (structured trait_key/value) ---
    def _phys_trait(self, character, key, value, name=None):
        """Build a canon has_trait row with the structured key/value split."""
        return {
            "character": character,
            "trait": name or value,
            "category": "physical",
            "trait_key": key,
            "trait_value": value,
        }

    def test_physical_trait_contradiction_trips(self) -> None:
        """Same character, same trait_key, different trait_value → violation."""
        canon = _canon_rel(
            {"has_trait": [self._phys_trait("Mara", "eye_color", "blue")]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.HAS_TRAIT,
                source="Mara",
                target="green",
                category="physical",
                trait_key="eye_color",
                trait_value="green",
            )
        )
        violations = tier1_check(ext, canon)
        assert any("Trait contradiction" in v for v in violations)

    def test_paraphrase_under_different_key_is_clean(self) -> None:
        """A near-synonym modeled under a different trait_key must NOT trip:
        height=tall vs build=towering are never compared (keys differ)."""
        canon = _canon_rel({"has_trait": [self._phys_trait("Mara", "height", "tall")]})
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.HAS_TRAIT,
                source="Mara",
                target="towering",
                category="physical",
                trait_key="build",
                trait_value="towering",
            )
        )
        assert tier1_check(ext, canon) == []

    def test_same_key_same_value_is_clean(self) -> None:
        """Re-asserting the same physical trait is not a contradiction."""
        canon = _canon_rel({"has_trait": [self._phys_trait("Mara", "height", "tall")]})
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.HAS_TRAIT,
                source="Mara",
                target="tall",
                category="physical",
                trait_key="height",
                trait_value="tall",
            )
        )
        assert tier1_check(ext, canon) == []

    def test_null_trait_key_defers_to_review(self) -> None:
        """trait_key missing → no deterministic decision. Tier-1 stays silent so
        the case routes to the Tier-2/NEEDS_REVIEW path (not a hard reject)."""
        canon = _canon_rel(
            {"has_trait": [self._phys_trait("Mara", "eye_color", "blue")]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.HAS_TRAIT,
                source="Mara",
                target="green eyes",  # unstructured; no stable key to compare
                category="physical",
                trait_key=None,
                trait_value=None,
            )
        )
        violations = tier1_check(ext, canon)
        assert not any("Trait contradiction" in v for v in violations)
        assert violations == []  # nothing trips → deferred to Tier-2

    def test_real_shaped_extraction_trips(self) -> None:
        """Regression guard against 'dead on real input': a model-shaped relation
        whose `target` is free text ('green eyes', NO delimiter — the exact shape
        that the old string-parser silently skipped) still trips, because the
        check reads the structured trait_key/trait_value, not the target string."""
        canon = _canon_rel(
            {"has_trait": [self._phys_trait("Mara", "eye_color", "blue")]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.HAS_TRAIT,
                source="Mara",
                target="green eyes",  # free text, no 'key: value' delimiter
                category="physical",
                trait_key="eye_color",
                trait_value="green",
            )
        )
        violations = tier1_check(ext, canon)
        assert any("Trait contradiction" in v for v in violations)

    # --- Check 7: identity collision ---
    def test_identity_collision_trips(self) -> None:
        canon = _canon_rel(
            {"identified_as": [{"character": "Brann", "alias": "The Captain"}]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.IDENTIFIED_AS,
                source="Mara",
                target="The Captain",
                kind="alias",
            )
        )
        violations = tier1_check(ext, canon)
        assert any("Identity collision" in v for v in violations)

    def test_same_owner_alias_is_clean(self) -> None:
        canon = _canon_rel(
            {"identified_as": [{"character": "Brann", "alias": "The Captain"}]}
        )
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.IDENTIFIED_AS,
                source="Brann",
                target="The Captain",
                kind="alias",
            )
        )
        assert tier1_check(ext, canon) == []

    def test_no_relations_block_is_clean(self) -> None:
        """Backward compat: canon without a 'relations' key never trips new checks."""
        ext = _ext_rel(
            ExtractedRelation(
                type=RelationType.RELATES_TO,
                source="A",
                target="B",
                stance="ally",
            )
        )
        assert tier1_check(ext, {"characters": [], "rules": [], "events": []}) == []
