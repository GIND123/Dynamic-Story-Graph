"""Unit tests for the Tier-1 deterministic canon gate.

tier1_check is a pure function: no database, no LLM, no fixtures required.
It works entirely from the canon dict returned by graph.get_canon.
"""

from app.graph import tier1_check
from app.models import ChapterExtraction, ExtractedCharacter, ExtractedEvent


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
