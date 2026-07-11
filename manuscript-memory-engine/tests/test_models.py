import pytest
from pydantic import ValidationError

from app.models import (
    ChapterExtraction,
    ChapterRequest,
    CharacterSeed,
    Contradiction,
    ExtractedCharacter,
    ExtractedEvent,
    JudgeVerdict,
    ManuscriptCreate,
)


class TestCharacterSeed:
    def test_defaults_to_alive(self) -> None:
        c = CharacterSeed(name="Mara", traits="pragmatic smuggler", location="docks")
        assert c.status == "alive"

    def test_explicit_dead(self) -> None:
        c = CharacterSeed(
            name="Brann", traits="loyal guard", location="citadel", status="dead"
        )
        assert c.status == "dead"

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CharacterSeed(name="X", traits="y", location="z", status="undead")


class TestManuscriptCreate:
    def test_rules_default_empty(self) -> None:
        m = ManuscriptCreate(
            title="The Ashen Crown",
            premise="A city after its king's death",
            characters=[
                CharacterSeed(name="Mara", traits="smuggler", location="docks")
            ],
        )
        assert m.rules == []

    def test_with_rules(self) -> None:
        m = ManuscriptCreate(
            title="The Ashen Crown",
            premise="A city after its king's death",
            characters=[],
            rules=["The dead do not return"],
        )
        assert len(m.rules) == 1


class TestChapterRequest:
    def test_scene_hint_optional(self) -> None:
        r = ChapterRequest()
        assert r.scene_hint is None

    def test_scene_hint_provided(self) -> None:
        r = ChapterRequest(scene_hint="Mara meets Sera at the docks")
        assert r.scene_hint == "Mara meets Sera at the docks"


class TestJudgeVerdict:
    def test_pass_verdict(self) -> None:
        v = JudgeVerdict(
            verdict="PASS",
            contradictions=[],
            coherence_score=0.9,
            reasoning="No contradictions found",
        )
        assert v.verdict == "PASS"
        assert v.coherence_score == 0.9

    def test_fail_with_contradictions(self) -> None:
        c = Contradiction(
            violated_fact="Brann is dead",
            draft_evidence="Brann walked into the room",
        )
        v = JudgeVerdict(
            verdict="FAIL",
            contradictions=[c],
            coherence_score=0.1,
            reasoning="Dead character appears",
        )
        assert v.verdict == "FAIL"
        assert len(v.contradictions) == 1

    def test_invalid_verdict_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdict(
                verdict="MAYBE",
                contradictions=[],
                coherence_score=0.5,
                reasoning="uncertain",
            )

    def test_coherence_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdict(
                verdict="PASS",
                contradictions=[],
                coherence_score=1.5,
                reasoning="out of range",
            )


class TestChapterExtraction:
    def test_full_extraction(self) -> None:
        e = ChapterExtraction(
            characters=[
                ExtractedCharacter(
                    name="Mara", status="alive", note="survived the fight"
                ),
                ExtractedCharacter(
                    name="Brann", status="dead", note="fell at the gate"
                ),
            ],
            events=[
                ExtractedEvent(
                    summary="Battle at the gate", involves=["Mara", "Brann"]
                ),
            ],
        )
        assert len(e.characters) == 2
        assert len(e.events) == 1
        assert e.characters[1].status == "dead"

    def test_invalid_character_status(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedCharacter(name="X", status="zombie", note="risen")
