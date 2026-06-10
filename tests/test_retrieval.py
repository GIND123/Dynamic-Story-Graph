"""Pure unit tests for the token-budget context assembly logic.

No services required — all functions are pure Python.
"""

from app.retrieval import _truncate_at_sentence, assemble_context, format_canon


_SAMPLE_CANON = {
    "characters": [
        {
            "name": "Mara",
            "status": "alive",
            "traits": "pragmatic smuggler",
            "location": "Duskwall docks",
            "recent_events": ["Mara joined forces with Sera"],
        },
        {
            "name": "Brann",
            "status": "dead",
            "traits": "royal guard captain",
            "location": "Citadel",
            "recent_events": ["Brann fell defending the Citadel gate"],
        },
    ],
    "rules": ["The dead do not return; Duskwall has no resurrection magic."],
    "events": [
        {"summary": "Mara and Sera discovered the king's death", "sequence_index": 0},
        {"summary": "Brann fell defending the Citadel gate", "sequence_index": 1},
    ],
}


class TestFormatCanon:
    def test_contains_character_names(self) -> None:
        result = format_canon(_SAMPLE_CANON)
        assert "Mara" in result
        assert "Brann" in result

    def test_contains_status_pipe_format(self) -> None:
        result = format_canon(_SAMPLE_CANON)
        assert "| alive |" in result
        assert "| dead |" in result

    def test_contains_rules(self) -> None:
        result = format_canon(_SAMPLE_CANON)
        assert "The dead do not return" in result

    def test_contains_events(self) -> None:
        result = format_canon(_SAMPLE_CANON)
        assert "Brann fell defending" in result

    def test_recent_events_per_character(self) -> None:
        result = format_canon(_SAMPLE_CANON)
        assert "Recent:" in result

    def test_empty_canon(self) -> None:
        result = format_canon({"characters": [], "rules": [], "events": []})
        assert "CHARACTERS:" in result


class TestTruncateAtSentence:
    def test_short_text_unchanged(self) -> None:
        text = "Short text."
        assert _truncate_at_sentence(text, 1000) == text

    def test_truncates_at_period(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        # 16 chars budget → fits "First sentence."
        result = _truncate_at_sentence(text, 4)
        assert result == "First sentence."
        assert "Second" not in result

    def test_truncates_at_question_mark(self) -> None:
        text = "Is this working? Yes it is. More text here."
        result = _truncate_at_sentence(text, 5)  # 20 chars
        assert result.endswith("?")

    def test_hard_cut_when_no_boundary(self) -> None:
        text = "NoSentenceBoundaryAtAllJustABigChunkOfText"
        result = _truncate_at_sentence(text, 2)  # 8 chars
        assert len(result) == 8

    def test_exact_fit(self) -> None:
        text = "Exactly fits."  # 13 chars, 4 tokens heuristic = 16 chars budget
        result = _truncate_at_sentence(text, 4)
        assert result == text


class TestAssembleContext:
    def test_always_includes_canon_facts(self) -> None:
        result = assemble_context(_SAMPLE_CANON, [])
        assert "=== CANON FACTS ===" in result
        assert "Mara" in result

    def test_no_passages_section_when_empty(self) -> None:
        result = assemble_context(_SAMPLE_CANON, [])
        assert "=== SIMILAR PASSAGES ===" not in result

    def test_passages_section_included(self) -> None:
        passages = [{"text": "Once upon a time in Duskwall.", "chapter": 1}]
        result = assemble_context(_SAMPLE_CANON, passages)
        assert "=== SIMILAR PASSAGES ===" in result
        assert "Once upon a time" in result

    def test_passage_budget_enforced(self) -> None:
        # Create a passage that is far larger than the budget
        long_text = ("A" * 100 + ". ") * 200  # ~200 sentences, ~24 000 chars
        passages = [{"text": long_text, "chapter": 1}]
        result = assemble_context(_SAMPLE_CANON, passages)
        passage_part = result.split("=== SIMILAR PASSAGES ===")[1]
        # Default TOKEN_BUDGET_PASSAGES = 2000 → 8000 chars; allow overhead for label
        assert len(passage_part) < 9000

    def test_multiple_passages_respect_budget(self) -> None:
        passages = [
            {"text": "Short passage one.", "chapter": 1},
            {"text": "Short passage two.", "chapter": 2},
        ]
        result = assemble_context(_SAMPLE_CANON, passages)
        assert "Short passage one" in result
        assert "Short passage two" in result

    def test_canon_facts_appear_before_passages(self) -> None:
        passages = [{"text": "A passage excerpt.", "chapter": 1}]
        result = assemble_context(_SAMPLE_CANON, passages)
        canon_pos = result.index("=== CANON FACTS ===")
        passage_pos = result.index("=== SIMILAR PASSAGES ===")
        assert canon_pos < passage_pos
