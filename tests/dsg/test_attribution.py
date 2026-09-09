"""The leakage guard is the whole causal claim, so it is tested adversarially.

If any test here can be made to pass while future text reaches a prompt, the E1
result is not a causal result and must not be reported as one.
"""

from __future__ import annotations

import pytest

from dsg.eval.attribution import (
    AttributionScore,
    CausalContext,
    LeakageError,
    assert_no_leakage,
    normalise_name,
    score_attribution,
)

TEXT = (
    "Elizabeth walked into the drawing room and found her sister waiting by the "
    "window with a letter in her hand. "
    '"I have been expecting you all morning," she said quietly. '
    '"Mr. Darcy called again while you were out, and he would not stay for tea."'
)
QUOTE_START = TEXT.index('"I have been')


class TestCausalContext:
    def test_prefix_stops_exactly_at_the_quote(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START)
        assert ctx.prefix.endswith("in her hand. ")
        assert "I have been expecting" not in ctx.prefix

    def test_tail_is_bounded_and_never_reaches_forward(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START)
        tail = ctx.tail(50)
        assert len(tail) == 50
        assert tail == TEXT[QUOTE_START - 50 : QUOTE_START]
        assert "she said" not in tail

    def test_tail_clamps_at_the_start_of_the_book(self) -> None:
        ctx = CausalContext(TEXT, 10)
        assert ctx.tail(500) == TEXT[:10]

    def test_zero_and_negative_tail_are_empty(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START)
        assert ctx.tail(0) == ""
        assert ctx.tail(-5) == ""

    def test_offsets_outside_the_text_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            CausalContext(TEXT, len(TEXT) + 1)
        with pytest.raises(ValueError):
            CausalContext(TEXT, -1)

    def test_start_at_end_of_text_is_allowed(self) -> None:
        ctx = CausalContext(TEXT, len(TEXT))
        assert ctx.prefix == TEXT


class TestLeakageGuard:
    def test_a_clean_prefix_prompt_passes(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START)
        prompt = f"Context:\n{ctx.tail(80)}\n\nWho speaks next?"
        assert ctx.verify(prompt) is prompt

    def test_the_speech_tag_after_the_quote_is_caught(self) -> None:
        """The exact failure mode E1 exists to prevent."""
        ctx = CausalContext(TEXT, QUOTE_START)
        leaked = (
            f"Context:\n{ctx.tail(80)}\n"
            '"I have been expecting you all morning," she said quietly.'
        )
        with pytest.raises(LeakageError, match="leaks future text"):
            ctx.verify(leaked, label="q1")

    def test_reformatted_leak_is_still_caught(self) -> None:
        """Rewrapping the copied text must not defeat the check."""
        ctx = CausalContext(TEXT, QUOTE_START)
        future = TEXT[QUOTE_START : QUOTE_START + 120]
        reflowed = "\n   ".join(future.split(" "))
        with pytest.raises(LeakageError):
            ctx.verify(f"Context:\n{reflowed}")

    def test_short_incidental_overlap_does_not_trip_it(self) -> None:
        """Ordinary English collocations must not produce false positives."""
        ctx = CausalContext(TEXT, QUOTE_START)
        ctx.verify(f"{ctx.tail(60)}\nAnswer with a name. she said")

    def test_guard_is_a_noop_at_end_of_text(self) -> None:
        assert_no_leakage("anything at all", TEXT, len(TEXT))

    def test_error_names_the_quote_under_test(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START)
        with pytest.raises(LeakageError, match="q42"):
            ctx.verify(TEXT[QUOTE_START : QUOTE_START + 90], label="q42")


SPLIT = 'Before. "Hello there," said Elizabeth, "how are you today?" After that.'
S1 = (SPLIT.index('"Hello'), SPLIT.index('," said') + 2)
S2 = (SPLIT.index('"how'), SPLIT.index('?"') + 2)


class TestSplitQuotes:
    """28.9% of PDNC is multi-segment; the interjection holds the speech tag."""

    def test_quote_text_excludes_the_interjected_tag(self) -> None:
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        assert "Elizabeth" not in ctx.quote
        assert "Hello there," in ctx.quote
        assert "how are you today?" in ctx.quote

    def test_the_interjection_is_forbidden_evidence(self) -> None:
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        assert "said Elizabeth" in ctx.forbidden

    def test_showing_the_naive_span_is_caught(self) -> None:
        """text[first_start:last_end] contains the answer -- must not pass."""
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        naive = SPLIT[S1[0] : S2[1]]
        with pytest.raises(LeakageError):
            ctx.verify(f"Who speaks?\n{naive}")

    def test_showing_only_the_spoken_words_passes(self) -> None:
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        ctx.verify(f"Context:\n{ctx.tail(20)}\nLine:\n{ctx.quote}\nSpeaker:")

    def test_text_after_the_quote_is_still_forbidden(self) -> None:
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        assert "After that." in ctx.forbidden

    def test_single_segment_defaults_and_still_guards(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START, len(TEXT))
        assert ctx.spans == ((QUOTE_START, len(TEXT)),)
        assert ctx.quote.startswith('"I have been')

    def test_forbidden_separator_blocks_cross_region_shingles(self) -> None:
        """A shingle must not straddle two disjoint forbidden regions."""
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        assert "\x00" in ctx.forbidden

    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            CausalContext(SPLIT, 30, 5)

    def test_short_speech_tag_is_caught_despite_being_under_shingle_width(self) -> None:
        """The regression this guard exists for: tags are ~20 chars, shingles are 40."""
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        tag = ctx.forbidden_regions[0]
        assert len(tag.strip()) < 40, "fixture must exercise the short-region path"
        with pytest.raises(LeakageError, match="forbidden region"):
            ctx.verify(f"Who speaks?{tag}")


class TestWhitelistGuard:
    """verify_evidence asks where a block came from, not whether it looks bad."""

    def test_a_prefix_slice_is_accepted(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START, len(TEXT))
        ctx.verify_evidence(ctx.tail(60))

    def test_the_spoken_words_are_accepted(self) -> None:
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        ctx.verify_evidence(ctx.quote)

    def test_the_interjected_tag_is_rejected(self) -> None:
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        with pytest.raises(LeakageError, match="not a slice of allowed text"):
            ctx.verify_evidence("said Elizabeth")

    def test_text_after_the_quote_is_rejected(self) -> None:
        ctx = CausalContext(SPLIT, S1[0], S2[1], (S1, S2))
        with pytest.raises(LeakageError):
            ctx.verify_evidence("After that.")

    def test_invented_text_is_rejected_even_though_it_is_not_in_the_novel(self) -> None:
        """A whitelist rejects anything unsanctioned, not just known-bad strings."""
        ctx = CausalContext(TEXT, QUOTE_START, len(TEXT))
        with pytest.raises(LeakageError):
            ctx.verify_evidence("a sentence from some entirely different book")

    def test_empty_blocks_are_ignored(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START, len(TEXT))
        ctx.verify_evidence("", "   ")

    def test_whitespace_reflow_does_not_break_acceptance(self) -> None:
        ctx = CausalContext(TEXT, QUOTE_START, len(TEXT))
        ctx.verify_evidence("\n  ".join(ctx.tail(60).split()))


class TestNormalisation:
    def test_case_punctuation_and_spacing_fold(self) -> None:
        assert normalise_name("  ELIZABETH   Bennet! ") == normalise_name("elizabeth bennet")

    def test_accents_fold(self) -> None:
        assert normalise_name("Zoë") == normalise_name("Zoe")

    def test_titles_are_preserved_because_they_distinguish_people(self) -> None:
        """Mr./Mrs./Miss Bennet are three people; folding titles merges them."""
        assert normalise_name("Mr. Bennet") != normalise_name("Mrs. Bennet")
        assert normalise_name("Miss Bennet") != normalise_name("Mr. Bennet")

    def test_empty_and_none_are_safe(self) -> None:
        assert normalise_name("") == ""
        assert normalise_name(None) == ""  # type: ignore[arg-type]


class TestScoring:
    def test_exact_and_alias_matches_count(self) -> None:
        score = score_attribution(
            ["elizabeth bennet", "Lizzy", "Mr. Darcy"],
            ["Elizabeth Bennet", "Elizabeth Bennet", "Elizabeth Bennet"],
            ["Explicit", "Implicit", "Implicit"],
            alias_sets={"Elizabeth Bennet": {"Lizzy", "Eliza"}},
        )
        assert score.n == 3
        assert score.correct == 2
        assert score.accuracy == pytest.approx(2 / 3)

    def test_an_alias_of_another_character_is_not_accepted(self) -> None:
        score = score_attribution(
            ["Lizzy"], ["Jane Bennet"], ["Implicit"],
            alias_sets={"Elizabeth Bennet": {"Lizzy"}},
        )
        assert score.correct == 0

    def test_none_prediction_is_wrong_not_an_error(self) -> None:
        score = score_attribution([None], ["Elizabeth"], ["Implicit"])
        assert score.n == 1 and score.correct == 0

    def test_explicit_and_non_explicit_split_separately(self) -> None:
        score = score_attribution(
            ["A", "wrong", "B", "C"],
            ["A", "B", "B", "C"],
            ["Explicit", "Explicit", "Implicit", "Anaphoric"],
        )
        assert score.accuracy_for("Explicit") == pytest.approx(0.5)
        assert score.accuracy_non_explicit == pytest.approx(1.0)
        assert score.accuracy == pytest.approx(0.75)

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            score_attribution(["A"], ["A", "B"], ["Explicit", "Implicit"])

    def test_empty_score_does_not_divide_by_zero(self) -> None:
        score = AttributionScore()
        assert score.accuracy == 0.0
        assert score.accuracy_non_explicit == 0.0
        assert score.as_dict()["n"] == 0.0


class TestPromptFairness:
    """Each condition must be asked a question that matches what it was shown.

    The causal conditions end exactly at the quote, so "the line that comes
    next" is literally true. The oracle's context brackets the quote, so the
    same wording is false there and the model has to guess which line is meant.
    That is a prompt artefact, not a property of causality, and it would depress
    the oracle and could manufacture a "causality is free" result.
    """

    def _novel(self):
        from dsg.eval.attribution_run import NovelData, Quote

        text = (
            "Elizabeth waited by the window. "
            '"I have been expecting you," she said. '
            "The clock struck four and nobody moved."
        )
        start = text.index('"I have been')
        end = text.index('," she') + 2
        q = Quote(
            novel="T", quote_id="q1", start=start, end=end,
            text=text[start:end], speaker="Elizabeth",
            quote_type="Explicit", spans=((start, end),),
        )
        nd = NovelData("T", text, [q], {"Elizabeth": {"Elizabeth"}}, ["Elizabeth", "Jane"])
        return nd, q

    def test_causal_conditions_say_the_line_comes_next(self) -> None:
        from dsg.eval.attribution_run import build_prompt

        nd, q = self._novel()
        p = build_prompt(nd, q, "text-causal", window=40, recent=None)
        assert "comes next" in p
        assert ">>>" not in p

    def test_the_oracle_marks_the_target_in_its_context(self) -> None:
        from dsg.eval.attribution_run import build_prompt

        nd, q = self._novel()
        p = build_prompt(nd, q, "oracle-noncausal", window=60, recent=None)
        body = p.split("Surrounding text", 1)[1].split("The line of dialogue", 1)[0]
        assert ">>>" in body and "<<<" in body

    def test_the_oracle_asks_about_the_marked_line_not_the_next_one(self) -> None:
        from dsg.eval.attribution_run import build_prompt

        nd, q = self._novel()
        p = build_prompt(nd, q, "oracle-noncausal", window=60, recent=None)
        assert "marked >>>...<<< above" in p
        assert "comes next" not in p

    def test_the_oracle_still_sees_text_after_the_quote(self) -> None:
        """It is the control that prices causality, so it must read forward."""
        from dsg.eval.attribution_run import build_prompt

        nd, q = self._novel()
        p = build_prompt(nd, q, "oracle-noncausal", window=60, recent=None)
        assert "clock struck four" in p

    def test_causal_conditions_never_see_text_after_the_quote(self) -> None:
        from dsg.eval.attribution_run import build_prompt

        nd, q = self._novel()
        for cond in ("text-causal", "state-causal", "state-retrieval"):
            p = build_prompt(nd, q, cond, window=200, recent=None)
            assert "clock struck four" not in p, cond
