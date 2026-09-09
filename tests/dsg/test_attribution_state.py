"""State snapshots must be causal by provenance, since substrings cannot vouch for them.

A text block is checked by asking whether it is a slice of allowed text. A state
digest is derived -- the extractor wrote it -- so that check does not apply and
the guarantee has to come from where the snapshot was taken instead.
"""

from __future__ import annotations

import pytest

from dsg.eval.attribution_state import EMPTY, CausalityError, Snapshot, snapshot_for


def snap(window: int, covers_to: int, **kw) -> Snapshot:
    return Snapshot(
        window=window,
        covers_to=covers_to,
        entities=kw.get("entities", ""),
        facts=kw.get("facts", ""),
        recent_speakers=kw.get("recent_speakers", ()),
    )


SNAPS = [snap(0, 1000), snap(1, 2000), snap(2, 3000), snap(3, 4000)]


class TestCausality:
    def test_a_snapshot_ending_before_the_quote_is_accepted(self) -> None:
        snap(0, 500).assert_causal(1200)

    def test_a_snapshot_ending_exactly_at_the_quote_is_accepted(self) -> None:
        """Coverage to s means text[:s] -- the quote itself is not included."""
        snap(0, 1200).assert_causal(1200)

    def test_a_snapshot_reaching_past_the_quote_is_rejected(self) -> None:
        with pytest.raises(CausalityError, match="has not reached"):
            snap(5, 1201).assert_causal(1200)

    def test_the_error_names_the_quote(self) -> None:
        with pytest.raises(CausalityError, match="q7"):
            snap(5, 999).assert_causal(10, label="q7")


class TestSnapshotSelection:
    def test_picks_the_latest_snapshot_that_does_not_overrun(self) -> None:
        assert snapshot_for(SNAPS, 2500).window == 1

    def test_boundary_is_inclusive(self) -> None:
        assert snapshot_for(SNAPS, 2000).window == 1

    def test_one_char_before_a_boundary_falls_back(self) -> None:
        assert snapshot_for(SNAPS, 1999).window == 0

    def test_a_quote_before_any_boundary_gets_the_empty_state(self) -> None:
        chosen = snapshot_for(SNAPS, 10)
        assert chosen is EMPTY
        assert chosen.covers_to == 0

    def test_a_quote_past_the_end_gets_the_last_snapshot(self) -> None:
        assert snapshot_for(SNAPS, 99999).window == 3

    def test_no_snapshots_at_all_is_the_empty_state(self) -> None:
        assert snapshot_for([], 500) is EMPTY

    def test_every_selection_is_causal_across_the_whole_range(self) -> None:
        """The property that matters, checked densely rather than at a few points."""
        for pos in range(0, 5000, 37):
            assert snapshot_for(SNAPS, pos).covers_to <= pos

    def test_the_window_containing_the_quote_is_excluded(self) -> None:
        """Window 2 spans (2000, 3000]; a quote at 2500 must not see it.

        That window's proposals were extracted from the whole window, including
        text after the quote -- the exact leak this experiment exists to avoid.
        """
        assert snapshot_for(SNAPS, 2500).window == 1


class TestRendering:
    def test_the_empty_state_renders_to_nothing(self) -> None:
        assert EMPTY.render() == ""

    def test_a_populated_snapshot_renders_all_three_sections(self) -> None:
        out = snap(
            1, 100,
            entities="- Brenda",
            facts="- Brenda: knows Tony",
            recent_speakers=("Tony", "Brenda"),
        ).render()
        assert "Characters the reader has met" in out
        assert "What the reader believes" in out
        assert "Most recent speakers" in out
        assert "Tony, Brenda" in out

    def test_missing_sections_are_omitted_not_left_blank(self) -> None:
        out = snap(1, 100, entities="- Brenda").render()
        assert "Characters the reader has met" in out
        assert "believes" not in out
        assert "Most recent speakers" not in out

    def test_recent_speakers_are_capped(self) -> None:
        out = snap(1, 100, recent_speakers=tuple("ABCDEFGH")).render()
        assert "E, F, G, H" in out
        assert "A," not in out


class TestSpanOrdering:
    """PDNC does not guarantee document order; callers assume it.

    5 of 37,131 quotes list a later segment first. Callers take spans[0][0] as
    the start and spans[-1][1] as the end, which inverts the range for those and
    crashes CausalContext -- it did, on the first full run.
    """

    def test_spans_come_back_sorted(self) -> None:
        from dsg.eval.causal_audit import quote_spans

        row = {"quoteByteSpans": "[[83859, 83964], [83789, 83837]]"}
        assert quote_spans(row) == ((83789, 83837), (83859, 83964))

    def test_the_outer_range_is_not_inverted(self) -> None:
        from dsg.eval.causal_audit import quote_span

        row = {"quoteByteSpans": "[[83859, 83964], [83789, 83837]]"}
        start, end = quote_span(row)
        assert start < end

    def test_a_three_segment_out_of_order_quote_sorts(self) -> None:
        from dsg.eval.causal_audit import quote_spans

        row = {"quoteByteSpans": "[[570083, 570395], [568680, 568956], [569161, 569198]]"}
        spans = quote_spans(row)
        assert [s for s, _ in spans] == sorted(s for s, _ in spans)
        assert spans[0][0] < spans[-1][1]

    def test_already_sorted_spans_are_unchanged(self) -> None:
        from dsg.eval.causal_audit import quote_spans

        row = {"quoteByteSpans": "[[100, 200], [300, 400]]"}
        assert quote_spans(row) == ((100, 200), (300, 400))

    def test_degenerate_spans_are_dropped(self) -> None:
        from dsg.eval.causal_audit import quote_spans

        assert quote_spans({"quoteByteSpans": "[[100, 100], [200, 300]]"}) == ((200, 300),)
