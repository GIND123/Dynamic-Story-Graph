"""Unit tests for graph functions that need a mocked Neo4j driver.

No live Neo4j required — the driver, session, and transaction are mocked so we
can capture the exact Cypher and parameters issued by commit_chapter.
"""

from unittest.mock import MagicMock

import pytest

from app import graph
from app.models import ChapterExtraction, ExtractedCharacter, ExtractedEvent


class _RecordingTx:
    """Captures every tx.run(query, **params) call for later assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        result = MagicMock()
        # Reads in commit_chapter call .single(); -1 means "no prior events"
        result.single.return_value = {"max_seq": -1}
        return result


def _mock_driver(tx: _RecordingTx) -> MagicMock:
    drv = MagicMock()
    session = MagicMock()
    session.execute_write.side_effect = lambda fn, *a, **kw: fn(tx, *a, **kw)
    drv.session.return_value.__enter__.return_value = session
    drv.session.return_value.__exit__.return_value = False
    return drv


class TestCommitChapterAutoMergesInvolves:
    def test_name_only_in_involves_is_merged_as_character(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tx = _RecordingTx()
        monkeypatch.setattr("app.graph.init_driver", lambda: _mock_driver(tx))

        extraction = ChapterExtraction(
            characters=[
                ExtractedCharacter(name="Mara", status="alive", note="appears")
            ],
            events=[
                ExtractedEvent(
                    summary="Mara meets a courier",
                    involves=["Mara", "Kira"],  # Kira is NOT in characters
                )
            ],
        )

        graph.commit_chapter("mid_x", 3, "some prose", extraction, embedding=None)

        # An ON CREATE SET MERGE for the undeclared name must have been issued.
        on_create_merges = [
            (q, p)
            for q, p in tx.calls
            if "ON CREATE SET" in q and p.get("name") == "Kira"
        ]
        assert on_create_merges, "Kira (only in involves) should be auto-merged"

        # The declared character must NOT go through the ON CREATE SET path.
        assert not any(
            "ON CREATE SET" in q and p.get("name") == "Mara" for q, p in tx.calls
        )

    def test_no_auto_merge_when_all_involves_declared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tx = _RecordingTx()
        monkeypatch.setattr("app.graph.init_driver", lambda: _mock_driver(tx))

        extraction = ChapterExtraction(
            characters=[
                ExtractedCharacter(name="Mara", status="alive", note="appears"),
                ExtractedCharacter(name="Sera", status="alive", note="appears"),
            ],
            events=[
                ExtractedEvent(summary="Mara and Sera plan", involves=["Mara", "Sera"])
            ],
        )

        graph.commit_chapter("mid_y", 4, "prose", extraction, embedding=None)

        assert not any("ON CREATE SET" in q for q, _ in tx.calls)


class TestNextChapterNumberFiltersCommitted:
    def test_query_filters_to_committed_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def _execute_query(query, **params):
            captured["query"] = query
            result = MagicMock()
            result.records = [{"next_n": 5}]
            return result

        drv = MagicMock()
        drv.execute_query.side_effect = _execute_query
        monkeypatch.setattr("app.graph.init_driver", lambda: drv)

        n = graph.next_chapter_number("mid_z")

        assert n == 5
        # The bug being fixed: skip past NEEDS_REVIEW by counting COMMITTED only.
        assert "status: 'COMMITTED'" in captured["query"]
