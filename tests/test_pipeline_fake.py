"""End-to-end pipeline tests in FAKE LLM mode.

All external dependencies (Redis lock, Neo4j graph functions, vectors) are
monkeypatched so no live services are required. Tasks run synchronously via
Celery's apply() method.
"""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_canon():
    return {
        "characters": [
            {
                "name": "Mara",
                "status": "alive",
                "traits": "pragmatic smuggler",
                "location": "Duskwall docks",
                "recent_events": [],
            },
            {
                "name": "Sera",
                "status": "alive",
                "traits": "archivist",
                "location": "Old Library",
                "recent_events": [],
            },
            {
                "name": "Brann",
                "status": "dead",
                "traits": "guard captain",
                "location": "Citadel",
                "recent_events": ["Brann fell defending the Citadel gate"],
            },
        ],
        "rules": ["The dead do not return; Duskwall has no resurrection magic."],
        "events": [
            {"summary": "Brann fell defending the Citadel gate", "sequence_index": 0}
        ],
    }


@pytest.fixture
def mock_redis_factory():
    """Returns a factory that builds a mock Redis client backed by an in-memory dict."""

    def _make(initial: dict | None = None):
        store: dict[str, str] = dict(initial or {})
        r = MagicMock()
        r.set.side_effect = lambda key, val, nx=False, ex=None: (
            None if (nx and key in store) else (store.update({key: val}) or True)
        )
        r.get.side_effect = lambda key: store.get(key)
        r.delete.side_effect = lambda key: store.pop(key, None)

        def _register_script(_lua_src):
            def _eval(keys, args):
                if store.get(keys[0]) == args[0]:
                    store.pop(keys[0], None)
                    return 1
                return 0

            return _eval

        r.register_script.side_effect = _register_script
        r._store = store  # expose for assertions
        return r

    return _make


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineHappyPath:
    def test_chapter_committed_in_fake_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_canon: dict,
        mock_redis_factory,
    ) -> None:
        """With FAKE_VIOLATION=0, the pipeline commits the chapter on attempt 1."""
        committed: dict = {}

        monkeypatch.setattr("app.tasks._redis", mock_redis_factory())
        monkeypatch.setattr("app.graph.get_canon", lambda mid: fake_canon)
        monkeypatch.setattr(
            "app.graph.is_chapter_committed", lambda mid, n: (mid, n) in committed
        )
        monkeypatch.setattr(
            "app.graph.commit_chapter",
            lambda mid, n, text, extraction, emb: committed.update({(mid, n): text}),
        )
        monkeypatch.setattr("app.graph.mark_needs_review", lambda *a, **kw: None)
        monkeypatch.setattr("app.vectors.similar_passages", lambda *a, **kw: [])
        monkeypatch.setattr("app.vectors.embed_one", lambda text: None)
        monkeypatch.setattr("app.config.settings.fake_violation", 0)
        monkeypatch.setattr("app.config.settings.llm_mode", "fake")

        from app.tasks import generate_chapter_task

        result = generate_chapter_task.apply(args=["mid_test", 3, None]).get()

        assert result["status"] == "committed"
        assert result["chapter_number"] == 3
        assert result["attempts"] == 1
        assert ("mid_test", 3) in committed

    def test_prose_is_non_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_canon: dict,
        mock_redis_factory,
    ) -> None:
        committed: dict = {}

        monkeypatch.setattr("app.tasks._redis", mock_redis_factory())
        monkeypatch.setattr("app.graph.get_canon", lambda mid: fake_canon)
        monkeypatch.setattr(
            "app.graph.is_chapter_committed", lambda mid, n: (mid, n) in committed
        )
        monkeypatch.setattr(
            "app.graph.commit_chapter",
            lambda mid, n, text, extraction, emb: committed.update({(mid, n): text}),
        )
        monkeypatch.setattr("app.graph.mark_needs_review", lambda *a, **kw: None)
        monkeypatch.setattr("app.vectors.similar_passages", lambda *a, **kw: [])
        monkeypatch.setattr("app.vectors.embed_one", lambda text: None)
        monkeypatch.setattr("app.config.settings.fake_violation", 0)
        monkeypatch.setattr("app.config.settings.llm_mode", "fake")

        from app.tasks import generate_chapter_task

        generate_chapter_task.apply(
            args=["mid_prose", 1, "Mara heads to the docks"]
        ).get()

        stored_text = committed[("mid_prose", 1)]
        assert len(stored_text) > 50
        assert "Duskwall" in stored_text


class TestIdempotency:
    def test_already_committed_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_canon: dict,
        mock_redis_factory,
    ) -> None:
        """Second call returns already_committed without re-running the pipeline."""
        committed: dict = {("mid_idem", 1): "existing text"}
        commit_call_count = {"n": 0}

        monkeypatch.setattr("app.tasks._redis", mock_redis_factory())
        monkeypatch.setattr("app.graph.get_canon", lambda mid: fake_canon)
        monkeypatch.setattr(
            "app.graph.is_chapter_committed", lambda mid, n: (mid, n) in committed
        )
        monkeypatch.setattr(
            "app.graph.commit_chapter",
            lambda *a, **kw: commit_call_count.update(
                {"n": commit_call_count["n"] + 1}
            ),
        )
        monkeypatch.setattr("app.graph.mark_needs_review", lambda *a, **kw: None)
        monkeypatch.setattr("app.vectors.similar_passages", lambda *a, **kw: [])
        monkeypatch.setattr("app.vectors.embed_one", lambda text: None)
        monkeypatch.setattr("app.config.settings.llm_mode", "fake")

        from app.tasks import generate_chapter_task

        result = generate_chapter_task.apply(args=["mid_idem", 1, None]).get()

        assert result["status"] == "already_committed"
        assert commit_call_count["n"] == 0  # commit never called

    def test_duplicate_lock_suppressed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_redis_factory,
    ) -> None:
        """If the lock is already held, the task returns duplicate_suppressed."""
        # Pre-populate the lock as if another worker holds it
        mock_redis = mock_redis_factory({"lock:mid_dup:chapter:5": "other_task_id"})
        monkeypatch.setattr("app.tasks._redis", mock_redis)

        from app.tasks import generate_chapter_task

        result = generate_chapter_task.apply(args=["mid_dup", 5, None]).get()
        assert result["status"] == "duplicate_suppressed"

    def test_lock_released_after_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_canon: dict,
        mock_redis_factory,
    ) -> None:
        """The idempotency lock must be deleted after a successful commit."""
        committed: dict = {}
        mock_redis = mock_redis_factory()

        monkeypatch.setattr("app.tasks._redis", mock_redis)
        monkeypatch.setattr("app.graph.get_canon", lambda mid: fake_canon)
        monkeypatch.setattr("app.graph.is_chapter_committed", lambda mid, n: False)
        monkeypatch.setattr(
            "app.graph.commit_chapter",
            lambda mid, n, text, extraction, emb: committed.update({(mid, n): text}),
        )
        monkeypatch.setattr("app.graph.mark_needs_review", lambda *a, **kw: None)
        monkeypatch.setattr("app.vectors.similar_passages", lambda *a, **kw: [])
        monkeypatch.setattr("app.vectors.embed_one", lambda text: None)
        monkeypatch.setattr("app.config.settings.fake_violation", 0)
        monkeypatch.setattr("app.config.settings.llm_mode", "fake")

        from app.tasks import generate_chapter_task

        generate_chapter_task.apply(args=["mid_lock", 7, None]).get()

        lock_key = "lock:mid_lock:chapter:7"
        assert mock_redis._store.get(lock_key) is None  # lock was released


class TestGate:
    def test_fake_violation_lands_needs_review(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_canon: dict,
        mock_redis_factory,
    ) -> None:
        """FAKE_VIOLATION=1: tier1 catches the dead-Brann violation on every attempt,
        exhausting all 3 retries and landing NEEDS_REVIEW."""
        committed: dict = {}
        review_calls: list = []

        monkeypatch.setattr("app.tasks._redis", mock_redis_factory())
        monkeypatch.setattr("app.graph.get_canon", lambda mid: fake_canon)
        monkeypatch.setattr("app.graph.is_chapter_committed", lambda mid, n: False)
        monkeypatch.setattr(
            "app.graph.commit_chapter",
            lambda mid, n, text, extraction, emb: committed.update({(mid, n): text}),
        )
        monkeypatch.setattr(
            "app.graph.mark_needs_review",
            lambda mid, n, text, feedback: review_calls.append(
                (mid, n, text, feedback)
            ),
        )
        monkeypatch.setattr("app.vectors.similar_passages", lambda *a, **kw: [])
        monkeypatch.setattr("app.vectors.embed_one", lambda text: None)
        monkeypatch.setattr("app.config.settings.fake_violation", 1)
        monkeypatch.setattr("app.config.settings.llm_mode", "fake")

        from app.tasks import generate_chapter_task

        result = generate_chapter_task.apply(args=["mid_gate", 3, None]).get()

        assert result["status"] == "needs_review"
        assert result["attempts"] == 3
        assert not committed  # commit_chapter must not have been called
        assert len(review_calls) == 1
        _mid, _n, last_text, last_feedback = review_calls[0]
        assert last_text  # last draft stored for human review
        assert last_feedback  # violation strings passed through

    def test_judge_fail_then_pass_commits_on_second_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_canon: dict,
        mock_redis_factory,
    ) -> None:
        """Tier1 passes; judge FAIL on attempt 1 then PASS on attempt 2 → committed."""
        from app.models import Contradiction, JudgeVerdict

        committed: dict = {}
        judge_call_count: list[int] = [0]

        def fake_evaluate(canon: dict, draft: str, llm) -> JudgeVerdict:
            judge_call_count[0] += 1
            if judge_call_count[0] == 1:
                return JudgeVerdict(
                    verdict="FAIL",
                    contradictions=[
                        Contradiction(
                            violated_fact="Brann is dead",
                            draft_evidence="Brann appears in the scene",
                        )
                    ],
                    coherence_score=0.1,
                    reasoning="dead character found",
                )
            return JudgeVerdict(
                verdict="PASS",
                contradictions=[],
                coherence_score=0.9,
                reasoning="no violations",
            )

        monkeypatch.setattr("app.tasks._redis", mock_redis_factory())
        monkeypatch.setattr("app.graph.get_canon", lambda mid: fake_canon)
        monkeypatch.setattr("app.graph.is_chapter_committed", lambda mid, n: False)
        monkeypatch.setattr(
            "app.graph.commit_chapter",
            lambda mid, n, text, extraction, emb: committed.update({(mid, n): text}),
        )
        monkeypatch.setattr("app.graph.mark_needs_review", lambda *a, **kw: None)
        monkeypatch.setattr("app.judge.evaluate", fake_evaluate)
        monkeypatch.setattr("app.vectors.similar_passages", lambda *a, **kw: [])
        monkeypatch.setattr("app.vectors.embed_one", lambda text: None)
        monkeypatch.setattr("app.config.settings.fake_violation", 0)
        monkeypatch.setattr("app.config.settings.llm_mode", "fake")

        from app.tasks import generate_chapter_task

        result = generate_chapter_task.apply(args=["mid_judge_retry", 5, None]).get()

        assert result["status"] == "committed"
        assert result["attempts"] == 2
        assert result["coherence_score"] == 0.9
        assert ("mid_judge_retry", 5) in committed
        assert judge_call_count[0] == 2  # judge called twice: FAIL then PASS


class TestRetryAfterNeedsReview:
    def test_retry_path_recommits_same_chapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_canon: dict,
        mock_redis_factory,
    ) -> None:
        """A NEEDS_REVIEW chapter is retried (same number) and commits cleanly."""
        committed: dict = {}
        review: dict = {}

        monkeypatch.setattr("app.tasks._redis", mock_redis_factory())
        monkeypatch.setattr("app.graph.get_canon", lambda mid: fake_canon)
        monkeypatch.setattr(
            "app.graph.is_chapter_committed", lambda mid, n: (mid, n) in committed
        )
        monkeypatch.setattr(
            "app.graph.commit_chapter",
            lambda mid, n, text, extraction, emb: committed.update({(mid, n): text}),
        )
        monkeypatch.setattr(
            "app.graph.mark_needs_review",
            lambda mid, n, text, feedback: review.update({(mid, n): (text, feedback)}),
        )
        monkeypatch.setattr("app.vectors.similar_passages", lambda *a, **kw: [])
        monkeypatch.setattr("app.vectors.embed_one", lambda text: None)
        monkeypatch.setattr("app.config.settings.llm_mode", "fake")

        from app.tasks import generate_chapter_task

        # First run: FAKE_VIOLATION=1 → chapter 3 lands NEEDS_REVIEW
        monkeypatch.setattr("app.config.settings.fake_violation", 1)
        r1 = generate_chapter_task.apply(args=["mid_retry", 3, None]).get()
        assert r1["status"] == "needs_review"
        assert r1["attempts"] == 3
        assert ("mid_retry", 3) not in committed

        # Second run: FAKE_VIOLATION=0 → SAME chapter 3 commits on attempt 1
        monkeypatch.setattr("app.config.settings.fake_violation", 0)
        r2 = generate_chapter_task.apply(args=["mid_retry", 3, None]).get()
        assert r2["status"] == "committed"  # not already_committed
        assert r2["attempts"] == 1
        assert ("mid_retry", 3) in committed


class TestFakeLLM:
    def test_draft_mentions_alive_characters(self) -> None:
        from app.llm import FakeLLM

        context = (
            "=== CANON FACTS ===\nCHARACTERS:\n"
            "- Mara | alive | docks | smuggler\n"
            "- Brann | dead | citadel | guard\n"
        )
        draft = FakeLLM().draft_chapter(context, None)
        assert "Mara" in draft
        assert "Brann" not in draft  # dead, must not appear without FAKE_VIOLATION

    def test_draft_violation_includes_dead_character(self, monkeypatch) -> None:
        from app.llm import FakeLLM

        monkeypatch.setattr("app.config.settings.fake_violation", 1)
        context = (
            "=== CANON FACTS ===\nCHARACTERS:\n"
            "- Mara | alive | docks | smuggler\n"
            "- Brann | dead | citadel | guard\n"
        )
        draft = FakeLLM().draft_chapter(context, None)
        assert "Brann" in draft

    def test_judge_pass_when_no_dead_in_draft(self) -> None:
        from app.llm import FakeLLM

        canon = "CHARACTERS:\n- Mara | alive | docks | smuggler\n- Brann | dead | citadel | guard"
        draft = "Mara walked along the docks in the early morning."
        verdict = FakeLLM().judge(canon, draft)
        assert verdict.verdict == "PASS"
        assert verdict.coherence_score == 0.9

    def test_judge_fail_when_dead_in_draft(self) -> None:
        from app.llm import FakeLLM

        canon = "CHARACTERS:\n- Mara | alive | docks | smuggler\n- Brann | dead | citadel | guard"
        draft = "Brann walked through the door as if nothing had happened."
        verdict = FakeLLM().judge(canon, draft)
        assert verdict.verdict == "FAIL"
        assert any("Brann" in c.violated_fact for c in verdict.contradictions)

    def test_extract_excludes_violation_name_from_characters(self, monkeypatch) -> None:
        """The dead violator must be in event.involves but NOT in characters."""
        from app.llm import FakeLLM

        monkeypatch.setattr("app.config.settings.fake_violation", 1)
        context = (
            "CHARACTERS:\n"
            "- Mara | alive | docks | smuggler\n"
            "- Brann | dead | citadel | guard\n"
        )
        llm = FakeLLM()
        draft = llm.draft_chapter(context, None)
        extraction = llm.extract(draft)

        char_names = {c.name for c in extraction.characters}
        all_involves = {name for e in extraction.events for name in e.involves}

        assert "Brann" not in char_names  # must not corrupt the graph
        assert "Brann" in all_involves  # tier1_check needs to see this
