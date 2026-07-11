"""Offline tests for the per-model profile runner (no Neo4j, no LLM, no network).

Verifies the assembled JSON shape, rate wiring, quote-attribution skipping when
there is no graph, and resume/skip behaviour (cached tasks are not recomputed).
"""

from pathlib import Path
from types import SimpleNamespace

from evals.run_model_profile import run_profile


class CountingAsk:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, prompt: str, meta: dict) -> str:
        self.calls += 1
        return self.answer


class CountingJudge:
    """Always PASS -> every planted contradiction is 'missed' (rate = 1.0)."""

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, canon: str, draft: str):
        self.calls += 1
        return SimpleNamespace(verdict="PASS")


def _cloze_root(tmp_path: Path) -> str:
    d = tmp_path / "model_output" / "chatgpt_results"
    d.mkdir(parents=True, exist_ok=True)
    rows = ["<name>x</name>\tx\tAlice\t[MASK] a", "<name>x</name>\tx\tBob\t[MASK] b"]
    (d / "book.txt").write_text("\n".join(rows), encoding="utf-8")
    return str(tmp_path)


def _run(tmp_path, ask, judge, existing=None, force=False, coref_docs=None):
    return run_profile(
        "fake-model",
        ask=ask,
        judge_llm=judge,
        driver=None,  # no graph -> quote attribution skipped
        novel_dir=None,
        manuscript_id="pdnc:PrideAndPrejudice",
        books=["book"],
        cloze_root=_cloze_root(tmp_path),
        coref_docs=coref_docs if coref_docs is not None else [],  # no coref by default
        litbank_root=str(tmp_path),
        sample_quotes=5,
        max_cloze=100,
        max_pairs=20,
        seed=0,
        base_url="http://tunnel",
        existing=existing,
        force=force,
    )


def test_profile_shape_and_rates(tmp_path):
    ask, judge = CountingAsk("Alice"), CountingJudge()
    profile = _run(tmp_path, ask, judge)

    assert profile["model"] == "fake-model"
    assert profile["base_url"] == "http://tunnel"

    cloze = profile["tasks"]["name_cloze"]["aggregate"]
    assert cloze["n"] == 2
    assert cloze["accuracy"] == 0.5  # Alice correct, Bob -> Alice (drift)
    assert cloze["entity_drift_rate"] == 0.5

    # quote attribution skipped without a driver
    assert "skipped" in profile["tasks"]["quote_attribution"]

    rates = profile["rates"]
    assert rates["entity_drift"] == 0.5  # no coref -> cloze proxy
    assert rates["entity_drift_source"] == "cloze_proxy"
    assert rates["contradiction"] == 1.0  # judge always PASS -> all missed
    assert rates["location_inconsistency"] == 1.0


def _coref_root(tmp_path: Path) -> None:
    """Write a tiny CoNLL doc: cluster 0 {John, He}, cluster 1 {Mary, her}."""
    d = tmp_path / "coref" / "conll"
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        "d 0 0 John _ (0)",
        "d 0 1 saw _ _",
        "d 0 2 Mary _ (1)",
        "d 0 3 . _ _",
        "",
        "d 0 0 He _ (0)",
        "d 0 1 smiled _ _",
        "d 0 2 at _ _",
        "d 0 3 her _ (1)",
        "d 0 4 . _ _",
    ]
    (d / "doc.conll").write_text("\n".join(lines), encoding="utf-8")


def test_coreference_overrides_cloze_for_entity_drift(tmp_path):
    _coref_root(tmp_path)
    # "yes" to every coref pair -> merges negatives, accuracy 0.5, drift 0.5.
    # As a cloze answer "yes" is never a valid character -> cloze drift 0.0.
    ask, judge = CountingAsk("yes"), CountingJudge()
    profile = _run(tmp_path, ask, judge, coref_docs=["doc"])

    co = profile["tasks"]["coreference"]["aggregate"]
    assert co["n"] >= 3  # positives + at least one hard negative
    assert co["false_merge_rate"] == 1.0  # "yes" merges every negative pair
    # cloze "yes" is never a valid character -> cloze proxy drift is 0.0
    assert profile["tasks"]["name_cloze"]["aggregate"]["entity_drift_rate"] == 0.0
    # but the headline entity_drift now comes from coref, not the cloze proxy
    assert profile["rates"]["entity_drift"] == co["entity_drift_rate"]
    assert profile["rates"]["entity_drift"] > 0.0
    assert profile["rates"]["entity_drift_source"] == "coreference"


def test_resume_skips_completed_tasks(tmp_path):
    ask, judge = CountingAsk("Alice"), CountingJudge()
    first = _run(tmp_path, ask, judge)
    ask_calls, judge_calls = ask.calls, judge.calls
    assert ask_calls > 0 and judge_calls > 0

    # Re-run with the previous profile as `existing`; nothing should recompute.
    second = _run(tmp_path, ask, judge, existing=first, force=False)
    assert ask.calls == ask_calls
    assert judge.calls == judge_calls
    assert second["tasks"]["name_cloze"] == first["tasks"]["name_cloze"]


def test_force_recomputes(tmp_path):
    ask, judge = CountingAsk("Alice"), CountingJudge()
    first = _run(tmp_path, ask, judge)
    ask_calls = ask.calls
    _run(tmp_path, ask, judge, existing=first, force=True)
    assert ask.calls > ask_calls  # cloze ran again
