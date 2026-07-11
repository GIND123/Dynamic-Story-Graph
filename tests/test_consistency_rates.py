"""Offline tests for score_contradiction (model-as-judge over planted cases).

Verifies the confusion matrix, per-class recall, the two headline rates, and
malformed handling — all with a stub judge_fn, no LLM.
"""

from types import SimpleNamespace

from evals.metrics.contradiction import load_cases, make_judge_fn, score_contradiction


def test_real_cases_json_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 20
    for c in cases:
        assert {"id", "cls", "difficulty", "expected_verdict", "canon", "draft"} <= set(
            c
        )
        assert c["cls"] in ("contradiction", "location", "clean")
        assert c["expected_verdict"] in ("PASS", "FAIL")
        # every planted case is tiered; clean controls are labelled 'clean'
        if c["expected_verdict"] == "FAIL":
            assert c["difficulty"] in ("blatant", "moderate", "subtle")
    # a perfect judge scores zero rates and full recall across every tier
    r = score_contradiction(
        cases, lambda canon, draft: "FAIL" if _is_planted(cases, draft) else "PASS"
    )
    assert r["recall"] == 1.0
    for tier in ("blatant", "moderate", "subtle"):
        assert r["per_difficulty"][tier]["recall"] == 1.0


def _is_planted(cases, draft):
    return any(c["draft"] == draft and c["expected_verdict"] == "FAIL" for c in cases)


def _cases():
    return [
        {
            "expected_verdict": "FAIL",
            "cls": "contradiction",
            "difficulty": "blatant",
            "canon": "c",
            "draft": "dead acts",
        },
        {
            "expected_verdict": "FAIL",
            "cls": "contradiction",
            "difficulty": "subtle",
            "canon": "c",
            "draft": "resurrected",
        },
        {
            "expected_verdict": "FAIL",
            "cls": "location",
            "difficulty": "subtle",
            "canon": "c",
            "draft": "wrong place",
        },
        {"expected_verdict": "PASS", "cls": "clean", "canon": "c", "draft": "all fine"},
        {
            "expected_verdict": "PASS",
            "cls": "clean",
            "canon": "c",
            "draft": "also fine",
        },
    ]


def test_rates_and_confusion_matrix():
    # Judge catches one contradiction, misses the other + the location one,
    # returns None (malformed) on one clean control, PASS on the other.
    def judge_fn(canon, draft):
        if draft == "dead acts":
            return "FAIL"
        if draft == "also fine":
            return None  # malformed
        return "PASS"

    r = score_contradiction(_cases(), judge_fn)

    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 0, 2, 2)
    assert r["malformed"] == 1
    assert r["precision"] == 1.0
    assert r["recall"] == 1 / 3
    # contradiction_rate = 1 - overall recall (planted misses across all classes)
    assert abs(r["contradiction_rate"] - (2 / 3)) < 1e-9
    # location class: 0 of 1 caught -> rate 1.0
    assert r["per_class"]["location"]["recall"] == 0.0
    assert r["location_inconsistency_rate"] == 1.0
    assert r["per_class"]["contradiction"]["n_planted"] == 2
    # per-difficulty: blatant caught (1/1), subtle missed (0/2)
    assert r["per_difficulty"]["blatant"]["recall"] == 1.0
    assert r["per_difficulty"]["subtle"]["recall"] == 0.0
    assert r["per_difficulty"]["subtle"]["n_planted"] == 2


def test_all_caught_zero_rates():
    r = score_contradiction(
        _cases(), lambda canon, draft: "FAIL" if "fine" not in draft else "PASS"
    )
    assert r["recall"] == 1.0
    assert r["contradiction_rate"] == 0.0
    assert r["location_inconsistency_rate"] == 0.0
    assert r["over_flag_rate"] == 0.0  # no clean control was wrongly flagged


def test_flag_everything_is_exposed_by_over_flag_rate():
    # A rubber-stamp-FAIL model: contradiction_rate looks perfect (0.0) but it
    # wrongly rejects every clean control -> over_flag_rate = 1.0 reveals it.
    r = score_contradiction(_cases(), lambda canon, draft: "FAIL")
    assert r["recall"] == 1.0
    assert r["contradiction_rate"] == 0.0  # deceptively "perfect"
    assert r["over_flag_rate"] == 1.0  # ...but flags all clean controls
    assert r["precision"] < 1.0


def test_make_judge_fn_maps_failure_to_none():
    class Boom:
        def judge(self, canon, draft):
            raise RuntimeError("invalid JSON")

    assert make_judge_fn(Boom())("c", "d") is None

    class Ok:
        def judge(self, canon, draft):
            return SimpleNamespace(verdict="FAIL")

    assert make_judge_fn(Ok())("c", "d") == "FAIL"
