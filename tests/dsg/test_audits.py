"""The audit exporters observe the detector and the calculus without changing them."""

from __future__ import annotations

from dsg.eval import calculus_audit as ca
from dsg.eval import detector_audit as da
from dsg.generate.canon import CanonFact, Premise
from dsg.policies.runner import run_policy
from dsg.proposals import FactProposal, WindowProposal
from tests.dsg.test_canon_detector import FACTS, INNOCENT, NOT_RESTATED, RESTATED, VIOLATIONS


def test_the_exported_hit_is_the_one_v1_flags():
    for fact_key, text in INNOCENT + VIOLATIONS:
        fact = FACTS[fact_key]
        hit = da.v1_violation_hit(fact, text)
        violated, evidence = fact.check(text)
        assert (hit is not None) == violated
        if hit:
            assert hit.evidence == evidence
            assert text[hit.start : hit.end] == hit.word


def test_the_exported_restatement_is_the_one_v1_counts():
    for fact_key, text in RESTATED + NOT_RESTATED:
        fact = FACTS[fact_key]
        assert (da.v1_restatement_hit(fact, text) is not None) == fact.mentioned(text)


def test_another_subjects_true_value_is_recognised():
    premise = Premise(
        "s", "t", "x", ["Miriam", "Silas"], [],
        canon=[CanonFact("f0", "Miriam", "eyes", "green", ""),
               CanonFact("f1", "Silas", "eyes", "blue", "")],
    )
    text = "Miriam watched as Silas turned, his blue eyes cold."
    hit = da.v1_violation_hit(premise.canon[0], text)
    assert hit is not None
    other, nearby = da._other_subject(premise, premise.canon[0], hit.word, text, hit.start)
    assert (other, nearby) == ("Silas", True)


def test_a_match_inside_a_hyphenated_word_is_reported():
    text = "her silver-white hair"
    at = text.index("silver")
    assert da._hyphen_compound(text, at, at + 6) == "silver-white"
    assert da._hyphen_compound("a silver locket", 2, 8) == ""


def _proposals() -> list[WindowProposal]:
    facts = [
        [FactProposal("Pip", "location", "the forge", evidence="at the forge")],
        [FactProposal("Pip", "location", "London", evidence="went to London")],
        [FactProposal("Pip", "location", "Paris", "narrated", "in fact he was in Paris")],
    ]
    return [
        WindowProposal(i, i * 100, i * 100 + 100,
                       entities=[{"surface": "Pip"}], facts=f)
        for i, f in enumerate(facts)
    ]


def test_recording_state_leaves_the_log_unchanged_and_sees_the_old_value():
    plain = run_policy(_proposals(), "dsg-full")
    rec = run_policy(_proposals(), "dsg-full", state_cls=ca.RecordingState)
    assert ca._log_rows(rec.state) == ca._log_rows(plain.state)
    assert [(e["old_object"], e["new_object"], e["rule"]) for e in rec.state.events] == [
        ("the forge", "London", "supersede:mutable"),
        ("London", "Paris", "revise:revelation-marker"),
    ]
    assert rec.state.events[1]["old_evidence"] == "went to London"


def _event(i: int, rule: str, book: str, corpus: str = "pdnc", marker: str = "") -> dict:
    return {"rule": rule, "book": book, "corpus": corpus,
            "markers_in_evidence": marker, "markers_in_sentence": "",
            "new_match": "exact", "old_match": "exact", "id": i}


def test_sample_caps_provisional_takes_small_strata_whole_and_marks_doubles():
    events = (
        [_event(i, "elaborate:provisional", f"b{i % 7}") for i in range(400)]
        + [_event(i, "supersede:mutable", f"b{i % 7}", "pdnc" if i % 3 else "litbank")
           for i in range(400)]
        + [_event(i, "revise:immutable", "b1") for i in range(12)]
        + [_event(0, "supersede:mutable", "b2", marker=r"\bin fact\b")]
    )
    rows, design = ca.sample(events, seed=1)
    assert len(rows) == ca.N_ITEMS
    assert design["provisional_share"] <= ca.MAX_PROVISIONAL_SHARE
    assert design["sampled_by_rule"]["revise:immutable"] == 12
    assert any(r["markers_in_evidence"] for r in rows)
    assert sum(r["double_annotate"] == "yes" for r in rows) == ca.N_DOUBLE
    assert rows == ca.sample(events, seed=1)[0], "seeded"
