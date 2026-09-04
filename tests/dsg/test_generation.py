"""Planted canon, the memory conditions, and the scorer."""

from __future__ import annotations

from dsg.generate.canon import CanonFact, build_premises, check_chapter
from dsg.generate.conditions import (
    CONDITIONS,
    StoryRun,
    build_chapter_prompt,
    build_memory,
)
from dsg.generate.run import POLICY_FOR, clean_chapter, init_runs
from dsg.generate.score import score_runs


def _fact(**kw):
    base = dict(fact_id="f0", subject="Hesper", kind="eyes", value="grey",
                statement="Hesper has grey eyes.")
    base.update(kw)
    return CanonFact(**base)


def test_a_contradicting_value_near_the_subject_is_a_violation():
    violated, evidence = _fact().check("Hesper turned, her blue eyes cold.")
    assert violated and "blue" in evidence


def test_the_correct_value_is_not_a_violation():
    assert not _fact().check("Hesper turned, her grey eyes cold.")[0]


def test_a_contradicting_word_far_from_the_subject_is_not_attributed():
    text = "Hesper left." + " padding." * 200 + " The blue eyes of the sea."
    assert not _fact().check(text)[0]


def test_a_nearby_correction_is_not_counted_against_the_story():
    """'not gold but silver' asserts the canon, it does not break it."""
    fact = _fact(subject="locket", kind="metal", value="silver",
                 statement="The locket is silver.")
    assert not fact.check("The locket was not gold but silver.")[0]


def test_restatement_is_detected_separately_from_violation():
    fact = _fact()
    assert fact.mentioned("Hesper watched, grey eyes steady.")
    assert not fact.mentioned("Hesper watched in silence.")


def test_aliases_count_as_the_value():
    fact = _fact(subject="ring", kind="metal", value="gold",
                 statement="The ring is gold.")
    assert fact.mentioned("She wore the golden ring.")
    assert fact.check("She wore the iron ring.")[0]


def test_premises_are_deterministic_and_carry_the_requested_canon():
    a = build_premises(n=4, n_canon=8, seed=0)
    b = build_premises(n=4, n_canon=8, seed=0)
    assert [p.to_json() for p in a] == [p.to_json() for p in b]
    assert all(len(p.canon) == 8 for p in a)
    assert len({p.story_id for p in a}) == 4


def test_canon_is_given_in_chapter_one_and_withheld_afterwards():
    premise = build_premises(1)[0]
    run = StoryRun(story_id=premise.story_id, condition="none")
    first = build_chapter_prompt(premise, run, 1, 8)
    assert premise.canon[0].statement in first

    run.chapters.append("prose")
    later = build_chapter_prompt(premise, run, 2, 8)
    assert premise.canon[0].statement not in later, "memory, not prompt-following"


def test_each_condition_carries_a_different_memory():
    premise = build_premises(1)[0]
    memories = {}
    for condition in CONDITIONS:
        run = StoryRun(story_id=premise.story_id, condition=condition,
                       chapters=["chapter one text"], summary="a summary")
        if condition in POLICY_FOR:
            run.state = init_runs([premise], (condition,))[0].state
        memories[condition] = build_memory(run)
    assert memories["none"] == ""
    assert "chapter one text" in memories["last-chapter"]
    assert "a summary" in memories["rolling-summary"]
    assert "chapter one text" in memories["full-context"]
    for condition in ("append-only-state", "dsg-state"):
        assert "canon" in memories[condition].lower()


def test_state_conditions_use_the_intended_policies():
    premise = build_premises(1)[0]
    runs = {r.condition: r for r in init_runs([premise])}
    assert runs["append-only-state"].state.policy.name == "append-only"
    assert runs["dsg-state"].state.policy.name == "dsg-full"
    assert runs["none"].state is None


def test_full_context_truncation_keeps_the_opening():
    """The canon is established in chapter 1, so a fair baseline keeps it."""
    run = StoryRun(story_id="s", condition="full-context",
                   chapters=["OPENING " + "x" * 4000] + ["y" * 4000 for _ in range(9)])
    memory = build_memory(run, budget_chars=6000)
    assert "OPENING" in memory
    assert len(memory) < 8000


def test_chapter_cleaning_strips_scaffolding():
    assert clean_chapter("CHAPTER 3\nThe rain came.").startswith("The rain came")
    assert clean_chapter("  The rain came.  ") == "The rain came."


def test_scoring_is_monotone_and_uses_first_violation():
    premise = build_premises(1)[0]
    fact = premise.canon[0]
    payload = {
        "meta": {"chapters": 3, "stories": 1, "conditions": ["none"]},
        "premises": [premise.to_json()],
        "records": [
            {"story_id": premise.story_id, "condition": "none", "chapter": 1,
             "chars": 100, "violations": [], "restated": [fact.fact_id],
             "evidence": {}},
            {"story_id": premise.story_id, "condition": "none", "chapter": 2,
             "chars": 100, "violations": [fact.fact_id], "restated": [],
             "evidence": {fact.fact_id: "blue eyes"}},
            {"story_id": premise.story_id, "condition": "none", "chapter": 3,
             "chars": 100, "violations": [fact.fact_id], "restated": [],
             "evidence": {fact.fact_id: "blue eyes"}},
        ],
    }
    score = score_runs(payload)[0]
    assert score.cumulative == sorted(score.cumulative), "violations never un-break"
    assert score.first_violation[fact.fact_id] == 2
    assert score.violated == {fact.fact_id}
    assert score.total_chars == 300
