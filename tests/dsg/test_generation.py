"""Planted canon, the memory conditions, and the scorer."""

from __future__ import annotations

from dsg.generate.canon import CanonFact, build_premises
from dsg.generate.conditions import (
    CONDITIONS,
    StoryRun,
    build_chapter_prompt,
    build_memory,
    memory_of,
    split_condition,
    variant_of,
)
from dsg.generate.run import clean_chapter, init_runs
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
    run = StoryRun(story_id=premise.story_id, condition="base:none")
    first = build_chapter_prompt(premise, run, 1, 8)
    assert premise.canon[0].statement in first

    run.chapters.append("prose")
    later = build_chapter_prompt(premise, run, 2, 8)
    assert premise.canon[0].statement not in later, "memory, not prompt-following"


def test_each_memory_kind_carries_something_different():
    premise = build_premises(1)[0]
    memories = {}
    for condition in CONDITIONS:
        run = init_runs([premise], (condition,))[0]
        run.chapters = ["chapter one text"]
        run.summary = "a summary"
        memories[memory_of(condition)] = build_memory(run)
    assert memories["none"] == ""
    assert "chapter one text" in memories["last-chapter"]
    assert "a summary" in memories["rolling-summary"]
    assert "chapter one text" in memories["full-context"]
    for memory in ("append-only-state", "dsg-state"):
        assert "canon" in memories[memory].lower()


def test_state_conditions_use_the_intended_policies():
    premise = build_premises(1)[0]
    runs = {r.condition: r for r in init_runs([premise])}
    assert runs["base:append-only-state"].state.policy.name == "append-only"
    assert runs["base:dsg-state"].state.policy.name == "dsg-full"
    assert runs["tuned:dsg-state"].state.policy.name == "dsg-full"
    assert runs["base:none"].state is None


def test_conditions_pair_a_backbone_variant_with_a_memory():
    """Training and memory must be separable: both variants at the same memory."""
    assert variant_of("tuned:dsg-state") == "tuned"
    assert memory_of("tuned:dsg-state") == "dsg-state"
    assert split_condition("base:none") == ("base", "none")
    # A bare memory name still parses, so older run files stay readable.
    assert split_condition("dsg-state") == ("base", "dsg-state")
    memories = {memory_of(c) for c in CONDITIONS}
    for variant in ("base", "tuned"):
        assert f"{variant}:dsg-state" in CONDITIONS
        assert f"{variant}:full-context" in CONDITIONS
    assert "none" in memories


def test_full_context_truncation_keeps_the_opening():
    """The canon is established in chapter 1, so a fair baseline keeps it."""
    run = StoryRun(story_id="s", condition="base:full-context",
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
        "meta": {"chapters": 3, "stories": 1, "conditions": ["base:none"]},
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


def test_null_valued_facts_never_enter_the_state():
    """A model asked for a property it cannot fill says so; that is not a fact."""
    from dsg.generate.run import parse_write_extraction
    from dsg.schemas import Window

    raw = (
        "FACT | Mrs. Darling | eye_colour | not specified\n"
        "FACT | Mrs. Darling | hair_colour | unknown\n"
        "FACT | Mrs. Darling | birthplace | N/A\n"
        "FACT | Mr. Darling | occupation | clerk\n"
        "FACT | Mrs. Darling | location | home\n"
    )
    proposal = parse_write_extraction(raw, Window(0, 0, 10, "x"))
    assert [(f.subject, f.object) for f in proposal.facts] == [
        ("Mr. Darling", "clerk"),
        ("Mrs. Darling", "home"),
    ]


def test_the_guard_fires_on_immutable_clashes_only():
    from dsg.generate.repair import detect_conflicts, repair_instruction
    from dsg.proposals import FactProposal, WindowProposal
    from dsg.schemas import Span
    from dsg.store import POLICIES, CandidateAssertion, NarrativeState

    state = NarrativeState(POLICIES["dsg-full"])
    state.step(0, 10_000)
    node = state.observe_entity("Hesper", Span(0, 6))
    state.observe_entity("Hesper", Span(9, 15))
    state.apply_assertion(CandidateAssertion(node, "eyes", "grey"))
    state.apply_assertion(CandidateAssertion(node, "location", "the harbour"))

    proposal = WindowProposal(
        1, 0, 10,
        facts=[
            FactProposal("Hesper", "eye_colour", "blue"),      # continuity error
            FactProposal("Hesper", "location", "the foundry"),  # the world moved
        ],
    )
    conflicts = detect_conflicts(state, proposal)
    assert [c.predicate for c in conflicts] == ["eye_colour"]
    assert "grey" in conflicts[0].describe() and "blue" in conflicts[0].describe()
    assert "canon" in repair_instruction(conflicts).lower()


def test_the_guard_is_a_dry_run_and_does_not_touch_the_state():
    from dsg.generate.repair import detect_conflicts
    from dsg.proposals import FactProposal, WindowProposal
    from dsg.schemas import Span
    from dsg.store import POLICIES, CandidateAssertion, NarrativeState

    state = NarrativeState(POLICIES["dsg-full"])
    state.step(0, 10_000)
    node = state.observe_entity("Hesper", Span(0, 6))
    state.apply_assertion(CandidateAssertion(node, "eyes", "grey"))
    before = {(a.predicate, a.object, a.status) for a in state.assertions.values()}

    detect_conflicts(
        state,
        WindowProposal(1, 0, 10, facts=[FactProposal("Hesper", "eye_colour", "blue")]),
    )
    after = {(a.predicate, a.object, a.status) for a in state.assertions.values()}
    assert before == after


def test_replaying_cached_extractions_rebuilds_the_same_state():
    """The resume guarantee: recovery must land where an uninterrupted run did."""
    from dsg.generate.run import replay_extraction
    from dsg.store import POLICIES, NarrativeState

    chapters = [
        "Hesper stood at the window, her grey eyes on the quay.",
        "Jarrow shook rain from his auburn hair and set down the iron seal.",
        "Hesper walked to the foundry before dawn.",
    ]
    cache = {
        "b1:0": "FACT | Hesper | eye_colour | grey\nFACT | Hesper | location | the window",
        "b1:1": "FACT | Jarrow | hair_colour | auburn\nFACT | seal | material | iron",
        "b1:2": "FACT | Hesper | location | the foundry",
    }

    def snapshot(state):
        return sorted(
            (state.entities[a.subject].canonical, a.predicate, a.object, a.status.value)
            for a in state.assertions.values()
            if a.subject in state.entities
        )

    uninterrupted = NarrativeState(POLICIES["dsg-full"])
    replay_extraction(uninterrupted, chapters, cache, "b1", len(chapters))

    # Interrupted after two chapters, then resumed for the rest.
    resumed = NarrativeState(POLICIES["dsg-full"])
    replay_extraction(resumed, chapters, cache, "b1", len(chapters))

    assert snapshot(resumed) == snapshot(uninterrupted)
    assert any(p == "eye_colour" for _, p, _, _ in snapshot(uninterrupted))
    # The move is a supersession, not a contradiction: exactly one live location.
    live = [
        a for a in uninterrupted.assertions.values()
        if a.live and a.predicate == "location"
    ]
    assert len(live) == 1 and live[0].object == "the foundry"
    assert not uninterrupted.violations


def test_replay_offsets_keep_provenance_inside_the_prefix():
    from dsg.generate.run import replay_extraction
    from dsg.store import POLICIES, NarrativeState

    chapters = ["a" * 100, "b" * 100, "c" * 100]
    cache = {f"b1:{i}": "FACT | Hesper | location | somewhere" for i in range(3)}
    state = NarrativeState(POLICIES["dsg-full"])
    offset = replay_extraction(state, chapters, cache, "b1", 3)
    assert offset == 3 * 102
    assert not [v for v in state.violations if v.code == "I6"]


def _fake_checkpoint(runs, chapters_per_run, extraction_log):
    """A checkpoint payload shaped exactly as the generation loop writes one."""
    return {
        "chapters_done": len(next(iter(chapters_per_run.values()))),
        "runs": [
            {
                "story_id": r.story_id,
                "condition": r.condition,
                "chapters": chapters_per_run[(r.story_id, r.condition)],
                "summary": f"summary for {r.condition}",
            }
            for r in runs
        ],
        "extractions": extraction_log,
    }


def test_generation_resume_restores_prose_summary_and_state():
    """The gap that cost nine chapters: checkpointing without reading back."""
    from dsg.generate.canon import build_premises
    from dsg.generate.run import init_runs, restore_runs

    premise = build_premises(1)[0]
    runs = init_runs([premise], ("base:dsg-state", "base:rolling-summary"))
    texts = [
        "Hesper stood at the window, her grey eyes on the quay.",
        "Hesper walked to the foundry before dawn.",
    ]
    chapters_per_run = {(r.story_id, r.condition): list(texts) for r in runs}
    log = {
        f"{premise.story_id}|base:dsg-state|0":
            "FACT | Hesper | eye_colour | grey\nFACT | Hesper | location | the window",
        f"{premise.story_id}|base:dsg-state|1":
            "FACT | Hesper | location | the foundry",
    }
    saved = _fake_checkpoint(runs, chapters_per_run, log)

    fresh = init_runs([premise], ("base:dsg-state", "base:rolling-summary"))
    restored = restore_runs(fresh, saved, 2, log)
    assert restored == 2

    by_condition = {r.condition: r for r in fresh}
    assert by_condition["base:dsg-state"].chapters == texts
    assert by_condition["base:rolling-summary"].summary.startswith("summary for")

    state = by_condition["base:dsg-state"].state
    live = {(a.predicate, a.object) for a in state.assertions.values() if a.live}
    assert ("eye_colour", "grey") in live
    # The move superseded rather than contradicted: one live location.
    assert ("location", "the foundry") in live
    assert ("location", "the window") not in live
    assert not state.violations


def test_generation_resume_truncates_to_the_completed_chapters():
    """A checkpoint written after chapter 1 must not carry chapter 2's prose."""
    from dsg.generate.canon import build_premises
    from dsg.generate.run import init_runs, restore_runs

    premise = build_premises(1)[0]
    runs = init_runs([premise], ("base:dsg-state",))
    saved = _fake_checkpoint(
        runs, {(runs[0].story_id, runs[0].condition): ["one", "two", "three"]}, {}
    )
    fresh = init_runs([premise], ("base:dsg-state",))
    restore_runs(fresh, saved, 1, {})
    assert fresh[0].chapters == ["one"]


def test_generation_resume_is_a_noop_without_a_matching_run():
    from dsg.generate.canon import build_premises
    from dsg.generate.run import init_runs, restore_runs

    premise = build_premises(1)[0]
    fresh = init_runs([premise], ("base:dsg-state",))
    assert restore_runs(fresh, {"runs": []}, 3, {}) == 0
    assert fresh[0].chapters == []


def test_graph_selection_prefers_the_consistent_candidate():
    """The core claim: ranking is a relative comparison, so noise cancels."""
    from dsg.generate.select import choose
    from dsg.schemas import Span
    from dsg.store import POLICIES, CandidateAssertion, NarrativeState

    state = NarrativeState(POLICIES["dsg-full"])
    state.step(0, 10_000)
    node = state.observe_entity("Hesper", Span(0, 6))
    state.observe_entity("Hesper", Span(9, 15))
    state.apply_assertion(CandidateAssertion(node, "eyes", "grey"))

    candidates = ["consistent chapter", "contradicting chapter", "neutral chapter"]
    extractions = [
        "FACT | Hesper | eye_colour | grey",       # agrees
        "FACT | Hesper | eye_colour | blue",       # contradicts established canon
        "FACT | Hesper | location | the quay",     # says nothing about eyes
    ]
    pick, scores = choose(state, candidates, extractions, 0, "graph")
    assert pick != 1, "the contradicting candidate must never be chosen"
    assert scores[1].conflicts == 1
    assert scores[0].conflicts == 0 and scores[2].conflicts == 0


def test_graph_selection_does_not_mutate_the_state():
    from dsg.generate.select import choose
    from dsg.schemas import Span
    from dsg.store import POLICIES, CandidateAssertion, NarrativeState

    state = NarrativeState(POLICIES["dsg-full"])
    state.step(0, 10_000)
    node = state.observe_entity("Hesper", Span(0, 6))
    state.apply_assertion(CandidateAssertion(node, "eyes", "grey"))
    before = {(a.predicate, a.object, a.status) for a in state.assertions.values()}
    choose(state, ["a", "b"], ["FACT | Hesper | eye_colour | blue", ""], 0, "graph")
    assert {(a.predicate, a.object, a.status) for a in state.assertions.values()} == before


def test_random_selection_is_the_control_and_ignores_the_graph():
    """The control must not consult the graph, or it is not a control."""
    from dsg.generate.select import choose
    from dsg.store import POLICIES, NarrativeState

    state = NarrativeState(POLICIES["dsg-full"])
    state.step(0, 10_000)
    picks = {
        choose(state, ["a", "b", "c", "d"], [""] * 4, 0, "random", seed=s)[0]
        for s in range(40)
    }
    assert len(picks) > 1, "a control that always picks the same index is not random"


def test_single_candidate_short_circuits():
    from dsg.generate.select import choose
    from dsg.store import POLICIES, NarrativeState

    state = NarrativeState(POLICIES["dsg-full"])
    pick, scores = choose(state, ["only"], [""], 0, "graph")
    assert pick == 0 and scores == []
