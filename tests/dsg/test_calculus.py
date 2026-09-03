"""The revision calculus: the behaviours the paper's claims rest on."""

from __future__ import annotations

import pytest

from dsg.schemas import CommitLevel, Op, Span, Status
from dsg.store import POLICIES, CandidateAssertion, NarrativeState


def state(policy: str = "dsg-full") -> NarrativeState:
    s = NarrativeState(POLICIES[policy])
    s.step(0, 10_000)
    return s


def test_underspecified_object_is_elaborated_not_contradicted():
    s = state()
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(CandidateAssertion(e, "location", "a house", evidence="in a house"))
    s.step(1, 10_000)
    op = s.apply_assertion(
        CandidateAssertion(e, "location", "Satis House", evidence="at Satis House")
    )
    assert op is Op.ELABORATE
    live = [a for a in s.assertions.values() if a.live]
    assert len(live) == 1 and live[0].object == "Satis House"
    assert not any(a.status is Status.RETRACTED for a in s.assertions.values())


def test_world_change_supersedes_and_keeps_history():
    s = state()
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(CandidateAssertion(e, "location", "the forge", evidence="at the forge"))
    s.step(5, 10_000)
    op = s.apply_assertion(CandidateAssertion(e, "location", "London", evidence="went to London"))
    assert op is Op.SUPERSEDE
    old = next(a for a in s.assertions.values() if a.object == "the forge")
    assert old.status is Status.SUPERSEDED
    assert old.valid_to == 5 and old.valid_from == 0  # history preserved, not deleted
    assert sum(1 for a in s.assertions.values() if a.live) == 1


def test_immutable_predicate_conflict_is_a_revision():
    s = state()
    e = s.observe_entity("Estella", Span(0, 7))
    s.apply_assertion(CandidateAssertion(e, "parent_of", "Miss Havisham", evidence="her mother"))
    s.step(9, 10_000)
    op = s.apply_assertion(
        CandidateAssertion(e, "parent_of", "Magwitch", evidence="her real father")
    )
    assert op is Op.REVISE
    old = next(a for a in s.assertions.values() if a.object == "Miss Havisham")
    assert old.status is Status.RETRACTED  # never was true, so not kept as history


def test_revelation_marker_promotes_supersede_to_revise():
    s = state()
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(CandidateAssertion(e, "occupation", "blacksmith", evidence="a blacksmith"))
    s.step(3, 10_000)
    op = s.apply_assertion(
        CandidateAssertion(
            e, "occupation", "gentleman", evidence="in fact he had never been a smith"
        )
    )
    assert op is Op.REVISE, "an explicit correction is not a world change"


def test_reported_belief_yields_to_narration():
    from dsg.schemas import Certainty

    s = state()
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(
        CandidateAssertion(
            e, "occupation", "clerk", certainty=Certainty.REPORTED, evidence="he claimed"
        )
    )
    s.step(4, 10_000)
    op = s.apply_assertion(
        CandidateAssertion(
            e, "occupation", "thief", certainty=Certainty.NARRATED, evidence="he was"
        )
    )
    assert op is Op.REVISE


def test_deferred_commitment_makes_identity_reveal_monotone():
    s = state("dsg-full")
    stranger = s.observe_entity("the stranger", Span(0, 12))
    assert s.entities[stranger].commit is CommitLevel.PROVISIONAL
    s.step(7, 10_000)
    named = s.observe_entity("Magwitch", Span(50, 58))
    op = s.merge_entities(named, stranger, reason="revealed")
    assert op is Op.MERGE_PROVISIONAL
    assert not any(r.rollback for r in s.log), "deferral should avoid rollbacks"


def test_eager_commitment_forces_a_rollback_for_the_same_reveal():
    s = state("dsg-eager")
    stranger = s.observe_entity("the stranger", Span(0, 12))
    assert s.entities[stranger].commit is CommitLevel.COMMITTED
    s.step(7, 10_000)
    named = s.observe_entity("Magwitch", Span(50, 58))
    op = s.merge_entities(named, stranger, reason="revealed")
    assert op is Op.MERGE_COMMITTED
    assert any(r.rollback for r in s.log)


def test_append_only_leaves_the_contradiction_live():
    s = state("append-only")
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(CandidateAssertion(e, "location", "the forge", evidence="at the forge"))
    s.close_step()
    s.step(5, 10_000)
    s.apply_assertion(CandidateAssertion(e, "location", "London", evidence="went to London"))
    violations = s.close_step()
    assert sum(1 for a in s.assertions.values() if a.live) == 2
    assert any(v.code == "I1" for v in violations)


def test_merge_repoints_assertions_and_reconciles_the_slot():
    s = state()
    a = s.observe_entity("the stranger", Span(0, 12))
    b = s.observe_entity("Magwitch", Span(20, 28))
    s.apply_assertion(CandidateAssertion(a, "location", "the marshes", evidence="on the marshes"))
    s.step(4, 10_000)
    s.apply_assertion(CandidateAssertion(b, "location", "London", evidence="in London"))
    s.merge_entities(b, a, reason="revealed")
    kept = s.deref(b)
    assert all(x.subject == kept for x in s.assertions.values())
    assert sum(1 for x in s.assertions.values() if x.live) == 1
    assert not s.close_step(), "a merge must not leave the store inconsistent"


def test_surname_alone_never_merges_two_characters():
    s = state()
    jane = s.observe_entity("Jane Bennet", Span(0, 11))
    liz = s.observe_entity("Elizabeth Bennet", Span(20, 36))
    assert s.deref(jane) != s.deref(liz)
    s.observe_entity("Bennet", Span(40, 46))
    assert s.deref(jane) != s.deref(liz)


def test_prefix_causality_violation_is_detected():
    s = state()
    s.step(0, 100)
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(
        CandidateAssertion(e, "location", "London", span=Span(500, 520), evidence="later")
    )
    violations = s.close_step()
    assert any(v.code == "I6" for v in violations)


@pytest.mark.parametrize("policy", list(POLICIES))
def test_every_policy_runs_and_logs(policy):
    s = NarrativeState(POLICIES[policy])
    s.step(0, 1000)
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(CandidateAssertion(e, "location", "the forge"))
    s.close_step()
    assert s.log and s.trace


def test_append_only_cannot_elaborate_and_so_holds_both_values():
    """Refinement is still a write. A store that only appends must hold both."""
    s = state("append-only")
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(CandidateAssertion(e, "location", "a house", evidence="in a house"))
    s.close_step()
    s.step(1, 10_000)
    op = s.apply_assertion(
        CandidateAssertion(e, "location", "Satis House", evidence="at Satis House")
    )
    assert op is Op.ASSERT
    assert sum(1 for a in s.assertions.values() if a.live) == 2
    assert any(v.code == "I1" for v in s.close_step())


def test_the_same_refinement_is_monotone_under_the_full_calculus():
    s = state("dsg-full")
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(CandidateAssertion(e, "location", "a house", evidence="in a house"))
    s.close_step()
    s.step(1, 10_000)
    assert (
        s.apply_assertion(
            CandidateAssertion(e, "location", "Satis House", evidence="at Satis House")
        )
        is Op.ELABORATE
    )
    assert sum(1 for a in s.assertions.values() if a.live) == 1
    assert not s.close_step()


def test_inconsistent_slot_rate_is_bounded_and_reflects_breakage():
    from dsg import invariants

    broken = state("append-only")
    e = broken.observe_entity("Pip", Span(0, 3))
    broken.apply_assertion(CandidateAssertion(e, "location", "the forge"))
    broken.step(1, 10_000)
    broken.apply_assertion(CandidateAssertion(e, "location", "London"))
    rate, bad, total = invariants.inconsistent_slot_rate(broken.assertions.values())
    assert (bad, total) == (1, 1) and rate == 1.0

    clean = state("dsg-full")
    e = clean.observe_entity("Pip", Span(0, 3))
    clean.apply_assertion(CandidateAssertion(e, "location", "the forge"))
    clean.step(1, 10_000)
    clean.apply_assertion(CandidateAssertion(e, "location", "London"))
    rate, bad, total = invariants.inconsistent_slot_rate(clean.assertions.values())
    assert (bad, total) == (0, 1) and rate == 0.0


def test_merge_of_separately_named_characters_is_refused():
    """A passing mention may not assert that two named characters are one."""
    s = state("dsg-full")
    jane = s.observe_entity("Jane Bennet", Span(0, 11))
    s.observe_entity("Jane Bennet", Span(20, 31))
    liz = s.observe_entity("Elizabeth Bennet", Span(40, 56))
    s.observe_entity("Elizabeth Bennet", Span(60, 76))
    op = s.merge_entities(liz, jane, reason="link")
    assert op is Op.NOOP
    assert s.deref(jane) != s.deref(liz)
    # An explicit revelation is the one thing that licenses it.
    assert s.merge_entities(liz, jane, reason="revealed", revealed=True) is not Op.NOOP


def test_titles_distinguish_people_who_share_a_surname():
    s = state("dsg-full")
    mrs = s.observe_entity("Mrs. Bennet", Span(0, 11))
    miss = s.observe_entity("Miss Bennet", Span(20, 31))
    mr = s.observe_entity("Mr. Bennet", Span(40, 50))
    assert len({s.deref(mrs), s.deref(miss), s.deref(mr)}) == 3
