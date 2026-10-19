"""SUPERSEDE versus REVISE, and which rule decided.

SUPERSEDE means the story world changed; REVISE means the earlier belief was
never true. A marker that only reports *finding out* ("learned that",
"realised") routinely introduces news of a real change, so on a mutable,
single-valued slot it must not force REVISE by itself. v1 lets any revelation
marker decide before mutability is consulted; those cases are strict-xfail
until the classifier is fixed.
"""

from __future__ import annotations

import pytest

from dsg.schemas import Certainty, Op, Span
from dsg.store import POLICIES, CandidateAssertion, NarrativeState


def _state() -> NarrativeState:
    s = NarrativeState(POLICIES["dsg-full"])
    s.step(0, 10_000)
    return s


def _two_step(
    surface: str, predicate: str, before: tuple[str, str], after: tuple[str, str],
    *, certainty_before: Certainty = Certainty.NARRATED,
) -> tuple[Op, NarrativeState]:
    s = _state()
    e = s.observe_entity(surface, Span(0, len(surface)))
    s.apply_assertion(
        CandidateAssertion(e, predicate, before[0], evidence=before[1],
                           certainty=certainty_before)
    )
    s.step(1, 10_000)
    op = s.apply_assertion(CandidateAssertion(e, predicate, after[0], evidence=after[1]))
    return op, s


# (surface, predicate, (earlier object, evidence), (later object, evidence)):
# a revelation-style verb reporting a genuine change on a mutable slot.
WORLD_CHANGES = [
    ("her father", "alive",
     ("alive", "her father was well enough to ride"),
     ("dead", "She learned that her father had died in the night.")),
    ("Harriet", "location",
     ("Highbury", "Harriet was at Highbury"),
     ("London", "Emma realised Harriet had gone to London.")),
    ("Jane", "resides_in",
     ("Longbourn", "Jane lived at Longbourn"),
     ("Netherfield", "He discovered that Jane had moved to Netherfield.")),
    ("Wickham", "location",
     ("Meryton", "Wickham was quartered at Meryton"),
     ("Brighton", "Wickham admitted he was now living in Brighton.")),
]


@pytest.mark.parametrize(
    ("surface", "predicate", "before", "after"),
    [
        pytest.param(
            *c,
            marks=pytest.mark.xfail(
                strict=True, reason="v1: any revelation marker forces REVISE"
            ),
        )
        for c in WORLD_CHANGES
    ],
    ids=[c[0] for c in WORLD_CHANGES],
)
def test_v1_news_of_a_world_change_supersedes(surface, predicate, before, after):
    op, _ = _two_step(surface, predicate, before, after)
    assert op is Op.SUPERSEDE


@pytest.mark.parametrize(
    ("surface", "predicate", "before", "after"), WORLD_CHANGES, ids=[c[0] for c in WORLD_CHANGES]
)
def test_v1_world_change_misclassification_is_attributed_to_the_marker(
    surface, predicate, before, after
):
    """Documents the defect: the log names the rule that produced REVISE."""
    op, s = _two_step(surface, predicate, before, after)
    assert op is Op.REVISE
    assert s.log[-1].rule == "revise:revelation-marker"


def test_an_explicit_correction_still_revises_and_logs_the_marker():
    op, s = _two_step(
        "Pip", "location", ("the forge", "at the forge"),
        ("London", "in fact he had never left London"),
    )
    assert op is Op.REVISE
    assert s.log[-1].rule == "revise:revelation-marker"


def test_reported_to_narrated_logs_its_rule():
    op, s = _two_step(
        "Pip", "location", ("Cairo", "he claimed"), ("Marseille", "he was"),
        certainty_before=Certainty.REPORTED,
    )
    assert op is Op.REVISE
    assert s.log[-1].rule == "revise:reported-to-narrated"


def test_immutable_predicate_logs_its_rule():
    op, s = _two_step(
        "Estella", "parent_of", ("Miss Havisham", "her mother"),
        ("Magwitch", "her real father"),
    )
    assert op is Op.REVISE
    assert s.log[-1].rule == "revise:immutable"


def test_plain_world_change_logs_the_default_rule():
    op, s = _two_step(
        "Pip", "location", ("the forge", "at the forge"), ("London", "went to London")
    )
    assert op is Op.SUPERSEDE
    assert s.log[-1].rule == "supersede:mutable"


@pytest.mark.parametrize(
    ("before", "after", "rule"),
    [
        (("a house", "in a house"), ("Satis House", "at Satis House"),
         "elaborate:underspecified"),
        (("dog", "a dog"), ("Newfoundland dog", "a Newfoundland dog"), "elaborate:refines"),
    ],
)
def test_elaboration_logs_which_rule_licensed_it(before, after, rule):
    op, s = _two_step("Pip", "location", before, after)
    assert op is Op.ELABORATE
    assert s.log[-1].rule == rule


def test_first_assertion_and_reinforcement_are_logged():
    s = _state()
    e = s.observe_entity("Pip", Span(0, 3))
    s.apply_assertion(CandidateAssertion(e, "location", "the forge", evidence="at the forge"))
    assert s.log[-1].op is Op.ASSERT and s.log[-1].rule == "assert:open-slot"
    s.apply_assertion(CandidateAssertion(e, "location", "the forge", evidence="at the forge"))
    assert s.log[-1].op is Op.NOOP and s.log[-1].rule == "noop:reinforced"


def test_a_policy_that_cannot_revise_still_logs_the_rule_it_needed():
    s = NarrativeState(POLICIES["append-only"])
    s.step(0, 10_000)
    e = s.observe_entity("Estella", Span(0, 7))
    s.apply_assertion(CandidateAssertion(e, "parent_of", "Miss Havisham", evidence="x"))
    s.step(1, 10_000)
    op = s.apply_assertion(CandidateAssertion(e, "parent_of", "Magwitch", evidence="y"))
    assert op is Op.ASSERT
    assert s.log[-1].rule == "revise:immutable"
    assert "needed revise" in s.log[-1].detail
