"""Probes must answer from the prefix only -- the property the protocol rests on."""

from __future__ import annotations

from dsg.data.pdnc import GoldCharacter, GoldMention, Novel
from dsg.eval.mention import build_probes, is_scorable, score_mentions
from dsg.policies.runner import Probe, run_policy
from dsg.proposals import WindowProposal


def _stream():
    return [
        WindowProposal(index=0, start=0, end=100,
                       entities=[{"surface": "Elizabeth", "kind": "name"}]),
        WindowProposal(index=1, start=100, end=200,
                       entities=[{"surface": "Mr. Darcy", "kind": "name"}]),
        WindowProposal(index=2, start=200, end=300,
                       entities=[{"surface": "Mr. Darcy", "kind": "name"}]),
    ]


def test_a_probe_cannot_see_a_name_introduced_later():
    """Asking about Darcy at position 50 must fail: he appears at 100."""
    early = Probe(position=50, surface="Mr. Darcy", gold="2")
    late = Probe(position=250, surface="Mr. Darcy", gold="2")
    result = run_policy(_stream(), "dsg-full", text_length=300, probes=[early, late])

    answers = {a.position: a for a in result.probe_answers}
    assert answers[50].node_id is None, "future text leaked into an earlier probe"
    assert answers[250].node_id is not None


def test_every_probe_is_answered_exactly_once_in_order():
    probes = [Probe(position=p, surface="Elizabeth", gold="1") for p in (10, 120, 210, 290)]
    result = run_policy(_stream(), "dsg-full", text_length=300, probes=probes)
    assert len(result.probe_answers) == len(probes)
    positions = [a.position for a in result.probe_answers]
    assert positions == sorted(positions)
    assert [a.window for a in result.probe_answers] == [0, 1, 2, 2]


def test_probes_are_shared_across_policies_so_only_state_differs():
    probes = [Probe(position=250, surface="Mr. Darcy", gold="2")]
    stateful = run_policy(_stream(), "dsg-full", text_length=300, probes=probes)
    stateless = run_policy(_stream(), "window-only", text_length=300, probes=probes)
    assert len(stateful.probe_answers) == len(stateless.probe_answers) == 1


def test_pronouns_are_excluded_from_probing():
    assert not is_scorable("she")
    assert not is_scorable("Your")
    assert is_scorable("Mr. Darcy")
    assert is_scorable("the young man")


def test_probe_construction_skips_unlinkable_gold():
    novel = Novel(
        book_id="toy",
        text="x" * 300,
        characters=[GoldCharacter("1", "Elizabeth", {"Elizabeth"})],
        mentions=[
            GoldMention(0, 9, "Elizabeth", "Elizabeth"),   # kept
            GoldMention(20, 23, "she", "Elizabeth"),       # pronoun, dropped
            GoldMention(40, 45, "Darcy", "Mr. Darcy"),     # no such gold character
        ],
    )
    probes = build_probes(novel)
    assert [p.surface for p in probes] == ["Elizabeth"]


def test_unanswered_probes_count_against_accuracy_not_coverage():
    novel = Novel(
        book_id="toy", text="x" * 300,
        characters=[GoldCharacter("1", "Elizabeth", {"Elizabeth"})],
        mentions=[GoldMention(10, 19, "Elizabeth", "Elizabeth"),
                  GoldMention(250, 259, "Elizabeth", "Elizabeth")],
    )
    result = run_policy(_stream(), "dsg-full", text_length=300, probes=build_probes(novel))
    score = score_mentions(result, novel)
    assert score.n_probes == 2
    assert score.accuracy <= score.accuracy_answered
    assert 0.0 <= score.coverage <= 1.0
