"""Windowing, proposal parsing, and an end-to-end run with no model weights."""

from __future__ import annotations

from pathlib import Path

from dsg.llm import build_backend
from dsg.proposals import extract_book, load_proposals, parse_proposal, save_proposals
from dsg.schemas import Window
from dsg.windows import iter_windows

TEXT = (
    "Mr. Bennet was so odd a mixture of quick parts. "
    "“My dear Mr. Bennet,” said his lady to him one day. "
    "The stranger walked in from the marshes. He said nothing at all. "
) * 40


def test_windows_tile_the_text_without_gaps_or_overlap():
    windows = [w for w, _ in iter_windows(TEXT, 400, 50)]
    assert windows[0].start == 0
    assert windows[-1].end == len(TEXT)
    for a, b in zip(windows, windows[1:]):
        assert a.end == b.start
    assert "".join(w.text for w in windows) == TEXT


def test_lead_in_only_ever_shows_earlier_text():
    for window, lead in iter_windows(TEXT, 400, 50):
        assert TEXT[max(0, window.start - 50) : window.start] == lead


def test_line_parser_survives_truncation_and_repetition():
    window = Window(0, 0, 10, "x")
    raw = (
        "CHAR | Mr. Bennet | name\n"
        "CHAR | Mr. Bennet | name\n"          # duplicate, must be dropped
        "FACT | Mr. Bennet | location | Longbourn | narrated | at home\n"
        "SAID | My dear Mr | Mrs. Bennet\n"
        "FACT | Mr. Bennet | locat"           # truncated mid-line
    )
    p = parse_proposal(raw, window)
    assert p.parse_ok
    assert [e["surface"] for e in p.entities] == ["Mr. Bennet"]
    assert len(p.facts) == 1 and p.facts[0].object == "Longbourn"
    assert len(p.speech) == 1


def test_json_fallback_still_works():
    window = Window(0, 0, 10, "x")
    raw = '```json\n{"entities":[{"surface":"Pip","kind":"name"}],"facts":[],"speech":[]}\n```'
    p = parse_proposal(raw, window)
    assert p.parse_ok and p.entities[0]["surface"] == "Pip"


def test_unparseable_output_is_flagged_not_swallowed():
    p = parse_proposal("I am sorry, I cannot help with that.", Window(0, 0, 10, "x"))
    assert not p.parse_ok and not p.entities


def test_proposals_round_trip(tmp_path: Path):
    proposals = extract_book(TEXT, build_backend("mock"), window_chars=400, max_windows=5)
    path = tmp_path / "book.jsonl"
    save_proposals(path, proposals, {"model": "mock"})
    meta, back = load_proposals(path)
    assert meta["model"] == "mock"
    assert len(back) == len(proposals)
    assert [p.index for p in back] == [p.index for p in proposals]


def test_policies_diverge_on_the_same_proposal_stream():
    from dsg.policies.runner import run_all

    proposals = extract_book(TEXT, build_backend("mock"), window_chars=400, max_windows=12)
    results = run_all(proposals, text_length=len(TEXT))
    assert set(results) >= {"window-only", "append-only", "dsg-full", "retrospective"}
    # Without cross-window resolution every re-mention starts a fresh node,
    # so the stateless baseline must end up with strictly more entities.
    assert len(results["window-only"].state.live_entities()) > len(
        results["dsg-full"].state.live_entities()
    )
    # And the revising policy must not leave contradictions standing.
    assert len(results["dsg-full"].state.violations) <= len(
        results["append-only"].state.violations
    )
    for name, result in results.items():
        assert result.windows == len(proposals)
        assert len(result.state.trace) == len(proposals)


def test_causal_policies_never_cite_unread_text():
    from dsg.policies.runner import run_policy

    proposals = extract_book(TEXT, build_backend("mock"), window_chars=400, max_windows=12)
    for policy in ("append-only", "dsg-eager", "dsg-full"):
        result = run_policy(proposals, policy, causal=True, text_length=len(TEXT))
        assert not [v for v in result.state.violations if v.code == "I6"]


def _stream_with_links():
    """A minimal book where a description is later revealed to be a named person."""
    from dsg.proposals import FactProposal, SpeechProposal, WindowProposal

    return [
        WindowProposal(
            index=0, start=0, end=100,
            entities=[{"surface": "the stranger", "kind": "description"}],
            facts=[FactProposal("the stranger", "location", "the marshes", evidence="on the marshes")],
        ),
        WindowProposal(
            index=1, start=100, end=200,
            entities=[{"surface": "Magwitch", "kind": "name"},
                      {"surface": "the stranger", "kind": "description"}],
            links=[{"surface": "the stranger", "same_as": "Magwitch"}],
            speech=[SpeechProposal("I am no stranger", "the stranger")],
        ),
        WindowProposal(
            index=2, start=200, end=300,
            entities=[{"surface": "Magwitch", "kind": "name"}],
            facts=[FactProposal("Magwitch", "location", "London", evidence="went to London")],
        ),
    ]


def test_identity_link_resolves_under_every_policy_that_allows_it():
    """Regression: the non-causal seeding path must handle links too."""
    from dsg.policies.runner import run_all

    results = run_all(_stream_with_links(), text_length=300)
    for policy in ("dsg-merge", "dsg-eager", "dsg-full", "retrospective"):
        state = results[policy].state
        named = [n for n in state.live_entities() if "Magwitch" in n.proper_names]
        assert len(named) == 1, policy
        assert "the stranger" in named[0].surfaces, f"{policy} lost the description"
    # append-only cannot merge at all, so the two references stay separate.
    append = results["append-only"].state
    assert len({append.deref(n.id) for n in append.live_entities()}) == 2
    # window-only cannot even carry the description across the boundary.
    assert len(results["window-only"].state.live_entities()) >= 2


def test_deferral_makes_the_reveal_monotone_but_eager_pays_a_rollback():
    from dsg.policies.runner import run_policy

    full = run_policy(_stream_with_links(), "dsg-full", text_length=300)
    eager = run_policy(_stream_with_links(), "dsg-eager", text_length=300)
    assert full.state.op_counts()["rollback"] == 0
    assert eager.state.op_counts()["rollback"] >= 1
    # Same end state, different cost of getting there.
    assert len(full.state.live_entities()) == len(eager.state.live_entities())


def test_link_diagnostics_are_counted():
    from dsg.policies.runner import run_policy

    result = run_policy(_stream_with_links(), "dsg-full", text_length=300)
    assert result.links_proposed == 1
    assert result.links_bound + result.links_merged == 1


def test_identity_questions_are_asked_only_about_people():
    """A model offered 'the pool' beside a cast list will link it to a character."""
    from dsg.proposals import SurfaceLedger, needs_resolution, parse_proposal
    from dsg.schemas import Window

    raw = (
        "CHAR | the pool | description\n"
        "CHAR | a very large jar | description\n"
        "CHAR | the young man | description\n"
        "CHAR | Mrs. Miller | description\n"      # a name the model mislabelled
        "CHAR | Winterbourne | name\n"
    )
    proposal = parse_proposal(raw, Window(0, 0, 10, "x"))
    kinds = {e["surface"]: e["kind"] for e in proposal.entities}
    assert kinds["Mrs. Miller"] == "name", "a proper name is never a description"

    ledger = SurfaceLedger()
    ledger.update(proposal)
    unnamed, names = needs_resolution(proposal, ledger)
    assert unnamed == ["the young man"]
    assert "Mrs. Miller" in names and "Winterbourne" in names
