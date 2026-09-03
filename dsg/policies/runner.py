"""Drive the store from a cached proposal stream under one policy.

The same ``WindowProposal`` list is replayed for every policy, so extraction is
held exactly constant and any difference in the results is attributable to the
representation and its update rules alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dsg.lexicon import is_proper_name
from dsg.proposals import WindowProposal
from dsg.schemas import Certainty, EntityNode, Op, Span
from dsg.store import POLICIES, CandidateAssertion, NarrativeState, PolicyConfig

_CERTAINTY = {
    "narrated": Certainty.NARRATED,
    "reported": Certainty.REPORTED,
    "implied": Certainty.IMPLIED,
}


@dataclass(slots=True)
class SpeakerCall:
    """A quote the extractor attributed, plus how the state resolved it."""

    window: int
    char_start: int
    char_end: int
    quote_cue: str
    speaker_surface: str
    node_id: str | None


@dataclass(slots=True)
class Probe:
    """A question put to the state at a fixed point in the book."""

    position: int
    surface: str
    gold: str


@dataclass(slots=True)
class ProbeAnswer:
    position: int
    surface: str
    gold: str
    window: int
    node_id: str | None


@dataclass(slots=True)
class RunResult:
    policy: str
    state: NarrativeState
    speaker_calls: list[SpeakerCall] = field(default_factory=list)
    windows: int = 0
    parse_failures: int = 0
    links_proposed: int = 0
    links_bound: int = 0      # description attached to a known node
    links_merged: int = 0     # two existing nodes folded together
    links_unresolved: int = 0 # target not in state yet
    probe_answers: list[ProbeAnswer] = field(default_factory=list)

    @property
    def trace(self) -> list[dict[str, int]]:
        return self.state.trace


def _confidence(certainty: str) -> float:
    return {"narrated": 0.7, "reported": 0.45, "implied": 0.4}.get(certainty, 0.6)


def run_policy(
    proposals: list[WindowProposal],
    policy: str | PolicyConfig,
    causal: bool = True,
    text_length: int | None = None,
    probes: list[Probe] | None = None,
) -> RunResult:
    config = POLICIES[policy] if isinstance(policy, str) else policy
    state = NarrativeState(config)
    result = RunResult(policy=config.name, state=state, windows=len(proposals))

    if not causal:
        # The oracle's only advantage: it may see every mention and identity
        # link in the book before committing to an entity inventory. Fact
        # application still runs in order, so its state remains interpretable.
        _seed_lookahead(state, proposals)

    pending = sorted(probes or [], key=lambda p: p.position)
    cursor = 0

    for proposal in proposals:
        if not proposal.parse_ok:
            result.parse_failures += 1
        prefix_end = text_length if not causal and text_length else proposal.end
        state.step(proposal.index, prefix_end)
        span = Span(proposal.start, proposal.end)

        for entity in proposal.entities:
            state.observe_entity(entity.get("surface", ""), span=span)

        for link in proposal.links:
            result.links_proposed += 1
            outcome = _apply_link(state, link, span)
            if outcome == "bound":
                result.links_bound += 1
            elif outcome == "merged":
                result.links_merged += 1
            else:
                result.links_unresolved += 1

        for fact in proposal.facts:
            subject = state.observe_entity(fact.subject, span=span)
            if not subject:
                continue
            obj_id = state.resolve(fact.object)
            state.apply_assertion(
                CandidateAssertion(
                    subject=subject,
                    predicate=fact.predicate,
                    object=obj_id or fact.object,
                    certainty=_CERTAINTY.get(fact.certainty, Certainty.NARRATED),
                    evidence=fact.evidence,
                    span=span,
                    confidence=_confidence(fact.certainty),
                )
            )

        for speech in proposal.speech:
            node = state.resolve(speech.speaker)
            if node is None and speech.speaker:
                # The speaker was named but not yet in state; observing binds it.
                node = state.observe_entity(speech.speaker, span=span) or None
            result.speaker_calls.append(
                SpeakerCall(
                    window=proposal.index,
                    char_start=proposal.start,
                    char_end=proposal.end,
                    quote_cue=speech.quote,
                    speaker_surface=speech.speaker,
                    node_id=state.deref(node) if node else None,
                )
            )

        # Answer every probe that falls in this window, using the state the
        # reader has just finished building. Nothing later than this window
        # has been read, so the answer is prefix-causal by construction.
        while cursor < len(pending) and pending[cursor].position < proposal.end:
            probe = pending[cursor]
            node = state.resolve(probe.surface)
            result.probe_answers.append(
                ProbeAnswer(
                    position=probe.position, surface=probe.surface, gold=probe.gold,
                    window=proposal.index,
                    node_id=state.deref(node) if node else None,
                )
            )
            cursor += 1

        state.close_step()
    return result


def _apply_link(state: NarrativeState, link: dict[str, str], span: Span) -> str:
    """Apply one proposed identity link, subject to the same guard as a merge.

    Binding a new surface onto an existing node *is* an identity claim -- it is
    a merge whose source node does not exist yet -- so it goes through the same
    constraints. Two consequences matter. A policy that may not merge does not
    get identity resolution smuggled in through this path; it simply observes
    the surface, which creates a separate node. And a proper name is never bound
    onto a node whose established names it conflicts with, however confidently
    the extractor asserts it.
    """
    surface, same_as = link.get("surface", ""), link.get("same_as", "")
    if not surface or not same_as:
        return "skipped"
    if not state.policy.allow_merge:
        return "blocked"
    target = state.resolve(same_as)
    if target is None:
        return "unresolved"

    source = state.resolve(surface)
    if source is None:
        node = state.entities[state.deref(target)]
        if is_proper_name(surface) and node.proper_names:
            probe = EntityNode(id="__probe__", canonical=surface, proper_names={surface})
            if not state.names_are_compatible(node, probe):
                return "blocked"
        node.observe(surface, state.index, span, proper=is_proper_name(surface))
        state._record(  # noqa: SLF001 - the store owns the log, this is its writer
            Op.ELABORATE, node.id, f"bound '{surface}' to {node.canonical}", payload=surface
        )
        return "bound"

    if state.deref(source) != state.deref(target):
        op = state.merge_entities(target, source, reason=f"link '{surface}' -> '{same_as}'")
        return "merged" if op.value.startswith("merge") else "blocked"
    return "bound"


def _seed_lookahead(state: NarrativeState, proposals: list[WindowProposal]) -> None:
    """Non-causal pre-pass: register every surface and link in the book."""
    last = proposals[-1] if proposals else None
    state.step(0, last.end if last else 0)
    for proposal in proposals:
        span = Span(proposal.start, proposal.end)
        for entity in proposal.entities:
            state.observe_entity(entity.get("surface", ""), span=span)
    for proposal in proposals:
        span = Span(proposal.start, proposal.end)
        for link in proposal.links:
            _apply_link(state, link, span)
    # Clear the trace so lookahead does not pollute the per-window curves.
    state.trace.clear()
    state.log.clear()
    state.violations.clear()


def run_all(
    proposals: list[WindowProposal],
    policies: list[str] | None = None,
    text_length: int | None = None,
    probes: list[Probe] | None = None,
) -> dict[str, RunResult]:
    names = policies or [
        "window-only", "append-only", "dsg-merge", "dsg-eager", "dsg-full", "retrospective",
    ]
    return {
        name: run_policy(
            proposals, name, causal=(name != "retrospective"),
            text_length=text_length, probes=probes,
        )
        for name in names
    }
