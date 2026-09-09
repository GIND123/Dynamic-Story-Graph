"""Causal DSG state snapshots, for the E1 ``state-causal`` condition.

Replays a novel's cached proposal stream under a policy and records the state
after every window. A quote at character ``s`` then reads the snapshot taken
after the last window that **ends at or before** ``s``, so the state it sees was
built from text the reader has genuinely passed.

Why this needs a different guarantee from the text conditions. A text block is
verified by substring: it must be a slice of ``text[:s]``. A state digest cannot
be, because it is *derived* -- the extractor wrote it, and none of its wording
need appear in the novel. Its causality is therefore established by **provenance**
instead: every snapshot records the window index and character offset it was
taken at, and ``Snapshot.assert_causal`` refuses any snapshot whose coverage
reaches past the quote. Provenance is checked on every use, not sampled.

Reuses the cached ``proposals/pdnc-qwen7b-w3200`` stream, so building state for
all 28 novels costs no GPU at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dsg.policies.runner import apply_window
from dsg.proposals import load_proposals
from dsg.schemas import Span
from dsg.store import POLICIES, NarrativeState

DEFAULT_PROPOSALS = Path("artifacts/proposals/pdnc-qwen7b-w3200")


class CausalityError(AssertionError):
    """Raised when a state snapshot covers text at or beyond the quote."""


@dataclass(slots=True, frozen=True)
class Snapshot:
    """State as of a window boundary, with the provenance to prove it is causal."""

    window: int
    covers_to: int  # character offset; state saw text[:covers_to] and no more
    entities: str  # who the reader knows about
    facts: str  # what the reader believes about them
    recent_speakers: tuple[str, ...]

    def assert_causal(self, quote_start: int, *, label: str = "") -> None:
        if self.covers_to > quote_start:
            raise CausalityError(
                f"state snapshot covers char {self.covers_to} > quote start "
                f"{quote_start}{' for ' + label if label else ''}: "
                "this snapshot saw text the reader has not reached"
            )

    def render(self, max_speakers: int = 4) -> str:
        """The block that goes in the prompt. Empty when the state knows nothing."""
        parts = []
        if self.entities:
            parts.append(f"Characters the reader has met so far:\n{self.entities}")
        if self.facts:
            parts.append(f"What the reader believes about them:\n{self.facts}")
        if self.recent_speakers:
            recent = ", ".join(self.recent_speakers[-max_speakers:])
            parts.append(f"Most recent speakers, oldest first: {recent}")
        return "\n\n".join(parts)


# The empty state, for quotes that precede the first window boundary.
EMPTY = Snapshot(window=-1, covers_to=0, entities="", facts="", recent_speakers=())


def build_snapshots(
    proposals_path: Path,
    policy: str = "dsg-full",
    *,
    max_entities: int = 12,
    max_facts_per_entity: int = 4,
) -> list[Snapshot]:
    """Replay one novel and snapshot the state after each window.

    Mirrors ``run_policy``'s application order exactly -- ``step``,
    ``apply_window``, speaker binding, ``close_step`` -- so the state here is
    the state that study would have had at the same point. Only the snapshot is
    added.
    """
    _meta, proposals = load_proposals(proposals_path)
    config = POLICIES[policy] if isinstance(policy, str) else policy
    state = NarrativeState(config)
    snapshots: list[Snapshot] = []
    speakers: list[str] = []

    for proposal in proposals:
        state.step(proposal.index, proposal.end)
        span = Span(proposal.start, proposal.end)
        apply_window(state, proposal, span)

        for speech in proposal.speech:
            node = state.resolve(speech.speaker)
            if node is None and speech.speaker:
                node = state.observe_entity(speech.speaker, span=span) or None
            if node:
                resolved = state.deref(node)
                name = next(
                    (n.canonical for n in state.live_entities() if n.id == resolved),
                    speech.speaker,
                )
                if name:
                    speakers.append(name)

        state.close_step()
        snapshots.append(
            Snapshot(
                window=proposal.index,
                covers_to=proposal.end,
                entities=state.digest(max_entities=max_entities),
                facts=state.fact_digest(
                    max_entities=max_entities,
                    max_facts_per_entity=max_facts_per_entity,
                ),
                recent_speakers=tuple(speakers[-8:]),
            )
        )
    return snapshots


def snapshot_for(snapshots: list[Snapshot], quote_start: int) -> Snapshot:
    """The latest snapshot whose coverage ends at or before ``quote_start``.

    Deliberately strict. The window *containing* the quote is excluded, because
    that window's proposals were extracted from the whole window -- including
    the text after the quote, which is exactly the leak this experiment exists
    to avoid. The cost is that state lags by up to one window; the alternative
    is not causal.
    """
    chosen = EMPTY
    for snap in snapshots:
        if snap.covers_to <= quote_start:
            chosen = snap
        else:
            break
    chosen.assert_causal(quote_start)
    return chosen


def load_all(
    root: Path = DEFAULT_PROPOSALS, policy: str = "dsg-full", novels: list[str] | None = None
) -> dict[str, list[Snapshot]]:
    """Snapshots for every novel with a cached proposal stream."""
    out: dict[str, list[Snapshot]] = {}
    for path in sorted(root.glob("*.jsonl")):
        name = path.stem
        if novels and name not in novels:
            continue
        out[name] = build_snapshots(path, policy)
    return out
