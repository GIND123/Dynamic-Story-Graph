"""Selecting between candidate chapters with the graph.

Why this design follows from the project's own negative results.

Result three established that the *absolute* contradiction count is dominated by
extraction noise: 215 published novels and 30 generated stories do not separate
(AUC 0.394). Result one established that *relative* comparisons on identical
input are sound, because a shared noise floor cancels when only the policy
varies.

Every generation design tried so far used the graph absolutely. As prompt
context it must be a faithful summary, which extraction is not good enough to
produce. As a threshold guard it must decide whether one chapter is bad in
isolation, which asks the noisy absolute quantity again.

Ranking k candidates for the *same* chapter position is the relative case. Every
candidate is scored by the same extractor against the same prior state, so the
noise that destroys absolute measurement is common to all of them and cancels in
the ordering. The graph is not asked whether a chapter is consistent; it is
asked which of these is *more* consistent, which is the question the instrument
can answer.

The scorer never sees the planted canon. It sees only what the text itself
established earlier.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from dsg.generate.repair import detect_conflicts
from dsg.generate.run import parse_write_extraction
from dsg.schemas import Window
from dsg.store import NarrativeState


@dataclass(slots=True)
class CandidateScore:
    index: int
    conflicts: int
    novel_entities: int
    chars: int

    def key(self) -> tuple[int, int, int]:
        """Fewest contradictions first, then fewest invented entities.

        The second term is a tie break, not a target: a chapter that introduces
        a crowd of new names has usually drifted from the story, and among
        candidates that contradict nothing equally it is the weaker one. Ties
        after that fall to candidate order, which is neutral.
        """
        return self.conflicts, self.novel_entities, self.index


def score_candidates(
    state: NarrativeState, candidates: list[str], extractions: list[str], start: int
) -> list[CandidateScore]:
    """Score each candidate against the state as it stands, without mutating it."""
    known = {s.lower() for node in state.live_entities() for s in node.surfaces}
    scores: list[CandidateScore] = []
    for index, (text, raw) in enumerate(zip(candidates, extractions, strict=False)):
        window = Window(index=0, start=start, end=start + len(text), text=text)
        proposal = parse_write_extraction(raw, window)
        conflicts = detect_conflicts(state, proposal, max_report=64)
        fresh = sum(
            1
            for e in proposal.entities
            if (e.get("surface") or "").lower() not in known
        )
        scores.append(
            CandidateScore(
                index=index, conflicts=len(conflicts), novel_entities=fresh,
                chars=len(text),
            )
        )
    return scores


def choose(
    state: NarrativeState, candidates: list[str], extractions: list[str],
    start: int, strategy: str, seed: int = 0,
) -> tuple[int, list[CandidateScore]]:
    """Return the chosen candidate index and the scores behind the choice."""
    if len(candidates) == 1:
        return 0, []
    scores = score_candidates(state, candidates, extractions, start)
    if strategy == "graph":
        return min(scores, key=CandidateScore.key).index, scores
    if strategy == "random":
        # The control that matters: same k samples, same budget, chosen blind.
        # Without it, any gain could be the extra sampling rather than the graph.
        return random.Random(seed).randrange(len(candidates)), scores
    return 0, scores
