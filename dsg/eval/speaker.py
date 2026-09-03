"""Plane B -- speaker attribution answered from state, at a fixed reading point.

For every gold quotation the system must name the speaker using only the state
it had built from text *before that point in the book*. Because the extraction
stream is shared across policies, the speaker *string* proposed for a given
quote is identical everywhere; what differs is whether the state can bind that
string to the right person. The measure therefore isolates entity resolution
rather than re-measuring the extractor.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

from dsg.data.pdnc import Novel
from dsg.eval.identity import node_to_character
from dsg.policies.runner import RunResult
from dsg.store import NarrativeState

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _similarity(cue: str, gold_text: str) -> float:
    c, g = _tokens(cue)[:12], _tokens(gold_text)[:24]
    if not c or not g:
        return 0.0
    if g[: len(c)] == c:
        return 1.0
    overlap = len(set(c) & set(g))
    return overlap / len(set(c))


@dataclass(slots=True)
class SpeakerScore:
    n_gold: int
    matched: int
    correct: int
    resolved: int
    accuracy_overall: float
    accuracy_matched: float
    coverage: float
    by_decile: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "n_gold": float(self.n_gold), "matched": float(self.matched),
            "correct": float(self.correct), "resolved": float(self.resolved),
            "speaker_acc": self.accuracy_overall,
            "speaker_acc_matched": self.accuracy_matched,
            "speaker_coverage": self.coverage,
        }


def score_speakers(
    result: RunResult, novel: Novel, min_similarity: float = 0.34, n_deciles: int = 10
) -> SpeakerScore:
    state: NarrativeState = result.state
    node_map = node_to_character(state, novel)
    gold_by_name = {c.name: c.char_id for c in novel.characters}

    # Index gold quotes by the window they start in.
    calls_by_window: dict[int, list] = {}
    for call in result.speaker_calls:
        calls_by_window.setdefault(call.window, []).append(call)
    windows = sorted({(c.char_start, c.char_end, c.window) for c in result.speaker_calls})
    starts = [w[0] for w in windows]

    def window_of(pos: int) -> int | None:
        """Binary search, since this runs once per gold quotation per policy."""
        i = bisect_right(starts, pos) - 1
        if i < 0:
            return None
        start, end, index = windows[i]
        return index if start <= pos < end else None

    text_len = max(1, len(novel.text))
    correct = matched = resolved = 0
    buckets = [{"n": 0, "matched": 0, "correct": 0} for _ in range(n_deciles)]
    used: set[int] = set()

    for quote in novel.quotes:
        bucket = min(n_deciles - 1, int(quote.start / text_len * n_deciles))
        buckets[bucket]["n"] += 1
        index = window_of(quote.start)
        if index is None:
            continue
        best, best_score = None, min_similarity
        for call in calls_by_window.get(index, []):
            key = id(call)
            if key in used:
                continue
            score = _similarity(call.quote_cue, quote.text)
            if score >= best_score:
                best, best_score = call, score
        if best is None:
            continue
        used.add(id(best))
        matched += 1
        buckets[bucket]["matched"] += 1
        predicted = node_map.get(best.node_id) if best.node_id else None
        if predicted is not None:
            resolved += 1
        gold_id = gold_by_name.get(quote.speaker)
        if predicted is not None and gold_id is not None and predicted == gold_id:
            correct += 1
            buckets[bucket]["correct"] += 1

    n_gold = len(novel.quotes)
    by_decile = [
        {
            "decile": i,
            "n": b["n"],
            "matched": b["matched"],
            "accuracy": (b["correct"] / b["matched"]) if b["matched"] else 0.0,
            "accuracy_overall": (b["correct"] / b["n"]) if b["n"] else 0.0,
        }
        for i, b in enumerate(buckets)
    ]
    return SpeakerScore(
        n_gold=n_gold, matched=matched, correct=correct, resolved=resolved,
        accuracy_overall=correct / n_gold if n_gold else 0.0,
        accuracy_matched=correct / matched if matched else 0.0,
        coverage=matched / n_gold if n_gold else 0.0,
        by_decile=by_decile,
    )
