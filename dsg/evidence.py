"""Find the source sentence behind an extractor's evidence string.

The extractor quotes its evidence, but a small model often paraphrases, trims or
re-punctuates the quote, so the string is not reliably a substring of the
window it came from. Locating it is deterministic: an exact match after
normalising case, whitespace and quote marks; failing that, the sentence (or two
adjacent sentences) that contains the largest share of the evidence's content
words. Below a fixed threshold the evidence is reported as not found rather
than pinned to a guess.

This is the single implementation. The audit sheets use it to show labellers
the sentence, and any rule that reads the source sentence at runtime must call
the same function, so what was labelled is what the rule sees. It only ever
reads the window it is given, so a caller that passes ``text[start:end]`` of the
current window stays prefix-causal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EXACT = "exact"
FUZZY = "fuzzy"
NOT_FOUND = "not_found"

# Share of the evidence's content words a sentence must contain, and the
# minimum number of them, for a fuzzy match to count.
MIN_CONTAINMENT = 0.6
MIN_SHARED = 2
# Content words are compared on this many leading characters.
STEM = 5
# A fuzzy match may not be a span much longer than the evidence: a long
# sentence contains most common words by chance.
MAX_SPAN_WORDS = 25
MAX_SPAN_RATIO = 3

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "—": "-", "–": "-"})
_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n\s*\n")
_STOP = frozenset((
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by", "for", "with",
    "from", "as", "is", "was", "were", "be", "been", "had", "has", "have", "he", "she", "it",
    "they", "his", "her", "their", "its", "that", "this", "there", "then", "than", "so", "not",
    "no", "i", "you", "we", "me", "him", "them", "my", "your", "our", "who", "which", "what",
    "when"
))


@dataclass(frozen=True, slots=True)
class Located:
    """Where in the window the evidence was found; offsets are window-relative."""

    start: int
    end: int
    method: str           # EXACT, FUZZY or NOT_FOUND
    score: float = 0.0    # content-word containment, 1.0 for an exact match

    @property
    def found(self) -> bool:
        return self.method != NOT_FOUND


def _normalise(text: str) -> tuple[str, list[int]]:
    """Lower-cased, quote-folded, whitespace-collapsed text plus an offset map."""
    out: list[str] = []
    index: list[int] = []
    prev_space = True
    for i, ch in enumerate(text.translate(_QUOTES).lower()):
        if ch.isspace():
            if prev_space:
                continue
            ch = " "
            prev_space = True
        else:
            prev_space = False
        out.append(ch)
        index.append(i)
    return "".join(out), index


def sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans by a fixed punctuation rule; deterministic, no model."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        if text[start : m.start()].strip():
            spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _content(text: str) -> list[str]:
    """Content-word stems: a fixed truncation, so 'realised' meets 'realizes'."""
    words = _WORD.findall(text.translate(_QUOTES).lower())
    return [w.replace("z", "s")[:STEM] for w in words if w not in _STOP]


def _enclosing(spans: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    lo = next((s for s, e in spans if s <= start < e), start)
    hi = next((e for s, e in spans if s < end <= e), end)
    return lo, hi


def locate(window: str, evidence: str) -> Located:
    """Find ``evidence`` in ``window``; returns the enclosing sentence span."""
    needle, _ = _normalise((evidence or "").strip(" .,;:\"'"))
    if not needle or not window:
        return Located(0, 0, NOT_FOUND)
    spans = sentences(window)

    hay, index = _normalise(window)
    at = hay.find(needle)
    if at >= 0:
        start, end = index[at], index[at + len(needle) - 1] + 1
        lo, hi = _enclosing(spans, start, end)
        return Located(lo, hi, EXACT, 1.0)

    wanted = _content(evidence)
    if len(set(wanted)) < MIN_SHARED:
        return Located(0, 0, NOT_FOUND)
    best: tuple[float, int, int, int] | None = None   # (score, -length, start, end)
    for i in range(len(spans)):
        for j in (i, i + 1):
            if j >= len(spans):
                continue
            lo, hi = spans[i][0], spans[j][1]
            words = _content(window[lo:hi])
            if len(words) > max(MAX_SPAN_WORDS, MAX_SPAN_RATIO * len(wanted)):
                continue
            have = set(words)
            shared = sum(1 for w in set(wanted) if w in have)
            score = shared / len(set(wanted))
            if shared < MIN_SHARED:
                continue
            cand = (score, -(hi - lo), lo, hi)
            if best is None or cand > best:
                best = cand
    if best is None or best[0] < MIN_CONTAINMENT:
        return Located(0, 0, NOT_FOUND, best[0] if best else 0.0)
    return Located(best[2], best[3], FUZZY, best[0])
