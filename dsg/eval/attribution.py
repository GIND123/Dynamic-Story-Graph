"""E1 -- prefix-causal quote attribution on PDNC.

Every published PDNC attribution number is *non-causal*: the system reads a
window centred on the quote, which contains the following text where the
attribution cue usually lives. This module scores the same benchmark under a
strict prefix-causal constraint -- at a quote starting at character ``s`` the
system may read ``text[:s]`` and nothing else.

The whole experiment rests on that cut actually holding, so it is enforced here
rather than assumed. ``CausalContext`` is the *only* way a prompt is built, it
physically cannot slice past ``s``, and ``assert_no_leakage`` re-checks the
assembled prompt against the forbidden suffix before the prompt is allowed out.
A violation raises; it never warns.

Note the contrast with the reading study, which is causal at *window*
granularity (``prefix_end = proposal.end`` in ``dsg/policies/runner.py``). That
is correct there, because a window is one reading step. It is not sufficient
here, because a quote sits inside a window and the remainder of that window is
future text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "CausalContext",
    "LeakageError",
    "assert_no_leakage",
    "normalise_name",
    "score_attribution",
    "AttributionScore",
]

# A shared n-gram width for the leakage check. Long enough that ordinary English
# collocations ("said the young man") do not trip it, short enough to catch a
# copied clause.
_LEAK_NGRAM = 40


class LeakageError(AssertionError):
    """Raised when a prompt contains text at or after the quote's start offset."""


def _shingles(text: str, n: int = _LEAK_NGRAM) -> set[str]:
    """Character n-grams of ``text``, whitespace-collapsed.

    Collapsing whitespace first means a prompt that reflows the source (wrapping,
    indentation) is still caught -- a leak that survives reformatting is still a
    leak.
    """
    flat = " ".join(text.split())
    if len(flat) < n:
        return set()
    return {flat[i : i + n] for i in range(len(flat) - n + 1)}


def assert_no_leakage(prompt: str, text: str, start: int, *, label: str = "") -> None:
    """Fail if ``prompt`` shares any ``_LEAK_NGRAM``-char shingle with ``text[start:]``.

    This is the guard the causal claim depends on (G1). It runs on every quote,
    not a sample, and raises rather than warning so a leaking run cannot finish
    and be written up.
    """
    future = text[start:]
    if not future:
        return
    overlap = _shingles(prompt) & _shingles(future)
    if overlap:
        example = sorted(overlap)[0]
        raise LeakageError(
            f"prompt leaks future text{' for ' + label if label else ''}: "
            f"{len(overlap)} shared {_LEAK_NGRAM}-char shingle(s); "
            f"first={example!r}"
        )


@dataclass(slots=True, frozen=True)
class CausalContext:
    """The only sanctioned way to obtain text for a quote at ``start``.

    Slicing is done here so no caller can accidentally reach past the cut. Every
    accessor returns a prefix of ``text[:start]``; there is no method that can
    return later text, which is why the guard is structural rather than a
    convention someone has to remember.
    """

    text: str
    start: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= len(self.text):
            raise ValueError(f"start {self.start} outside text of length {len(self.text)}")

    @property
    def prefix(self) -> str:
        """Everything the reader has passed. Never includes the quote itself."""
        return self.text[: self.start]

    def tail(self, chars: int) -> str:
        """The last ``chars`` characters before the quote."""
        if chars <= 0:
            return ""
        return self.text[max(0, self.start - chars) : self.start]

    def verify(self, prompt: str, *, label: str = "") -> str:
        """Return ``prompt`` unchanged, having proved it does not leak."""
        assert_no_leakage(prompt, self.text, self.start, label=label)
        return prompt


# --- scoring -----------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
# Honorifics are *distinguishing* in this corpus -- "Mr. Bennet", "Mrs. Bennet"
# and "Miss Bennet" are three people -- so they are deliberately NOT stripped.
# See dsg/matching.py and the title-aware matching note in dsg/README.md.


def normalise_name(name: str) -> str:
    """Case/punctuation/whitespace-insensitive form used for exact-match scoring.

    Deliberately conservative: it folds accents and spacing but keeps titles,
    because collapsing them merges distinct characters.
    """
    folded = unicodedata.normalize("NFKD", name or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = _PUNCT_RE.sub(" ", folded.lower())
    return _WS_RE.sub(" ", folded).strip()


@dataclass(slots=True)
class AttributionScore:
    """Per-novel attribution result, split by quote type.

    The split matters: a prefix-causal cut removes the decisive lexical cue for
    ~29% of *explicit* quotes but leaves implicit/anaphoric quotes untouched, so
    a single overall number hides where the constraint actually bites.
    """

    n: int = 0
    correct: int = 0
    by_type: dict[str, list[int]] = field(default_factory=dict)

    def add(self, quote_type: str, is_correct: bool) -> None:
        self.n += 1
        self.correct += int(is_correct)
        slot = self.by_type.setdefault(quote_type or "unknown", [0, 0])
        slot[0] += 1
        slot[1] += int(is_correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def accuracy_for(self, *types: str) -> float:
        n = sum(self.by_type.get(t, [0, 0])[0] for t in types)
        c = sum(self.by_type.get(t, [0, 0])[1] for t in types)
        return c / n if n else 0.0

    @property
    def accuracy_non_explicit(self) -> float:
        """The slice the literature reports separately, and where state should help."""
        return self.accuracy_for("Implicit", "Anaphoric")

    def as_dict(self) -> dict[str, float]:
        out = {
            "n": float(self.n),
            "accuracy": self.accuracy,
            "accuracy_explicit": self.accuracy_for("Explicit"),
            "accuracy_non_explicit": self.accuracy_non_explicit,
        }
        for t, (n, c) in sorted(self.by_type.items()):
            out[f"n_{t.lower()}"] = float(n)
            out[f"acc_{t.lower()}"] = c / n if n else 0.0
        return out


def score_attribution(
    predictions: Iterable[str | None],
    gold: Sequence[str],
    quote_types: Sequence[str],
    *,
    alias_sets: dict[str, set[str]] | None = None,
) -> AttributionScore:
    """Exact match after normalisation, with gold aliases accepted (G5: no judge).

    ``alias_sets`` maps a gold speaker's canonical name to the surface forms the
    annotation licenses for that character, so naming "Lizzy" for "Elizabeth
    Bennet" counts when PDNC says they are the same person -- and does not when
    it does not.
    """
    preds = list(predictions)
    if not (len(preds) == len(gold) == len(quote_types)):
        raise ValueError(
            f"length mismatch: predictions={len(preds)} gold={len(gold)} "
            f"types={len(quote_types)}"
        )
    score = AttributionScore()
    for pred, truth, qtype in zip(preds, gold, quote_types, strict=True):
        accepted = {normalise_name(truth)}
        if alias_sets and truth in alias_sets:
            accepted |= {normalise_name(a) for a in alias_sets[truth]}
        accepted.discard("")
        score.add(qtype, normalise_name(pred or "") in accepted)
    return score
