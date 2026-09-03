"""Prefix-causal windowing.

A window is one reading step. The contract the whole study rests on is that at
step ``t`` a system may condition on ``w_t`` and on state derived from
``w_1..w_{t-1}`` -- never on later text. ``lead_in`` carries a short tail of the
previous window so pronouns at a boundary are resolvable, but it is always text
the reader has *already* passed, so causality holds.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from dsg.schemas import Window

_BOUNDARY_RE = re.compile(r"(?<=[.!?\"'”])\s+|\n\n+")


def _snap(text: str, target: int, search: int = 600) -> int:
    """Move a cut point to the nearest sentence/paragraph boundary."""
    if target >= len(text):
        return len(text)
    lo, hi = max(0, target - search), min(len(text), target + search)
    best, best_d = target, search + 1
    for m in _BOUNDARY_RE.finditer(text, lo, hi):
        d = abs(m.end() - target)
        if d < best_d:
            best, best_d = m.end(), d
    return best


def iter_windows(
    text: str, size_chars: int = 4800, lead_chars: int = 500
) -> Iterator[tuple[Window, str]]:
    """Yield ``(window, lead_in)`` pairs covering ``text`` without gaps."""
    n, start, index = len(text), 0, 0
    while start < n:
        end = _snap(text, min(n, start + size_chars))
        if end <= start:
            end = min(n, start + size_chars)
        lead = text[max(0, start - lead_chars) : start]
        yield Window(index=index, start=start, end=end, text=text[start:end]), lead
        start, index = end, index + 1


def window_count(text: str, size_chars: int = 4800) -> int:
    return sum(1 for _ in iter_windows(text, size_chars, 0))
