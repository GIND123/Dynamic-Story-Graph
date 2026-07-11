"""Shared predictor scaffolding: telemetry, candidate parsing, prose I/O.

`LLMAsk` is the injected `(prompt, meta) -> answer_text` callable. `meta` carries
non-gold routing info only (quote_id, quote_type, task) so tests can vary
behaviour by quote_type WITHOUT the predictor ever seeing the gold answer.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from baselines.config import BaselineConfig, count_tokens

# (prompt, meta) -> answer text. meta never contains the gold answer.
LLMAsk = Callable[[str, dict], str]


class _Driver(Protocol):
    def execute_query(self, query: str, **params): ...


@dataclass
class CallRecord:
    input_tokens: int
    output_tokens: int
    latency_seconds: float


@dataclass
class Telemetry:
    n_calls: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_latency_seconds: float
    total_cost: float


class BasePredictor:
    """Common telemetry + helpers. Subclasses implement predict_quote/predict_cloze."""

    def __init__(
        self,
        manuscript_id: str,
        driver: _Driver,
        ask: LLMAsk | None = None,
        config: BaselineConfig | None = None,
    ) -> None:
        self.manuscript_id = manuscript_id
        self.driver = driver
        self.ask = ask
        self.config = config or BaselineConfig()
        self.records: list[CallRecord] = []

    # -- telemetry ---------------------------------------------------------
    def _record(self, input_tokens: int, output_tokens: int, latency: float) -> None:
        self.records.append(CallRecord(input_tokens, output_tokens, latency))

    def _ask_llm(self, prompt: str, meta: dict) -> str:
        """Call the injected LLM, timing it and recording token accounting."""
        if self.ask is None:
            # No LLM available (e.g. offline default): record the prompt cost,
            # return empty so the caller resolves to None.
            self._record(count_tokens(prompt), 0, 0.0)
            return ""
        t0 = time.perf_counter()
        answer = self.ask(prompt, meta) or ""
        latency = time.perf_counter() - t0
        self._record(count_tokens(prompt), count_tokens(answer), latency)
        return answer

    def telemetry(self) -> Telemetry:
        n = len(self.records)
        if n == 0:
            return Telemetry(0, 0.0, 0.0, 0.0, 0.0)
        ti = sum(r.input_tokens for r in self.records)
        to = sum(r.output_tokens for r in self.records)
        lat = sum(r.latency_seconds for r in self.records)
        return Telemetry(
            n_calls=n,
            avg_input_tokens=ti / n,
            avg_output_tokens=to / n,
            avg_latency_seconds=lat / n,
            total_cost=self.config.cost(ti, to),
        )

    # -- candidate parsing -------------------------------------------------
    @staticmethod
    def pick_candidate(answer: str, candidates: list[str]) -> str | None:
        """Return the candidate name that appears in the answer (longest match)."""
        if not answer:
            return None
        low = answer.lower()
        hits = [c for c in candidates if re.search(rf"\b{re.escape(c.lower())}\b", low)]
        return max(hits, key=len) if hits else None


# ---------------------------------------------------------------------------
# Prose I/O (manuscript-scoped) shared by long-context and cloze paths
# ---------------------------------------------------------------------------


def locate_quote_chapter(
    driver: _Driver, manuscript_id: str, quote_text: str, cache: dict
) -> int | None:
    """Map a quote to its prose Chapter `number` by text-match (cached).

    There is no edge from a quote Event to its prose Chapter and the two
    orderings differ, so we locate the quote by matching its leading text inside
    Chapter.text (verified: Q500 -> chapter 206).
    """
    snippet = quote_text.strip()[:40]
    if snippet in cache:
        return cache[snippet]
    result = driver.execute_query(
        "MATCH (ch:Chapter {manuscript_id: $mid}) WHERE ch.text CONTAINS $snippet "
        "RETURN ch.number AS number ORDER BY ch.number LIMIT 1",
        mid=manuscript_id,
        snippet=snippet,
    )
    number = result.records[0]["number"] if result.records else None
    cache[snippet] = number
    return number


def fetch_chapter_window(
    driver: _Driver, manuscript_id: str, lo: int, hi: int
) -> list[dict]:
    """Return Chapter {number, text} in [lo, hi] ordered by number (scoped)."""
    result = driver.execute_query(
        "MATCH (ch:Chapter {manuscript_id: $mid}) "
        "WHERE ch.number >= $lo AND ch.number <= $hi "
        "RETURN ch.number AS number, ch.text AS text ORDER BY ch.number",
        mid=manuscript_id,
        lo=lo,
        hi=hi,
    )
    return [{"number": r["number"], "text": r["text"]} for r in result.records]


def join_to_budget(chunks: list[str], budget: int) -> str:
    """Concatenate chunks in order up to `budget` tokens (whole chunks only)."""
    out: list[str] = []
    used = 0
    for chunk in chunks:
        t = count_tokens(chunk)
        if used + t > budget and out:
            break
        out.append(chunk)
        used += t
    return "\n\n".join(out)
