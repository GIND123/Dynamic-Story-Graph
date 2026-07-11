"""METRIC 1 — PDNC quote attribution accuracy.

GOLD: PDNC speaker labels are ground truth. Each dialogue is an Event
{is_dialogue:true, quote_type, summary=<text>} with INVOLVES role='agent'
(speaker) / role='patient' (addressee) edges and source_ref=<QID>. We pull these
directly from the graph (NOT get_canon, which caps events and hides dialogue).

A predictor is a pluggable callable predict(quote_text, context) -> name. This
module ships one reference baseline (majority-class speaker); the graph / vector
/ long-context predictors drop into the same signature in a later change.

Accuracy is reported per quote_type and overall, with the standard hard split:
explicit vs non-explicit (implicit + anaphoric) — non-explicit is the
discriminating case. Names are alias-normalized before matching (one character =
one identity), and scoring is filtered to major+intermediate characters (Category
from character_info.csv), with a >=10-quotes fallback when Category is missing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol

from evals.metrics.pdnc_meta import PdncChars, read_pdnc_characters

# Context passed to a predictor; carries everything a predictor may use so the
# signature is stable as smarter predictors are added later.
Predictor = Callable[[str, dict], "str | None"]

_KNOWN_TYPES = ("explicit", "implicit", "anaphoric")
_NON_EXPLICIT = ("implicit", "anaphoric")
_FILTER_CATEGORIES = ("major", "intermediate")
_FALLBACK_MIN_QUOTES = 10


@dataclass
class GoldQuote:
    quote_id: str
    quote_type: str
    speaker: str
    addressees: list[str]
    sequence_index: int
    text: str


class _Driver(Protocol):
    def execute_query(self, query: str, **params): ...


_FETCH_CYPHER = (
    "MATCH (e:Event {manuscript_id: $mid, is_dialogue: true})"
    "-[:INVOLVES {role: 'agent'}]->(spk:Character) "
    "OPTIONAL MATCH (e)-[:INVOLVES {role: 'patient'}]->(adr:Character) "
    "WITH e, spk, collect(DISTINCT adr.name) AS addressees "
    "RETURN e.uid AS uid, coalesce(e.quote_type, '') AS quote_type, "
    "spk.name AS speaker, addressees AS addressees, "
    "coalesce(e.summary, '') AS text, e.sequence_index AS seq "
    "ORDER BY e.sequence_index"
)


def _quote_id(uid: str) -> str:
    """Extract the PDNC quote id from a '…:Event:quote:<QID>' uid."""
    marker = ":Event:quote:"
    return uid.split(marker, 1)[1] if marker in uid else uid


def fetch_gold_dialogue(driver: _Driver, manuscript_id: str) -> list[GoldQuote]:
    """Pull every gold dialogue quotation for the manuscript from the graph."""
    result = driver.execute_query(_FETCH_CYPHER, mid=manuscript_id)
    gold: list[GoldQuote] = []
    for r in result.records:
        gold.append(
            GoldQuote(
                quote_id=_quote_id(r["uid"]),
                quote_type=(r["quote_type"] or "").strip().lower(),
                speaker=r["speaker"],
                addressees=[a for a in (r["addressees"] or []) if a],
                sequence_index=r["seq"],
                text=r["text"] or "",
            )
        )
    return gold


# ---------------------------------------------------------------------------
# Reference predictor (baseline floor; smarter predictors arrive later)
# ---------------------------------------------------------------------------


class MajorityClassPredictor:
    """Always predicts the most frequent gold speaker among allowed characters.

    The trivial classification floor — it ignores the quote text and tells you
    the accuracy a no-information model achieves. Real graph/vector/long-context
    predictors replace it via the same (quote_text, context) signature.
    """

    def __init__(self, label: str | None) -> None:
        self.label = label

    def __call__(self, quote_text: str, context: dict) -> str | None:
        return self.label

    @classmethod
    def fit(
        cls, gold: list[GoldQuote], chars: PdncChars, allowed: set[str]
    ) -> "MajorityClassPredictor":
        counts = Counter(
            n
            for q in gold
            if (n := chars.normalize(q.speaker)) in allowed and n is not None
        )
        return cls(counts.most_common(1)[0][0] if counts else None)


# ---------------------------------------------------------------------------
# Scoring (pure)
# ---------------------------------------------------------------------------


@dataclass
class QuoteAttributionReport:
    overall: float
    explicit: float
    implicit: float
    anaphoric: float
    non_explicit: float
    n_quotes: int
    n_characters: int
    filter_rule: str
    addressee_accuracy: float | None = None
    per_type_counts: dict = field(default_factory=dict)


def _ratio(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def score_quote_attribution(
    gold: list[GoldQuote],
    predict: Predictor,
    chars: PdncChars,
    *,
    allowed: set[str],
    filter_rule: str,
    manuscript_id: str = "",
    addressee_predict: Predictor | None = None,
) -> QuoteAttributionReport:
    """Score a predictor against gold, restricted to `allowed` speakers.

    Pure: no graph, no files. `allowed` is the set of canonical names to score
    (filtered by Category or the >=10 fallback). previous_speaker (gold of the
    preceding quote) is exposed in context for continuity-style predictors.
    """
    overall = [0, 0]  # [correct, total]
    by_type: dict[str, list[int]] = {t: [0, 0] for t in _KNOWN_TYPES}
    addr_correct = addr_total = 0
    scored_speakers: set[str] = set()
    prev_speaker: str | None = None

    for q in gold:
        gold_speaker = chars.normalize(q.speaker)
        context = {
            "manuscript_id": manuscript_id,
            "quote_id": q.quote_id,
            "quote_type": q.quote_type,
            "sequence_index": q.sequence_index,
            "previous_speaker": prev_speaker,
            "candidates": sorted(allowed),
        }
        prev_speaker = gold_speaker  # for the NEXT quote's context

        if gold_speaker not in allowed:
            continue
        scored_speakers.add(gold_speaker)

        predicted = chars.normalize(predict(q.text, context))
        correct = predicted == gold_speaker
        overall[1] += 1
        overall[0] += int(correct)
        if q.quote_type in by_type:
            by_type[q.quote_type][1] += 1
            by_type[q.quote_type][0] += int(correct)

        if addressee_predict is not None and q.addressees:
            gold_addr = {chars.normalize(a) for a in q.addressees}
            predicted_addr = chars.normalize(addressee_predict(q.text, context))
            addr_total += 1
            addr_correct += int(predicted_addr in gold_addr)

    non_explicit = [
        sum(by_type[t][0] for t in _NON_EXPLICIT),
        sum(by_type[t][1] for t in _NON_EXPLICIT),
    ]
    return QuoteAttributionReport(
        overall=_ratio(*overall),
        explicit=_ratio(*by_type["explicit"]),
        implicit=_ratio(*by_type["implicit"]),
        anaphoric=_ratio(*by_type["anaphoric"]),
        non_explicit=_ratio(*non_explicit),
        n_quotes=overall[1],
        n_characters=len(scored_speakers),
        filter_rule=filter_rule,
        addressee_accuracy=(_ratio(addr_correct, addr_total) if addr_total else None),
        per_type_counts={t: tuple(by_type[t]) for t in _KNOWN_TYPES},
    )


def _resolve_allowed(gold: list[GoldQuote], chars: PdncChars) -> tuple[set[str], str]:
    """Pick the character filter: Category if available, else >=10-quotes."""
    category_filtered = chars.filtered_names(_FILTER_CATEGORIES)
    if category_filtered:
        return category_filtered, "category: major+intermediate"
    counts = Counter(chars.normalize(q.speaker) for q in gold)
    allowed = {n for n, k in counts.items() if n and k >= _FALLBACK_MIN_QUOTES}
    return allowed, f">={_FALLBACK_MIN_QUOTES} quotes spoken (Category unavailable)"


def run_quote_attribution(
    manuscript_id: str,
    novel_dir: str,
    *,
    predict: Predictor | None = None,
    addressee_predict: Predictor | None = None,
    driver: _Driver | None = None,
) -> dict:
    """End-to-end: read chars, fetch gold, score filtered AND all-character.

    `driver` defaults to the live Neo4j driver; tests inject a fake. Returns
    {"filtered": report, "all": report} so both numbers are visible.
    """
    if driver is None:
        from app import graph

        driver = graph.init_driver()

    chars = read_pdnc_characters(novel_dir)
    gold = fetch_gold_dialogue(driver, manuscript_id)
    allowed, filter_rule = _resolve_allowed(gold, chars)
    predict = predict or MajorityClassPredictor.fit(gold, chars, allowed)

    return {
        "filtered": score_quote_attribution(
            gold,
            predict,
            chars,
            allowed=allowed,
            filter_rule=filter_rule,
            manuscript_id=manuscript_id,
            addressee_predict=addressee_predict,
        ),
        "all": score_quote_attribution(
            gold,
            predict,
            chars,
            allowed=set(chars.canonical_names),
            filter_rule="all characters",
            manuscript_id=manuscript_id,
            addressee_predict=addressee_predict,
        ),
    }
