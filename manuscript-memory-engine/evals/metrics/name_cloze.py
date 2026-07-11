"""METRIC 2 — Name-cloze harness.

A memory-source-agnostic harness: select passages with EXACTLY ONE proper-name
character mention and no other named entities, mask the name with [MASK], ask a
pluggable predict(masked_passage, manuscript_id) -> name, and score exact
alias-normalized accuracy.

Cloze targets are filtered to proper-name PERs (every token capitalized, not
ALL-CAPS, multi-occurrence in the corpus), which excludes the generic/common-noun
spans the LitBank loader surfaces as Character nodes (e.g. "Foot passengers",
"ancient Greenwich pensioners"). Selection is reproducible via a fixed seed.

This change ships one reference predictor (corpus-frequency baseline). The
graph / vector / long-context predictors plug into the same signature later.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol

MASK = "[MASK]"
ClozePredictor = Callable[[str, str], "str | None"]

_DEFAULT_MIN_TOKENS = 40
_DEFAULT_MAX_TOKENS = 60
_DEFAULT_MIN_OCCURRENCES = 2
_DEFAULT_MAX_PASSAGES = 100


class _Driver(Protocol):
    def execute_query(self, query: str, **params): ...


@dataclass
class ClozePassage:
    number: int
    masked_text: str
    answer: str  # the canonical-ish name that was masked


@dataclass
class NameClozeReport:
    accuracy: float
    n_passages: int
    n_skipped: int
    seed: int


def is_proper_name(name: str) -> bool:
    """True if every whitespace token is capitalized and none is ALL-CAPS.

    Rejects common-noun-led spans ("Foot passengers" -> 'passengers' lowercase)
    and header tokens ("CHAPTER"). Conservative: a multi-word title like
    "Lord Chancellor" passes, which is acceptable for a cloze target.
    """
    tokens = [t for t in name.split() if any(ch.isalpha() for ch in t)]
    if not tokens:
        return False
    for token in tokens:
        first_alpha = next((ch for ch in token if ch.isalpha()), "")
        if not first_alpha.isupper():
            return False
        if token.isupper():  # ALL-CAPS header-like token
            return False
    return True


def _contains(text: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def proper_name_targets(
    passages: list[dict],
    character_names: set[str],
    *,
    min_occurrences: int = _DEFAULT_MIN_OCCURRENCES,
) -> set[str]:
    """Subset of character_names that are proper names AND recur in the corpus."""
    passage_counts = Counter(
        name
        for name in character_names
        if is_proper_name(name)
        for p in passages
        if _contains(p["text"], name)
    )
    return {n for n, c in passage_counts.items() if c >= min_occurrences}


def mask_passage(text: str, name: str) -> str:
    return re.sub(rf"\b{re.escape(name)}\b", MASK, text)


# ---------------------------------------------------------------------------
# Passage construction: sentence-windowing
# ---------------------------------------------------------------------------
#
# Raw Chapter nodes are the wrong granularity for a 40-60 token window: LitBank
# passages are one sentence each (too short) and PDNC passages are ~1200-char
# chunks (too long). We instead concatenate the ordered passage text into a
# single sentence stream and slide a window over consecutive sentences, combining
# adjacent short sentences until the window lands in [min_tokens, max_tokens] and
# never splitting a sentence. Windows are NON-OVERLAPPING (greedy from the front);
# when a window cannot land in range the start advances by one sentence. This is
# deterministic, so the existing seed governs only the final shuffle/selection.

# Title abbreviations whose trailing period must NOT end a sentence ("Mr. Bennet").
_ABBREV = frozenset(
    {"mr", "mrs", "ms", "dr", "st", "sir", "lady", "lord", "capt", "col", "rev", "hon"}
)


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences on .!? boundaries, guarding title abbreviations."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    merged: list[str] = []
    for part in parts:
        if not part:
            continue
        if merged:
            words = merged[-1].split()
            last = words[-1].rstrip(".").lower() if words else ""
            if last in _ABBREV:  # "Mr." was not a sentence end — re-join
                merged[-1] = f"{merged[-1]} {part}"
                continue
        merged.append(part)
    return [s for s in merged if s.strip()]


def build_sentence_windows(
    passages: list[dict],
    *,
    min_tokens: int = _DEFAULT_MIN_TOKENS,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list[dict]:
    """Build [min,max]-token passages by windowing consecutive sentences in order.

    Passages are read in `number` order, split into sentences, and concatenated
    into one stream; consecutive sentences are accumulated until the running token
    count is in range. Returns {number, text} dicts (number = window index) shaped
    exactly like raw passages, so the unchanged selection constraints apply.
    """
    sentences: list[str] = []
    for p in sorted(passages, key=lambda x: x["number"]):
        sentences.extend(_split_sentences(p["text"]))

    windows: list[dict] = []
    i = 0
    n = len(sentences)
    while i < n:
        j = i
        tokens = 0
        while j < n and tokens < min_tokens:
            tokens += len(sentences[j].split())
            j += 1
        if min_tokens <= tokens <= max_tokens:
            windows.append({"number": len(windows), "text": " ".join(sentences[i:j])})
            i = j  # non-overlapping
        else:
            i += 1  # window can't land in range; slide start forward by one
    return windows


def select_cloze_passages(
    passages: list[dict],
    character_names: set[str],
    *,
    min_tokens: int = _DEFAULT_MIN_TOKENS,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    min_occurrences: int = _DEFAULT_MIN_OCCURRENCES,
    max_passages: int = _DEFAULT_MAX_PASSAGES,
    seed: int = 0,
) -> tuple[list[ClozePassage], int]:
    """Select maskable passages. Returns (selected, n_skipped).

    A passage qualifies when it is `min_tokens`..`max_tokens` long and contains
    EXACTLY ONE named entity. "Named entity" is defined by the proper-name target
    set (`proper_name_targets`: passes is_proper_name + recurs >= min_occurrences)
    — the SINGLE source of truth for "is this a name". Generic common-noun spans
    the loader surfaces as Character nodes ("Foot passengers") are NOT named
    entities under the Name Cloze task, so they are not counted toward the
    one-entity / no-other-entities rule. Order is shuffled deterministically by seed.
    """
    targets = proper_name_targets(
        passages, character_names, min_occurrences=min_occurrences
    )
    eligible: list[ClozePassage] = []
    for p in passages:
        text = p["text"]
        n_tokens = len(text.split())
        if not (min_tokens <= n_tokens <= max_tokens):
            continue
        # Count only proper-name entities; generic spans do not count.
        present = [n for n in targets if _contains(text, n)]
        if len(present) != 1:
            continue
        name = present[0]
        eligible.append(
            ClozePassage(
                number=p["number"], masked_text=mask_passage(text, name), answer=name
            )
        )

    random.Random(seed).shuffle(eligible)
    selected = eligible[:max_passages]
    n_skipped = len(passages) - len(eligible)
    return selected, n_skipped


def score_cloze(
    selected: list[ClozePassage],
    predict: ClozePredictor,
    manuscript_id: str,
    *,
    n_skipped: int,
    seed: int,
    normalize: Callable[[str | None], "str | None"] | None = None,
) -> NameClozeReport:
    """Exact alias-normalized accuracy over selected passages (pure)."""
    norm = normalize or (lambda x: (x or "").strip().lower() or None)
    correct = 0
    for passage in selected:
        predicted = norm(predict(passage.masked_text, manuscript_id))
        if predicted is not None and predicted == norm(passage.answer):
            correct += 1
    return NameClozeReport(
        accuracy=(correct / len(selected) if selected else 0.0),
        n_passages=len(selected),
        n_skipped=n_skipped,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Reference predictor (corpus-frequency baseline)
# ---------------------------------------------------------------------------


class FrequencyClozePredictor:
    """Predicts the single most frequent proper-name character in the corpus.

    A no-context floor: it ignores the masked passage entirely. Real predictors
    (graph neighbourhood, vector retrieval, long context) plug in later.
    """

    def __init__(self, label: str | None) -> None:
        self.label = label

    def __call__(self, masked_passage: str, manuscript_id: str) -> str | None:
        return self.label

    @classmethod
    def fit(cls, passages: list[dict], targets: set[str]) -> "FrequencyClozePredictor":
        counts = Counter(
            name for name in targets for p in passages if _contains(p["text"], name)
        )
        return cls(counts.most_common(1)[0][0] if counts else None)


# ---------------------------------------------------------------------------
# Graph fetch + runner
# ---------------------------------------------------------------------------


def fetch_passages(driver: _Driver, manuscript_id: str) -> list[dict]:
    result = driver.execute_query(
        "MATCH (ch:Chapter {manuscript_id: $mid}) WHERE ch.text IS NOT NULL "
        "RETURN ch.number AS number, ch.text AS text ORDER BY ch.number",
        mid=manuscript_id,
    )
    return [{"number": r["number"], "text": r["text"]} for r in result.records]


def fetch_character_names(driver: _Driver, manuscript_id: str) -> set[str]:
    result = driver.execute_query(
        "MATCH (c:Character {manuscript_id: $mid}) RETURN c.name AS name",
        mid=manuscript_id,
    )
    return {r["name"] for r in result.records if r["name"]}


def run_name_cloze(
    manuscript_id: str,
    *,
    predict: ClozePredictor | None = None,
    driver: _Driver | None = None,
    min_tokens: int = _DEFAULT_MIN_TOKENS,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_passages: int = _DEFAULT_MAX_PASSAGES,
    seed: int = 0,
) -> NameClozeReport:
    if driver is None:
        from app import graph

        driver = graph.init_driver()

    passages = fetch_passages(driver, manuscript_id)
    names = fetch_character_names(driver, manuscript_id)
    # Construct 40-60 token passages by sentence-windowing the raw Chapter text,
    # then apply the unchanged selection constraints to each window.
    windows = build_sentence_windows(
        passages, min_tokens=min_tokens, max_tokens=max_tokens
    )
    selected, n_skipped = select_cloze_passages(
        windows,
        names,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        max_passages=max_passages,
        seed=seed,
    )
    if predict is None:
        targets = proper_name_targets(windows, names)
        predict = FrequencyClozePredictor.fit(windows, targets)
    return score_cloze(selected, predict, manuscript_id, n_skipped=n_skipped, seed=seed)
