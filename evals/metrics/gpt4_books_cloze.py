"""METRIC 2 — GPT4-Books name cloze (self-contained; no Neo4j).

Source: github.com/bamman-group/gpt4-books, paper aclanthology.org/2023.emnlp-main.453.
Each `data/model_output/chatgpt_results/<book>.txt` is TAB-separated, ~100 rows:

    <name>PRED</name> \t PRED \t GOLD \t passage-with-[MASK]

col 1-2 are ChatGPT's prediction (kept only as a reference baseline); **col 3 is
the gold** masked name — we score against col 3. col 4 is the passage. The
passages are pre-masked and self-contained, so this metric reads the file
directly (like pdnc_meta reads character_info.csv) and needs no graph.

Scoring (matching the GPT4-Books exact-match protocol, with symmetric honorific
stripping so "Mr. Darcy" vs "Darcy" is not a spurious miss):
  - accuracy          : normalized(pred) == normalized(gold)
  - entity_drift_rate : wrong, but the prediction is a DIFFERENT valid character
                        of this book (identity confusion — the entity-drift proxy)
  - miss_rate         : wrong and not a known character of this book
  - abstain_rate      : the predictor returned nothing / raised (robustness signal)

Every predictor call is wrapped: a raising or empty predictor is counted as an
abstain, never crashes the run — the failure IS part of the profile.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# predict(masked_passage, book_id) -> a name, or None to abstain.
ClozePredictor = Callable[[str, str], "str | None"]

MASK = "[MASK]"

# Leading honorifics stripped from BOTH sides before comparison (symmetric, so it
# cannot bias accuracy up or down — it only removes a formatting mismatch).
_HONORIFICS = frozenset(
    {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord", "st", "mme", "mlle"}
)

_CLOZE_PROMPT = """\
A single character name in the PASSAGE below has been replaced with [MASK]. \
Name the character that fills [MASK].

PASSAGE:
{masked}

Answer with ONLY that name, exactly as it would appear in the text — no other \
words, no explanation.
Name:"""


@dataclass
class ClozeRow:
    chatgpt_pred: str
    gold: str
    passage: str


@dataclass
class ClozeReport:
    book_id: str
    n: int
    n_total: int
    accuracy: float
    entity_drift_rate: float
    miss_rate: float
    abstain_rate: float
    chatgpt_reference_accuracy: float
    correct: int
    entity_drift: int
    miss: int
    abstain: int
    chatgpt_correct: int
    sample_indices: list[int]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _normalize(name: str | None) -> str:
    """Lowercase, strip quotes/punctuation/whitespace, drop a leading honorific."""
    if not name:
        return ""
    text = name.strip().strip("\"'“”‘’.,!?;:()[]").lower()
    text = re.sub(r"\s+", " ", text).strip()
    parts = text.split(" ")
    if len(parts) > 1 and parts[0].rstrip(".") in _HONORIFICS:
        parts = parts[1:]
    return " ".join(parts).strip()


def book_path(book_id: str, root: str | Path) -> Path:
    """Resolve a book id (e.g. '1342_pride_and_prejudice') to its results file."""
    return Path(root) / "model_output" / "chatgpt_results" / f"{book_id}.txt"


def read_book(path: str | Path) -> list[ClozeRow]:
    """Parse a chatgpt_results TSV; gold is column 3, passage column 4."""
    rows: list[ClozeRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue  # defensive: skip malformed rows
        rows.append(ClozeRow(chatgpt_pred=cols[1], gold=cols[2], passage=cols[3]))
    return rows


def select_rows(
    rows: list[ClozeRow], max_rows: int, seed: int
) -> tuple[list[ClozeRow], list[int]]:
    """Deterministically pick up to `max_rows` rows; returns (rows, sorted indices).

    Fixed seed so every model is scored on the SAME passages (fair comparison).
    """
    n = len(rows)
    if max_rows <= 0 or max_rows >= n:
        return rows, list(range(n))
    indices = sorted(random.Random(seed).sample(range(n), max_rows))
    return [rows[i] for i in indices], indices


def make_ask_cloze_predictor(ask: Callable[[str, dict], str]) -> ClozePredictor:
    """Wrap an `(prompt, meta) -> text` LLM-ask into a cloze predictor.

    Graph-free (unlike the baseline cloze predictor): the passage is the whole
    input. Returns the first non-empty line of the answer, or None on empty.
    """

    def predict(masked_passage: str, book_id: str) -> str | None:
        answer = ask(_CLOZE_PROMPT.format(masked=masked_passage), {"task": "cloze"})
        if not answer or not answer.strip():
            return None
        return answer.strip().splitlines()[0].strip() or None

    return predict


def score_cloze(
    rows: list[ClozeRow],
    predict: ClozePredictor,
    *,
    book_id: str,
    n_total: int,
    sample_indices: list[int],
) -> ClozeReport:
    """Score selected rows. Predictor exceptions/empties count as abstains."""
    valid_names = {_normalize(r.gold) for r in rows if _normalize(r.gold)}

    correct = entity_drift = miss = abstain = 0
    chatgpt_correct = 0
    for row in rows:
        gold = _normalize(row.gold)
        if _normalize(row.chatgpt_pred) == gold:
            chatgpt_correct += 1

        try:
            raw = predict(row.passage, book_id)
        except Exception:  # any predictor/transport failure = abstain, never crash
            raw = None
        pred = _normalize(raw)

        if not pred:
            abstain += 1
        elif pred == gold:
            correct += 1
        elif pred in valid_names:
            entity_drift += 1  # wrong-but-valid character = identity confusion
        else:
            miss += 1

    n = len(rows)
    denom = n or 1
    return ClozeReport(
        book_id=book_id,
        n=n,
        n_total=n_total,
        accuracy=correct / denom,
        entity_drift_rate=entity_drift / denom,
        miss_rate=miss / denom,
        abstain_rate=abstain / denom,
        chatgpt_reference_accuracy=chatgpt_correct / denom,
        correct=correct,
        entity_drift=entity_drift,
        miss=miss,
        abstain=abstain,
        chatgpt_correct=chatgpt_correct,
        sample_indices=sample_indices,
    )


def run_book_cloze(
    book_id: str,
    *,
    predict: ClozePredictor,
    root: str | Path,
    max_rows: int = 100,
    seed: int = 0,
) -> ClozeReport:
    """Read one book file, sample deterministically, and score `predict`."""
    rows = read_book(book_path(book_id, root))
    selected, indices = select_rows(rows, max_rows, seed)
    return score_cloze(
        selected,
        predict,
        book_id=book_id,
        n_total=len(rows),
        sample_indices=indices,
    )
