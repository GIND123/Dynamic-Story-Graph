"""METRIC 4 — LitBank coreference (pairwise identity tracking; no Neo4j).

Source: github.com/dbamman/litbank, gold coref in CoNLL-2012 format under
`data/litbank/coref/conll/<doc>.conll` (last column = cluster markers `(N`, `N)`,
`(N)`, pipe-separated when nested). Read directly, like pdnc_meta reads CSV.

The probe: sample labeled mention pairs from ONE document and ask the model
whether the two mentions refer to the SAME person. This is the "entity drift"
task in its purest form — a wrong answer means the model either **split** one
character into two (missed a coreference) or **merged** two characters into one
(a false link). We filter to PERSON clusters (a cluster is a person if it
contains a singular personal pronoun) so both sides of a negative pair are people
— a fair, hard negative, not person-vs-place.

Reported: accuracy, precision/recall/F1 on the coreferent class, positive-miss
rate (splits), false-merge rate (merges), and the headline
`entity_drift_rate = 1 - accuracy` (captures both drift directions). A predictor
that abstains/raises is counted, never crashes the run.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# predict(context, mention_a, mention_b) -> True (same) / False (different) / None.
CorefPredictor = Callable[[str, str, str], "bool | None"]

_PERSON_PRONOUNS = frozenset(
    {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
)
_MARK_SINGLE = re.compile(r"^\((\d+)\)$")
_MARK_OPEN = re.compile(r"^\((\d+)$")
_MARK_CLOSE = re.compile(r"^(\d+)\)$")

_COREF_PROMPT = """\
Read the PASSAGE, then decide whether the two mentions refer to the SAME person.

PASSAGE:
{context}

Mention A: "{a}"
Mention B: "{b}"

Answer with exactly one word: yes or no.
Answer:"""


@dataclass
class Mention:
    cluster: int
    sent_idx: int
    text: str


@dataclass
class Sentence:
    idx: int
    text: str


@dataclass
class CorefDoc:
    sentences: list[Sentence]
    clusters: dict[int, list[Mention]] = field(default_factory=dict)


@dataclass
class CorefPair:
    mention_a: Mention
    mention_b: Mention
    same: bool  # gold label
    context: str


@dataclass
class CorefReport:
    doc_id: str
    n: int
    n_positive: int
    n_negative: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    positive_miss_rate: float
    false_merge_rate: float
    abstain_rate: float
    entity_drift_rate: float
    correct: int
    abstain: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def conll_path(doc_id: str, root: str | Path) -> Path:
    return Path(root) / "coref" / "conll" / f"{doc_id}.conll"


def parse_conll(path: str | Path) -> CorefDoc:
    """Parse a CoNLL-2012 coref file into sentences + clusters of mentions."""
    sentences: list[Sentence] = []
    clusters: dict[int, list[Mention]] = {}
    cur_tokens: list[str] = []
    cur_labels: list[str] = []
    sent_idx = 0

    def flush() -> None:
        nonlocal sent_idx
        if not cur_tokens:
            return
        sentences.append(Sentence(idx=sent_idx, text=" ".join(cur_tokens)))
        for cluster, start, end in _mentions_in_sentence(cur_labels):
            text = " ".join(cur_tokens[start : end + 1])
            clusters.setdefault(cluster, []).append(
                Mention(cluster=cluster, sent_idx=sent_idx, text=text)
            )
        sent_idx += 1

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        if not line.strip():
            flush()
            cur_tokens, cur_labels = [], []
            continue
        fields = line.split()
        cur_tokens.append(fields[3])
        cur_labels.append(fields[-1])
    flush()
    return CorefDoc(sentences=sentences, clusters=clusters)


def _mentions_in_sentence(labels: list[str]) -> list[tuple[int, int, int]]:
    """Return (cluster, start_tok, end_tok) mentions, handling nesting + pipes."""
    open_stacks: dict[int, list[int]] = {}
    out: list[tuple[int, int, int]] = []
    for pos, label in enumerate(labels):
        if label == "_":
            continue
        for part in label.split("|"):
            if m := _MARK_SINGLE.match(part):
                out.append((int(m.group(1)), pos, pos))
            elif m := _MARK_OPEN.match(part):
                open_stacks.setdefault(int(m.group(1)), []).append(pos)
            elif m := _MARK_CLOSE.match(part):
                cid = int(m.group(1))
                if open_stacks.get(cid):
                    start = open_stacks[cid].pop()
                    out.append((cid, start, pos))
    return out


def _is_person_cluster(mentions: list[Mention]) -> bool:
    return any(m.text.strip().lower() in _PERSON_PRONOUNS for m in mentions)


def person_clusters(doc: CorefDoc) -> dict[int, list[Mention]]:
    return {c: ms for c, ms in doc.clusters.items() if _is_person_cluster(ms)}


def _context(doc: CorefDoc, a: Mention, b: Mention) -> str:
    idxs = sorted({a.sent_idx, b.sent_idx})
    return " ".join(doc.sentences[i].text for i in idxs)


def build_pairs(doc: CorefDoc, max_pairs: int, seed: int) -> list[CorefPair]:
    """Balanced positive/negative person-mention pairs; deterministic by seed."""
    rng = random.Random(seed)
    people = person_clusters(doc)

    # Positives: two DIFFERENT surface forms from the same cluster (non-trivial).
    positives: list[tuple[Mention, Mention]] = []
    for mentions in people.values():
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                if mentions[i].text.lower() != mentions[j].text.lower():
                    positives.append((mentions[i], mentions[j]))

    # Negatives: one mention from each of two different person clusters.
    negatives: list[tuple[Mention, Mention]] = []
    cluster_ids = list(people.keys())
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            negatives.append((people[cluster_ids[i]][0], people[cluster_ids[j]][0]))

    rng.shuffle(positives)
    rng.shuffle(negatives)
    half = max(1, max_pairs // 2)
    chosen = [(a, b, True) for a, b in positives[:half]]
    chosen += [(a, b, False) for a, b in negatives[:half]]
    rng.shuffle(chosen)
    return [
        CorefPair(mention_a=a, mention_b=b, same=same, context=_context(doc, a, b))
        for a, b, same in chosen
    ]


def make_ask_coref_predictor(ask: Callable[[str, dict], str]) -> CorefPredictor:
    """Wrap an `(prompt, meta) -> text` ask into a yes/no coref predictor."""

    def predict(context: str, a: str, b: str) -> bool | None:
        answer = ask(_COREF_PROMPT.format(context=context, a=a, b=b), {"task": "coref"})
        low = (answer or "").strip().lower()
        if low.startswith("yes") or low.startswith("same"):
            return True
        if low.startswith("no") or low.startswith("diff"):
            return False
        return None

    return predict


def score_coref(
    pairs: list[CorefPair], predict: CorefPredictor, *, doc_id: str
) -> CorefReport:
    """Score pairwise coreference; predictor errors/abstains never crash."""
    tp = fp = fn = tn = abstain = 0
    pos_miss = neg_merge = 0
    n_pos = sum(1 for p in pairs if p.same)
    n_neg = len(pairs) - n_pos

    for pair in pairs:
        try:
            pred = predict(pair.context, pair.mention_a.text, pair.mention_b.text)
        except Exception:
            pred = None
        if pred is None:
            abstain += 1
            if pair.same:
                fn += 1
                pos_miss += 1
            else:
                fp += 1  # abstain on a negative is not a correct "different"
            continue
        if pair.same and pred:
            tp += 1
        elif pair.same and not pred:
            fn += 1
            pos_miss += 1
        elif not pair.same and pred:
            fp += 1
            neg_merge += 1
        else:
            tn += 1

    n = len(pairs)
    denom = n or 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    accuracy = (tp + tn) / denom
    return CorefReport(
        doc_id=doc_id,
        n=n,
        n_positive=n_pos,
        n_negative=n_neg,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        positive_miss_rate=pos_miss / n_pos if n_pos else 0.0,
        false_merge_rate=neg_merge / n_neg if n_neg else 0.0,
        abstain_rate=abstain / denom,
        entity_drift_rate=1.0 - accuracy,  # split + merge, symmetric headline
        correct=tp + tn,
        abstain=abstain,
    )


def run_document_coref(
    doc_id: str,
    *,
    predict: CorefPredictor,
    root: str | Path,
    max_pairs: int = 40,
    seed: int = 0,
) -> CorefReport:
    """Parse one document, build balanced pairs, and score `predict`."""
    doc = parse_conll(conll_path(doc_id, root))
    pairs = build_pairs(doc, max_pairs, seed)
    return score_coref(pairs, predict, doc_id=doc_id)
