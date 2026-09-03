"""Plane B' -- mention linking, probed at the point the mention occurs.

The strongest extrinsic measure available here, and the one least entangled with
the extractor. PDNC annotates every referring expression inside a quotation with
a character offset and the character it denotes. Each becomes a question put to
the state at exactly the reading position where a human reader meets it: *given
only what has been read, who is this?*

Pronouns are excluded. Resolving them needs syntactic coreference, which this
system does not attempt and does not claim; including them would report a
capability gap as a representation difference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dsg.data.pdnc import Novel
from dsg.eval.identity import node_to_character
from dsg.policies.runner import Probe, RunResult

_PRONOUNS = frozenset(
    {
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves", "who", "whom", "whose",
        "this", "that", "these", "those", "one", "someone", "somebody",
    }
)
_WORD_RE = re.compile(r"[^\w\s']+")


def is_scorable(text: str) -> bool:
    s = _WORD_RE.sub(" ", (text or "").strip().lower()).strip()
    return bool(s) and s not in _PRONOUNS and len(s) > 1


def build_probes(novel: Novel) -> list[Probe]:
    """One probe per non-pronominal gold mention, in reading order."""
    gold_by_name = {c.name: c.char_id for c in novel.characters}
    out: list[Probe] = []
    for mention in novel.mentions:
        if not is_scorable(mention.text):
            continue
        char_id = gold_by_name.get(mention.entity)
        if char_id is None:
            continue
        out.append(Probe(position=mention.start, surface=mention.text, gold=char_id))
    return out


@dataclass(slots=True)
class MentionScore:
    n_probes: int
    answered: int
    correct: int
    accuracy: float          # over all probes; unanswered counts as wrong
    accuracy_answered: float # over the probes the state was willing to answer
    coverage: float
    by_decile: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "mention_probes": float(self.n_probes),
            "mention_answered": float(self.answered),
            "mention_correct": float(self.correct),
            "mention_acc": self.accuracy,
            "mention_acc_answered": self.accuracy_answered,
            "mention_coverage": self.coverage,
        }


def score_mentions(result: RunResult, novel: Novel, n_deciles: int = 10) -> MentionScore:
    node_map = node_to_character(result.state, novel)
    text_len = max(1, len(novel.text))
    buckets = [{"n": 0, "answered": 0, "correct": 0} for _ in range(n_deciles)]
    correct = answered = 0

    for answer in result.probe_answers:
        bucket = buckets[min(n_deciles - 1, int(answer.position / text_len * n_deciles))]
        bucket["n"] += 1
        if answer.node_id is None:
            continue
        predicted = node_map.get(answer.node_id)
        if predicted is None:
            continue
        answered += 1
        bucket["answered"] += 1
        if predicted == answer.gold:
            correct += 1
            bucket["correct"] += 1

    total = len(result.probe_answers)
    return MentionScore(
        n_probes=total,
        answered=answered,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        accuracy_answered=correct / answered if answered else 0.0,
        coverage=answered / total if total else 0.0,
        by_decile=[
            {
                "decile": i, "n": b["n"], "answered": b["answered"],
                "accuracy": (b["correct"] / b["n"]) if b["n"] else 0.0,
                "accuracy_answered": (b["correct"] / b["answered"]) if b["answered"] else 0.0,
            }
            for i, b in enumerate(buckets)
        ],
    )
