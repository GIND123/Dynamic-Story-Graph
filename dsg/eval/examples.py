"""Qualitative output: the actual revision events, with the text that caused them.

Aggregate metrics can hide a calculus that fires for the wrong reasons. This
pulls every non-trivial update out of the log with its evidence span and the
surrounding prose, so the behaviour can be read rather than trusted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dsg.data.pdnc import Novel
from dsg.policies.runner import RunResult
from dsg.schemas import Op

INTERESTING = (Op.REVISE, Op.SUPERSEDE, Op.MERGE_PROVISIONAL, Op.MERGE_COMMITTED, Op.ELABORATE)


@dataclass(slots=True)
class Example:
    book: str
    policy: str
    op: str
    window: int
    position: float
    detail: str
    subject: str = ""
    predicate: str = ""
    old_value: str = ""
    new_value: str = ""
    evidence: str = ""
    context: str = ""

    def as_dict(self) -> dict:
        return {
            "book": self.book, "policy": self.policy, "op": self.op,
            "window": self.window, "position": round(self.position, 3),
            "detail": self.detail, "subject": self.subject, "predicate": self.predicate,
            "old_value": self.old_value, "new_value": self.new_value,
            "evidence": self.evidence, "context": self.context,
        }


def _context(novel: Novel, start: int, end: int, pad: int = 160) -> str:
    lo, hi = max(0, start - pad), min(len(novel.text), end + pad)
    return " ".join(novel.text[lo:hi].split())


def collect(result: RunResult, novel: Novel, limit: int = 40) -> list[Example]:
    state = result.state
    windows = max(1, result.windows)
    out: list[Example] = []
    for record in state.log:
        if record.op not in INTERESTING:
            continue
        example = Example(
            book=novel.book_id, policy=result.policy, op=record.op.value,
            window=record.index, position=record.index / windows, detail=record.detail,
        )
        assertion = state.assertions.get(record.target)
        if assertion is not None:
            subject_node = state.entities.get(assertion.subject)
            example.subject = (
                subject_node.canonical if subject_node else assertion.subject
            )
            example.predicate = assertion.predicate
            example.new_value = assertion.object
            prior = state.assertions.get(record.payload or "")
            if prior is not None:
                example.old_value = prior.object
            if assertion.provenance is not None:
                example.evidence = _context(
                    novel, assertion.provenance.start, assertion.provenance.end, pad=0
                )[:400]
                example.context = _context(
                    novel, assertion.provenance.start, assertion.provenance.start + 1
                )
        else:
            node = state.entities.get(record.target)
            if node is not None:
                example.subject = node.canonical
                example.new_value = ", ".join(sorted(node.proper_names)[:3])
                if node.provenance:
                    span = node.provenance[-1]
                    example.context = _context(novel, span.start, span.start + 1)
        out.append(example)

    # Prefer the operations that actually distinguish the policies.
    priority = {
        Op.REVISE.value: 0, Op.MERGE_COMMITTED.value: 1, Op.MERGE_PROVISIONAL.value: 2,
        Op.SUPERSEDE.value: 3, Op.ELABORATE.value: 4,
    }
    out.sort(key=lambda e: (priority.get(e.op, 9), -e.position))
    return out[:limit]


def dump(
    results: dict[str, RunResult],
    novels: dict[str, Novel],
    out_path: Path,
    policy: str = "dsg-full",
    per_book: int = 25,
) -> Path:
    rows: list[dict] = []
    for book_id, result in results.items():
        novel = novels.get(book_id)
        if novel is None or result.policy != policy:
            continue
        rows.extend(e.as_dict() for e in collect(result, novel, limit=per_book))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    return out_path


def to_markdown(rows: list[dict], limit: int = 25) -> str:
    lines = ["| op | book | position | change | evidence |", "|---|---|---|---|---|"]
    for row in rows[:limit]:
        change = (
            f"{row['subject']}.{row['predicate']}: "
            f"'{row['old_value']}' → '{row['new_value']}'"
            if row["predicate"]
            else row["detail"]
        )
        context = (row["context"] or row["evidence"])[:150].replace("|", "/")
        lines.append(
            f"| `{row['op']}` | {row['book']} | {row['position']:.2f} | "
            f"{change[:90]} | …{context}… |"
        )
    return "\n".join(lines)
