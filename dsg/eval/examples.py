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


def _name(state, value: str) -> str:
    """Show a person's name rather than the internal node id it resolves to."""
    node = state.entities.get(value)
    return node.canonical if node is not None else value


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
            example.new_value = _name(state, assertion.object)
            prior = state.assertions.get(record.payload or "")
            if prior is not None:
                example.old_value = _name(state, prior.object)
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


@dataclass(slots=True)
class TrajectoryEntry:
    d_open: int
    d_close: int | None
    predicate: str
    object: str
    status: str
    certainty: str
    position: float
    evidence: str = ""

    def as_dict(self) -> dict:
        return {
            "d_open": self.d_open, "d_close": self.d_close,
            "predicate": self.predicate, "object": self.object,
            "status": self.status, "certainty": self.certainty,
            "position": round(self.position, 3), "evidence": self.evidence,
        }


def trajectory(
    result: RunResult, novel: Novel, character: str, limit: int = 40
) -> list[dict]:
    """Everything the reader came to believe about one character, in reading order.

    Superseded and retracted beliefs are kept, with the discourse interval over
    which they were held. This is the view that shows the two clocks doing their
    work: a fact can leave the believed set because the world moved on, or
    because the reader was corrected, and the record says which.
    """
    state = result.state
    node_id = state.resolve(character)
    if node_id is None:
        return []
    node_id = state.deref(node_id)
    windows = max(1, result.windows)

    rows = [
        TrajectoryEntry(
            d_open=a.d_open,
            d_close=a.d_close,
            predicate=a.predicate,
            object=(
                state.entities[a.object].canonical
                if a.object in state.entities
                else a.object
            ),
            status=a.status.value,
            certainty=a.certainty.value,
            position=a.d_open / windows,
            evidence=(
                _context(novel, a.provenance.start, a.provenance.start + 1, pad=90)
                if a.provenance is not None
                else ""
            ),
        )
        for a in state.assertions.values()
        if a.subject == node_id
    ]
    rows.sort(key=lambda r: (r.d_open, r.predicate))
    return [r.as_dict() for r in rows[:limit]]


def trajectory_markdown(rows: list[dict], character: str) -> str:
    if not rows:
        return f"_no state recorded for {character}_"
    lines = [
        f"**{character}** — what the reader believes, and when.",
        "",
        "| read at | predicate | value | held until | status | source |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        held = "—" if r["d_close"] is None else f"window {r['d_close']}"
        lines.append(
            f"| {r['position']:.2f} | {r['predicate']} | {str(r['object'])[:40]} | "
            f"{held} | {r['status']} | {r['certainty']} |"
        )
    return "\n".join(lines)
