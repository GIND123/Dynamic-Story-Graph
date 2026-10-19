"""A blind hand-check of flagged canon violations.

The violation checker is a proximity test: a contradicting word near the
subject's name counts, whoever or whatever it describes. The smoke runs showed
it charging shutter paint to a character's eyes and one man's trade to another.
Prose that packs characters together is charged more, so a guard on
``violation_rate`` cannot be read until the flags themselves are.

This samples flagged spans from every arm, shuffles them together with the arm
hidden, and asks a person: real, false positive, or unsure. Unblinding gives a
false-positive rate per arm and a corrected violation rate, with bounds.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from dsg.generate.canon import Premise

VERDICTS = {
    "real": "real", "r": "real",
    "false positive": "false positive", "false": "false positive",
    "fp": "false positive", "f": "false positive",
    "unsure": "unsure", "u": "unsure", "?": "unsure",
}
SHEET_COLUMNS = ("item_id", "flagged_fact", "context", "verdict", "note")
CONTEXT = 100   # characters of padding outside the subject mention and the flagged word

INSTRUCTION = """# Violation hand-check

Each row in `handcheck.csv` is a passage the automatic checker flagged as
contradicting one planted fact. The flagged word is marked `[[like this]]`, and
the mention of the fact's subject that the checker matched is marked
`{{like this}}`.
Decide whether the passage really contradicts **that fact about that subject**,
and write one of: `real`, `false positive`, `unsure`.

A false positive is a flagged word that describes someone or something else
(another character's eyes, a wall's paint, another person's trade), or that is
not a claim about the subject at all. Rate every row before discussing any.
"""


def _span_key(story: str, fact: str, condition: str, chapter: int) -> str:
    return f"{story}|{fact}|{condition}|{chapter}"


def _context(text: str, s_start: int, s_end: int, start: int, end: int) -> str:
    """The flagged word and the subject mention it was charged to, both marked.

    The checker allows up to 220 characters between them, so a fixed window
    round the word alone can hide whose attribute it is.
    """
    lo = max(0, min(s_start, start) - CONTEXT)
    hi = min(len(text), max(s_end, end) + CONTEXT)
    marks = sorted([(s_start, s_end, "{{", "}}"), (start, end, "[[", "]]")])
    body, cursor = "", lo
    for a, b, left, right in marks:
        body += text[cursor:a] + left + text[a:b] + right
        cursor = b
    body = " ".join((body + text[cursor:hi]).split())
    return ("…" if lo > 0 else "") + body + ("…" if hi < len(text) else "")


def build_sheet(
    payload: dict, out_dir: Path, per_arm: int = 40, seed: int = 0
) -> Path:
    """Write handcheck.csv, HANDCHECK.md and HANDCHECK_KEY.json; return the sheet.

    Up to ``per_arm`` flagged spans are sampled from each condition. Spans with
    identical fact and context (chapter 1 is often word-for-word the same in two
    memories of one template) become one row, rated once for every arm it
    belongs to.
    """
    rng = random.Random(seed)
    premises = {p["story_id"]: Premise.from_json(p) for p in payload["premises"]}
    facts = {
        (story, f.fact_id): f for story, p in premises.items() for f in p.canon
    }

    spans_by_arm: dict[str, list[dict]] = {}
    arms: dict[str, dict] = {}
    for rec in payload["records"]:
        condition, story = rec["condition"], rec["story_id"]
        arm = arms.setdefault(condition, {"stories": set(), "flagged": {}})
        arm["stories"].add(story)
        for fact_id in rec.get("violations", []):
            fact = facts[(story, fact_id)]
            where = fact.locate(rec.get("text", ""))
            if where is None:      # text missing or no longer matching: cannot show it
                continue
            key = _span_key(story, fact_id, condition, rec["chapter"])
            arm["flagged"].setdefault(f"{story}|{fact_id}", []).append(key)
            spans_by_arm.setdefault(condition, []).append({
                "key": key, "story": story, "fact_id": fact_id,
                "statement": fact.statement,
                "context": _context(rec["text"], *where),
            })

    sampled = []
    for condition in sorted(spans_by_arm):
        spans = spans_by_arm[condition]
        sampled += rng.sample(spans, per_arm) if len(spans) > per_arm else spans

    items: dict[tuple[str, str], dict] = {}
    for span in sampled:
        item = items.setdefault(
            (span["statement"], span["context"]),
            {"statement": span["statement"], "context": span["context"], "spans": []},
        )
        item["spans"].append(span["key"])
    rows = list(items.values())
    rng.shuffle(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key: dict[str, list[str]] = {}
    ids: set[str] = set()
    sheet = out_dir / "handcheck.csv"
    with sheet.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SHEET_COLUMNS)
        for row in rows:
            item_id = f"{rng.getrandbits(32):08x}"
            while item_id in ids:
                item_id = f"{rng.getrandbits(32):08x}"
            ids.add(item_id)
            key[item_id] = row["spans"]
            writer.writerow([item_id, row["statement"], row["context"], "", ""])
    (out_dir / "HANDCHECK.md").write_text(INSTRUCTION)
    (out_dir / "HANDCHECK_KEY.json").write_text(json.dumps({
        "seed": seed, "per_arm": per_arm,
        "items": key,
        "arms": {
            condition: {
                "total_facts": sum(len(premises[s].canon) for s in arm["stories"]),
                "flagged": arm["flagged"],
            }
            for condition, arm in sorted(arms.items())
        },
    }, indent=2))
    return sheet


def unblind_sheet(sheet_csv: Path, key_json: Path) -> dict:
    """False-positive rate per arm, and the violation rate corrected for it.

    A planted fact counts as violated in a story when any of its flagged spans
    is real. Per arm:

    - ``confirmed``: facts with at least one span rated real;
    - ``cleared``: facts whose every flagged span was rated false positive;
    - ``unresolved``: the rest (unsampled, unsure, or partly rated).

    The corrected rate counts confirmed facts in full and unresolved facts at
    one minus the arm's span false-positive rate; the bounds count unresolved
    facts as none and as all violated.
    """
    key = json.loads(Path(key_json).read_text())
    verdict_of: dict[str, str] = {}
    with Path(sheet_csv).open(newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            item_id = (row.get("item_id") or "").strip()
            if item_id not in key["items"]:
                raise ValueError(f"handcheck row {row_number}: unknown item id {item_id!r}")
            raw = (row.get("verdict") or "").strip().lower()
            if not raw:
                continue
            if raw not in VERDICTS:
                raise ValueError(
                    f"handcheck row {row_number}: verdict must be real, "
                    f"false positive or unsure, not {raw!r}"
                )
            for span in key["items"][item_id]:
                verdict_of[span] = VERDICTS[raw]

    out: dict[str, dict] = {}
    for condition, arm in key["arms"].items():
        total = arm["total_facts"]
        counts = {"real": 0, "false positive": 0, "unsure": 0}
        confirmed = cleared = unresolved = 0
        for spans in arm["flagged"].values():
            verdicts = [verdict_of.get(s) for s in spans]
            for v in verdicts:
                if v:
                    counts[v] += 1
            if "real" in verdicts:
                confirmed += 1
            elif all(v == "false positive" for v in verdicts):
                cleared += 1
            else:
                unresolved += 1
        decided = counts["real"] + counts["false positive"]
        fp_rate = counts["false positive"] / decided if decided else None
        keep = 1 - fp_rate if fp_rate is not None else 1.0
        flagged = len(arm["flagged"])
        out[condition] = {
            "spans_rated": dict(counts),
            "false_positive_rate": fp_rate,
            "facts_flagged": flagged,
            "facts_confirmed": confirmed,
            "facts_cleared": cleared,
            "facts_unresolved": unresolved,
            "raw_violation_rate": flagged / total if total else 0.0,
            "corrected_violation_rate": (confirmed + unresolved * keep) / total if total else 0.0,
            "corrected_bounds": (
                confirmed / total if total else 0.0,
                (confirmed + unresolved) / total if total else 0.0,
            ),
        }
    return out


def summary_markdown(result: dict) -> str:
    lines = [
        "# Violation hand-check",
        "",
        "| arm | spans real / FP / unsure | FP rate | raw violation rate | "
        "corrected | bounds | facts confirmed / cleared / unresolved |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition, r in result.items():
        c = r["spans_rated"]
        fp = "n/a" if r["false_positive_rate"] is None else f"{r['false_positive_rate']:.2f}"
        lo, hi = r["corrected_bounds"]
        lines.append(
            f"| `{condition}` | {c['real']} / {c['false positive']} / {c['unsure']} | {fp} | "
            f"{r['raw_violation_rate']:.3f} | {r['corrected_violation_rate']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {r['facts_confirmed']} / {r['facts_cleared']} / "
            f"{r['facts_unresolved']} |"
        )
    lines += [
        "",
        "The raw rate is the share of planted facts flagged at least once, as the "
        "report's `violation_rate` counts it. The corrected rate keeps confirmed "
        "facts and discounts unresolved ones by the arm's false-positive rate; the "
        "bounds treat every unresolved fact as clean, then as violated.",
    ]
    return "\n".join(lines) + "\n"
