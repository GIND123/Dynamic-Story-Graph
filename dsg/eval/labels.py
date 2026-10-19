"""Split an audit sheet across annotators, check what comes back, merge it.

People label; this module only moves rows around and refuses bad values. The
blind sheet is split so every row has one primary annotator and each
double-annotation row a second, *different* one, with the pairs rotated so no
two people share all the overlap. Row text is hashed at split time, so a file
that comes back with an edited passage is caught rather than merged.

    python -m dsg.eval.labels split --sheet audit/detector --annotators 3
    python -m dsg.eval.labels check --sheet audit/detector
    python -m dsg.eval.labels merge --sheet audit/detector
    python -m dsg.eval.labels finalize --sheet audit/detector

Disagreements follow ``audit/ANALYSIS_PLAN.md``: agreement is computed on the
two independent labels first; the annotators then fill ``<label>_adjudicated``
(with ``adjudication_notes``) on every overlap row where they disagree, keeping
both originals; ``finalize`` takes the adjudicated label there, the shared label
where they agreed, and the single annotator's label elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

SEED = 20261018

# Allowed values per label column, by sheet. A value of None means free text.
SCHEMAS: dict[str, dict[str, tuple[str, ...] | None]] = {
    "detector": {
        "real_violation": ("yes", "no", "unsure"),
        "real_restatement": ("yes", "no", "unsure"),
        "labeller": None,
        "notes": None,
    },
    "calculus": {
        "label": ("ELABORATE", "SUPERSEDE", "REVISE", "EXTRACTION_ERROR", "UNSURE"),
        "correct_value": ("old", "new", "both", "neither", "unsure"),
        "correct_value_text": None,
        "labeller": None,
        "notes": None,
    },
}
# Columns that may be left empty.
OPTIONAL = {"correct_value_text", "notes"}
# Answers that the guides say must carry a note.
NEEDS_NOTE = {"unsure", "UNSURE"}
NOT_ASKED = "n/a"
ADJ = "_adjudicated"
NOTES = "adjudication_notes"
HIDDEN = ("double_annotate",)    # in the blind sheet, never in an annotator file


def schema_of(columns: list[str]) -> str:
    if "real_violation" in columns:
        return "detector"
    if "correct_value" in columns:
        return "calculus"
    raise ValueError("not an audit sheet: no known label columns")


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _content_hash(row: dict, label_columns: set[str]) -> str:
    body = {k: v for k, v in sorted(row.items()) if k not in label_columns and k not in HIDDEN}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def annotator_name(i: int) -> str:
    return f"annotator_{i + 1:02d}"


def split(sheet_dir: Path, n: int, seed: int = SEED) -> dict:
    columns, rows = _read(sheet_dir / "sheet_blind.csv")
    schema = SCHEMAS[schema_of(columns)]
    doubles = [r for r in rows if r.get("double_annotate") == "yes"]
    if doubles and n < 2:
        raise ValueError("double annotation needs at least two annotators")
    rng = random.Random(seed)
    order = rows[:]
    rng.shuffle(order)
    assignment: dict[str, list[str]] = {}
    d = 0
    for i, r in enumerate(order):
        primary = i % n
        who = [annotator_name(primary)]
        if r.get("double_annotate") == "yes":
            # Rotate the partner so the overlap spreads over every pair.
            who.append(annotator_name((primary + 1 + d % (n - 1)) % n))
            d += 1
        assignment[r["item_id"]] = who
    out_cols = [c for c in columns if c not in HIDDEN]
    by_id = {r["item_id"]: r for r in rows}
    files: dict[str, list[str]] = {annotator_name(i): [] for i in range(n)}
    for item_id in (r["item_id"] for r in order):
        for who in assignment[item_id]:
            files[who].append(item_id)
    for who, ids in files.items():
        _write(sheet_dir / "annotators" / f"{who}.csv", out_cols, [by_id[i] for i in ids])
    plan = {
        "seed": seed, "annotators": n, "sheet": "sheet_blind.csv",
        "command": " ".join(["python", "-m", "dsg.eval.labels", *sys.argv[1:]]),
        "rows_per_annotator": {w: len(ids) for w, ids in files.items()},
        "double_pairs": dict(Counter("+".join(sorted(v)) for v in assignment.values()
                                     if len(v) == 2)),
        "assignment": assignment,
        "content_sha256": {i: _content_hash(by_id[i], set(schema)) for i in by_id},
    }
    (sheet_dir / "annotators" / "assignment.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def _canonical(value: str, allowed: tuple[str, ...]) -> str | None:
    v = value.strip()
    for a in allowed:
        if v.lower() == a.lower():
            return a
    return None


def check(sheet_dir: Path, partial: bool = False) -> tuple[list[str], dict[str, list[dict]]]:
    """Every problem found, and the cleaned rows per annotator."""
    plan = json.loads((sheet_dir / "annotators" / "assignment.json").read_text())
    _, blind = _read(sheet_dir / plan["sheet"])
    blind_by_id = {r["item_id"]: r for r in blind}
    schema = SCHEMAS[schema_of(list(blind[0]))]
    problems: list[str] = []
    cleaned: dict[str, list[dict]] = {}
    for i in range(plan["annotators"]):
        who = annotator_name(i)
        path = sheet_dir / "annotators" / f"{who}.csv"
        expected = {k for k, v in plan["assignment"].items() if who in v}
        _, rows = _read(path)
        seen = Counter(r.get("item_id", "") for r in rows)
        for item_id, k in seen.items():
            if k > 1:
                problems.append(f"{who}: {item_id} appears {k} times")
        missing, extra = expected - set(seen), set(seen) - expected
        problems += [f"{who}: missing row {x}" for x in sorted(missing)]
        problems += [f"{who}: unexpected row {x}" for x in sorted(extra)]
        out = []
        for r in rows:
            item_id = r.get("item_id", "")
            if item_id not in expected:
                continue
            where = f"{who}:{item_id}"
            if _content_hash(r, set(schema)) != plan["content_sha256"][item_id]:
                problems.append(f"{where}: row text was edited")
            clean = dict(r)
            for col, allowed in schema.items():
                raw = (r.get(col) or "").strip()
                preset = (blind_by_id[item_id].get(col) or "").strip()
                if preset == NOT_ASKED:
                    if raw != NOT_ASKED:
                        problems.append(f"{where}: {col} must stay '{NOT_ASKED}'")
                    continue
                if not raw:
                    if col not in OPTIONAL and not partial:
                        problems.append(f"{where}: {col} is empty")
                    continue
                if allowed is not None:
                    value = _canonical(raw, allowed)
                    if value is None:
                        problems.append(f"{where}: {col}='{raw}' not in {list(allowed)}")
                        continue
                    clean[col] = value
                    if value in NEEDS_NOTE and not (r.get("notes") or "").strip():
                        problems.append(f"{where}: {col}={value} needs a note")
            out.append(clean)
        cleaned[who] = out
    return problems, cleaned


def decision_columns(schema: dict[str, tuple[str, ...] | None]) -> list[str]:
    """Label columns with a closed set of values: the ones that get adjudicated."""
    return [c for c, allowed in schema.items() if allowed is not None]


def _norm(value: str) -> str:
    return (value or "").strip().lower()


def merge(sheet_dir: Path) -> Path:
    """One row per item: both annotators' labels, and a slot to adjudicate.

    ``<label>_adjudicated`` is blank where the two labels disagree and must be
    filled by the annotators jointly; it is ``n/a`` where there is nothing to
    adjudicate. Re-merging keeps adjudications already written, as long as the
    two original labels they resolved have not changed.
    """
    problems, cleaned = check(sheet_dir)
    if problems:
        raise SystemExit("refusing to merge:\n  " + "\n  ".join(problems))
    plan = json.loads((sheet_dir / "annotators" / "assignment.json").read_text())
    columns, blind = _read(sheet_dir / plan["sheet"])
    schema = SCHEMAS[schema_of(columns)]
    decisions = decision_columns(schema)
    path = sheet_dir / "labels_merged.csv"
    previous = {r["item_id"]: r for r in _read(path)[1]} if path.exists() else {}
    by_who = {w: {r["item_id"]: r for r in rows} for w, rows in cleaned.items()}
    merged, dropped = [], []
    for r in blind:
        who = plan["assignment"][r["item_id"]]
        row = {"item_id": r["item_id"], "annotator_a": who[0]}
        row.update({c: by_who[who[0]][r["item_id"]].get(c, "") for c in schema})
        old = previous.get(r["item_id"], {})
        if len(who) > 1:
            row["annotator_b"] = who[1]
            row.update({f"{c}_b": by_who[who[1]][r["item_id"]].get(c, "") for c in schema})
        for c in decisions:
            if len(who) < 2 or _norm(row[c]) == _norm(row[f"{c}_b"]):
                row[c + ADJ] = NOT_ASKED
                continue
            kept = (old.get(c + ADJ, "") if _norm(old.get(c)) == _norm(row[c])
                    and _norm(old.get(f"{c}_b")) == _norm(row[f"{c}_b"]) else "")
            if old.get(c + ADJ, "") not in ("", NOT_ASKED) and not kept:
                dropped.append(f"{r['item_id']}:{c}")
            row[c + ADJ] = kept
        row[NOTES] = old.get(NOTES, "") if any(row[c + ADJ] not in ("", NOT_ASKED)
                                               for c in decisions) else ""
        merged.append(row)
    out_cols = ["item_id", "annotator_a", *schema, "annotator_b", *(f"{c}_b" for c in schema),
                *(c + ADJ for c in decisions), NOTES]
    _write(path, out_cols, merged)
    if dropped:
        print("adjudications dropped because an original label changed: " + ", ".join(dropped))
    return path


def finalize(sheet_dir: Path) -> tuple[list[str], Path | None]:
    """Problems, and the final-label file once there are none."""
    columns, rows = _read(sheet_dir / "labels_merged.csv")
    schema = SCHEMAS[schema_of(columns)]
    decisions = decision_columns(schema)
    problems: list[str] = []
    final = []
    for r in rows:
        out = {"item_id": r["item_id"]}
        for c in decisions:
            adj = (r.get(c + ADJ) or "").strip()
            if not r.get("annotator_b"):
                out[c], out[f"{c}_source"] = r[c], "single"
            elif adj == NOT_ASKED:
                out[c], out[f"{c}_source"] = r[c], "agreed"
            elif not adj:
                problems.append(f"{r['item_id']}: {c} disagreement not adjudicated "
                                f"('{r[c]}' vs '{r[f'{c}_b']}')")
            else:
                value = _canonical(adj, schema[c]) if adj != NOT_ASKED else None
                if value is None:
                    problems.append(f"{r['item_id']}: {c}{ADJ}='{adj}' not in {list(schema[c])}")
                    continue
                if not (r.get(NOTES) or "").strip():
                    problems.append(f"{r['item_id']}: adjudication needs a note")
                out[c], out[f"{c}_source"] = value, "adjudicated"
        final.append(out)
    if problems:
        return problems, None
    path = sheet_dir / "labels_final.csv"
    _write(path, ["item_id", *(x for c in decisions for x in (c, f"{c}_source"))], final)
    return [], path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.labels", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--sheet", required=True)
    s.add_argument("--annotators", type=int, required=True)
    s.add_argument("--seed", type=int, default=SEED)
    c = sub.add_parser("check")
    c.add_argument("--sheet", required=True)
    c.add_argument("--partial", action="store_true", help="allow unlabelled rows")
    m = sub.add_parser("merge")
    m.add_argument("--sheet", required=True)
    f = sub.add_parser("finalize", help="final labels once disagreements are adjudicated")
    f.add_argument("--sheet", required=True)
    args = p.parse_args(argv)
    sheet = Path(args.sheet)
    if args.cmd == "split":
        plan = split(sheet, args.annotators, args.seed)
        print(json.dumps({k: plan[k] for k in ("rows_per_annotator", "double_pairs")}, indent=2))
        return 0
    if args.cmd == "check":
        problems, _ = check(sheet, partial=args.partial)
        print("\n".join(problems) if problems else "ok")
        return 1 if problems else 0
    if args.cmd == "finalize":
        problems, path = finalize(sheet)
        print("\n".join(problems) if problems else f"final labels -> {path}")
        return 1 if problems else 0
    print(f"merged -> {merge(sheet)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
