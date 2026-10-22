"""Read one annotator's rows, and write back only the label columns.

The questions follow the labelling guides: a detector row asks whichever of
``real_violation`` / ``real_restatement`` is not preset to ``n/a`` (an
unflagged mention asks both, independently); a calculus row asks ``label`` and
``correct_value``. Nothing is preselected. A save is refused unless every
asked question has an allowed answer and any ``unsure`` carries a note, so a
saved row always passes ``python -m dsg.eval.labels check``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dsg.eval.labels import NEEDS_NOTE, NOT_ASKED, SCHEMAS, _read, _write

SHEETS = ("detector", "calculus")
QUESTIONS = {
    "detector": ("real_violation", "real_restatement"),
    "calculus": ("label", "correct_value"),
}
FREE_TEXT = {"detector": ("notes",), "calculus": ("correct_value_text", "notes")}


class LabelError(ValueError):
    """A save the guides do not allow."""


def sheet_path(root: Path, sheet: str, me: str) -> Path:
    if sheet not in SHEETS:
        raise LabelError(f"unknown sheet {sheet!r}")
    return Path(root) / "audit" / sheet / "annotators" / f"{me}.csv"


def asked(sheet: str, row: dict) -> list[str]:
    """The label columns this row asks about."""
    return [c for c in QUESTIONS[sheet] if (row.get(c) or "").strip() != NOT_ASKED]


def is_done(sheet: str, row: dict) -> bool:
    return all((row.get(c) or "").strip() for c in asked(sheet, row))


def load(root: Path, sheet: str, me: str) -> tuple[list[str], list[dict]]:
    return _read(sheet_path(root, sheet, me))


def progress(root: Path, sheet: str, me: str) -> dict:
    _, rows = load(root, sheet, me)
    done = [is_done(sheet, r) for r in rows]
    return {"done": sum(done), "total": len(rows),
            "next_open": next((i for i, d in enumerate(done) if not d), None)}


def validate(sheet: str, row: dict, answers: dict) -> dict:
    """The cleaned values to write, or LabelError."""
    schema = SCHEMAS[sheet]
    out: dict[str, str] = {}
    for col in asked(sheet, row):
        value = (answers.get(col) or "").strip()
        if value not in (schema[col] or ()):
            raise LabelError(f"{col}: choose one of {list(schema[col])}")
        out[col] = value
    for col in FREE_TEXT[sheet]:
        out[col] = " ".join((answers.get(col) or "").split())
    if any(v in NEEDS_NOTE for v in out.values()) and not out["notes"]:
        raise LabelError("the guide asks for a note whenever you answer unsure")
    return out


def save(root: Path, sheet: str, me: str, item_id: str, answers: dict,
         initials: str) -> dict:
    """Write one row's labels; every other cell in the file is left as it was."""
    if not initials.strip():
        raise LabelError("initials are required for the labeller column")
    path = sheet_path(root, sheet, me)
    columns, rows = _read(path)
    row = next((r for r in rows if r.get("item_id") == item_id), None)
    if row is None:
        raise LabelError(f"{item_id} is not in {path.name}")
    row.update(validate(sheet, row, answers))
    row["labeller"] = initials.strip().upper()
    tmp = path.with_name(path.name + ".tmp")
    _write(tmp, columns, rows)
    os.replace(tmp, path)
    return row
