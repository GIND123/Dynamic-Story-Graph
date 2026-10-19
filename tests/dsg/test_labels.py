"""Splitting, checking and merging annotator files."""

from __future__ import annotations

import csv
import json

import pytest

from dsg.eval import labels


def _sheet(tmp_path, n_rows=12, doubles=(0, 3, 6, 9)):
    cols = ["item_id", "double_annotate", "item_type", "context",
            "real_violation", "real_restatement", "labeller", "notes"]
    rows = [{"item_id": f"D{i:04d}", "double_annotate": "yes" if i in doubles else "no",
             "item_type": "flagged_violation", "context": f"text {i}",
             "real_violation": "", "real_restatement": "n/a", "labeller": "", "notes": ""}
            for i in range(n_rows)]
    with (tmp_path / "sheet_blind.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def _fill(tmp_path, who, value="yes", edit=None):
    path = tmp_path / "annotators" / f"{who}.csv"
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols, rows = reader.fieldnames, list(reader)
    for r in rows:
        r["real_violation"], r["labeller"] = value, who
        if edit:
            edit(r)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_split_gives_overlap_rows_to_two_different_people(tmp_path):
    plan = labels.split(_sheet(tmp_path), 3, seed=1)
    assert sum(plan["rows_per_annotator"].values()) == 12 + 4
    for item_id, who in plan["assignment"].items():
        assert len(set(who)) == len(who)
        assert len(who) == (2 if item_id in {"D0000", "D0003", "D0006", "D0009"} else 1)
    assert len(plan["double_pairs"]) > 1, "overlap spread over more than one pair"
    header = (tmp_path / "annotators" / "annotator_01.csv").read_text().splitlines()[0]
    assert "double_annotate" not in header
    assert plan == labels.split(_sheet(tmp_path), 3, seed=1) | {"command": plan["command"]}


def test_check_rejects_bad_values_edits_and_unnoted_unsure(tmp_path):
    labels.split(_sheet(tmp_path), 2, seed=1)
    _fill(tmp_path, "annotator_01", "maybe")
    _fill(tmp_path, "annotator_02", "unsure",
          edit=lambda r: r.update(context=r["context"] + "!") if r["item_id"] == "D0001" else None)
    problems, _ = labels.check(tmp_path)
    text = "\n".join(problems)
    assert "'maybe' not in" in text
    assert "needs a note" in text
    assert "row text was edited" in text


def test_not_asked_columns_must_stay_not_asked(tmp_path):
    labels.split(_sheet(tmp_path), 2, seed=1)
    _fill(tmp_path, "annotator_01", edit=lambda r: r.update(real_restatement="yes"))
    _fill(tmp_path, "annotator_02")
    problems, _ = labels.check(tmp_path)
    assert any("must stay 'n/a'" in p for p in problems)


def test_merge_writes_both_labels_for_overlap_rows(tmp_path):
    labels.split(_sheet(tmp_path), 2, seed=1)
    _fill(tmp_path, "annotator_01", "Yes")
    _fill(tmp_path, "annotator_02", "no")
    path = labels.merge(tmp_path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 12
    doubled = [r for r in rows if r["annotator_b"]]
    assert len(doubled) == 4
    assert all({r["real_violation"], r["real_violation_b"]} == {"yes", "no"} for r in doubled)
    plan = json.loads((tmp_path / "annotators" / "assignment.json").read_text())
    assert all(r["annotator_a"] == plan["assignment"][r["item_id"]][0] for r in rows)


def test_merge_refuses_incomplete_files(tmp_path):
    labels.split(_sheet(tmp_path), 2, seed=1)
    _fill(tmp_path, "annotator_01")
    with pytest.raises(SystemExit):
        labels.merge(tmp_path)
