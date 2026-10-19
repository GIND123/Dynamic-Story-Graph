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


def _merged(tmp_path):
    return {r["item_id"]: r for r in csv.DictReader((tmp_path / "labels_merged.csv").open())}


def _set_adjudication(tmp_path, values, note="agreed on re-reading"):
    path = tmp_path / "labels_merged.csv"
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols, rows = reader.fieldnames, list(reader)
    for r in rows:
        if r["item_id"] in values:
            r["real_violation_adjudicated"] = values[r["item_id"]]
            r["adjudication_notes"] = note
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_merge_opens_an_adjudication_slot_only_where_annotators_disagree(tmp_path):
    labels.split(_sheet(tmp_path), 2, seed=1)
    _fill(tmp_path, "annotator_01", "yes")
    _fill(tmp_path, "annotator_02", "no")
    labels.merge(tmp_path)
    rows = _merged(tmp_path)
    doubled = {i for i, r in rows.items() if r["annotator_b"]}
    assert all(rows[i]["real_violation_adjudicated"] == "" for i in doubled)
    single = [r for i, r in rows.items() if i not in doubled]
    assert all(r["real_violation_adjudicated"] == "n/a" for r in single)
    # real_restatement is n/a for both annotators, so it never needs adjudicating.
    assert all(r["real_restatement_adjudicated"] == "n/a" for r in rows.values())


def test_finalize_refuses_until_every_disagreement_is_adjudicated(tmp_path):
    labels.split(_sheet(tmp_path), 2, seed=1)
    _fill(tmp_path, "annotator_01", "yes")
    _fill(tmp_path, "annotator_02", "no")
    labels.merge(tmp_path)
    problems, path = labels.finalize(tmp_path)
    assert path is None and len(problems) == 4
    doubled = [i for i, r in _merged(tmp_path).items() if r["annotator_b"]]
    _set_adjudication(tmp_path, {i: "No" for i in doubled})
    problems, path = labels.finalize(tmp_path)
    assert problems == []
    final = {r["item_id"]: r for r in csv.DictReader(path.open())}
    assert all(final[i]["real_violation"] == "no" for i in doubled)
    assert all(final[i]["real_violation_source"] == "adjudicated" for i in doubled)
    single = [i for i in final if i not in doubled]
    assert all(final[i]["real_violation_source"] == "single" for i in single)


def test_finalize_requires_a_note_and_an_allowed_value(tmp_path):
    labels.split(_sheet(tmp_path), 2, seed=1)
    _fill(tmp_path, "annotator_01", "yes")
    _fill(tmp_path, "annotator_02", "no")
    labels.merge(tmp_path)
    doubled = [i for i, r in _merged(tmp_path).items() if r["annotator_b"]]
    _set_adjudication(tmp_path, {doubled[0]: "maybe", **{i: "yes" for i in doubled[1:]}}, note="")
    problems, _ = labels.finalize(tmp_path)
    assert any("'maybe' not in" in p for p in problems)
    assert any("needs a note" in p for p in problems)


def test_remerge_keeps_adjudications_whose_originals_did_not_change(tmp_path):
    labels.split(_sheet(tmp_path), 2, seed=1)
    _fill(tmp_path, "annotator_01", "yes")
    _fill(tmp_path, "annotator_02", "no")
    labels.merge(tmp_path)
    doubled = [i for i, r in _merged(tmp_path).items() if r["annotator_b"]]
    _set_adjudication(tmp_path, {i: "yes" for i in doubled})
    labels.merge(tmp_path)
    rows = _merged(tmp_path)
    assert all(rows[i]["real_violation_adjudicated"] == "yes" for i in doubled)
    assert all(rows[i]["adjudication_notes"] for i in doubled)
