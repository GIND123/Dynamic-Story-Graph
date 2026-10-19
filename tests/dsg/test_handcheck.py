"""The blind hand-check of flagged violations."""

from __future__ import annotations

import csv
import json
import random

import pytest

from dsg.generate.canon import VOCAB, CanonFact, build_premises
from dsg.generate.handcheck import build_sheet, unblind_sheet


def test_locate_agrees_with_check_on_every_text():
    """locate must flag exactly what the published check flags, and point at it."""
    rng = random.Random(0)
    fact = CanonFact(fact_id="f", subject="Hesper", kind="eyes", value="grey",
                     statement="Hesper has grey eyes.")
    filler = ["the quay", "said", "Jarrow", "rain", "the cottage", "not", "but"]
    for _ in range(400):
        words = [rng.choice(filler + list(VOCAB["eyes"]) + ["Hesper"]) for _ in range(60)]
        text = " ".join(words)
        flagged, _ = fact.check(text)
        where = fact.locate(text)
        assert (where is not None) == flagged, text
        if where:
            s_start, s_end, start, end = where
            assert text[s_start:s_end].lower() == "hesper"
            assert text[start:end].lower() in fact.contradictions()


def _payload(n_stories=3, chapters=3):
    premises = build_premises(n=n_stories)
    records = []
    for premise in premises:
        hesper, other = premise.canon[0].subject, premise.canon[5].subject
        eyes = premise.canon[0]
        wrong = next(v for v in VOCAB["eyes"] if v != eyes.value)
        for n, condition in enumerate(("base:last-chapter", "base:last-chapter~prose",
                                        "base:dsg-hybrid", "base:dsg-hybrid~prose")):
            for chapter in range(1, chapters + 1):
                # Chapter 1 is word-for-word the same across memories of a template.
                text = (f"{hesper} turned. {other} looked up, {wrong} eyes bright. "
                        + (f"Part {chapter}, take {n}." if chapter > 1 else ""))
                records.append({
                    "story_id": premise.story_id, "condition": condition,
                    "chapter": chapter, "text": text, "violations": [eyes.fact_id],
                })
    return {"premises": [p.to_json() for p in premises], "records": records}


def _rows(out):
    return list(csv.DictReader((out / "handcheck.csv").open()))


def test_the_sheet_hides_the_arm_and_marks_word_and_subject(tmp_path):
    build_sheet(_payload(), tmp_path)
    shown = (tmp_path / "handcheck.csv").read_text() + (tmp_path / "HANDCHECK.md").read_text()
    for leak in ("prose", "classic", "base:", "last-chapter", "dsg-hybrid", "~"):
        assert leak not in shown, leak
    for row in _rows(tmp_path):
        assert "[[" in row["context"] and "{{" in row["context"]
        assert row["flagged_fact"].endswith("eyes.")
        assert row["verdict"] == ""


def test_identical_spans_are_rated_once_for_every_arm(tmp_path):
    build_sheet(_payload(), tmp_path)
    key = json.loads((tmp_path / "HANDCHECK_KEY.json").read_text())
    rows = _rows(tmp_path)
    assert len({r["context"] for r in rows}) == len(rows)
    merged = [spans for spans in key["items"].values() if len(spans) > 1]
    assert merged, "chapter 1 is shared by both memories of a template"
    assert all({s.split("|")[3] for s in spans} == {"1"} for spans in merged)


def test_sampling_is_capped_per_arm_and_seeded(tmp_path):
    payload = _payload(n_stories=10, chapters=5)       # 50 flagged spans per arm
    build_sheet(payload, tmp_path / "a", per_arm=12, seed=3)
    build_sheet(payload, tmp_path / "b", per_arm=12, seed=3)
    assert _rows(tmp_path / "a") == _rows(tmp_path / "b")
    key = json.loads((tmp_path / "a" / "HANDCHECK_KEY.json").read_text())
    per_arm: dict[str, int] = {}
    for spans in key["items"].values():
        for span in spans:
            arm = span.split("|")[2]
            per_arm[arm] = per_arm.get(arm, 0) + 1
    assert set(per_arm) == {"base:last-chapter", "base:last-chapter~prose",
                            "base:dsg-hybrid", "base:dsg-hybrid~prose"}
    assert all(n == 12 for n in per_arm.values())


def _rate(tmp_path, verdict_for):
    rows = _rows(tmp_path)
    for row in rows:
        row["verdict"] = verdict_for(row)
    with (tmp_path / "handcheck.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return unblind_sheet(tmp_path / "handcheck.csv", tmp_path / "HANDCHECK_KEY.json")


def test_all_false_positives_clear_every_fact(tmp_path):
    build_sheet(_payload(), tmp_path)          # 9 spans per arm, all sampled
    result = _rate(tmp_path, lambda row: "fp")
    for arm in result.values():
        assert arm["false_positive_rate"] == 1.0
        assert arm["raw_violation_rate"] == pytest.approx(3 / 24)
        assert arm["corrected_violation_rate"] == 0.0
        assert arm["facts_cleared"] == 3 and arm["corrected_bounds"] == (0.0, 0.0)


def test_one_real_span_confirms_its_fact(tmp_path):
    build_sheet(_payload(), tmp_path)
    key = json.loads((tmp_path / "HANDCHECK_KEY.json").read_text())

    def verdict(row):
        spans = key["items"][row["item_id"]]
        return "real" if any(s[-2:] == "|2" for s in spans) else "false positive"

    result = _rate(tmp_path, verdict)
    for arm in result.values():
        assert arm["facts_confirmed"] == 3, "chapter 2 is real in every story"
        assert arm["corrected_violation_rate"] == pytest.approx(3 / 24)
        assert arm["false_positive_rate"] == pytest.approx(6 / 9)


def test_unrated_facts_are_discounted_by_the_false_positive_rate(tmp_path):
    build_sheet(_payload(), tmp_path)
    key = json.loads((tmp_path / "HANDCHECK_KEY.json").read_text())
    story0 = build_premises(1)[0].story_id

    def verdict(row):
        spans = key["items"][row["item_id"]]
        if all(s.startswith(story0) for s in spans):
            return ""                                  # leave story 0 unrated
        return "real" if any(s.endswith("|2") for s in spans) else "fp"

    result = _rate(tmp_path, verdict)
    arm = result["base:last-chapter"]
    assert arm["facts_unresolved"] == 1 and arm["facts_confirmed"] == 2
    fp = arm["false_positive_rate"]
    assert fp == pytest.approx(4 / 6)
    assert arm["corrected_violation_rate"] == pytest.approx((2 + (1 - fp)) / 24)
    assert arm["corrected_bounds"] == (pytest.approx(2 / 24), pytest.approx(3 / 24))


@pytest.mark.parametrize("bad", [{"verdict": "maybe"}, {"item_id": "deadbeef", "verdict": "fp"}])
def test_malformed_sheets_are_refused(tmp_path, bad):
    build_sheet(_payload(), tmp_path)
    rows = _rows(tmp_path)
    rows[0].update(bad)
    with (tmp_path / "handcheck.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        unblind_sheet(tmp_path / "handcheck.csv", tmp_path / "HANDCHECK_KEY.json")


def test_the_cli_round_trip(tmp_path, capsys):
    from dsg.cli import main

    source = tmp_path / "generation.json"
    source.write_text(json.dumps(_payload()))
    out = tmp_path / "hc"
    assert main(["gen-handcheck", "--generation", str(source), "--out", str(out)]) == 0
    _rate(out, lambda row: "unsure")
    assert main(["gen-handcheck-unblind", "--sheet", str(out / "handcheck.csv"),
                 "--key", str(out / "HANDCHECK_KEY.json"),
                 "--out", str(tmp_path / "hc.json")]) == 0
    assert "| `base:last-chapter~prose` |" in capsys.readouterr().out
    saved = json.loads((tmp_path / "hc.json").read_text())
    assert saved["base:dsg-hybrid"]["false_positive_rate"] is None
