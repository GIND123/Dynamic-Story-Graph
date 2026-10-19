"""The blinded reading packet: nothing but the prose reaches a reviewer."""

from __future__ import annotations

import csv
import json

import pytest

from dsg.generate.blind import (
    INSTRUCTION,
    REVIEWERS,
    STORY_BREAK,
    build_packet,
    normalise,
    unblind,
)
from dsg.generate.canon import build_premises

BODY = "Hesper set the iron seal on the table and waited for Jarrow to speak."


@pytest.mark.parametrize("heading", [
    "CHAPTER 3", "Chapter Three: The Debt", "**Chapter IV**", "# Chapter 2 -- Arrival",
    "Chapter twenty-one", "chapter 7.",
])
def test_leading_chapter_headings_are_stripped(heading):
    assert normalise(f"  {heading}\n\n{BODY}  \n") == BODY


def test_prose_that_opens_with_the_word_is_kept():
    text = "Chapter three of her life began in the rain."
    assert normalise(text) == text


def test_stacked_headings_and_whitespace_go_too():
    assert normalise(f"\n\nChapter 3\n\nChapter 3: again\n{BODY}\n\n") == BODY


def _payload(classic_text, prose_text, n_stories=3, chapters=3):
    premises = build_premises(n=n_stories)
    records = []
    for premise in premises:
        for chapter in range(1, chapters + 1):
            for condition, text in (("base:last-chapter", classic_text),
                                    ("base:last-chapter~prose", prose_text)):
                records.append({
                    "story_id": premise.story_id, "condition": condition,
                    "chapter": chapter,
                    "text": text.format(n=chapter, story=premise.story_id),
                })
    return {"meta": {}, "premises": [p.to_json() for p in premises], "records": records}


def _pairs(out):
    return [json.loads(line) for line in (out / "pairs.jsonl").read_text().splitlines()]


def test_one_arm_heading_style_cannot_reveal_the_template(tmp_path):
    """Classic writes "CHAPTER 2", prose writes "Chapter Two: ..."; readers see neither."""
    body = BODY + " ({story}, part {n})"
    payload = _payload("CHAPTER {n}\n" + body, "Chapter {n}: The Request\n\n" + body)
    packet = build_packet(payload, tmp_path, max_pairs=50).read_text()
    for pair in _pairs(tmp_path):
        assert pair["A"] == pair["B"], "identical prose must reach the reader identically"
        assert pair["A"].startswith("Hesper set the iron seal")
    assert "CHAPTER" not in packet and "The Request" not in packet


def test_the_packet_opens_with_the_reading_instruction(tmp_path):
    packet = build_packet(_payload(BODY, BODY), tmp_path).read_text()
    assert packet.startswith(INSTRUCTION)
    assert "readability and how natural the prose is" in INSTRUCTION
    assert "Ignore differences in length" in INSTRUCTION
    assert "list of facts" not in INSTRUCTION, "the instruction must not lead"


def test_nothing_in_the_packet_names_an_arm(tmp_path):
    packet = build_packet(_payload("Old {story} {n}.", "New {story} {n}."), tmp_path)
    readable = packet.read_text().replace(INSTRUCTION, "")
    readable += (tmp_path / "pairs.jsonl").read_text()
    for reviewer in REVIEWERS:
        readable += (tmp_path / f"ratings_{reviewer}.csv").read_text()
    for leak in ("prose", "classic", "base:", "last-chapter", "~", "template"):
        assert leak not in readable, leak


def test_sides_are_random_and_the_key_maps_them_back(tmp_path):
    build_packet(_payload("Old {story} {n}.", "New {story} {n}."), tmp_path, max_pairs=50)
    key = json.loads((tmp_path / "KEY.json").read_text())["pairs"]
    for pair in _pairs(tmp_path):
        entry = key[pair["pair_id"]]
        prose_side = entry["prose_side"]
        assert pair[prose_side].startswith("New")
        assert entry[prose_side] == "base:last-chapter~prose"
    assert {e["prose_side"] for e in key.values()} == {"A", "B"}


def test_the_packet_is_deterministic_per_seed_and_capped(tmp_path):
    payload = _payload("Old {story} {n}.", "New {story} {n}.", n_stories=10, chapters=5)
    build_packet(payload, tmp_path / "a", max_pairs=20, seed=0)
    build_packet(payload, tmp_path / "b", max_pairs=20, seed=0)
    build_packet(payload, tmp_path / "c", max_pairs=20, seed=1)
    assert _pairs(tmp_path / "a") == _pairs(tmp_path / "b")
    assert _pairs(tmp_path / "a") != _pairs(tmp_path / "c")
    assert len(_pairs(tmp_path / "a")) == 20


def test_each_reviewer_gets_their_own_ratings_file(tmp_path):
    build_packet(_payload(BODY, BODY), tmp_path)
    ids = [p["pair_id"] for p in _pairs(tmp_path)]
    assert not (tmp_path / "ratings.csv").exists()
    for reviewer in REVIEWERS:
        rows = list(csv.DictReader((tmp_path / f"ratings_{reviewer}.csv").open()))
        assert [r["pair_id"] for r in rows] == ids
        assert {r["reviewer"] for r in rows} == {reviewer}


def test_story_unit_joins_normalised_chapters(tmp_path):
    payload = _payload("CHAPTER {n}\nOld {story} {n}.", "Chapter {n}\n\nNew {story} {n}.")
    build_packet(payload, tmp_path, unit="story")
    pairs = _pairs(tmp_path)
    assert len(pairs) == 3
    for pair in pairs:
        assert pair["A"].count(STORY_BREAK) == 2 and "hapter" not in pair["A"] + pair["B"]
        assert pair["beat"] == ""


def test_an_unknown_arm_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        build_packet(_payload(BODY, BODY), tmp_path, classic="base:none")


def _fill(path, choose):
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        row.update(choose(row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _packet(tmp_path, n_stories=3):
    build_packet(_payload("Old {story} {n}.", "New {story} {n}.", n_stories=n_stories),
                 tmp_path, max_pairs=50)
    key = json.loads((tmp_path / "KEY.json").read_text())["pairs"]
    files = [tmp_path / f"ratings_{r}.csv" for r in REVIEWERS]
    return key, files


def _side(key, row, arm):
    prose = key[row["pair_id"]]["prose_side"]
    return prose if arm == "prose" else ("B" if prose == "A" else "A")


def test_reviewers_are_reported_separately_never_pooled(tmp_path):
    key, (r1, r2) = _packet(tmp_path)
    ids = sorted(key)

    def r1_choice(row):
        side = _side(key, row, "prose")
        other = "B" if side == "A" else "A"
        return {"prefer": side.lower(), f"readability_{side}": "4",
                f"readability_{other}": "2", "buildup_prefer": side}

    _fill(r1, r1_choice)
    # R2 prefers prose on the first 4 pairs, classic on the next 3, ties the
    # rest, and leaves one blank: blank is "not rated", never a tie.
    def r2_choice(row):
        i = ids.index(row["pair_id"])
        if i == len(ids) - 1:
            return {"prefer": ""}
        arm = "prose" if i < 4 else "classic" if i < 7 else None
        return {"prefer": _side(key, row, arm) if arm else "TIE"}

    _fill(r2, r2_choice)
    result = unblind([r1, r2], tmp_path / "KEY.json")
    one, two = result["reviewers"]["R1"], result["reviewers"]["R2"]
    assert one["votes"] == {"prose": 9, "classic": 0, "tie": 0}
    assert one["readability_mean"] == {"prose": 4.0, "classic": 2.0}
    assert one["buildup"]["prose"] == 9 and one["prose_win_rate"] == 1.0
    assert two["rated"] == 8 and two["votes"] == {"prose": 4, "classic": 3, "tie": 1}
    assert "votes" not in result, "no pooled count across reviewers"


def test_a_pair_counts_for_an_arm_only_when_both_reviewers_prefer_it(tmp_path):
    key, (r1, r2) = _packet(tmp_path)
    ids = sorted(key)
    plan = {  # pair index -> (R1, R2)
        0: ("prose", "prose"), 1: ("prose", "prose"), 2: ("classic", "classic"),
        3: ("prose", "classic"), 4: ("prose", "tie"), 5: ("tie", "tie"),
        6: ("prose", None),
    }

    def fill(who):
        def choose(row):
            i = ids.index(row["pair_id"])
            arm = plan.get(i, (None, None))[who]
            if arm is None:
                return {"prefer": ""}
            return {"prefer": "tie" if arm == "tie" else _side(key, row, arm)}
        return choose

    _fill(r1, fill(0))
    _fill(r2, fill(1))
    result = unblind([r1, r2], tmp_path / "KEY.json")
    # Pair 6 has only R1's answer, so it is not rated by both.
    assert result["pairs"]["rated_by_all"] == 6
    assert {k: result["pairs"][k] for k in ("prose", "classic", "split")} == {
        "prose": 2, "classic": 1, "split": 3
    }
    # Same choice on pairs 0, 1, 2 and 5 (a shared tie counts as agreement).
    assert result["agreement"] == pytest.approx(4 / 6)


@pytest.mark.parametrize("bad", [
    {"prefer": "C"}, {"prefer": "A", "readability_A": "9"},
    {"prefer": "A", "readability_B": "good"}, {"pair_id": "deadbeef", "prefer": "A"},
    {"reviewer": "R2", "prefer": "A"},
])
def test_malformed_ratings_are_refused(tmp_path, bad):
    _, (r1, r2) = _packet(tmp_path)
    rows = list(csv.DictReader(r1.open()))
    rows[0].update(bad)   # one bad row is enough to stop the read
    with r1.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        unblind([r1, r2], tmp_path / "KEY.json")


def test_one_reviewer_in_two_files_is_refused(tmp_path):
    _, (r1, _) = _packet(tmp_path)
    with pytest.raises(ValueError, match="more than one file"):
        unblind([r1, r1], tmp_path / "KEY.json")


def test_the_cli_round_trip(tmp_path, capsys):
    from dsg.cli import main

    source = tmp_path / "generation.json"
    source.write_text(json.dumps(_payload("Old {story} {n}.", "New {story} {n}.")))
    out = tmp_path / "p"
    assert main(["gen-blind", "--generation", str(source), "--out", str(out),
                 "--max-pairs", "5"]) == 0
    for reviewer in REVIEWERS:
        _fill(out / f"ratings_{reviewer}.csv", lambda row: {"prefer": "tie"})
    assert main(["gen-unblind", "--ratings", str(out / "ratings_R1.csv"),
                 str(out / "ratings_R2.csv"), "--key", str(out / "KEY.json"),
                 "--out", str(tmp_path / "summary.json")]) == 0
    printed = capsys.readouterr().out
    assert "| R1 | 5 | 0 | 0 | 5 |" in printed and "split 5" in printed
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["agreement"] == 1.0 and saved["pairs"]["split"] == 5
