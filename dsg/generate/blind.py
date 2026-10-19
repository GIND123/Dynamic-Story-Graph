"""A blinded, shuffled reading packet: classic against prose, judged by people.

The readability proxies explain a result; the reviewers' judgement is the
result. So the packet must give a reader nothing but the prose: both arms are
normalised identically, sides are assigned at random, pairs are shuffled, and
the only record of which side is which lives in ``KEY.json``, kept away from
reviewers. No model judges anything here.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

from dsg.eval.bootstrap import _sign_test
from dsg.generate.score import classic_twin

INSTRUCTION = """# Reading packet

Each pair below shows two versions of the same passage, **Text A** and **Text B**.
Judge **readability and how natural the prose is**. **Ignore differences in length.**

Record your answers in your own ratings file (`ratings_R1.csv` or
`ratings_R2.csv`), one row per pair: which you prefer (A, B or tie), a 1-5
readability score for each, and for whole stories which builds better across
its chapters. Do not look at the other reviewer's file or discuss any pair
until both files are complete.
"""

REVIEWERS = ("R1", "R2")
RATING_COLUMNS = (
    "pair_id", "reviewer", "prefer", "readability_A", "readability_B",
    "buildup_prefer", "note",
)
STORY_BREAK = "\n\n* * *\n\n"

_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)
# A leading chapter heading in any of the forms a model emits: "CHAPTER 3",
# "Chapter Three: The Debt", "**Chapter IV**", "# Chapter 2 --". Only a whole
# heading line matches, never prose that happens to open with the word.
_HEADING = re.compile(
    rf"^[ \t]*(?:#+[ \t]*)?[*_]*[ \t]*chapter[ \t]+(?:\d+|[ivxlc]+|(?:{_NUMBER_WORDS})"
    rf"(?:-(?:{_NUMBER_WORDS}))?)\b[*_]*(?:[ \t]*[:.\-–—][^\n]{{0,60}})?[*_]*[ \t]*(?:\n|$)",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    """Strip leading chapter headings and surrounding whitespace, for every arm."""
    text = (text or "").strip()
    while True:
        stripped = _HEADING.sub("", text, count=1).strip()
        if stripped == text:
            return text
        text = stripped


def prose_twin(classic: str) -> str:
    """``base:last-chapter@bo4`` -> ``base:last-chapter~prose@bo4``."""
    base, at, selection = classic.partition("@")
    return f"{base}~prose{at}{selection}"


def _arms(payload: dict, classic: str) -> tuple[dict, dict]:
    prose = prose_twin(classic)
    if classic_twin(prose) != classic:
        raise ValueError(f"{classic!r} is not a classic condition")
    texts: dict[str, dict[tuple[str, int], str]] = {classic: {}, prose: {}}
    for rec in payload["records"]:
        if rec["condition"] in texts:
            texts[rec["condition"]][(rec["story_id"], rec["chapter"])] = rec.get("text", "")
    if not texts[classic] or not texts[prose]:
        raise ValueError(f"the run has no records for both {classic!r} and {prose!r}")
    return texts[classic], texts[prose]


def _units(payload: dict, classic: str, unit: str) -> list[dict]:
    """Every (story, chapter) or story both arms wrote, with what a reader is shown."""
    old, new = _arms(payload, classic)
    premises = {p["story_id"]: p for p in payload["premises"]}
    keys = sorted(set(old) & set(new))
    if unit == "chapter":
        return [
            {"story_id": s, "chapter": c, "title": premises[s]["title"],
             "beat": premises[s]["beats"][(c - 1) % len(premises[s]["beats"])],
             "classic": normalise(old[(s, c)]), "prose": normalise(new[(s, c)])}
            for s, c in keys
        ]
    if unit == "story":
        stories = sorted({s for s, _ in keys})
        out = []
        for s in stories:
            chapters = sorted(c for t, c in keys if t == s)
            out.append({
                "story_id": s, "chapter": None, "title": premises[s]["title"], "beat": "",
                "classic": STORY_BREAK.join(normalise(old[(s, c)]) for c in chapters),
                "prose": STORY_BREAK.join(normalise(new[(s, c)]) for c in chapters),
            })
        return out
    raise ValueError(f"unit must be 'chapter' or 'story', not {unit!r}")


def build_packet(
    payload: dict, out_dir: Path, classic: str = "base:last-chapter",
    unit: str = "chapter", max_pairs: int = 20, seed: int = 0,
) -> Path:
    """Write packet.md, pairs.jsonl, ratings.csv and KEY.json; return the packet."""
    rng = random.Random(seed)
    units = _units(payload, classic, unit)
    if len(units) > max_pairs:
        units = rng.sample(units, max_pairs)
    rng.shuffle(units)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs, key, ids = [], {}, set()
    for item in units:
        pair_id = f"{rng.getrandbits(32):08x}"
        while pair_id in ids:
            pair_id = f"{rng.getrandbits(32):08x}"
        ids.add(pair_id)
        prose_is_a = rng.random() < 0.5
        a, b = ("prose", "classic") if prose_is_a else ("classic", "prose")
        pairs.append({"pair_id": pair_id, "unit": unit, "title": item["title"],
                      "beat": item["beat"], "A": item[a], "B": item[b]})
        key[pair_id] = {
            "A": prose_twin(classic) if prose_is_a else classic,
            "B": classic if prose_is_a else prose_twin(classic),
            "prose_side": "A" if prose_is_a else "B",
            "story_id": item["story_id"], "chapter": item["chapter"],
        }

    lines = [INSTRUCTION]
    for n, pair in enumerate(pairs, 1):
        lines += ["---", "", f"## Pair {n} (id {pair['pair_id']})", "",
                  f"*{pair['title']}*" + (f" -- what happens: {pair['beat']}"
                                          if pair["beat"] else ""), ""]
        for side in ("A", "B"):
            lines += [f"### Text {side}", "", pair[side], ""]
    packet = out_dir / "packet.md"
    packet.write_text("\n".join(lines))
    (out_dir / "pairs.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in pairs)
    )
    # One file per reviewer, so neither can see the other's answers.
    for reviewer in REVIEWERS:
        with (out_dir / f"ratings_{reviewer}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(RATING_COLUMNS)
            for pair in pairs:
                writer.writerow([pair["pair_id"], reviewer] + [""] * 5)
    (out_dir / "KEY.json").write_text(json.dumps({
        "seed": seed, "unit": unit, "classic": classic,
        "prose": prose_twin(classic), "pairs": key,
    }, indent=2))
    return packet


def _score(value: str, where: str, column: str) -> float | None:
    if not value.strip():
        return None
    try:
        score = float(value)
    except ValueError:
        raise ValueError(f"{where}: {column} {value!r} is not a number") from None
    if not 1 <= score <= 5:
        raise ValueError(f"{where}: {column} {score} is outside 1-5")
    return score


def _arm(choice: str, prose_side: str, where: str, column: str) -> str | None:
    """A/B/tie as written by the reviewer -> "prose", "classic", "tie", or None if blank."""
    choice = (choice or "").strip().upper()
    if not choice:
        return None
    if choice == "TIE":
        return "tie"
    if choice not in ("A", "B"):
        raise ValueError(f"{where}: {column} must be A, B or tie")
    return "prose" if choice == prose_side else "classic"


def read_ratings(path: Path, key: dict) -> tuple[str, dict[str, dict]]:
    """One reviewer's file -> (reviewer, {pair_id: unblinded answers})."""
    reviewer, answers = "", {}
    with Path(path).open(newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            where = f"{Path(path).name} row {row_number}"
            name = (row.get("reviewer") or "").strip()
            if not name or (reviewer and name != reviewer):
                raise ValueError(f"{where}: each file must hold exactly one reviewer")
            reviewer = name
            pair_id = (row.get("pair_id") or "").strip()
            if pair_id not in key:
                raise ValueError(f"{where}: unknown pair id {pair_id!r}")
            if pair_id in answers:
                raise ValueError(f"{where}: pair {pair_id} is rated twice")
            side = key[pair_id]["prose_side"]
            prefer = _arm(row.get("prefer", ""), side, where, "prefer")
            if prefer is None:
                continue    # left blank: not rated, which is not a tie
            readability = {}
            for column in ("readability_A", "readability_B"):
                score = _score(row.get(column, ""), where, column)
                if score is not None:
                    readability["prose" if column[-1] == side else "classic"] = score
            answers[pair_id] = {
                "prefer": prefer,
                "buildup": _arm(row.get("buildup_prefer", ""), side, where, "buildup_prefer"),
                "readability": readability,
            }
    if not reviewer:
        raise ValueError(f"{Path(path).name}: no rows")
    return reviewer, answers


def _tally(answers: dict[str, dict]) -> dict:
    votes = {"prose": 0, "classic": 0, "tie": 0}
    buildup = {"prose": 0, "classic": 0, "tie": 0}
    scores: dict[str, list[float]] = {"prose": [], "classic": []}
    for answer in answers.values():
        votes[answer["prefer"]] += 1
        if answer["buildup"]:
            buildup[answer["buildup"]] += 1
        for arm, score in answer["readability"].items():
            scores[arm].append(score)
    decided = votes["prose"] + votes["classic"]
    return {
        "rated": len(answers),
        "votes": votes,
        "prose_win_rate": votes["prose"] / decided if decided else None,
        "p_sign": _sign_test(votes["prose"], votes["classic"]),
        "readability_mean": {k: (sum(v) / len(v) if v else None) for k, v in scores.items()},
        "buildup": buildup,
    }


def unblind(ratings: list[Path], key_json: Path) -> dict:
    """Each reviewer on their own, then pairs, then how often the reviewers agree.

    Two reviewers reading the same pair are not two independent observations,
    so their rows are never pooled. A pair counts for an arm only when both
    reviewers prefer it; anything else, a tie included, is a split.
    """
    key = json.loads(Path(key_json).read_text())["pairs"]
    by_reviewer: dict[str, dict[str, dict]] = {}
    for path in ratings:
        reviewer, answers = read_ratings(path, key)
        if reviewer in by_reviewer:
            raise ValueError(f"reviewer {reviewer} appears in more than one file")
        by_reviewer[reviewer] = answers

    both = sorted(set.intersection(*(set(a) for a in by_reviewer.values())))
    pairs = {"prose": 0, "classic": 0, "split": 0}
    agree = 0
    for pair_id in both:
        choices = {a[pair_id]["prefer"] for a in by_reviewer.values()}
        agree += len(choices) == 1
        only = next(iter(choices)) if len(choices) == 1 else None
        pairs[only if only in ("prose", "classic") else "split"] += 1
    return {
        "reviewers": {r: _tally(a) for r, a in sorted(by_reviewer.items())},
        "pairs": {
            "rated_by_all": len(both),
            **pairs,
            "p_sign": _sign_test(pairs["prose"], pairs["classic"]),
        },
        "agreement": agree / len(both) if both else None,
    }


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summary_markdown(result: dict) -> str:
    pairs = result["pairs"]
    lines = [
        "# Blind read: prose vs classic",
        "",
        "## Per reviewer",
        "",
        "| reviewer | rated | prose | classic | tie | prose win rate | sign p | "
        "readability prose | readability classic | build-up prose–classic–tie |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for reviewer, t in result["reviewers"].items():
        v, r, b = t["votes"], t["readability_mean"], t["buildup"]
        lines.append(
            f"| {reviewer} | {t['rated']} | {v['prose']} | {v['classic']} | {v['tie']} | "
            f"{_rate(t['prose_win_rate'])} | {t['p_sign']:.3f} | {_rate(r['prose'])} | "
            f"{_rate(r['classic'])} | {b['prose']}–{b['classic']}–{b['tie']} |"
        )
    lines += [
        "",
        "## Per pair",
        "",
        f"Of {pairs['rated_by_all']} pairs rated by every reviewer: prose {pairs['prose']}, "
        f"classic {pairs['classic']}, split {pairs['split']} (a pair counts for an arm "
        f"only when all reviewers prefer it; a tie counts as split). Sign test on "
        f"prose vs classic pairs: p = {pairs['p_sign']:.3f}.",
        "",
        f"Reviewer agreement (same choice on the same pair): {_rate(result['agreement'])}.",
    ]
    return "\n".join(lines) + "\n"
