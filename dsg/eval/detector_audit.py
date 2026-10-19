"""Audit sheets for the planted-canon detector, built from saved generation runs.

The v1 detector (``CanonFact.check``) flags a fact whenever any other value from
its vocabulary appears within the window of a subject mention. Whether a flag is
a real continuity error is a human judgement, so this module only builds the
sheets people label and the counts that need no label:

* a seeded, stratified sample of v1 violation flags,
* a seeded, stratified sample of v1 restatements (``CanonFact.mentioned``),
* a seeded sample of subject mentions v1 did not flag, for recall: a few
  drawn at random, most from sentences that also carry an attribute cue,
* a bound on the generation result with other-subject-value flags removed,
* exposure: subject mentions per chapter, by run and condition,
* overlap: how many v1 flags are another planted subject's true value.

Every number is computed from the stored chapter ``text``. The recomputed v1
decisions are asserted equal to the ones saved at generation time, so the sheet
audits exactly the detector the published numbers used.

    python -m dsg.eval.detector_audit export \\
        --generation artifacts/hub/generation --out audit/detector
    python -m dsg.eval.detector_audit bound \
        --generation artifacts/hub/generation --out audit/detector
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from dsg.eval.bootstrap import paired_bootstrap
from dsg.eval.provenance import record as provenance
from dsg.evidence import sentences
from dsg.generate.canon import VOCAB, CanonFact, Premise, _forms
from dsg.generate.score import score_runs

SEED = 20261018
N_VIOLATIONS = 300
N_RESTATEMENTS = 100
N_UNFLAGGED_RANDOM = 50     # unflagged mentions with no attribute cue
N_UNFLAGGED_CUED = 100      # unflagged mentions in a sentence with an attribute cue
BOOTSTRAP_SEED = 0          # dsg.eval.bootstrap's default, as every reported interval
CONTEXT = 300           # characters shown either side of the judged span

N_DOUBLE = 50               # rows labelled by two people, for agreement

LABEL_COLUMNS = ("real_violation", "real_restatement", "labeller", "notes")
# The blind sheet is an allow-list: anything that reveals the arm, the
# automatic overlap call or why a row was sampled stays out of it.
BLIND_COLUMNS = (
    "item_id", "double_annotate", "item_type", "story", "chapter", "fact_id", "subject",
    "kind", "true_value", "matched_word", "matched_value", "context", *LABEL_COLUMNS,
)

# Targeted stratum that settles the overlap bound: other-subject-value flags in
# the arms of the two headline within-run gaps. (run, condition, rows).
BOUND_TARGETS = (
    ("gen-v3-hybrid", "base:dsg-state", 40),
    ("gen-v3-hybrid", "base:none", 10),
    ("gen-v4-retrieval", "base:beat-retrieval", 40),
    ("gen-v4-retrieval", "base:none", 10),
)


def _words(*alts: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE)


def _vocab_words(kind: str) -> list[str]:
    return [re.escape(f) for v in VOCAB[kind] for f in _forms(v)]


# Values a story might give an attribute that the closed vocabulary lacks, so
# v1 cannot see them. Recall is estimated where these occur.
OUT_OF_VOCAB: dict[str, re.Pattern[str]] = {
    "eyes": _words("brown", "dark", "black", "violet", "gray", "golden", "silver", "pale",
                   "emerald", "sapphire", "steely", "steel-grey", "steel-gray"),
    "hair": _words("brown", "chestnut", "dark", "grey", "gray", "golden", "fair", "white",
                   "sandy", "raven", "copper", "ginger", "flaxen", "honey", "greying", "graying"),
    "metal": _words("copper", "bronze", "steel", "tin", "gilt", "gilded", "wooden", "wood",
                    "glass", "crystal", "leather", "bone", "ivory", "nickel", "lead"),
    "trade": _words("watchmaker", "mapmaker", "herbalist", "smith", "baker", "carpenter",
                    "weaver", "tailor", "cooper", "miller", "merchant", "fisherman",
                    "physician", "doctor", "scribe", "mason", "chandler", "potter", "farmer",
                    "shopkeeper", "innkeeper", "sailor", "clerk", "jeweller", "jeweler",
                    "locksmith", "printer", "healer", "tinker", "surveyor", "chemist"),
    "home": re.compile(r"(?!x)x"),   # place names are open-ended; not attempted
}

# A sentence carries an attribute cue when it names the attribute itself. For
# metal the attribute noun is the subject (the object), so the cue is a
# material word in the same sentence.
ATTRIBUTE_CUE: dict[str, re.Pattern[str]] = {
    "eyes": _words("eyes?", "gaze"),
    "hair": _words("hair", "haired", "locks", "curls"),
    "metal": _words(*_vocab_words("metal"), OUT_OF_VOCAB["metal"].pattern),
    "trade": _words(*_vocab_words("trade"), OUT_OF_VOCAB["trade"].pattern,
                    "worked as", "works as", "trade", "craft", "apprentice[ds]?", "workshop"),
    "home": _words("born", "from", "hailed", "native", "grew up", "raised", "home"),
}


def _subject_re(subject: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(subject)}(?!\w)", re.IGNORECASE)


def _value_of(kind: str, word: str) -> str | None:
    """The vocabulary value a surface form asserts, within one kind."""
    w = word.lower()
    for value in VOCAB[kind]:
        if w in (f.lower() for f in _forms(value)):
            return value
    return None


@dataclass(slots=True)
class Hit:
    mention_start: int
    mention_end: int
    start: int
    end: int
    word: str
    evidence: str = ""


def v1_violation_hit(fact: CanonFact, text: str) -> Hit | None:
    """The exact hit ``CanonFact.check`` (v1) bases its flag on, with offsets."""
    subject_re = _subject_re(fact.subject)
    bad = fact._pattern(fact.contradictions())
    good = fact._pattern(_forms(fact.value))
    for m in subject_re.finditer(text):
        lo = max(0, m.start() - fact.window)
        hi = min(len(text), m.end() + fact.window)
        span = text[lo:hi]
        hit = bad.search(span)
        if hit is None:
            continue
        near = span[max(0, hit.start() - 40) : hit.end() + 40]
        if good.search(near):
            continue
        evidence = " ".join(span[max(0, hit.start() - 90) : hit.end() + 90].split())
        return Hit(m.start(), m.end(), lo + hit.start(), lo + hit.end(), hit.group(0), evidence)
    return None


def v1_restatement_hit(fact: CanonFact, text: str) -> Hit | None:
    """The hit ``CanonFact.mentioned`` (v1) bases its restatement on."""
    good = fact._pattern(_forms(fact.value))
    for m in _subject_re(fact.subject).finditer(text):
        lo, hi = max(0, m.start() - fact.window), min(len(text), m.end() + fact.window)
        hit = good.search(text[lo:hi])
        if hit:
            return Hit(m.start(), m.end(), lo + hit.start(), lo + hit.end(), hit.group(0))
    return None


def candidate_hits(fact: CanonFact, text: str) -> int:
    """Distinct unsuppressed contradicting words inside any subject window."""
    bad = fact._pattern(fact.contradictions())
    good = fact._pattern(_forms(fact.value))
    seen: set[int] = set()
    for m in _subject_re(fact.subject).finditer(text):
        lo, hi = max(0, m.start() - fact.window), min(len(text), m.end() + fact.window)
        for h in bad.finditer(text, lo, hi):
            if not good.search(text, max(lo, h.start() - 40), min(hi, h.end() + 40)):
                seen.add(h.start())
    return len(seen)


def _hyphen_compound(text: str, start: int, end: int) -> str:
    """The hyphenated token a match sits inside, e.g. 'silver' in 'silver-white'."""
    lo, hi = start, end
    while lo > 0 and (text[lo - 1].isalnum() or text[lo - 1] == "-"):
        lo -= 1
    while hi < len(text) and (text[hi].isalnum() or text[hi] == "-"):
        hi += 1
    token = text[lo:hi]
    return token if token.lower() != text[start:end].lower() else ""


def _context(text: str, marks: list[tuple[int, int, str, str]]) -> str:
    """Text around the marked spans, with each span bracketed for the labeller."""
    lo = max(0, min(s for s, _, _, _ in marks) - CONTEXT)
    hi = min(len(text), max(e for _, e, _, _ in marks) + CONTEXT)
    out, cur = [], lo
    for s, e, left, right in sorted(marks):
        if s < cur:
            continue
        out += [text[cur:s], left, text[s:e], right]
        cur = e
    out.append(text[cur:hi])
    body = " ".join("".join(out).split())
    return ("... " if lo > 0 else "") + body + (" ..." if hi < len(text) else "")


def _other_subject(premise: Premise, fact: CanonFact, word: str, text: str, at: int):
    """Another planted subject whose true value of the same kind is ``word``."""
    value = _value_of(fact.kind, word)
    for other in premise.canon:
        if other.kind == fact.kind and other.subject != fact.subject and other.value == value:
            lo, hi = max(0, at - fact.window), min(len(text), at + fact.window)
            nearby = bool(_subject_re(other.subject).search(text, lo, hi))
            return other.subject, nearby
    return "", False


def _allocate(sizes: dict[tuple, int], total: int) -> dict[tuple, int]:
    """Equal allocation across strata, capped by each stratum's size."""
    alloc = dict.fromkeys(sizes, 0)
    remaining = total
    while remaining > 0:
        open_ = [k for k in sorted(sizes) if alloc[k] < sizes[k]]
        if not open_:
            break
        k = min(open_, key=lambda key: (alloc[key], key))
        alloc[k] += 1
        remaining -= 1
    return alloc


def _stratified(rows: list[dict], key, total: int, rng: random.Random):
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[key(r)].append(r)
    alloc = _allocate({k: len(v) for k, v in strata.items()}, total)
    picked: list[dict] = []
    design = []
    for k in sorted(strata):
        chosen = rng.sample(strata[k], alloc[k])
        for r in chosen:
            r["stratum"] = "|".join(k)
            r["stratum_population"] = len(strata[k])
            r["stratum_sampled"] = alloc[k]
        picked += chosen
        design.append({"stratum": list(k), "population": len(strata[k]), "sampled": alloc[k]})
    return picked, design


def recall_stratum(fact: CanonFact, text: str, start: int, end: int,
                   sentence: tuple[int, int]) -> str:
    """Why an unflagged mention is a likely place for a missed violation.

    In priority order: v1 found a contradicting word but suppressed it because
    the true value was within 40 characters; the mention's sentence carries an
    attribute cue and a value outside the vocabulary; it carries the cue and a
    vocabulary value v1 did not reach; it carries the cue alone. Empty when
    there is no cue.
    """
    lo, hi = max(0, start - fact.window), min(len(text), end + fact.window)
    span = text[lo:hi]
    hit = fact._pattern(fact.contradictions()).search(span)
    if hit is not None:
        near = span[max(0, hit.start() - 40) : hit.end() + 40]
        if fact._pattern(_forms(fact.value)).search(near):
            return "suppressed_by_true_value"
    sent = text[sentence[0] : sentence[1]]
    if not ATTRIBUTE_CUE[fact.kind].search(sent):
        return ""
    if OUT_OF_VOCAB[fact.kind].search(sent):
        return "out_of_vocab_value"
    if fact._pattern(fact.contradictions()).search(sent):
        return "vocab_value_in_sentence"
    return "attribute_cue"


def _sentence_of(spans: list[tuple[int, int]], at: int) -> tuple[int, int]:
    return next(((s, e) for s, e in spans if s <= at < e), (at, at))


def load_runs(generation_dir: Path) -> list[tuple[str, dict]]:
    return [(p.stem, json.loads(p.read_text())) for p in sorted(generation_dir.glob("*.json"))]


def collect(runs: list[tuple[str, dict]]):
    """Every v1 flag, restatement and unflagged subject mention, plus counts."""
    flags, restates, unflagged = [], [], []
    exposure: dict[tuple[str, str], list[int]] = defaultdict(list)
    exposure_kind: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for run_id, payload in runs:
        premises = {p["story_id"]: Premise.from_json(p) for p in payload["premises"]}
        for rec in payload["records"]:
            text = rec.get("text") or ""
            premise = premises[rec["story_id"]]
            base = {"run": run_id, "condition": rec["condition"],
                    "story": rec["story_id"], "chapter": rec["chapter"]}
            subjects = {f.subject for f in premise.canon}
            exposure[(run_id, rec["condition"])].append(
                sum(len(_subject_re(s).findall(text)) for s in subjects)
            )
            for kind in sorted({f.kind for f in premise.canon}):
                kind_subjects = {f.subject for f in premise.canon if f.kind == kind}
                exposure_kind[(run_id, rec["condition"], kind)].append(
                    sum(len(_subject_re(s).findall(text)) for s in kind_subjects)
                )
            spans = None
            saved_v = set(rec["violations"])
            saved_r = set(rec["restated"])
            for fact in premise.canon:
                fbase = {**base, "fact_id": fact.fact_id, "subject": fact.subject,
                         "kind": fact.kind, "true_value": fact.value}
                vhit = v1_violation_hit(fact, text)
                rhit = v1_restatement_hit(fact, text)
                # The recomputation must be the detector the numbers came from.
                assert (vhit is not None) == (fact.fact_id in saved_v), (run_id, rec["story_id"])
                assert (rhit is not None) == (fact.fact_id in saved_r), (run_id, rec["story_id"])
                if vhit is not None:
                    assert vhit.evidence == fact.check(text)[1]
                    other, nearby = _other_subject(premise, fact, vhit.word, text, vhit.start)
                    flags.append({
                        **fbase, "item_type": "flagged_violation",
                        "matched_word": vhit.word,
                        "matched_value": _value_of(fact.kind, vhit.word) or "",
                        "context": _context(text, [
                            (vhit.mention_start, vhit.mention_end, "[[", "]]"),
                            (vhit.start, vhit.end, "<<", ">>"),
                        ]),
                        "other_subject_same_value": other,
                        "other_subject_nearby": "yes" if nearby else "no" if other else "",
                        "hyphen_compound": _hyphen_compound(text, vhit.start, vhit.end),
                        "v1_other_candidate_hits": candidate_hits(fact, text) - 1,
                        "v1_restated_in_chapter": "yes" if rhit else "no",
                    })
                if rhit is not None:
                    restates.append({
                        **fbase, "item_type": "flagged_restatement",
                        "matched_word": rhit.word, "matched_value": fact.value,
                        "context": _context(text, [
                            (rhit.mention_start, rhit.mention_end, "[[", "]]"),
                            (rhit.start, rhit.end, "<<", ">>"),
                        ]),
                        "hyphen_compound": _hyphen_compound(text, rhit.start, rhit.end),
                        "v1_flagged_in_chapter": "yes" if vhit else "no",
                    })
                if vhit is None:
                    if spans is None:
                        spans = sentences(text)
                    for m in _subject_re(fact.subject).finditer(text):
                        sent = _sentence_of(spans, m.start())
                        unflagged.append({
                            **fbase, "item_type": "unflagged_mention",
                            "matched_word": "", "matched_value": "",
                            "context": _context(text, [(m.start(), m.end(), "[[", "]]")]),
                            "v1_restated_in_chapter": "yes" if rhit else "no",
                            "recall_stratum": recall_stratum(
                                fact, text, m.start(), m.end(), sent
                            ) or "random",
                        })
    return flags, restates, unflagged, exposure, exposure_kind


def overlap_by_condition(flags: list[dict]) -> dict[str, dict[str, float]]:
    """Per condition: v1 flags whose matched word is another planted subject's
    true value of the same kind. Automatic, so it needs no label -- but it is a
    string coincidence, not a judgement, and those rows are still labelled."""
    out: dict[str, dict[str, float]] = {}
    by: dict[str, list[dict]] = defaultdict(list)
    for f in flags:
        by[f["condition"]].append(f)
    for condition in sorted(by):
        rows = by[condition]
        same = [r for r in rows if r["other_subject_same_value"]]
        near = [r for r in same if r["other_subject_nearby"] == "yes"]
        hyph = [r for r in rows if r["hyphen_compound"]]
        out[condition] = {
            "flags": len(rows),
            "other_subject_value": len(same),
            "other_subject_value_share": len(same) / len(rows),
            "other_subject_value_and_nearby": len(near),
            "inside_hyphenated_word": len(hyph),
        }
    return out


def overlap_by_kind(flags: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for kind in sorted({f["kind"] for f in flags}):
        rows = [f for f in flags if f["kind"] == kind]
        same = sum(1 for r in rows if r["other_subject_same_value"])
        out[kind] = {"flags": len(rows), "other_subject_value": same,
                     "other_subject_value_share": same / len(rows)}
    return out


def _mean(xs: list[int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


SHEET_COLUMNS = (
    "item_id", "item_type", "run", "condition", "story", "chapter", "fact_id",
    "subject", "kind", "true_value", "matched_word", "matched_value", "context",
    *LABEL_COLUMNS,
    "other_subject_same_value", "other_subject_nearby", "hyphen_compound",
    "v1_other_candidate_hits", "v1_restated_in_chapter", "v1_flagged_in_chapter",
    "recall_stratum", "stratum", "stratum_population", "stratum_sampled",
    "bound_stratum", "bound_stratum_population", "bound_stratum_sampled", "double_annotate",
)


def _flag_key(r: dict) -> tuple:
    return (r["item_type"], r["run"], r["condition"], r["story"], r["chapter"], r["fact_id"])


def _bound_sample(flags: list[dict], rows: list[dict], rng: random.Random) -> tuple[list, list]:
    """Draw BOUND_TARGETS; a flag already in the main sample is labelled once
    and carries both designs' weights."""
    by_key = {_flag_key(r): r for r in rows}
    added, design = [], []
    for run, condition, n in BOUND_TARGETS:
        pool = [f for f in flags if f["run"] == run and f["condition"] == condition
                and f["other_subject_same_value"]]
        k = min(n, len(pool))
        overlap = 0
        for f in rng.sample(pool, k):
            row = by_key.get(_flag_key(f))
            overlap += row is not None
            if row is None:
                row = {**f, "real_violation": "", "real_restatement": "n/a"}
                added.append(row)
                by_key[_flag_key(f)] = row
            row.update(bound_stratum=f"{run}|{condition}", bound_stratum_population=len(pool),
                       bound_stratum_sampled=k)
        design.append({"stratum": [run, condition], "population": len(pool), "sampled": k,
                       "already_in_main_sample": overlap})
    return added, design


def _doubles(rows: list[dict], rng: random.Random, n: int = N_DOUBLE) -> None:
    """Mark ``n`` rows for a second annotator, proportional to item type."""
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_type[r["item_type"]].append(i)
    alloc = {t: round(n * len(ix) / len(rows)) for t, ix in sorted(by_type.items())}
    while sum(alloc.values()) > n:
        alloc[max(alloc, key=lambda t: (alloc[t], t))] -= 1
    while sum(alloc.values()) < n:
        alloc[min(alloc, key=lambda t: (alloc[t] / len(by_type[t]), t))] += 1
    chosen = {i for t in sorted(by_type) for i in rng.sample(by_type[t], alloc[t])}
    for i, r in enumerate(rows):
        r["double_annotate"] = "yes" if i in chosen else "no"


def export(generation_dir: Path, out_dir: Path, seed: int = SEED,
           argv: list[str] | None = None) -> dict:
    runs = load_runs(generation_dir)
    flags, restates, unflagged, exposure, exposure_kind = collect(runs)
    rng = random.Random(seed)

    by_cond_kind = lambda r: (r["condition"], r["kind"])  # noqa: E731
    v_sample, v_design = _stratified(flags, by_cond_kind, N_VIOLATIONS, rng)
    r_sample, r_design = _stratified(restates, by_cond_kind, N_RESTATEMENTS, rng)
    # Recall: the two partitions are disjoint, so each row's stratum size is
    # its inverse inclusion weight.
    cued = [r for r in unflagged if r["recall_stratum"] != "random"]
    plain = [r for r in unflagged if r["recall_stratum"] == "random"]
    c_sample, c_design = _stratified(
        cued, lambda r: (r["kind"], r["recall_stratum"]), N_UNFLAGGED_CUED, rng
    )
    p_sample, p_design = _stratified(
        plain, lambda r: ("random",), N_UNFLAGGED_RANDOM, rng
    )
    u_sample, u_design = c_sample + p_sample, c_design + p_design

    for r in v_sample + u_sample:
        r["real_violation"] = ""
    for r in u_sample + r_sample:
        r["real_restatement"] = ""
    rows = v_sample + r_sample + u_sample
    for r in rows:
        r.setdefault("real_violation", "n/a")
        r.setdefault("real_restatement", "n/a")
    # Separate streams, so adding a design never moves the main sample.
    bound_rows, b_design = _bound_sample(flags, rows, random.Random(seed + 1))
    rows += bound_rows
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["item_id"] = f"D{i:04d}"
    _doubles(rows, random.Random(seed + 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "sheet.csv", rows, SHEET_COLUMNS)
    _write(out_dir / "sheet_blind.csv", rows, BLIND_COLUMNS)

    manifest = {
        "provenance": provenance(
            "dsg.eval.detector_audit", argv,
            {"sample": seed, "bound_target": seed + 1, "double": seed + 2},
            sorted(generation_dir.glob("*.json")),
        ),
        "seed": seed,
        "runs": [run_id for run_id, _ in runs],
        "population": {"flagged_violation": len(flags),
                       "flagged_restatement": len(restates),
                       "unflagged_mention": len(unflagged)},
        "sampled": dict(Counter(r["item_type"] for r in rows)),
        "rows": len(rows),
        "bound_target_design": b_design,
        "bound_target_added_rows": len(bound_rows),
        "double_annotate": sum(r["double_annotate"] == "yes" for r in rows),
        "double_by_type": dict(Counter(
            r["item_type"] for r in rows if r["double_annotate"] == "yes")),
        "recall_population": dict(Counter(r["recall_stratum"] for r in unflagged)),
        "recall_population_by_kind": {
            f"{k}|{s}": n for (k, s), n in sorted(
                Counter((r["kind"], r["recall_stratum"]) for r in unflagged).items())
        },
        "design": {"bound_target": b_design, "flagged_violation": v_design,
                   "flagged_restatement": r_design, "unflagged_mention": u_design},
        "overlap_by_condition": overlap_by_condition(flags),
        "overlap_by_kind": overlap_by_kind(flags),
        "overlap_in_sample": overlap_by_condition(v_sample),
        "exposure_mentions_per_chapter": [
            {"run": run, "condition": cond, "chapters": len(xs),
             "subject_mentions_per_chapter": _mean(xs)}
            for (run, cond), xs in sorted(exposure.items())
        ],
        "exposure_by_kind": [
            {"run": run, "condition": cond, "kind": kind,
             "subject_mentions_per_chapter": _mean(xs)}
            for (run, cond, kind), xs in sorted(exposure_kind.items())
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def v1_without_other_subject(premise: Premise, fact: CanonFact, text: str) -> bool:
    """v1, except that a hit which is another planted subject's true value of
    the same kind is treated as no hit. Not a detector: an upper bound on how
    much those coincidences can move a comparison."""
    bad = fact._pattern(fact.contradictions())
    good = fact._pattern(_forms(fact.value))
    for m in _subject_re(fact.subject).finditer(text):
        lo = max(0, m.start() - fact.window)
        hi = min(len(text), m.end() + fact.window)
        span = text[lo:hi]
        hit = bad.search(span)
        if hit is None:
            continue
        near = span[max(0, hit.start() - 40) : hit.end() + 40]
        if good.search(near):
            continue
        if _other_subject(premise, fact, hit.group(0), text, lo + hit.start())[0]:
            continue
        return True
    return False


# Within-run gaps the generation negative rests on: (run, focal, baseline).
BOUND_PAIRS = (
    ("gen-v3-hybrid", "base:dsg-state", "base:none"),
    ("gen-v4-retrieval", "base:beat-retrieval", "base:none"),
)


def bound(generation_dir: Path, out_dir: Path, argv: list[str] | None = None) -> dict:
    """The two headline gaps, under v1 and under v1 minus other-subject flags."""
    out: dict[str, object] = {
        "provenance": provenance(
            "dsg.eval.detector_audit", argv, {"bootstrap": BOOTSTRAP_SEED},
            [generation_dir / f"{run}.json" for run, _, _ in BOUND_PAIRS],
        ),
        "kind": "bound",
        "note": ("An upper bound on how far other-subject-value coincidences can move "
                 "each gap. Not a result: those rows are unlabelled, and some may be "
                 "real violations."),
        "metric": "violation_rate",
        "comparisons": {},
    }
    for run, focal, baseline in BOUND_PAIRS:
        payload = json.loads((generation_dir / f"{run}.json").read_text())
        premises = {p["story_id"]: Premise.from_json(p) for p in payload["premises"]}
        stripped = {**payload, "records": []}
        removed: Counter = Counter()
        for rec in payload["records"]:
            premise = premises[rec["story_id"]]
            kept = [f.fact_id for f in premise.canon
                    if v1_without_other_subject(premise, f, rec["text"])]
            removed[rec["condition"]] += len(rec["violations"]) - len(kept)
            stripped["records"].append({**rec, "violations": kept})
        entry: dict[str, object] = {}
        for variant, data in (("v1", payload), ("v1_minus_other_subject", stripped)):
            by = defaultdict(dict)
            for score in score_runs(data):
                by[score.condition][score.story_id] = score.violation_rate
            stories = sorted(set(by[focal]) & set(by[baseline]))
            entry[variant] = paired_bootstrap(
                [by[focal][s] for s in stories], [by[baseline][s] for s in stories],
                seed=BOOTSTRAP_SEED,
            ).as_dict()
        entry["chapter_flags_removed"] = {focal: removed[focal], baseline: removed[baseline]}
        out["comparisons"][f"{run}:{focal}_minus_{baseline}"] = entry
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bound.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def _write(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.detector_audit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="build the labelling sheet and manifest")
    e.add_argument("--generation", default="artifacts/hub/generation")
    e.add_argument("--out", default="audit/detector")
    e.add_argument("--seed", type=int, default=SEED)
    b = sub.add_parser("bound", help="headline gaps with other-subject flags removed")
    b.add_argument("--generation", default="artifacts/hub/generation")
    b.add_argument("--out", default="audit/detector")
    args = p.parse_args(argv)
    if args.cmd == "bound":
        print(json.dumps(bound(Path(args.generation), Path(args.out), argv)["comparisons"],
                         indent=2))
        return 0
    m = export(Path(args.generation), Path(args.out), args.seed, argv)
    print(json.dumps({k: m[k] for k in ("population", "sampled", "recall_population")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

