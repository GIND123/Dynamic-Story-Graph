"""Audit sheet for the revision calculus's SUPERSEDE / REVISE / ELABORATE calls.

SUPERSEDE says the story world changed; REVISE says an earlier belief was never
true (Katsuno & Mendelzon's update versus revision). Which one applies is a
human judgement about the book, so this module replays the cached proposals
under ``dsg-full`` on CPU, records every classified conflict with the rule that
decided it, and builds the sheet people label. It decides nothing itself.

Replay is observational: ``RecordingState`` only copies what the classifier saw
before handing control back, and the replayed update log is checked against a
plain replay of the same book so the audit describes the shipped calculus.

    python -m dsg.eval.calculus_audit export --out audit/calculus
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from dsg.data.registry import load as load_corpus
from dsg.eval.provenance import record as provenance
from dsg.evidence import NOT_FOUND, Located, locate
from dsg.experiments.run_study import load_book_proposals
from dsg.lexicon import revelation_markers
from dsg.policies.runner import run_policy
from dsg.schemas import Op
from dsg.store import CandidateAssertion, NarrativeState

SEED = 20261018
N_ITEMS = 250
N_DOUBLE = 50
# Low-confidence ELABORATE decides most classified clashes and rewrites the old
# value in place, so it is the main rule under audit.
N_PROVISIONAL = 90
CONTEXT = 300

RUNS = {
    "pdnc": "artifacts/hub/proposals/pdnc-qwen7b-w3200",
    "litbank": "artifacts/hub/proposals/litbank-qwen7b-w1200",
}

# Cues that a value changed in the story world. Recorded for analysis only.
CHANGE_CUES = (r"\bdied\b", r"\bdead\b", r"\bmarr(?:ied|y|ies)\b", r"\bmoved?\b",
               r"\bbecame\b", r"\bnow\b", r"\bno longer\b", r"\bsince\b")
_CHANGE_RES = tuple((c, re.compile(c, re.IGNORECASE)) for c in CHANGE_CUES)

CLASSIFIED_RULES = (
    "elaborate:underspecified", "elaborate:refines", "elaborate:provisional",
    "revise:revelation-marker", "revise:reported-to-narrated", "revise:immutable",
    "supersede:mutable",
)


def change_cues(text: str) -> list[str]:
    return [c for c, rx in _CHANGE_RES if rx.search(text or "")]


class RecordingState(NarrativeState):
    """A ``NarrativeState`` that remembers what each classification saw."""

    def __init__(self, policy) -> None:
        super().__init__(policy)
        self.events: list[dict] = []
        self._evidence: dict[str, tuple[str, int, int, int]] = {}
        self._pending: dict | None = None

    def _name(self, entity_id: str) -> str:
        node = self.entities.get(self.deref(entity_id))
        return node.canonical if node else entity_id

    def _assert_new(self, subject, predicate, obj, cand, record=True, rule=""):
        a = super()._assert_new(subject, predicate, obj, cand, record=record, rule=rule)
        span = cand.span
        self._evidence[a.id] = (cand.evidence, self.index,
                                span.start if span else -1, span.end if span else -1)
        return a

    def _classify_with_rule(self, existing, obj, cand, predicate):
        op, rule = super()._classify_with_rule(existing, obj, cand, predicate)
        old_ev, old_index, old_start, old_end = self._evidence.get(existing.id, ("", -1, -1, -1))
        span = cand.span
        self._pending = {
            "subject": self._name(existing.subject), "predicate": predicate,
            "raw_predicate": cand.predicate,
            "old_assertion": existing.id,
            "old_object": self._name(existing.object) if existing.object in self.entities
            else existing.object,
            "old_evidence": old_ev, "old_window_index": old_index,
            "old_start": old_start, "old_end": old_end,
            "old_certainty": existing.certainty.value, "old_commit": existing.commit.value,
            "old_confidence": existing.confidence,
            "new_object": self._name(obj) if obj in self.entities else obj,
            "new_evidence": cand.evidence, "new_window_index": self.index,
            "new_start": span.start if span else -1, "new_end": span.end if span else -1,
            "new_certainty": cand.certainty.value, "new_confidence": cand.confidence,
            "classified_op": op.value, "rule": rule,
        }
        return op, rule

    def apply_assertion(self, cand: CandidateAssertion) -> Op:
        self._pending = None
        op = super().apply_assertion(cand)
        if self._pending is not None:
            event = {**self._pending, "applied_op": op.value}
            self.events.append(event)
            if op is Op.ELABORATE:
                span = cand.span
                self._evidence[event["old_assertion"]] = (
                    cand.evidence, self.index,
                    span.start if span else -1, span.end if span else -1,
                )
        self._pending = None
        return op


def _log_rows(state: NarrativeState) -> list[tuple]:
    return [(r.index, r.op.value, r.target, r.detail, r.payload, r.rule) for r in state.log]


def replay(corpus: str, run_dir: Path) -> tuple[list[dict], dict]:
    """Every classified conflict in every book, plus an operation census."""
    by_book = load_book_proposals(run_dir)
    novels = {n.book_id: n for n in load_corpus(corpus) if n.book_id in by_book}
    events: list[dict] = []
    census: Counter = Counter()
    for book_id, proposals in sorted(by_book.items()):
        novel = novels.get(book_id)
        if novel is None:
            continue
        rec = run_policy(proposals, "dsg-full", causal=True,
                         text_length=len(novel.text), state_cls=RecordingState)
        plain = run_policy(proposals, "dsg-full", causal=True, text_length=len(novel.text))
        assert _log_rows(rec.state) == _log_rows(plain.state), f"recording changed {book_id}"
        for r in rec.state.log:
            census[(r.op.value, r.rule)] += 1
        for e in rec.state.events:
            events.append(_with_text(e, corpus, book_id, novel.text))
    return events, {f"{op}|{rule}": n for (op, rule), n in sorted(census.items())}


def _clip(text: str, lo: int, hi: int, limit: int) -> str:
    hi = min(hi, limit)
    lo = max(0, min(lo, hi))
    body = " ".join(text[lo:hi].split())
    return body


def _located(text: str, start: int, end: int, evidence: str, limit: int):
    """Locate evidence in its window, then show ~CONTEXT chars either side.

    Nothing at or after ``limit`` (the end of the new candidate's window) is
    ever shown, so the sheet stays prefix-causal.
    """
    if start < 0:
        return Located(0, 0, NOT_FOUND), "", ""
    window = text[start:end]
    loc = locate(window, evidence)
    if not loc.found:
        return loc, "", _clip(text, start, end, limit)
    s, e = start + loc.start, start + loc.end
    shown = (
        ("... " if s - CONTEXT > 0 else "")
        + _clip(text, s - CONTEXT, s, limit) + " >>>"
        + _clip(text, s, e, limit) + "<<< "
        + _clip(text, e, e + CONTEXT, limit)
        + (" ..." if e + CONTEXT < limit else "")
    )
    return loc, shown, _clip(text, start, end, limit)


def _with_text(e: dict, corpus: str, book_id: str, text: str) -> dict:
    limit = e["new_end"] if e["new_end"] >= 0 else len(text)
    old_loc, old_ctx, old_win = _located(
        text, e["old_start"], e["old_end"], e["old_evidence"], limit
    )
    new_loc, new_ctx, new_win = _located(
        text, e["new_start"], e["new_end"], e["new_evidence"], limit
    )
    new_sentence = (
        text[e["new_start"] + new_loc.start : e["new_start"] + new_loc.end]
        if new_loc.found else ""
    )
    markers_ev = revelation_markers(e["new_evidence"])
    markers_sent = revelation_markers(new_sentence)
    return {
        **e, "corpus": corpus, "book": book_id,
        "old_match": old_loc.method, "new_match": new_loc.method,
        "old_context": old_ctx, "new_context": new_ctx,
        "old_window": old_win, "new_window": new_win,
        "markers_in_evidence": ";".join(markers_ev),
        "markers_in_sentence": ";".join(markers_sent),
        "change_cues_in_evidence": ";".join(change_cues(e["new_evidence"])),
        "change_cues_in_sentence": ";".join(change_cues(new_sentence)),
    }


LITBANK_SHARE = 0.30    # minimum LitBank share of a sampled stratum, where available

SHEET_COLUMNS = (
    "item_id", "double_annotate", "corpus", "book", "subject", "predicate", "raw_predicate",
    "old_object", "old_evidence", "old_match", "old_context",
    "new_object", "new_evidence", "new_match", "new_context",
    "label", "correct_value", "correct_value_text", "labeller", "notes",
    "classified_op", "applied_op", "rule", "markers_in_evidence",
    "markers_in_sentence", "change_cues_in_evidence", "change_cues_in_sentence",
    "old_certainty", "new_certainty", "old_confidence", "old_commit",
    "old_window_index", "new_window_index", "old_start", "old_end", "new_start", "new_end",
    "sample_reason", "stratum", "stratum_population", "stratum_sampled",
    "old_window", "new_window",
)
# Everything a labeller could be steered by: the calculus's own call and the
# cues it keyed on.
BLIND_DROP = {
    "classified_op", "applied_op", "rule", "markers_in_evidence",
    "markers_in_sentence", "change_cues_in_evidence", "change_cues_in_sentence",
    "old_certainty", "new_certainty", "old_confidence", "old_commit",
    "sample_reason", "stratum", "stratum_population", "stratum_sampled",
}


def _by_novel(pool: list[dict], n: int, rng: random.Random, reason: str) -> list[dict]:
    """Seeded sample spread evenly over novels, at least LITBANK_SHARE LitBank."""
    by_corpus = {c: [e for e in pool if e["corpus"] == c] for c in ("pdnc", "litbank")}
    lit = min(len(by_corpus["litbank"]), max(round(n * LITBANK_SHARE),
                                             n - len(by_corpus["pdnc"])))
    quota = {"litbank": lit, "pdnc": min(len(by_corpus["pdnc"]), n - lit)}
    picked = []
    for corpus in ("pdnc", "litbank"):
        strata: dict[str, list[dict]] = defaultdict(list)
        for e in by_corpus[corpus]:
            strata[e["book"]].append(e)
        alloc = dict.fromkeys(strata, 0)
        for _ in range(quota[corpus]):
            open_ = [b for b in sorted(strata) if alloc[b] < len(strata[b])]
            if not open_:
                break
            alloc[min(open_, key=lambda b: (alloc[b], b))] += 1
        for book in sorted(strata):
            for e in rng.sample(strata[book], alloc[book]):
                picked.append({**e, "sample_reason": reason,
                               "stratum": f"{e['rule']}|{corpus}|{book}",
                               "stratum_population": len(strata[book]),
                               "stratum_sampled": alloc[book]})
    return picked


def sample(events: list[dict], seed: int = SEED) -> tuple[list[dict], dict]:
    """The labelling sample. Census strata are taken whole; the two large ones
    (``supersede:mutable`` and ``elaborate:provisional``) are sampled by novel,
    with N_PROVISIONAL rows of ``elaborate:provisional``."""
    rng = random.Random(seed)
    for i, e in enumerate(events):
        e["_key"] = i
    chosen: dict[int, dict] = {}

    def take(rows: list[dict], reason: str) -> None:
        for e in rows:
            if e["_key"] not in chosen:
                chosen[e["_key"]] = {**e, "sample_reason": reason, "stratum": e["rule"],
                                     "stratum_population": len(rows),
                                     "stratum_sampled": len(rows)}

    # Small strata in full.
    for rule in ("revise:revelation-marker", "revise:reported-to-narrated",
                 "revise:immutable", "elaborate:refines", "elaborate:underspecified"):
        take([e for e in events if e["rule"] == rule], "census")

    prov_pool = [e for e in events if e["rule"] == "elaborate:provisional"]
    for e in _by_novel(prov_pool, N_PROVISIONAL, rng, "provisional"):
        chosen[e["_key"]] = e

    sup_pool = [e for e in events
                if e["rule"] == "supersede:mutable" and e["_key"] not in chosen]
    for e in _by_novel(sup_pool, max(0, N_ITEMS - len(chosen)), rng, "supersede"):
        chosen[e["_key"]] = e

    rows = [chosen[k] for k in sorted(chosen)]
    rng.shuffle(rows)
    # Double annotation: proportional to rule, seeded.
    by_rule: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_rule[r["rule"]].append(i)
    doubles: set[int] = set()
    for rule in sorted(by_rule):
        k = round(N_DOUBLE * len(by_rule[rule]) / len(rows))
        doubles.update(rng.sample(by_rule[rule], min(k, len(by_rule[rule]))))
    spare = [i for i in range(len(rows)) if i not in doubles]
    while len(doubles) < N_DOUBLE and spare:
        doubles.add(spare.pop(rng.randrange(len(spare))))
    while len(doubles) > N_DOUBLE:
        doubles.remove(max(doubles))
    for i, r in enumerate(rows, 1):
        r["item_id"] = f"C{i:04d}"
        r["double_annotate"] = "yes" if (i - 1) in doubles else "no"
        for c in ("label", "correct_value", "correct_value_text", "labeller", "notes"):
            r[c] = ""
        r.pop("_key", None)

    marker_rows = Counter(m for r in rows for m in set(
        filter(None, (r["markers_in_evidence"] + ";" + r["markers_in_sentence"]).split(";"))))
    marker_population = Counter(m for e in events for m in set(
        filter(None, (e["markers_in_evidence"] + ";" + e["markers_in_sentence"]).split(";"))))
    design = {
        "seed": seed,
        "population_by_rule": dict(Counter(e["rule"] for e in events)),
        "population_by_corpus_rule": {f"{c}|{r}": n for (c, r), n in sorted(
            Counter((e["corpus"], e["rule"]) for e in events).items())},
        "sampled_by_rule": dict(Counter(r["rule"] for r in rows)),
        "sampled_by_corpus": dict(Counter(r["corpus"] for r in rows)),
        "sampled_by_reason": dict(Counter(r["sample_reason"] for r in rows)),
        "provisional_share": sum(r["rule"] == "elaborate:provisional" for r in rows) / len(rows),
        "double_annotate": sum(r["double_annotate"] == "yes" for r in rows),
        "double_by_rule": dict(Counter(r["rule"] for r in rows if r["double_annotate"] == "yes")),
        "markers": {m: {"population": marker_population[m], "sampled": marker_rows[m]}
                    for m in sorted(marker_population)},
        "evidence_not_found": {
            "population": _not_found(events), "sample": _not_found(rows),
        },
        "new_match_in_sample": dict(Counter(r["new_match"] for r in rows)),
        "old_match_in_sample": dict(Counter(r["old_match"] for r in rows)),
        "books_in_sample": len({r["book"] for r in rows}),
    }
    return rows, design


# Evidence strings that cannot be located because there is nothing to find:
# empty, or the extractor wrote the certainty label into the evidence field.
_NON_EVIDENCE = frozenset({"", "reported", "narrated", "inferred", "hypothetical"})


def _not_found(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    out: dict[str, float] = {"rows": n}
    for side in ("old", "new"):
        nf = [r for r in rows if r[f"{side}_match"] == NOT_FOUND]
        empty = sum(r[f"{side}_evidence"].strip().lower() in _NON_EVIDENCE for r in nf)
        out[f"{side}_not_found"] = len(nf)
        out[f"{side}_not_found_rate"] = len(nf) / n if n else 0.0
        out[f"{side}_not_found_no_evidence"] = empty
        out[f"{side}_not_found_real_evidence"] = len(nf) - empty
    either = sum(1 for r in rows if NOT_FOUND in (r["old_match"], r["new_match"]))
    out["either_not_found"] = either
    out["either_not_found_rate"] = either / n if n else 0.0
    return out


def _write(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def export(out_dir: Path, seed: int = SEED, argv: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    events, census = [], {}
    for corpus, run_dir in RUNS.items():
        ev, c = replay(corpus, Path(run_dir))
        events += ev
        census[corpus] = c
    with (out_dir / "events.jsonl").open("w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    rows, design = sample(events, seed)
    blind = tuple(c for c in SHEET_COLUMNS if c not in BLIND_DROP)
    _write(out_dir / "sheet.csv", rows, SHEET_COLUMNS)
    _write(out_dir / "sheet_blind.csv", rows, blind)
    inputs = [f for d in RUNS.values() for f in sorted(Path(d).glob("*.jsonl"))]
    manifest = {
        "provenance": provenance("dsg.eval.calculus_audit", argv, {"sample": seed}, inputs),
        "runs": RUNS, "policy": "dsg-full", "census": census, **design,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.calculus_audit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="replay, sample and write the labelling sheets")
    e.add_argument("--out", default="audit/calculus")
    e.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    m = export(Path(args.out), args.seed, argv)
    print(json.dumps({k: m[k] for k in (
        "population_by_rule", "sampled_by_rule", "sampled_by_corpus",
        "provisional_share", "double_annotate", "evidence_not_found")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
