"""The elaboration-vs-position trend with the low-confidence overwrite removed.

README §4.3 reports the elaboration rate's Spearman trend over discourse
position, with a shuffled-order control. The rate counts every ELABORATE, but
one ELABORATE path -- ``elaborate:provisional``, an overwrite of a provisional,
low-confidence value -- replaces a value rather than refining it
(``audit/calculus/elaborate_census.json``). This replays the study exactly
(``run_study``'s narrative run and ``_shuffled_run``), checks that the
unmodified profile reproduces the published trend, and recomputes it two ways:

* ``overwrite_excluded`` -- overwrite records dropped from numerator and total
* ``overwrite_not_elaboration`` -- kept in the total, removed from the numerator

The published ``results.json`` predates commit 0be8e0d (``occupation`` made
multi-valued, ``refines()`` widened), which changes the replay. So everything is
computed twice: under the lexicon as of ``0be8e0d^`` (read from git history and
rebound in the modules that import it), where the unmodified profile must
reproduce every published trend exactly, and under the current lexicon.

    python -m dsg.eval.elaboration_trend --out audit/calculus/elaboration_trend.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import dsg.invariants
import dsg.lexicon
import dsg.matching
import dsg.policies.runner
import dsg.proposals
import dsg.store

from dsg.data.registry import load as load_corpus
from dsg.eval.instrument import position_trend
from dsg.eval.provenance import record as provenance
from dsg.experiments.run_study import _shuffled_run, load_book_proposals
from dsg.policies.runner import run_policy

RUNS = {
    "pdnc-qwen3b": "artifacts/hub/proposals/pdnc-qwen3b-w3200",
    "pdnc-qwen7b": "artifacts/hub/proposals/pdnc-qwen7b-w3200",
}
OVERWRITE = "elaborate:provisional"
OPS = ("elaborate", "supersede", "revise", "merge_provisional", "merge_committed", "assert")


PRE_LEXICON = "0be8e0d^"
_IMPORTERS = (dsg.store, dsg.matching, dsg.proposals, dsg.policies.runner, dsg.invariants)


@contextlib.contextmanager
def lexicon_at(rev: str | None):
    """Rebind every lexicon name the calculus imports to its version at ``rev``."""
    if rev is None:
        yield
        return
    source = subprocess.run(["git", "show", f"{rev}:dsg/lexicon.py"], capture_output=True,
                            text=True, check=True).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source)
    spec = importlib.util.spec_from_file_location("lexicon_at_rev", fh.name)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    saved = []
    for mod in _IMPORTERS:
        for name in dir(old):
            current = getattr(dsg.lexicon, name, None)
            if not name.startswith("__") and current is not None \
                    and getattr(mod, name, None) is current:
                saved.append((mod, name, current))
                setattr(mod, name, getattr(old, name))
    try:
        yield
    finally:
        for mod, name, value in saved:
            setattr(mod, name, value)


def profile(result, variant: str, n_bins: int = 20) -> list[dict]:
    """``dsg.eval.process.revision_profile``, optionally without the overwrite."""
    windows = max(1, result.windows)
    bins = [{"bin": i, "overwrite": 0, **dict.fromkeys(OPS, 0)} for i in range(n_bins)]
    for record in result.state.log:
        b = min(n_bins - 1, int(record.index / windows * n_bins))
        if record.rule == OVERWRITE:
            bins[b]["overwrite"] += 1
            if variant == "overwrite_excluded":
                continue
        if record.op.value in bins[b]:
            bins[b][record.op.value] += 1
    for b in bins:
        total = sum(b[k] for k in OPS)
        elaborate = b["elaborate"] - (b["overwrite"] if variant == "overwrite_not_elaboration"
                                      else 0)
        b["total"] = total
        b["revision_rate"] = (b["revise"] + b["merge_committed"]) / total if total else 0.0
        b["elaboration_rate"] = (elaborate + b["merge_provisional"]) / total if total else 0.0
    return bins


def trends(run_dir: Path) -> dict:
    by_book = load_book_proposals(run_dir)
    novels = {n.book_id: n for n in load_corpus("pdnc") if n.book_id in by_book}
    variants = ("published", "overwrite_excluded", "overwrite_not_elaboration")
    narrative = {v: {} for v in variants}
    shuffled = {v: {} for v in variants}
    for book_id, proposals in by_book.items():
        novel = novels.get(book_id)
        if novel is None:
            continue
        fwd = run_policy(proposals, "dsg-full", causal=True, text_length=len(novel.text))
        shuf = _shuffled_run(proposals, novel)
        for v in variants:
            narrative[v][book_id] = {"dsg-full": profile(fwd, v)}
            shuffled[v][book_id] = {"dsg-full": profile(shuf, v)}
    out = {}
    for v in variants:
        out[v] = {}
        for label, prof in (("narrative", narrative[v]), ("shuffled", shuffled[v])):
            t = position_trend(prof)
            out[v][label] = {op: {"rho": t[op]["rho"], "p_value": t[op]["p_value"],
                                  "first_bin": t[op]["first_bin"], "last_bin": t[op]["last_bin"]}
                             for op in ("elaboration", "revision")}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.elaboration_trend", description=__doc__)
    p.add_argument("--out", default="audit/calculus/elaboration_trend.json")
    p.add_argument("--results", default="artifacts/hub/results")
    args = p.parse_args(argv)
    inputs = [f for d in RUNS.values() for f in sorted(Path(d).glob("*.jsonl"))]
    out = {"provenance": provenance("dsg.eval.elaboration_trend", argv, {"shuffle": 0}, inputs),
           "pre_lexicon_rev": PRE_LEXICON, "runs": {}}
    for name, run_dir in RUNS.items():
        inst = json.loads((Path(args.results) / name / "results.json").read_text())["instrument"]
        reported = {label: {op: inst[key][op]["rho"] for op in ("elaboration", "revision")}
                    for label, key in (("narrative", "position_trend"),
                                       ("shuffled", "position_trend_shuffled"))}
        entry = {"reported_in_results_json": reported}
        for lex, rev in (("lexicon_pre_0be8e0d", PRE_LEXICON), ("lexicon_current", None)):
            with lexicon_at(rev):
                res = trends(Path(run_dir))
            if rev is not None:
                for label in ("narrative", "shuffled"):
                    for op in ("elaboration", "revision"):
                        got, want = res["published"][label][op]["rho"], reported[label][op]
                        assert abs(got - want) < 1e-9, (name, lex, label, op, got, want)
            entry[lex] = res
            for v, r in res.items():
                print(name, lex, f"{v:26}", {lab: {op: (round(x["rho"], 3), round(x["p_value"], 4))
                                                   for op, x in ops.items()}
                                             for lab, ops in r.items()}, flush=True)
        out["runs"][name] = entry
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
