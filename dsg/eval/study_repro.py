"""Do the reading-study numbers the docs cite reproduce under current code?

The published ``results.json`` files (``artifacts/hub/results``) predate commit
0be8e0d, which made ``occupation`` multi-valued and widened ``refines()``. This
reads every reading-study number cited in README.md, docs/FINDINGS.md and
docs/METHODOLOGY.md from the published results and from a rerun under current
code, checks that the published value renders exactly as the doc cites it (so
the mapping is right), and records the difference and whether a sign or a
significance call changes. It edits no doc.

    python -m dsg study --proposals artifacts/hub/proposals/pdnc-qwen3b-w3200 \\
        --out artifacts/results-current/pdnc-qwen3b      # likewise 1.5B, 7B, LitBank 7B
    python -m dsg.eval.study_repro --out audit/repro/study_current_code.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsg.eval.provenance import record as provenance

RUNS = ("pdnc-qwen1.5b", "pdnc-qwen3b", "pdnc-qwen7b", "litbank-qwen7b")
TREND = "audit/calculus/elaboration_trend.json"
CALCULUS = ("dsg-eager", "dsg-full", "retrospective")
R, F, M = "README", "FINDINGS", "METHODOLOGY"

AO_WO, MERGE, REV = "append-only_vs_window-only", "dsg-merge_vs_append-only", \
    "dsg-eager_vs_dsg-merge"
DEFER, CAUSAL, F1 = "dsg-full_vs_dsg-eager", "retrospective_vs_dsg-full", \
    "dsg-full_vs_append-only"
INC, CONLL, MENTION, ROLL = "inconsistent_slot_rate", "conll_f1", "mention_acc", "rollback"


def cmp(run, pair, metric, field="mean_delta"):
    return ("cmp", run, pair, metric, field)


def trend(run, which, op, field="rho"):
    return ("trend", run, which, op, field)


# (where cited, what, quantity, cited text, format)
CLAIMS = [
    (f"{R} §3", "7B persistent state: CoNLL", cmp("pdnc-qwen7b", AO_WO, CONLL), "+0.209", "f3"),
    (f"{R} §3", "  CI low", cmp("pdnc-qwen7b", AO_WO, CONLL, "ci_low"), "+0.182", "f3"),
    (f"{R} §3", "  CI high", cmp("pdnc-qwen7b", AO_WO, CONLL, "ci_high"), "+0.240", "f3"),
    (f"{R} §3", "  wins", cmp("pdnc-qwen7b", AO_WO, CONLL, "wins"), "28–0", "wins"),
    (f"{R} §3", "7B persistent state: mention", cmp("pdnc-qwen7b", AO_WO, MENTION), "+0.152",
     "f3"),
    (f"{R} §3", "  CI low", cmp("pdnc-qwen7b", AO_WO, MENTION, "ci_low"), "+0.124", "f3"),
    (f"{R} §3", "  CI high", cmp("pdnc-qwen7b", AO_WO, MENTION, "ci_high"), "+0.183", "f3"),
    (f"{R} §3", "  wins", cmp("pdnc-qwen7b", AO_WO, MENTION, "wins"), "28–0", "wins"),
    (f"{R} §3", "7B persistent state: inconsistent slots", cmp("pdnc-qwen7b", AO_WO, INC),
     "+0.261", "f3"),
    (f"{R} §3", "  CI low", cmp("pdnc-qwen7b", AO_WO, INC, "ci_low"), "+0.233", "f3"),
    (f"{R} §3", "  CI high", cmp("pdnc-qwen7b", AO_WO, INC, "ci_high"), "+0.291", "f3"),
    (f"{R} §3", "  wins", cmp("pdnc-qwen7b", AO_WO, INC, "wins"), "28–0", "wins"),
    (f"{R} §3, {F} §1", "3B persistent state: CoNLL", cmp("pdnc-qwen3b", AO_WO, CONLL), "+0.088",
     "f3"),
    (f"{R} §3", "1.5B persistent state: CoNLL", cmp("pdnc-qwen1.5b", AO_WO, CONLL), "+0.063",
     "f3"),
    (f"{R} §3", "LitBank persistent state: CoNLL", cmp("litbank-qwen7b", AO_WO, CONLL),
     "+0.081", "f3"),
    (f"{R} §3, {F} §1/§3", "3B persistent state: mention", cmp("pdnc-qwen3b", AO_WO, MENTION),
     "+0.310", "f3"),
    (f"{R} §3", "1.5B persistent state: mention", cmp("pdnc-qwen1.5b", AO_WO, MENTION),
     "+0.314", "f3"),
    (f"{F} §1", "3B persistent state: inconsistent slots", cmp("pdnc-qwen3b", AO_WO, INC),
     "+0.323", "f3"),
    (f"{F} §1", "3B persistent state: rollbacks", cmp("pdnc-qwen3b", AO_WO, ROLL), "0", "f0"),
    (f"{R} §3", "7B window-only inconsistent", cmp("pdnc-qwen7b", AO_WO, INC, "mean_b"), "0.110",
     "f3"),
    (f"{R} §3", "7B append-only inconsistent", cmp("pdnc-qwen7b", AO_WO, INC, "mean_a"), "0.371",
     "f3"),
    (f"{R} §3, {F} §1", "3B window-only inconsistent", cmp("pdnc-qwen3b", AO_WO, INC, "mean_b"),
     "0.101", "f3"),
    (f"{R} §3, {F} §1/F3, {M} §2", "3B append-only inconsistent",
     cmp("pdnc-qwen3b", AO_WO, INC, "mean_a"), "0.423", "f3"),
    (f"{R} §3, {F} §1", "1.5B window-only inconsistent",
     cmp("pdnc-qwen1.5b", AO_WO, INC, "mean_b"), "0.004", "f3"),
    (f"{R} §3, {F} §1, {M} §2", "1.5B append-only inconsistent",
     cmp("pdnc-qwen1.5b", AO_WO, INC, "mean_a"), "0.107", "f3"),
    (f"{R} §3, {F} §1", "LitBank window-only inconsistent",
     cmp("litbank-qwen7b", AO_WO, INC, "mean_b"), "0.091", "f3"),
    (f"{R} §3, {F} §1", "LitBank append-only inconsistent",
     cmp("litbank-qwen7b", AO_WO, INC, "mean_a"), "0.181", "f3"),
    (f"{R} §3", "append-only inconsistent, lowest of 4 runs", ("ao_range", "min"), "11%",
     "pct0"),
    (f"{R} §3", "append-only inconsistent, highest of 4 runs", ("ao_range", "max"), "42%",
     "pct0"),
    (f"{R} §3, {F} §5", "1.5B unparseable windows", ("parse", "pdnc-qwen1.5b", "rate"), "20.6%",
     "pct1"),
    (f"{R} §3", "1.5B parsed windows", ("parse", "pdnc-qwen1.5b", "parsed"), "3,226", "int"),
    (f"{R} §3", "1.5B windows", ("parse", "pdnc-qwen1.5b", "windows"), "4,063", "int"),
    (f"{R} §3, {M} §2", "calculus book-runs with any inconsistent slot",
     ("nonzero", "dsg-full"), "0", "int"),
    (f"{R} §3, {M} §2", "book-runs per policy", ("book_runs",), "184", "int"),
    (f"{R} §3", "7B + merging: CoNLL", cmp("pdnc-qwen7b", MERGE, CONLL), "−0.002", "f3"),
    (f"{R} §3", "7B + merging: mention", cmp("pdnc-qwen7b", MERGE, MENTION), "−0.005", "f3"),
    (f"{R} §3", "7B + merging: inconsistent", cmp("pdnc-qwen7b", MERGE, INC), "+0.007", "f3"),
    (f"{R} §3", "7B + fact revision: CoNLL", cmp("pdnc-qwen7b", REV, CONLL), "0.000", "f3"),
    (f"{R} §3", "7B + fact revision: mention", cmp("pdnc-qwen7b", REV, MENTION), "0.000", "f3"),
    (f"{R} §3", "7B + fact revision: inconsistent", cmp("pdnc-qwen7b", REV, INC), "−0.377", "f3"),
    (f"{R} §3", "  wins", cmp("pdnc-qwen7b", REV, INC, "wins"), "0–28", "wins"),
    (f"{R} §3", "7B + deferred: CoNLL", cmp("pdnc-qwen7b", DEFER, CONLL), "−0.003", "f3"),
    (f"{R} §3", "7B + deferred: mention", cmp("pdnc-qwen7b", DEFER, MENTION), "+0.007", "f3"),
    (f"{R} §3", "7B + deferred: inconsistent", cmp("pdnc-qwen7b", DEFER, INC), "0.000", "f3"),
    (f"{R} §3", "7B causality: CoNLL", cmp("pdnc-qwen7b", CAUSAL, CONLL), "−0.002", "f3"),
    (f"{R} §3", "7B causality: mention", cmp("pdnc-qwen7b", CAUSAL, MENTION), "+0.028", "f3"),
    (f"{R} §3", "  wins", cmp("pdnc-qwen7b", CAUSAL, MENTION, "wins"), "25–2", "wins"),
    (f"{R} §3", "7B causality: inconsistent", cmp("pdnc-qwen7b", CAUSAL, INC), "0.000", "f3"),
    (f"{R} §3, {F} §1/§3", "3B + merging: mention", cmp("pdnc-qwen3b", MERGE, MENTION),
     "−0.016", "f3"),
    (f"{R} §3", "1.5B + merging: mention", cmp("pdnc-qwen1.5b", MERGE, MENTION), "−0.023", "f3"),
    (f"{R} §3, {F} §1/F2", "3B causality: mention", cmp("pdnc-qwen3b", CAUSAL, MENTION),
     "+0.031", "f3"),
    (f"{F} F2", "  CI low", cmp("pdnc-qwen3b", CAUSAL, MENTION, "ci_low"), "+0.018", "f3"),
    (f"{F} F2", "  CI high", cmp("pdnc-qwen3b", CAUSAL, MENTION, "ci_high"), "+0.045", "f3"),
    (f"{F} F2", "  wins", cmp("pdnc-qwen3b", CAUSAL, MENTION, "wins"), "23–4", "wins"),
    (f"{R} §3, {F} F2", "1.5B causality: mention", cmp("pdnc-qwen1.5b", CAUSAL, MENTION),
     "+0.050", "f3"),
    (f"{F} F2", "  CI low", cmp("pdnc-qwen1.5b", CAUSAL, MENTION, "ci_low"), "+0.038", "f3"),
    (f"{F} F2", "  CI high", cmp("pdnc-qwen1.5b", CAUSAL, MENTION, "ci_high"), "+0.064", "f3"),
    (f"{R} §3, {F} F2", "  wins", cmp("pdnc-qwen1.5b", CAUSAL, MENTION, "wins"), "28–0", "wins"),
    (f"{F} §1", "3B + merging: CoNLL", cmp("pdnc-qwen3b", MERGE, CONLL), "−0.004", "f3"),
    (f"{F} §1", "3B + merging: inconsistent", cmp("pdnc-qwen3b", MERGE, INC), "+0.010", "f3"),
    (f"{F} §1/§3", "3B + merging: rollbacks", cmp("pdnc-qwen3b", MERGE, ROLL), "+2.36", "f2"),
    (f"{F} §1", "3B + fact revision: CoNLL", cmp("pdnc-qwen3b", REV, CONLL), "0.000", "f3"),
    (f"{F} §1", "3B + fact revision: mention", cmp("pdnc-qwen3b", REV, MENTION), "0.000", "f3"),
    (f"{F} §1/§3", "3B + fact revision: inconsistent", cmp("pdnc-qwen3b", REV, INC), "−0.433",
     "f3"),
    (f"{F} §1/§3", "  wins", cmp("pdnc-qwen3b", REV, INC, "wins"), "0–28", "wins"),
    (f"{F} §1", "3B + fact revision: rollbacks", cmp("pdnc-qwen3b", REV, ROLL), "+33.1", "f1"),
    (f"{F} §1", "3B + deferred: CoNLL", cmp("pdnc-qwen3b", DEFER, CONLL), "−0.002", "f3"),
    (f"{F} §1/§3", "3B + deferred: mention", cmp("pdnc-qwen3b", DEFER, MENTION), "+0.014", "f3"),
    (f"{F} §1", "3B + deferred: inconsistent", cmp("pdnc-qwen3b", DEFER, INC), "0.000", "f3"),
    (f"{F} §1/§3", "3B + deferred: rollbacks", cmp("pdnc-qwen3b", DEFER, ROLL), "−2.57", "f2"),
    (f"{F} §3", "3B dsg-eager rollbacks per book", cmp("pdnc-qwen3b", DEFER, ROLL, "mean_b"),
     "35.5", "f1"),
    (f"{F} §3", "3B deferred: rollback reduction", cmp("pdnc-qwen3b", DEFER, ROLL, "reduction"),
     "7%", "pct0"),
    (f"{F} §1", "3B causality: CoNLL", cmp("pdnc-qwen3b", CAUSAL, CONLL), "+0.005", "f3"),
    (f"{F} §1", "3B causality: inconsistent", cmp("pdnc-qwen3b", CAUSAL, INC), "0.000", "f3"),
    (f"{F} §1", "3B causality: rollbacks (cited ns)", cmp("pdnc-qwen3b", CAUSAL, ROLL), None,
     "f3"),
    (f"{F} F1", "3B calculus vs append-only: CoNLL", cmp("pdnc-qwen3b", F1, CONLL), "−0.005",
     "f3"),
    (f"{F} F1", "  CI low", cmp("pdnc-qwen3b", F1, CONLL, "ci_low"), "−0.011", "f3"),
    (f"{F} F1", "  CI high", cmp("pdnc-qwen3b", F1, CONLL, "ci_high"), "+0.001", "f3"),
    (f"{F} F1", "1.5B calculus vs append-only: CoNLL", cmp("pdnc-qwen1.5b", F1, CONLL),
     "−0.003", "f3"),
    (f"{F} F1", "LitBank calculus vs append-only: CoNLL", cmp("litbank-qwen7b", F1, CONLL),
     "−0.003", "f3"),
    (f"{F} F1", "3B calculus vs append-only: mention (cited flat)",
     cmp("pdnc-qwen3b", F1, MENTION), None, "f3"),
    (f"{F} F1", "1.5B calculus vs append-only: mention (cited flat)",
     cmp("pdnc-qwen1.5b", F1, MENTION), None, "f3"),
    (f"{F} §3", "1.5B + deferred: rollbacks", cmp("pdnc-qwen1.5b", DEFER, ROLL), "−5.50", "f2"),
    (f"{F} §3", "  wins", cmp("pdnc-qwen1.5b", DEFER, ROLL, "wins"), "0–28", "wins"),
    (f"{F} §3", "  reduction", cmp("pdnc-qwen1.5b", DEFER, ROLL, "reduction"), "99%", "pct0"),
    (f"{F} §3", "LitBank + deferred: rollbacks", cmp("litbank-qwen7b", DEFER, ROLL), "−0.60",
     "f2"),
    (f"{F} §3", "  reduction", cmp("litbank-qwen7b", DEFER, ROLL, "reduction"), "92%", "pct0"),
    (f"{F} §3", "1.5B + deferred: mention", cmp("pdnc-qwen1.5b", DEFER, MENTION), "+0.006",
     "f3"),
    (f"{R} §4.3, {F} F4", "3B narrative elaboration ρ",
     trend("pdnc-qwen3b", "position_trend", "elaboration"), "−0.59", "f2"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen3b", "position_trend", "elaboration",
                                      "p_value"), "0.007", "f3"),
    (f"{R} §4.3, {F} F4", "3B narrative revision ρ",
     trend("pdnc-qwen3b", "position_trend", "revision"), "+0.82", "f2"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen3b", "position_trend", "revision", "p_value"),
     "9×10⁻⁶", "e0"),
    (f"{R} §4.3, {F} F4", "  first twentieth",
     trend("pdnc-qwen3b", "position_trend", "revision", "first_bin"), "4.6%", "pct1"),
    (f"{R} §4.3, {F} F4", "  last twentieth",
     trend("pdnc-qwen3b", "position_trend", "revision", "last_bin"), "29.8%", "pct1"),
    (f"{R} §4.3, {F} F4", "3B shuffled elaboration ρ",
     trend("pdnc-qwen3b", "position_trend_shuffled", "elaboration"), "+0.08", "f2"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen3b", "position_trend_shuffled", "elaboration",
                                      "p_value"), "0.75", "f2"),
    (f"{R} §4.3, {F} F4", "3B shuffled revision ρ",
     trend("pdnc-qwen3b", "position_trend_shuffled", "revision"), "+0.69", "f2"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen3b", "position_trend_shuffled", "revision",
                                      "p_value"), "8×10⁻⁴", "e0"),
    (f"{R} §4.3", "7B narrative elaboration ρ",
     trend("pdnc-qwen7b", "position_trend", "elaboration"), "−0.084", "f3"),
    (f"{R} §4.3", "  p", trend("pdnc-qwen7b", "position_trend", "elaboration", "p_value"),
     "0.72", "f2"),
]

# The approved §4.3/F4 rewrite cites the overwrite-excluded trend, which no
# results.json carries; elaboration_trend.json has it under both lexicons.
APPROVED = [
    ("approved C", f"{run} {label} elaboration {'ρ' if field == 'rho' else 'p'}, "
                   "overwrite excluded",
     ("overwrite_excluded", run, label, field), cited, fmt)
    for run, label, field, cited, fmt in (
        ("pdnc-qwen3b", "narrative", "rho", "−0.850", "f3"),
        ("pdnc-qwen3b", "narrative", "p_value", None, "e0"),
        ("pdnc-qwen3b", "shuffled", "rho", None, "f3"),
        ("pdnc-qwen3b", "shuffled", "p_value", None, "f2"),
        ("pdnc-qwen7b", "narrative", "rho", "−0.746", "f3"),
        ("pdnc-qwen7b", "narrative", "p_value", None, "e0"),
        ("pdnc-qwen7b", "shuffled", "rho", None, "f3"),
        ("pdnc-qwen7b", "shuffled", "p_value", None, "f2"),
    )
]

SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def render(value, fmt: str) -> str:
    if fmt == "wins":
        return f"{value[0]:.0f}–{value[1]:.0f}"
    if fmt == "e0":
        mantissa, exp = f"{value:.0e}".split("e")
        return f"{mantissa}×10^{int(exp)}"
    if fmt.startswith("pct"):
        return f"{100 * value:.{int(fmt[3:])}f}%"
    if fmt == "int":
        return f"{value:,.0f}"
    text = f"{value:.{int(fmt[1:])}f}"
    return text[1:] if text.startswith("-") and text.strip("-0.") == "" else text


def normalise(cited: str) -> str:
    text = cited.replace("−", "-").lstrip("+")
    if "×10" in text:
        mantissa, exp = text.split("×10")
        return f"{mantissa}×10^{int(exp.translate(SUPERSCRIPT))}"
    return text


class Source:
    def __init__(self, root: Path, trend_lexicon: str):
        self.root, self.trend_lexicon = root, trend_lexicon
        self._cache: dict[str, dict] = {}
        self._trend = json.loads(Path(TREND).read_text())

    def run(self, name: str) -> dict:
        if name not in self._cache:
            self._cache[name] = json.loads((self.root / name / "results.json").read_text())
        return self._cache[name]

    def get(self, q: tuple):
        kind = q[0]
        if kind == "cmp":
            _, run, pair, metric, field = q
            c = self.run(run)["comparisons"][pair][metric]
            if field == "wins":
                return (c["wins_a"], c["wins_b"])
            if field == "reduction":
                return -c["mean_delta"] / c["mean_b"]
            return c[field]
        if kind == "trend":
            _, run, which, op, field = q
            return self.run(run)["instrument"][which][op][field]
        if kind == "ao_range":
            vals = [self.run(r)["comparisons"][AO_WO][INC]["mean_a"] for r in RUNS]
            return min(vals) if q[1] == "min" else max(vals)
        if kind == "parse":
            recs = [x for x in self.run(q[1])["records"] if x["policy"] == "window-only"]
            windows = sum(x["windows"] for x in recs)
            failed = sum(x["parse_failures"] for x in recs)
            return {"rate": failed / windows, "parsed": windows - failed, "windows": windows}[q[2]]
        if kind == "nonzero":
            return sum(1 for r in RUNS for x in self.run(r)["records"]
                       if x["policy"] in CALCULUS and x[INC] > 0)
        if kind == "book_runs":
            return sum(self.run(r)["meta"]["books"] for r in RUNS)
        if kind == "overwrite_excluded":
            _, run, label, field = q
            return self._trend["runs"][run][self.trend_lexicon]["overwrite_excluded"][label][
                "elaboration"][field]
        raise ValueError(kind)

    def call(self, q: tuple):
        """Sign and significance of the quantity a row belongs to, if it has one."""
        if q[0] == "cmp":
            c = self.run(q[1])["comparisons"][q[2]][q[3]]
            return {"delta": c["mean_delta"], "significant": bool(c["excludes_zero"])}
        if q[0] == "trend":
            t = self.run(q[1])["instrument"][q[2]][q[3]]
            return {"delta": t["rho"], "significant": t["p_value"] < 0.05}
        if q[0] == "overwrite_excluded":
            t = self._trend["runs"][q[1]][self.trend_lexicon]["overwrite_excluded"][q[2]][
                "elaboration"]
            return {"delta": t["rho"], "significant": t["p_value"] < 0.05}
        return None


def sign(x: float) -> int:
    return (x > 1e-12) - (x < -1e-12)


def compare(published: Source, current: Source) -> list[dict]:
    rows = []
    for where, what, q, cited, fmt in CLAIMS + APPROVED:
        pub, cur = published.get(q), current.get(q)
        pc, cc = published.call(q), current.call(q)
        row = {"where": where, "what": what, "query": list(q), "cited": cited,
               "published": list(pub) if isinstance(pub, tuple) else pub,
               "current": list(cur) if isinstance(cur, tuple) else cur,
               "published_rendered": render(pub, fmt), "current_rendered": render(cur, fmt)}
        if not isinstance(pub, tuple):
            row["difference"] = cur - pub
        if cited is not None:
            row["published_matches_cited"] = row["published_rendered"] == normalise(cited)
            row["cited_text_changes"] = row["current_rendered"] != normalise(cited)
        else:
            row["cited_text_changes"] = row["current_rendered"] != row["published_rendered"]
        if pc is not None:
            row["sign_changes"] = sign(pc["delta"]) != sign(cc["delta"])
            row["significance_changes"] = pc["significant"] != cc["significant"]
            row["significant_published"], row["significant_current"] = \
                pc["significant"], cc["significant"]
        rows.append(row)
    return rows


def table(rows: list[dict]) -> str:
    out = ["| where | what | cited | published | current | difference | text changes "
           "| sign/significance changes |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        diff = f"{r['difference']:+.4f}" if "difference" in r else "—"
        call = "—" if "sign_changes" not in r else (
            "**yes**" if r["sign_changes"] or r["significance_changes"] else "no")
        out.append(f"| {r['where']} | {r['what']} | {r['cited'] or '(not cited)'} "
                   f"| {r['published_rendered']} | {r['current_rendered']} | {diff} "
                   f"| {'**yes**' if r['cited_text_changes'] else 'no'} | {call} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.study_repro", description=__doc__)
    p.add_argument("--published", default="artifacts/hub/results")
    p.add_argument("--current", default="artifacts/results-current")
    p.add_argument("--out", default="audit/repro/study_current_code.json")
    args = p.parse_args(argv)
    published = Source(Path(args.published), "lexicon_pre_0be8e0d")
    current = Source(Path(args.current), "lexicon_current")
    for run in ("pdnc-qwen3b", "pdnc-qwen7b"):
        # The trend replay and the study rerun must agree on the unmodified profile.
        replay = current._trend["runs"][run]["lexicon_current"]["published"]
        for label, key in (("narrative", "position_trend"),
                           ("shuffled", "position_trend_shuffled")):
            for op in ("elaboration", "revision"):
                got = current.run(run)["instrument"][key][op]["rho"]
                assert abs(got - replay[label][op]["rho"]) < 1e-9, (run, label, op)
    rows = compare(published, current)
    # A cited value the published results.json does not render to is a doc error
    # independent of the code change; it is recorded, not asserted away.
    unmatched = [{"where": r["where"], "what": r["what"], "cited": r["cited"],
                  "published_rendered": r["published_rendered"]}
                 for r in rows if r.get("published_matches_cited") is False]
    inputs = [Path(TREND)] + [root / r / "results.json" for root in
                              (Path(args.published), Path(args.current)) for r in RUNS]
    out = {"provenance": provenance("dsg.eval.study_repro", argv, {}, inputs),
           "cited_not_matching_published": unmatched, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    Path(args.out).with_suffix(".md").write_text(table(rows) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
