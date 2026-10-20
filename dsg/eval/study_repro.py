"""The reading-study numbers the docs cite, under current code, beside the published ones.

The published ``results.json`` files (``artifacts/hub/results``) predate commit
0be8e0d, which made ``occupation`` multi-valued and widened ``refines()``. The
study was rerun under current code, and those reruns are the reference: every
reading-study number in README.md, docs/FINDINGS.md, docs/METHODOLOGY.md and
dsg/README.md is a row below, and the run fails unless each cited text renders
exactly from the current results. Each row also carries the published value,
the doc text before the rerun where it changed, and whether a sign or a
significance call changed.

The current results are read from the Hub copy pinned in
``audit/repro/hub_manifest.json`` (``results-current/`` in the artifact repo),
falling back to ``$DSG_REPRO_LOCAL`` or ``artifacts/results-current``; a copy
must match its manifest hash.

    python -m dsg study --proposals artifacts/hub/proposals/pdnc-qwen3b-w3200 \\
        --out artifacts/results-current/pdnc-qwen3b      # likewise 1.5B, 7B, LitBank 7B
    HF_TOKEN=... python -m dsg.eval.study_repro --upload  # once: push, pin the revision
    python -m dsg.eval.study_repro --out audit/repro/study_current_code.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsg.eval.e1_artifacts import _sha256, resolve
from dsg.eval.provenance import record as provenance

RUNS = ("pdnc-qwen1.5b", "pdnc-qwen3b", "pdnc-qwen7b", "litbank-qwen7b")
PDNC = RUNS[:3]
TREND = "audit/calculus/elaboration_trend.json"
MANIFEST = Path("audit/repro/hub_manifest.json")
HUB_DIR = "results-current"
LOCAL_DIRS = ("artifacts/results-current",)
CALCULUS = ("dsg-eager", "dsg-full", "retrospective")
R, F, M, D = "README", "FINDINGS", "METHODOLOGY", "dsg/README"

AO_WO, MERGE, REV = "append-only_vs_window-only", "dsg-merge_vs_append-only", \
    "dsg-eager_vs_dsg-merge"
DEFER, CAUSAL, F1 = "dsg-full_vs_dsg-eager", "retrospective_vs_dsg-full", \
    "dsg-full_vs_append-only"
INC, CONLL, MENTION, ROLL = "inconsistent_slot_rate", "conll_f1", "mention_acc", "rollback"
NARR, SHUF = "position_trend", "position_trend_shuffled"


def cmp(run, pair, metric, field="mean_delta"):
    return ("cmp", run, pair, metric, field)


def trend(run, which, op, field="rho"):
    return ("trend", run, which, op, field)


def excl(run, label, field="rho"):
    """Elaboration trend with the low-confidence overwrite excluded (elaboration_trend.py)."""
    return ("overwrite_excluded", run, label, field)


# (where cited, what, quantity, cited text now, format[, cited text before the rerun])
# A row whose text is now None is no longer cited.
CLAIMS = [
    (f"{R} §3", "7B persistent state: CoNLL", cmp("pdnc-qwen7b", AO_WO, CONLL), "+0.209", "f3"),
    (f"{R} §3", "  CI low", cmp("pdnc-qwen7b", AO_WO, CONLL, "ci_low"), "+0.182", "f3"),
    (f"{R} §3", "  CI high", cmp("pdnc-qwen7b", AO_WO, CONLL, "ci_high"), "+0.240", "f3"),
    (f"{R} §3", "  wins", cmp("pdnc-qwen7b", AO_WO, CONLL, "wins"), "28–0", "wins"),
    (f"{R} §3", "7B persistent state: mention", cmp("pdnc-qwen7b", AO_WO, MENTION), "+0.152",
     "f3"),
    (f"{R} §3", "  CI low", cmp("pdnc-qwen7b", AO_WO, MENTION, "ci_low"), "+0.124", "f3"),
    (f"{R} §3", "  CI high", cmp("pdnc-qwen7b", AO_WO, MENTION, "ci_high"), "+0.182", "f3",
     "+0.183"),
    (f"{R} §3", "  wins", cmp("pdnc-qwen7b", AO_WO, MENTION, "wins"), "28–0", "wins"),
    (f"{R} §3", "7B persistent state: inconsistent slots", cmp("pdnc-qwen7b", AO_WO, INC),
     "+0.269", "f3", "+0.261"),
    (f"{R} §3", "  CI low", cmp("pdnc-qwen7b", AO_WO, INC, "ci_low"), "+0.239", "f3", "+0.233"),
    (f"{R} §3", "  CI high", cmp("pdnc-qwen7b", AO_WO, INC, "ci_high"), "+0.299", "f3",
     "+0.291"),
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
     "+0.320", "f3", "+0.323"),
    (f"{F} §1", "3B persistent state: rollbacks", cmp("pdnc-qwen3b", AO_WO, ROLL), "0", "f0"),
    (f"{R} §3", "7B window-only inconsistent", cmp("pdnc-qwen7b", AO_WO, INC, "mean_b"), "0.113",
     "f3", "0.110"),
    (f"{R} §3, {D} §3", "7B append-only inconsistent", cmp("pdnc-qwen7b", AO_WO, INC, "mean_a"),
     "0.382", "f3", "0.371"),
    (f"{R} §3, {F} §1", "3B window-only inconsistent", cmp("pdnc-qwen3b", AO_WO, INC, "mean_b"),
     "0.102", "f3", "0.101"),
    (f"{R} §3, {F} §1, {D} §3", "3B append-only inconsistent",
     cmp("pdnc-qwen3b", AO_WO, INC, "mean_a"), "0.422", "f3", "0.423"),
    (f"{F} F3", "3B append-only inconsistent, as a share",
     cmp("pdnc-qwen3b", AO_WO, INC, "mean_a"), "42.2%", "pct1", "42.3%"),
    (f"{R} §3, {F} §1", "1.5B window-only inconsistent",
     cmp("pdnc-qwen1.5b", AO_WO, INC, "mean_b"), "0.004", "f3"),
    (f"{R} §3, {F} §1, {D} §3", "1.5B append-only inconsistent",
     cmp("pdnc-qwen1.5b", AO_WO, INC, "mean_a"), "0.090", "f3", "0.107"),
    (f"{R} §3, {F} §1", "LitBank window-only inconsistent",
     cmp("litbank-qwen7b", AO_WO, INC, "mean_b"), "0.092", "f3", "0.091"),
    (f"{R} §3, {F} §1, {D} §3", "LitBank append-only inconsistent",
     cmp("litbank-qwen7b", AO_WO, INC, "mean_a"), "0.185", "f3", "0.181"),
    (f"{R} §3", "append-only inconsistent, lowest of 4 runs", ("ao_range", "min", RUNS), "9%",
     "pct0", "11%"),
    (f"{R} §3", "append-only inconsistent, highest of 4 runs", ("ao_range", "max", RUNS), "42%",
     "pct0"),
    (f"{M} §2", "append-only inconsistent, lowest PDNC run", ("ao_range", "min", PDNC), "0.090",
     "f3", "0.107"),
    (f"{M} §2", "append-only inconsistent, highest PDNC run", ("ao_range", "max", PDNC), "0.422",
     "f3", "0.423"),
    (f"{R} §3, {F} §5", "1.5B unparseable windows", ("parse", "pdnc-qwen1.5b", "rate"), "20.6%",
     "pct1"),
    (f"{R} §3", "1.5B parsed windows", ("parse", "pdnc-qwen1.5b", "parsed"), "3,226", "int"),
    (f"{R} §3", "1.5B windows", ("parse", "pdnc-qwen1.5b", "windows"), "4,063", "int"),
    (f"{R} §3, {M} §2, {D} §3", "calculus book-runs with any inconsistent slot",
     ("nonzero", "dsg-full"), "0", "int"),
    (f"{R} §3, {M} §2, {D} §3", "book-runs per policy", ("book_runs",), "184", "int"),
    (f"{R} §3", "7B + merging: CoNLL", cmp("pdnc-qwen7b", MERGE, CONLL), "−0.002", "f3"),
    (f"{R} §3", "7B + merging: mention", cmp("pdnc-qwen7b", MERGE, MENTION), "−0.005", "f3"),
    (f"{R} §3", "7B + merging: inconsistent", cmp("pdnc-qwen7b", MERGE, INC), "+0.005", "f3",
     "+0.007"),
    (f"{R} §3", "  CI low", cmp("pdnc-qwen7b", MERGE, INC, "ci_low"), "+0.001", "f3"),
    (f"{R} §3", "  CI high", cmp("pdnc-qwen7b", MERGE, INC, "ci_high"), "+0.010", "f3"),
    (f"{R} §3", "7B + fact revision: CoNLL", cmp("pdnc-qwen7b", REV, CONLL), "0.000", "f3"),
    (f"{R} §3", "7B + fact revision: mention", cmp("pdnc-qwen7b", REV, MENTION), "0.000", "f3"),
    (f"{R} §3", "7B + fact revision: inconsistent", cmp("pdnc-qwen7b", REV, INC), "−0.388", "f3",
     "−0.377"),
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
    (f"{F} §1", "3B + merging: CoNLL", cmp("pdnc-qwen3b", MERGE, CONLL), "−0.003", "f3",
     "−0.004"),
    (f"{R} §3, {F} §1/§3", "3B + merging: inconsistent", cmp("pdnc-qwen3b", MERGE, INC),
     "+0.017", "f3", "+0.010"),
    (f"{R} §3, {F} §3", "  CI low", cmp("pdnc-qwen3b", MERGE, INC, "ci_low"), "+0.003", "f3"),
    (f"{R} §3, {F} §3", "  CI high", cmp("pdnc-qwen3b", MERGE, INC, "ci_high"), "+0.031", "f3"),
    (f"{F} §1/§3", "3B + merging: rollbacks", cmp("pdnc-qwen3b", MERGE, ROLL), "+2.36", "f2"),
    (f"{F} §1", "3B + fact revision: CoNLL", cmp("pdnc-qwen3b", REV, CONLL), "0.000", "f3"),
    (f"{F} §1", "3B + fact revision: mention", cmp("pdnc-qwen3b", REV, MENTION), "0.000", "f3"),
    (f"{F} §1/§3", "3B + fact revision: inconsistent", cmp("pdnc-qwen3b", REV, INC), "−0.439",
     "f3", "−0.433"),
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
    (f"{F} F1", "  CI high", cmp("pdnc-qwen3b", F1, CONLL, "ci_high"), "+0.000", "f3", "+0.001"),
    (f"{F} F1", "1.5B calculus vs append-only: CoNLL", cmp("pdnc-qwen1.5b", F1, CONLL),
     "−0.003", "f3"),
    (f"{F} F1", "LitBank calculus vs append-only: CoNLL", cmp("litbank-qwen7b", F1, CONLL),
     "−0.003", "f3"),
    (f"{F} F1", "3B calculus vs append-only: mention", cmp("pdnc-qwen3b", F1, MENTION),
     "−0.002", "f3", "flat"),
    (f"{F} F1", "1.5B calculus vs append-only: mention", cmp("pdnc-qwen1.5b", F1, MENTION),
     "−0.017", "f3", "flat"),
    (f"{F} F1", "  CI low", cmp("pdnc-qwen1.5b", F1, MENTION, "ci_low"), "−0.034", "f3"),
    (f"{F} F1", "  CI high", cmp("pdnc-qwen1.5b", F1, MENTION, "ci_high"), "−0.004", "f3"),
    (f"{D} §3", "7B dsg-full CoNLL", cmp("pdnc-qwen7b", F1, CONLL, "mean_a"), "0.659", "f3"),
    (f"{D} §3", "7B append-only CoNLL", cmp("pdnc-qwen7b", F1, CONLL, "mean_b"), "0.664", "f3"),
    (f"{F} §3", "1.5B + deferred: rollbacks", cmp("pdnc-qwen1.5b", DEFER, ROLL), "−5.50", "f2"),
    (f"{F} §3", "  wins", cmp("pdnc-qwen1.5b", DEFER, ROLL, "wins"), "0–28", "wins"),
    (f"{F} §3", "  reduction", cmp("pdnc-qwen1.5b", DEFER, ROLL, "reduction"), "99%", "pct0"),
    (f"{F} §3", "LitBank + deferred: rollbacks", cmp("litbank-qwen7b", DEFER, ROLL), "−0.60",
     "f2"),
    (f"{F} §3", "  reduction", cmp("litbank-qwen7b", DEFER, ROLL, "reduction"), "92%", "pct0"),
    (f"{F} §3", "1.5B + deferred: mention", cmp("pdnc-qwen1.5b", DEFER, MENTION), "+0.006",
     "f3"),
    (f"{R} §4.3, {F} F4", "3B narrative revision ρ", trend("pdnc-qwen3b", NARR, "revision"),
     "+0.82", "f2"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen3b", NARR, "revision", "p_value"), "9×10⁻⁶",
     "e0"),
    (f"{R} §4.3, {F} F4", "  first twentieth",
     trend("pdnc-qwen3b", NARR, "revision", "first_bin"), "4.4%", "pct1", "4.6%"),
    (f"{R} §4.3, {F} F4", "  last twentieth",
     trend("pdnc-qwen3b", NARR, "revision", "last_bin"), "29.8%", "pct1"),
    (f"{R} §4.3, {F} F4", "3B shuffled revision ρ", trend("pdnc-qwen3b", SHUF, "revision"),
     "+0.68", "f2", "+0.69"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen3b", SHUF, "revision", "p_value"), "1×10⁻³",
     "e0", "8×10⁻⁴"),
    (f"{R} §4.3, {F} F4", "3B narrative elaboration ρ, overwrite counted",
     trend("pdnc-qwen3b", NARR, "elaboration"), "−0.59", "f2"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen3b", NARR, "elaboration", "p_value"), "0.007",
     "f3"),
    (f"{R} §4.3, {F} F4", "3B shuffled elaboration ρ, overwrite counted",
     trend("pdnc-qwen3b", SHUF, "elaboration"), None, "f2", "+0.08"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen3b", SHUF, "elaboration", "p_value"), None,
     "f2", "0.75"),
    (f"{R} §4.3, {F} F4", "7B narrative elaboration ρ, overwrite counted",
     trend("pdnc-qwen7b", NARR, "elaboration"), "−0.155", "f3", "−0.084"),
    (f"{R} §4.3, {F} F4", "  p", trend("pdnc-qwen7b", NARR, "elaboration", "p_value"), "0.51",
     "f2", "0.72"),
    (f"{R} §4.3, {F} F4, {D} §6", "3B narrative elaboration ρ, overwrite excluded",
     excl("pdnc-qwen3b", "narrative"), "−0.850", "f3"),
    (f"{R} §4.3, {F} F4", "  p", excl("pdnc-qwen3b", "narrative", "p_value"), "2×10⁻⁶", "e0"),
    (f"{R} §4.3, {F} F4, {D} §6", "3B shuffled elaboration ρ, overwrite excluded",
     excl("pdnc-qwen3b", "shuffled"), "−0.141", "f3"),
    (f"{R} §4.3, {F} F4, {D} §6", "  p", excl("pdnc-qwen3b", "shuffled", "p_value"), "0.55",
     "f2"),
    (f"{R} §4.3, {F} F4, {D} §6", "7B narrative elaboration ρ, overwrite excluded",
     excl("pdnc-qwen7b", "narrative"), "−0.741", "f3"),
    (f"{R} §4.3, {F} F4", "  p", excl("pdnc-qwen7b", "narrative", "p_value"), "2×10⁻⁴", "e0"),
    (f"{R} §4.3, {F} F4, {D} §6", "7B shuffled elaboration ρ, overwrite excluded",
     excl("pdnc-qwen7b", "shuffled"), "+0.292", "f3"),
    (f"{R} §4.3, {F} F4, {D} §6", "  p", excl("pdnc-qwen7b", "shuffled", "p_value"), "0.21",
     "f2"),
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
    """One set of study results, plus the elaboration-trend replay under one lexicon."""

    def __init__(self, path_of, trend_lexicon: str):
        self.path_of, self.trend_lexicon = path_of, trend_lexicon
        self._cache: dict[str, dict] = {}
        self._trend = json.loads(Path(TREND).read_text())

    def run(self, name: str) -> dict:
        if name not in self._cache:
            self._cache[name] = json.loads(Path(self.path_of(name)).read_text())
        return self._cache[name]

    def _excl(self, run: str, label: str) -> dict:
        return self._trend["runs"][run][self.trend_lexicon]["overwrite_excluded"][label][
            "elaboration"]

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
            vals = [self.run(r)["comparisons"][AO_WO][INC]["mean_a"] for r in q[2]]
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
            return self._excl(q[1], q[2])[q[3]]
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
            t = self._excl(q[1], q[2])
            return {"delta": t["rho"], "significant": t["p_value"] < 0.05}
        return None


def sign(x: float) -> int:
    return (x > 1e-12) - (x < -1e-12)


def compare(published: Source, current: Source) -> list[dict]:
    rows = []
    for where, what, q, cited, fmt, *was in CLAIMS:
        before = was[0] if was else cited
        pub, cur = published.get(q), current.get(q)
        pc, cc = published.call(q), current.call(q)
        row = {"where": where, "what": what, "query": list(q), "cited": cited,
               "cited_before_rerun": before,
               "published": list(pub) if isinstance(pub, tuple) else pub,
               "current": list(cur) if isinstance(cur, tuple) else cur,
               "published_rendered": render(pub, fmt), "current_rendered": render(cur, fmt)}
        if not isinstance(pub, tuple):
            row["difference"] = cur - pub
        if before is not None and before != "flat":
            row["before_matched_published"] = row["published_rendered"] == normalise(before)
        if cited is not None:
            row["cited_matches_current"] = row["current_rendered"] == normalise(cited)
        if pc is not None:
            row["sign_changes"] = sign(pc["delta"]) != sign(cc["delta"])
            row["significance_changes"] = pc["significant"] != cc["significant"]
            row["significant_published"], row["significant_current"] = \
                pc["significant"], cc["significant"]
        rows.append(row)
    return rows


def table(rows: list[dict]) -> str:
    out = ["| where | what | doc before | doc now | published | current | difference "
           "| sign/significance changes |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        diff = f"{r['difference']:+.4f}" if "difference" in r else "—"
        call = "—" if "sign_changes" not in r else (
            "**yes**" if r["sign_changes"] or r["significance_changes"] else "no")
        before = r["cited_before_rerun"] or "(not cited)"
        now = r["cited"] or "(not cited)"
        if now != before:
            now = f"**{now}**"
        out.append(f"| {r['where']} | {r['what']} | {before} | {now} "
                   f"| {r['published_rendered']} | {r['current_rendered']} | {diff} | {call} |")
    return "\n".join(out)


def current_path(run: str) -> Path:
    return resolve(f"{run}/results.json", MANIFEST, local_dirs=LOCAL_DIRS, env="DSG_REPRO_LOCAL")


def write_manifest(local: Path, revision: str | None) -> dict:
    """Pin each rerun's hash; ``revision`` is the Hub commit, None until uploaded."""
    from dsg.hub import ARTIFACT_REPO

    manifest = {"repo": ARTIFACT_REPO, "repo_type": "model", "revision": revision,
                "files": [{"name": f"{r}/results.json", "hub_path": f"{HUB_DIR}/{r}/results.json",
                           "bytes": (local / r / "results.json").stat().st_size,
                           "sha256": _sha256(local / r / "results.json")} for r in RUNS]}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def upload(local: Path) -> dict:
    """Push the reruns to ``results-current/`` in one Hub commit and pin it in the manifest."""
    from dsg.hub import ARTIFACT_REPO, _api

    api, token = _api()
    info = api.upload_folder(folder_path=str(local), repo_id=ARTIFACT_REPO, repo_type="model",
                             path_in_repo=HUB_DIR,
                             allow_patterns=[f"{r}/results.json" for r in RUNS], token=token,
                             commit_message="Reading study rerun under current code")
    return write_manifest(local, info.oid)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.study_repro", description=__doc__)
    p.add_argument("--published", default="artifacts/hub/results")
    p.add_argument("--out", default="audit/repro/study_current_code.json")
    p.add_argument("--upload", action="store_true",
                   help="push artifacts/results-current to the Hub and rewrite the manifest")
    args = p.parse_args(argv)
    if args.upload:
        print(json.dumps(upload(Path(LOCAL_DIRS[0])), indent=2))
        return 0
    published = Source(lambda run: Path(args.published) / run / "results.json",
                       "lexicon_pre_0be8e0d")
    current = Source(current_path, "lexicon_current")
    for run in ("pdnc-qwen3b", "pdnc-qwen7b"):
        # The trend replay and the study rerun must agree on the unmodified profile.
        replay = current._trend["runs"][run]["lexicon_current"]["published"]
        for label, key in (("narrative", NARR), ("shuffled", SHUF)):
            for op in ("elaboration", "revision"):
                got = current.run(run)["instrument"][key][op]["rho"]
                assert abs(got - replay[label][op]["rho"]) < 1e-9, (run, label, op)
    rows = compare(published, current)
    stale = [(r["where"], r["what"], r["cited"], r["current_rendered"])
             for r in rows if r.get("cited_matches_current") is False]
    if stale:
        raise SystemExit(f"cited text does not match current results: {stale}")
    inputs = [Path(TREND)] + [Path(args.published) / r / "results.json" for r in RUNS]
    out = {"provenance": provenance("dsg.eval.study_repro", argv, {}, inputs),
           "current_results": {f"{HUB_DIR}/{r}/results.json": _sha256(current_path(r))
                               for r in RUNS},
           "manifest": str(MANIFEST), "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    Path(args.out).with_suffix(".md").write_text(table(rows) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
