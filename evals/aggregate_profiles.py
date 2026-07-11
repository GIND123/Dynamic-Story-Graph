"""Combine per-model profiles into one failure-profile table.

Reads evals/results/profiles/*.json (written by run_model_profile.py) and emits:
  - an ASCII table to stdout,
  - evals/results/failure_profile.md   (markdown, for the README/report),
  - evals/results/failure_profile.csv  (for a spreadsheet / plot).

Each row is one model; columns are the three task accuracies and the three
headline failure rates (plus the consistency malformed count, a real signal for
weaker models). Missing/skipped values render as "n/a".
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
PROFILE_DIR = RESULTS_DIR / "profiles"

# (header, path-in-profile) — dotted paths resolved by _dig.
COLUMNS: list[tuple[str, str]] = [
    ("model", "model"),
    ("quote_acc", "tasks.quote_attribution.overall"),
    ("cloze_acc", "tasks.name_cloze.aggregate.accuracy"),
    ("coref_acc", "tasks.coreference.aggregate.accuracy"),
    ("cons_recall", "tasks.consistency.recall"),
    ("subtle_recall", "tasks.consistency.per_difficulty.subtle.recall"),
    ("contradiction", "rates.contradiction"),
    ("location_incon", "rates.location_inconsistency"),
    ("over_flag", "tasks.consistency.over_flag_rate"),  # counters flag-everything
    ("cons_prec", "tasks.consistency.precision"),
    ("entity_drift", "rates.entity_drift"),
    ("drift_src", "rates.entity_drift_source"),
    ("cons_malformed", "tasks.consistency.malformed"),
]


def _dig(obj: dict, dotted: str):
    cur = obj
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int):
        return str(v)
    return "n/a" if v is None else str(v)


def load_rows(profile_dir: Path = PROFILE_DIR) -> list[dict]:
    rows = []
    for path in sorted(profile_dir.glob("*.json")):
        profile = json.loads(path.read_text())
        row = {h: _dig(profile, p) for h, p in COLUMNS}
        # Fallback: compute over_flag from stored fp/tn for profiles written
        # before over_flag_rate existed (fp/(fp+tn) = share of clean wrongly flagged).
        if row.get("over_flag") is None:
            fp = _dig(profile, "tasks.consistency.fp")
            tn = _dig(profile, "tasks.consistency.tn")
            if isinstance(fp, int) and isinstance(tn, int) and (fp + tn):
                row["over_flag"] = fp / (fp + tn)
        rows.append(row)
    return rows


def render_ascii(rows: list[dict]) -> str:
    headers = [h for h, _ in COLUMNS]
    table = [[_fmt(r[h]) for h in headers] for r in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table))
        if table
        else len(headers[i])
        for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, "  ".join("-" * widths[i] for i in range(len(headers)))]
    for row in table:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def render_markdown(rows: list[dict]) -> str:
    headers = [h for h, _ in COLUMNS]
    out = [
        "# Vanilla-LLM failure profile",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r[h]) for h in headers) + " |")
    out += [
        "",
        "_Rates are share-of-failures (higher = worse). `contradiction` and "
        "`location_incon` = 1 − recall on planted cases; `entity_drift` = "
        "cloze wrong-but-valid-character rate (a proxy). Small n — directional._",
    ]
    return "\n".join(out)


def write_csv(rows: list[dict], path: Path) -> None:
    headers = [h for h, _ in COLUMNS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: _fmt(r[h]) for h in headers})


def main() -> int:
    rows = load_rows()
    if not rows:
        print(f"No profiles in {PROFILE_DIR}. Run run_model_profile.py first.")
        return 0
    print(render_ascii(rows))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "failure_profile.md").write_text(render_markdown(rows))
    write_csv(rows, RESULTS_DIR / "failure_profile.csv")
    print(f"\nWrote {RESULTS_DIR / 'failure_profile.md'} and .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
