"""Turn per-model context-experiment JSONs into the paper's figures + table.

Reads evals/results/context/*.json and writes:
  - evals/results/context/context_summary.md / .csv  (advertised vs effective)
  - evals/results/context/plots/*.png                (curves; needs matplotlib)

The table is always written; plots are skipped with a note if matplotlib is
absent (`pip install matplotlib`).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RESULT_DIR = Path(__file__).parent / "results" / "context"
PLOT_DIR = RESULT_DIR / "plots"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return "n/a" if v is None else str(v)


def load_profiles(result_dir: Path = RESULT_DIR) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(result_dir.glob("*.json"))]


# Parameter counts (not stored in the JSON) for the human-readable table.
_PARAMS = {
    "gemma2:9b": "9B",
    "qwen2.5:7b": "7B",
    "llama3.1:8b": "8B",
    "mistral:7b": "7B",
    "yi:9b": "9B",
    "qwen2.5:72b": "72B",
    "phi3.5": "3.8B",
    "qwen2.5:3b": "3B",
    "llama3.2:3b": "3B",
    "gemma2:2b": "2B",
}


def _effective_of(pts: list[dict], frac: float = 0.8) -> tuple[float | None, bool]:
    """Largest length still >= frac*peak with no earlier drop; (tokens, lower_bound).

    Stdlib mirror of evals.metrics.long_context.effective_context, duplicated on
    purpose so this reporting script stays dependency-free (it runs from just the
    JSON, with no Neo4j/baselines import). Keep the two definitions in sync.
    """
    if not pts:
        return None, False
    ordered = sorted(pts, key=lambda p: p["actual_tokens"])
    threshold = frac * max(p["overall"] for p in ordered)
    last_good: float | None = None
    for p in ordered:
        if p["overall"] < threshold:
            return last_good, False
        last_good = p["actual_tokens"]
    return last_good, True


def _fmt_effective(tokens: float | None, is_lower_bound: bool) -> str:
    if tokens is None:
        return "<first rung"
    return f"≥{round(tokens):,}" if is_lower_bound else f"{round(tokens):,}"


def _fmt_gap(advertised, tokens, is_lower_bound: bool) -> str:
    # Only meaningful where the effective limit was actually reached in range.
    if is_lower_bound or not tokens or not advertised:
        return "—"
    return f"{advertised / tokens:.1f}×"


def _md_cell(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"  # thousands separators for the readable table
    if isinstance(v, float):
        return f"{v:.3f}"
    return "n/a" if v is None else str(v)


def summary_rows(profiles: list[dict]) -> list[dict]:
    """One row per model: advertised window vs the effective (reliably-usable) one."""
    rows = []
    for prof in profiles:
        model = prof.get("model")
        qa = (prof.get("curves") or {}).get("quote_attribution") or {}
        pts = qa.get("points") or []
        advertised = prof.get("advertised_window")
        peak = max((p["overall"] for p in pts), default=None)
        deepest = max((p["actual_tokens"] for p in pts), default=None)
        eff_tokens, eff_lb = _effective_of(pts, frac=0.8)

        def _rnd(v):
            return None if v is None else round(v)

        rows.append(
            {
                "model": model,
                "params": _PARAMS.get(model, "?"),
                "advertised": advertised,
                "peak_overall": peak,
                "effective_window": _fmt_effective(eff_tokens, eff_lb),
                "reality_gap": _fmt_gap(advertised, eff_tokens, eff_lb),
                "deepest_tested": _rnd(deepest),
                # raw / provenance fields (CSV)
                "effective_tokens": _rnd(eff_tokens),
                "effective_is_lower_bound": eff_lb,
                "failure_point_50pct": _rnd(qa.get("failure_point")),
            }
        )
    return rows


# (row-key, column header) for the human-readable table the report embeds.
_MD_COLS = [
    ("model", "model"),
    ("params", "params"),
    ("advertised", "advertised window"),
    ("peak_overall", "peak accuracy"),
    ("effective_window", "effective window (≥80% of peak)"),
    ("reality_gap", "advertised ÷ effective"),
    ("deepest_tested", "deepest tested"),
]
# Fuller machine-readable dump (no thousands separators, valid CSV).
_CSV_COLS = [
    "model",
    "params",
    "advertised",
    "peak_overall",
    "effective_tokens",
    "effective_is_lower_bound",
    "reality_gap",
    "failure_point_50pct",
    "deepest_tested",
]


# ---------------------------------------------------------------------------
# Per-window matrices: the metric value at each context-window band, per model.
# Banded by the rung's TARGET length (the intended window); actual tokens run a
# little lower and are in the raw tables below. The deepest 32K rung sits beyond
# this 16K-capped grid (the 72B's ~29K quote point is in the failure-point table).
# ---------------------------------------------------------------------------
_WINDOW_BANDS = [1000, 2000, 4000, 8000, 16000]
_WINDOW_LABELS = ["1K", "2K", "4K", "8K", "16K"]


def _target_band(target_tokens: float) -> int | None:
    if target_tokens >= 24000:
        return None  # 32K rung — beyond the 16K grid
    return min(
        range(len(_WINDOW_BANDS)), key=lambda i: abs(_WINDOW_BANDS[i] - target_tokens)
    )


def _param_billions(model: str) -> float:
    try:
        return float(_PARAMS.get(model, "0B").rstrip("Bb"))
    except ValueError:
        return 0.0


def _by_window(profiles: list[dict]) -> list[dict]:
    """Profiles ordered by advertised window then size (small -> large) — readable rows."""
    return sorted(
        profiles,
        key=lambda p: (
            p.get("advertised_window") or 0,
            _param_billions(p.get("model")),
        ),
    )


def _acc_matrix_lines(
    profiles: list[dict], curve: str, value_key: str, title: str
) -> list[str]:
    lines = [
        "",
        f"### {title}",
        "",
        "| model | " + " | ".join(_WINDOW_LABELS) + " |",
        "| --- | " + " | ".join("---" for _ in _WINDOW_LABELS) + " |",
    ]
    for prof in _by_window(profiles):
        cells = ["—"] * len(_WINDOW_BANDS)
        for p in (prof.get("curves") or {}).get(curve, {}).get("points", []):
            b = _target_band(p["target_tokens"])
            if b is not None:
                cells[b] = f"{p[value_key]:.2f}"
        lines.append(f"| {prof.get('model')} | " + " | ".join(cells) + " |")
    return lines


def _needle_matrix_lines(profiles: list[dict], curve: str, title: str) -> list[str]:
    lines = [
        "",
        f"### {title} — detection recall / over-flag (avg over depths)",
        "",
        "| model | " + " | ".join(_WINDOW_LABELS) + " |",
        "| --- | " + " | ".join("---" for _ in _WINDOW_LABELS) + " |",
    ]
    for prof in _by_window(profiles):
        agg: dict[int, list] = {}
        for p in (prof.get("curves") or {}).get(curve, {}).get("points", []):
            b = _target_band(p["target_tokens"])
            if b is not None:
                agg.setdefault(b, []).append(
                    (p["detection_recall"], p["over_flag_rate"])
                )
        cells = ["—"] * len(_WINDOW_BANDS)
        for b, vs in agg.items():
            r = sum(x[0] for x in vs) / len(vs)
            o = sum(x[1] for x in vs) / len(vs)
            cells[b] = f"{r:.2f}/{o:.2f}"
        lines.append(f"| {prof.get('model')} | " + " | ".join(cells) + " |")
    return lines


def _per_window_matrix_lines(profiles: list[dict]) -> list[str]:
    """One compact matrix per metric (rows = models by window, cols = 1K..16K)."""
    out = ["", "## Metric value by context window (per-window matrices)"]
    out += _acc_matrix_lines(
        profiles, "quote_attribution", "overall", "Quote attribution (overall accuracy)"
    )
    out += _acc_matrix_lines(
        profiles, "coreference", "accuracy", "Coreference (accuracy)"
    )
    out += _acc_matrix_lines(
        profiles, "name_cloze", "accuracy", "Name cloze (accuracy)"
    )
    out += _needle_matrix_lines(profiles, "consistency_needle", "Consistency needle")
    out += _needle_matrix_lines(
        profiles, "location_needle", "Location-inconsistency needle"
    )
    return out


def _raw_attribution_lines(profiles: list[dict]) -> list[str]:
    """Every measured quote-attribution value, one row per (model, length) — the
    raw multi-value-per-length data, straight from the JSON (nothing derived)."""
    lines = [
        "",
        "## Raw quote-attribution accuracy by context length",
        "",
        "| model | context length (actual tok) | overall | explicit | non-explicit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for prof in profiles:
        model = prof.get("model")
        qa = (prof.get("curves") or {}).get("quote_attribution") or {}
        for p in qa.get("points") or []:
            lines.append(
                f"| {model} | {round(p['actual_tokens']):,} | {p['overall']:.3f} | "
                f"{p['explicit']:.3f} | {p['non_explicit']:.3f} |"
            )
    return lines


def _raw_needle_lines(profiles: list[dict], curve_key: str, title: str) -> list[str]:
    """Every measured needle value (recall/over-flag) per (model, length, depth).
    Shared by the general consistency needle and the location-inconsistency needle."""
    lines = [
        "",
        f"## {title}",
        "",
        "| model | context length (actual tok) | depth | detection recall | over-flag rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for prof in profiles:
        model = prof.get("model")
        nd = (prof.get("curves") or {}).get(curve_key) or {}
        for p in nd.get("points") or []:
            lines.append(
                f"| {model} | {round(p['actual_tokens']):,} | {p['depth']:.0%} | "
                f"{p['detection_recall']:.2f} | {p['over_flag_rate']:.2f} |"
            )
    return lines


def _raw_cloze_lines(profiles: list[dict]) -> list[str]:
    """Every measured name-cloze value, one row per (model, length)."""
    lines = [
        "",
        "## Raw name-cloze results by context length",
        "",
        "| model | context length (actual tok) | accuracy | entity drift | miss | abstain |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for prof in profiles:
        model = prof.get("model")
        cl = (prof.get("curves") or {}).get("name_cloze") or {}
        for p in cl.get("points") or []:
            lines.append(
                f"| {model} | {round(p['actual_tokens']):,} | {p['accuracy']:.3f} | "
                f"{p['entity_drift_rate']:.3f} | {p['miss_rate']:.3f} | "
                f"{p['abstain_rate']:.3f} |"
            )
    return lines


def _raw_coref_lines(profiles: list[dict]) -> list[str]:
    """Every measured coreference value, one row per (model, length)."""
    lines = [
        "",
        "## Raw coreference / entity-drift results by context length",
        "",
        "| model | context length (actual tok) | accuracy | entity drift | split (pos-miss) | merge (false) | abstain |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for prof in profiles:
        model = prof.get("model")
        co = (prof.get("curves") or {}).get("coreference") or {}
        for p in co.get("points") or []:
            lines.append(
                f"| {model} | {round(p['actual_tokens']):,} | {p['accuracy']:.3f} | "
                f"{p['entity_drift_rate']:.3f} | {p['positive_miss_rate']:.3f} | "
                f"{p['false_merge_rate']:.3f} | {p['abstain_rate']:.3f} |"
            )
    return lines


def write_table(rows: list[dict], profiles: list[dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    headers = [h for _, h in _MD_COLS]
    keys = [k for k, _ in _MD_COLS]
    md = [
        "# Advertised vs effective context window (quote-attribution task)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        md.append("| " + " | ".join(_md_cell(r[k]) for k in keys) + " |")
    md += [
        "",
        "**How to read this.** *Effective window* = the largest input length at "
        "which the model still scored within 20% of its own best (≥80% of peak), "
        "with no earlier drop — i.e. how far it can read and still track who said "
        "what. `≥N` means accuracy never fell that far in the tested range, so the "
        "true limit is at least N (we ran out of GPU/window before finding it). "
        "*Advertised ÷ effective* is the reality gap, shown only where the limit "
        "was actually reached; `—` means the model held to the deepest length we "
        "could test.",
        "",
        "_No model fully collapsed in range: accuracy never fell below 50% of its "
        "own peak. Degradation on this task is steady erosion, not a cliff, so the "
        "effective window above — not a collapse point — is the practical measure. "
        "Raw per-model curves are in the sibling `*.json` files._",
    ]
    md += _per_window_matrix_lines(profiles)
    md += _raw_attribution_lines(profiles)
    md += _raw_cloze_lines(profiles)
    md += _raw_coref_lines(profiles)
    md += _raw_needle_lines(
        profiles,
        "consistency_needle",
        "Raw consistency-needle results by context length and needle depth",
    )
    md += _raw_needle_lines(
        profiles,
        "location_needle",
        "Raw location-inconsistency needle results by context length and depth",
    )
    (RESULT_DIR / "context_summary.md").write_text("\n".join(md))
    with (RESULT_DIR / "context_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in _CSV_COLS})


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_quote_curves(profiles: list[dict], plt) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    # Combined overall-accuracy overlay across models.
    fig, ax = plt.subplots(figsize=(8, 5))
    for prof in profiles:
        qa = (prof.get("curves") or {}).get("quote_attribution") or {}
        pts = qa.get("points") or []
        if not pts:
            continue
        xs = [p["actual_tokens"] for p in pts]
        ax.plot(xs, [p["overall"] for p in pts], marker="o", label=prof.get("model"))
    ax.set_xscale("log")
    ax.set_xlabel("input context length (tokens, actual)")
    ax.set_ylabel("quote-attribution accuracy")
    ax.set_title("Story-tracking accuracy vs context length (vanilla LLMs)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "quote_curves_overall.png", dpi=130)
    plt.close(fig)

    # Per-model: overall + non-explicit (long-range) accuracy vs length.
    for prof in profiles:
        qa = (prof.get("curves") or {}).get("quote_attribution") or {}
        pts = qa.get("points") or []
        if not pts:
            continue
        xs = [p["actual_tokens"] for p in pts]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, [p["overall"] for p in pts], marker="o", label="overall")
        ax.plot(
            xs,
            [p["non_explicit"] for p in pts],
            marker="s",
            label="non-explicit (long-range)",
        )
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("input context length (tokens, actual)")
        ax.set_ylabel("accuracy")
        ax.set_title(
            f"{prof.get('model')} — advertised {prof.get('advertised_window')} tok"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            PLOT_DIR / f"{prof.get('model').replace(':', '_')}_quote.png", dpi=130
        )
        plt.close(fig)


def main() -> int:
    profiles = load_profiles()
    if not profiles:
        print(
            f"No context profiles in {RESULT_DIR}. Run run_context_experiment.py first."
        )
        return 0
    rows = summary_rows(profiles)
    write_table(rows, profiles)
    print(f"Wrote {RESULT_DIR / 'context_summary.md'} and .csv ({len(rows)} models)")
    plt = _plt()
    if plt is None:
        print("matplotlib not installed — skipped plots (pip install matplotlib).")
    else:
        plot_quote_curves(profiles, plt)
        print(f"Wrote plots to {PLOT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
