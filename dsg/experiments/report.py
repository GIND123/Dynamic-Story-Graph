"""Tables and publication figures from a scored study.

Figure design follows one rule throughout: at most three hues carry identity and
everything else is context grey. Six policies would need six categorical hues,
which no palette separates safely under colour-vision deficiency, so the
comparison is drawn as *emphasis* -- the system, its foil and the oracle in
colour, the ablation ladder in grey with direct labels.

Palette slots 1-3 of the reference categorical theme; that three-slot subset is
the documented all-pairs-safe set in both light and dark modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FOCAL = "#2a78d6"     # dsg-full  -- the system
FOIL = "#eb6834"      # append-only -- what everyone does today
ORACLE = "#1baf7a"    # retrospective -- the non-causal upper reference
CONTEXT = "#8a8a85"   # ablations
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdcd8"

POLICY_ORDER = [
    "window-only", "append-only", "dsg-merge", "dsg-eager", "dsg-full", "retrospective"
]
POLICY_LABEL = {
    "window-only": "window-only\n(no state)",
    "append-only": "append-only\n(standard)",
    "dsg-merge": "+ identity merge",
    "dsg-eager": "+ revision,\neager commit",
    "dsg-full": "DSG\n(deferred commit)",
    "retrospective": "retrospective\n(non-causal)",
}
COLOR_FOR = {
    "dsg-full": FOCAL, "append-only": FOIL, "retrospective": ORACLE,
    "window-only": CONTEXT, "dsg-merge": CONTEXT, "dsg-eager": CONTEXT,
}


def _style(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def _mean_by_policy(records: list[dict], metric: str) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for row in records:
        acc.setdefault(row["policy"], []).append(float(row.get(metric, 0.0)))
    return {p: sum(v) / len(v) for p, v in acc.items() if v}


def _present(records: list[dict]) -> list[str]:
    seen = {r["policy"] for r in records}
    return [p for p in POLICY_ORDER if p in seen]


def figure_main(records: list[dict], out: Path) -> Path:
    """Three horizontal bar panels: identity, speaker, integrity."""
    panels = [
        ("conll_f1", "Identity CoNLL F1", "higher is better", False),
        ("mention_acc", "Mention linking accuracy\n(probed in reading order)",
         "higher is better", False),
        ("inconsistent_slot_rate", "Self-contradictory slots\n(share of the state)",
         "lower is better", True),
    ]
    policies = _present(records)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, (metric, title, note, _lower_better) in zip(axes, panels, strict=False):
        means = _mean_by_policy(records, metric)
        vals = [means.get(p, 0.0) for p in policies]
        colors = [COLOR_FOR[p] for p in policies]
        y = range(len(policies))
        ax.barh(list(y), vals, color=colors, height=0.62, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels([POLICY_LABEL[p] for p in policies], fontsize=8)
        ax.invert_yaxis()
        span = max(vals) if max(vals) else 1.0
        for i, v in enumerate(vals):
            ax.text(v + span * 0.02, i, f"{v:.3g}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, span * 1.22)
        _style(ax, xlabel=note, title=title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out)


def figure_prefix_curve(curves: dict, out: Path, metric: str = "conll_f1") -> Path:
    """Identity quality against how much of the book has been read."""
    series: dict[str, dict[float, list[float]]] = {}
    for by_policy in curves.values():
        for policy, points in by_policy.items():
            for point in points:
                series.setdefault(policy, {}).setdefault(point["fraction"], []).append(
                    float(point[metric])
                )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for policy in POLICY_ORDER:
        if policy not in series:
            continue
        xs = sorted(series[policy])
        ys = [sum(series[policy][x]) / len(series[policy][x]) for x in xs]
        focal = policy in ("dsg-full", "append-only", "retrospective")
        ax.plot(
            xs, ys, color=COLOR_FOR[policy], linewidth=2.0 if focal else 1.2,
            alpha=1.0 if focal else 0.55,
            linestyle="-" if policy != "retrospective" else (0, (4, 2)),
            zorder=3 if focal else 2,
        )
        ax.text(
            xs[-1] + 0.012, ys[-1], POLICY_LABEL[policy].replace("\n", " "),
            fontsize=7.5, va="center",
            color=COLOR_FOR[policy] if focal else INK_SOFT,
        )
    ax.set_xlim(0, 1.34)
    _style(
        ax, xlabel="fraction of the book read", ylabel="identity CoNLL F1",
        title="State quality degrades with reading depth — unless it can be revised",
    )
    fig.tight_layout()
    return _save(fig, out)


def figure_paired_deltas(records: list[dict], out: Path, metric: str = "conll_f1",
                         a: str = "dsg-full", b: str = "append-only") -> Path:
    """Per-book dumbbells: the same novel under two policies."""
    by_book: dict[str, dict[str, float]] = {}
    for row in records:
        by_book.setdefault(row["book"], {})[row["policy"]] = float(row.get(metric, 0.0))
    books = sorted(
        (bk for bk, v in by_book.items() if a in v and b in v),
        key=lambda bk: by_book[bk][a] - by_book[bk][b],
    )
    if not books:
        return out
    fig, ax = plt.subplots(figsize=(6.6, max(3.2, 0.24 * len(books) + 1.2)))
    for i, book in enumerate(books):
        va, vb = by_book[book][a], by_book[book][b]
        ax.plot([vb, va], [i, i], color=GRID, linewidth=1.6, zorder=2)
        ax.scatter([vb], [i], color=FOIL, s=26, zorder=3)
        ax.scatter([va], [i], color=FOCAL, s=26, zorder=3)
    ax.set_yticks(range(len(books)))
    ax.set_yticklabels(books, fontsize=7)
    _style(ax, xlabel=metric.replace("_", " "),
           title=f"Per-book paired comparison: {a} vs {b}")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=FOCAL, label=a),
            Line2D([], [], marker="o", linestyle="", color=FOIL, label=b),
        ],
        frameon=False, fontsize=8, loc="lower right",
    )
    fig.tight_layout()
    return _save(fig, out)


def figure_revision_profile(profiles: dict, out: Path, policy: str = "dsg-full") -> Path:
    """Where in a book the reader's model is refined versus corrected."""
    bins: dict[int, dict[str, list[float]]] = {}
    for by_policy in profiles.values():
        for entry in by_policy.get(policy, []):
            slot = bins.setdefault(entry["bin"], {"elab": [], "rev": []})
            slot["elab"].append(float(entry["elaboration_rate"]))
            slot["rev"].append(float(entry["revision_rate"]))
    if not bins:
        return out
    xs = sorted(bins)
    elab = [sum(bins[x]["elab"]) / len(bins[x]["elab"]) for x in xs]
    rev = [sum(bins[x]["rev"]) / len(bins[x]["rev"]) for x in xs]
    pos = [x / (max(xs) or 1) for x in xs]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 1.0 / (len(xs) * 1.35)
    ax.bar([p - width / 2 for p in pos], elab, width=width, color=FOCAL,
           label="elaboration (monotone)", zorder=3)
    ax.bar([p + width / 2 for p in pos], rev, width=width, color=FOIL,
           label="revision (rollback)", zorder=3)
    _style(ax, xlabel="position in the book", ylabel="share of update operations",
           title="Information release: refinement early, correction late")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def figure_violation_growth(growth: dict, out: Path) -> Path:
    """Cumulative structural violations against reading position."""
    series: dict[str, dict[float, list[float]]] = {}
    for by_policy in growth.values():
        for policy, trace in by_policy.items():
            n = max(1, len(trace))
            for point in trace:
                frac = round(point["index"] / n, 2)
                series.setdefault(policy, {}).setdefault(frac, []).append(
                    float(point["violations_total"])
                )
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for policy in POLICY_ORDER:
        if policy not in series:
            continue
        xs = sorted(series[policy])
        ys = [sum(series[policy][x]) / len(series[policy][x]) for x in xs]
        focal = policy in ("dsg-full", "append-only", "retrospective")
        ax.plot(xs, ys, color=COLOR_FOR[policy], linewidth=2.0 if focal else 1.1,
                alpha=1.0 if focal else 0.5, zorder=3 if focal else 2)
        if ys:
            ax.text(xs[-1] + 0.01, ys[-1], POLICY_LABEL[policy].replace("\n", " "),
                    fontsize=7.5, va="center",
                    color=COLOR_FOR[policy] if focal else INK_SOFT)
    ax.set_xlim(0, 1.34)
    _style(ax, xlabel="fraction of the book read",
           ylabel="cumulative violations (mean per book)",
           title="Un-revisable state accumulates contradictions as the book goes on")
    fig.tight_layout()
    return _save(fig, out)


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out.with_suffix(".png")


def _comparison_table(comparisons: dict, metrics: tuple[str, ...]) -> str:
    lines = [
        "| comparison | metric | mean Δ | 95% CI | excludes 0 | wins | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for pair, entry in comparisons.items():
        n = int(entry.get("n_books", 0))
        for metric in metrics:
            stat = entry.get(metric)
            if not stat:
                continue
            lines.append(
                f"| {pair.replace('_vs_', ' vs ')} | {metric} | "
                f"{stat['mean_delta']:+.4f} | "
                f"[{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}] | "
                f"{'**yes**' if stat['excludes_zero'] else 'no'} | "
                f"{int(stat['wins_a'])}–{int(stat['wins_b'])} | {n} |"
            )
    return "\n".join(lines)


def _instrument_table(data: dict) -> str:
    rows = data.get("by_narrative_person") or []
    if not rows:
        return "_not computed_"
    lines = [
        "How long a novel holds a figure unnamed before the reader is told who "
        "they are, by narrative person. Unpaired bootstrap on the group means; "
        "the first-person group is small, so read the interval, not the point.",
        "",
        "| metric | first-person | third-person | Δ | 95% CI | excludes 0 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric']} | {r['mean_a']:.3f} (n={r['n_a']}) | "
            f"{r['mean_b']:.3f} (n={r['n_b']}) | {r['delta']:+.3f} | "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
            f"{'**yes**' if r['excludes_zero'] else 'no'} |"
        )
    return "\n".join(lines)


def _trajectory_section(trajectories: dict, n_books: int = 2, rows: int = 12) -> str:
    from dsg.eval.examples import trajectory_markdown

    if not trajectories:
        return "_none recorded_"
    out = []
    for book, entry in list(trajectories.items())[:n_books]:
        out.append(f"### {book}")
        out.append("")
        out.append(trajectory_markdown(entry["rows"][:rows], entry["character"]))
        out.append("")
    return "\n".join(out)


def _examples_table(rows: list[dict]) -> str:
    from dsg.eval.examples import to_markdown

    if not rows:
        return "_none recorded_"
    return to_markdown(rows, limit=20)


def build_report(results_dir: Path, out_dir: Path) -> Path:
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    payload = json.loads((results_dir / "results.json").read_text())
    records = payload["records"]
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = out_dir / "figures"

    figure_main(records, figs / "fig1-main-results")
    if payload.get("curves"):
        figure_prefix_curve(payload["curves"], figs / "fig2-prefix-curve")
    figure_paired_deltas(records, figs / "fig3-paired-deltas")
    if payload.get("profiles"):
        figure_revision_profile(payload["profiles"], figs / "fig4-revision-profile")
    if payload.get("growth"):
        figure_violation_growth(payload["growth"], figs / "fig5-violation-growth")

    from dsg.experiments.run_study import summarize

    meta = payload["meta"]
    body = [
        "# Dynamic Story Graph — results",
        "",
        f"Books: **{meta['books']}** · policies: {', '.join(meta['policies'])} · "
        f"scoring time: {meta['seconds']}s",
        "",
        "## Means over books",
        "",
        summarize(payload),
        "",
        "## Paired comparisons (bootstrap over books)",
        "",
        _comparison_table(
            payload.get("comparisons", {}),
            ("conll_f1", "mention_acc", "mention_acc_answered", "speaker_acc_matched",
             "violations_per_100w", "inconsistent_slot_rate", "rollback", "monotone_fraction",
             "fragmentation", "conflation"),
        ),
        "",
        "## Identity-resolution latency (measurement instrument)",
        "",
        _instrument_table(payload.get("instrument", {})),
        "",
        "## One character, traced (dsg-full)",
        "",
        _trajectory_section(payload.get("trajectories", {})),
        "",
        "## Revision events on real text (dsg-full)",
        "",
        _examples_table(payload.get("examples", [])),
        "",
        "## Figures",
        "",
        "![main](figures/fig1-main-results.png)",
        "![prefix](figures/fig2-prefix-curve.png)",
        "![paired](figures/fig3-paired-deltas.png)",
        "![revision](figures/fig4-revision-profile.png)",
        "![violations](figures/fig5-violation-growth.png)",
        "",
    ]
    path = out_dir / "results.md"
    path.write_text("\n".join(body))
    return path
