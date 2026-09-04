"""Figures and tables for the generation experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsg.experiments.report import CONTEXT, INK, INK_SOFT, _save, _style
from dsg.generate.score import compare, curves, score_runs, summarize

ORDER = (
    "base:none",
    "base:last-chapter",
    "base:rolling-summary",
    "base:full-context",
    "base:append-only-state",
    "base:dsg-state",
    "tuned:full-context",
    "tuned:dsg-state",
    "tuned:dsg-repair",
    "base:dsg-hybrid",
    "base:dsg-hybrid-repair",
)

FOCAL = "#2a78d6"    # the system: state under the full calculus
FOIL = "#eb6834"     # its ablation: state that cannot be revised
STRONG = "#1baf7a"   # the strong practical baseline: paste the transcript

COLOR_FOR = {
    "base:none": CONTEXT,
    "base:last-chapter": CONTEXT,
    "base:rolling-summary": CONTEXT,
    "base:full-context": STRONG,
    "base:append-only-state": FOIL,
    "base:dsg-state": FOCAL,
    "tuned:full-context": STRONG,
    "tuned:dsg-state": FOCAL,
    "tuned:dsg-repair": FOCAL,
    "base:dsg-hybrid": FOCAL,
    "base:dsg-hybrid-repair": FOCAL,
}
LABEL = {
    "base:none": "no memory",
    "base:last-chapter": "last chapter",
    "base:rolling-summary": "rolling summary",
    "base:full-context": "full context",
    "base:append-only-state": "state, no revision",
    "base:dsg-state": "DSG state",
    "tuned:full-context": "tuned + full context",
    "tuned:dsg-state": "tuned + DSG state",
    "tuned:dsg-repair": "tuned + DSG + guard",
    "base:dsg-hybrid": "DSG + recent text",
    "base:dsg-hybrid-repair": "DSG + recent + guard",
}
# Drawn with a marker and full weight; everything else is context grey.
EMPHASISED = (
    "base:full-context", "base:append-only-state", "base:dsg-state",
    "tuned:full-context", "tuned:dsg-state", "tuned:dsg-repair",
    "base:dsg-hybrid", "base:dsg-hybrid-repair",
)
# Dashed where the backbone is the fine-tuned one, so variant reads off the line.
TUNED = tuple(c for c in ORDER if c.startswith("tuned:"))


def figure_violation_curve(payload: dict, out: Path) -> Path:
    """The headline: how much of the planted canon each condition has broken."""
    scores = score_runs(payload)
    series = curves(scores)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for condition in ORDER:
        ys = series.get(condition)
        if not ys:
            continue
        xs = list(range(1, len(ys) + 1))
        focal = condition in EMPHASISED
        ax.plot(
            xs, ys, color=COLOR_FOR[condition],
            linewidth=2.2 if focal else 1.2, alpha=1.0 if focal else 0.55,
            marker="o" if focal else None, markersize=4,
            linestyle=(0, (5, 2)) if condition in TUNED else "-",
            zorder=3 if focal else 2,
        )
        ax.text(xs[-1] + 0.12, ys[-1], LABEL[condition], fontsize=8, va="center",
                color=COLOR_FOR[condition] if focal else INK_SOFT)
    ax.set_xlim(1, len(next(iter(series.values()))) + 6.5)
    ax.set_ylim(bottom=0)
    _style(
        ax,
        xlabel="chapter",
        ylabel="share of planted canon contradicted",
        title="Canon broken as the story gets longer (lower is better)",
    )
    fig.tight_layout()
    return _save(fig, out)


def figure_final_bars(payload: dict, out: Path) -> Path:
    scores = score_runs(payload)
    panels = [
        ("violation_rate", "Canon contradicted by the end", "lower is better"),
        ("restatement_rate", "Canon actively restated", "higher is better"),
        ("prompt_tokens_last", "Context tokens at the final chapter",
         "lower is cheaper"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    present = [c for c in ORDER if any(s.condition == c for s in scores)]
    for ax, (metric, title, note) in zip(axes, panels, strict=False):
        vals = []
        for condition in present:
            rows = [s.as_dict()[metric] for s in scores if s.condition == condition]
            vals.append(sum(rows) / len(rows) if rows else 0.0)
        ax.barh(range(len(present)), vals,
                color=[COLOR_FOR[c] for c in present], height=0.62, zorder=3)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([LABEL[c] for c in present], fontsize=8)
        ax.invert_yaxis()
        span = max(vals) if max(vals) else 1.0
        for i, v in enumerate(vals):
            ax.text(v + span * 0.02, i,
                    f"{v:,.0f}" if metric in ("total_chars", "prompt_tokens_last")
                    else f"{v:.3f}",
                    va="center", fontsize=8, color=INK)
        ax.set_xlim(0, span * 1.25)
        _style(ax, xlabel=note, title=title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out)


def _comparison_table(comparisons: dict) -> str:
    lines = ["| comparison | metric | mean Δ | 95% CI | excludes 0 | wins | n |",
             "|---|---|---|---|---|---|---|"]
    for pair, entry in comparisons.items():
        n = int(entry.get("n_stories", 0))
        for metric in ("violation_rate", "restatement_rate",
                       "mean_first_violation", "total_chars",
                       "prompt_tokens_last"):
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


def _examples(payload: dict, limit: int = 12) -> str:
    rows = [
        r for r in payload["records"]
        if r.get("evidence")
        and r["condition"] in ("base:full-context", "base:rolling-summary",
                               "base:none", "base:append-only-state")
    ]
    rows.sort(key=lambda r: -r["chapter"])
    if not rows:
        return "_none recorded_"
    lines = ["| condition | ch | fact | contradicting text |", "|---|---|---|---|"]
    for r in rows[:limit]:
        for fact_id, why in list(r["evidence"].items())[:1]:
            lines.append(
                f"| {r['condition']} | {r['chapter']} | `{fact_id}` | "
                f"…{why[:130].replace('|', '/')}… |"
            )
    return "\n".join(lines)


def build_report(generation_json: Path, out_dir: Path) -> Path:
    payload = json.loads(Path(generation_json).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = out_dir / "figures"
    figure_violation_curve(payload, figs / "gen1-canon-violation")
    figure_final_bars(payload, figs / "gen2-final")

    scores = score_runs(payload)
    meta = payload["meta"]
    body = [
        "# Generation: consistency at length",
        "",
        f"{meta['stories']} stories × {len(meta['conditions'])} memory conditions × "
        f"{meta['chapters']} chapters, {meta['model']}. "
        f"Canon is stated in chapter 1 only; from chapter 2 each condition keeps "
        f"whatever its memory carries. Violations are deterministic string tests "
        f"against facts we planted, so no model or human judgement enters the "
        f"measurement.",
        "",
        "## Means over stories",
        "",
        summarize(scores, ORDER),
        "",
        "## Paired comparisons (bootstrap over stories)",
        "",
        _comparison_table(compare(scores)),
        "",
        "## Continuity errors on real generated text",
        "",
        _examples(payload),
        "",
        "## Figures",
        "",
        "![canon](figures/gen1-canon-violation.png)",
        "![final](figures/gen2-final.png)",
        "",
    ]
    path = out_dir / "generation.md"
    path.write_text("\n".join(body))
    return path
