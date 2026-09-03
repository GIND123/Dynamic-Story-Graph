"""Combine several scored studies into the cross-corpus and cross-scale views.

Two of the headline claims live here rather than inside any single run:

* **Length dependence.** If premature commitment is the mechanism, the gap
  between policies should grow with how much has been read -- large over a
  500,000-character novel, near-zero over a 2,000-token excerpt. Comparing PDNC
  against LitBank tests that across corpora; the prefix curves test it within.
* **Scale.** If explicit structure substitutes for model capacity, the value of
  the calculus should be largest for the smallest model.

Both are differences of paired differences, so both are reported as effect
sizes with bootstrap intervals rather than as raw scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsg.eval.bootstrap import paired_bootstrap
from dsg.experiments.report import CONTEXT, FOCAL, INK, INK_SOFT, _save, _style

SCALE_ORDER = ["qwen1.5b", "qwen3b", "qwen7b", "qwen14b"]
SCALE_LABEL = {"qwen1.5b": "1.5B", "qwen3b": "3B", "qwen7b": "7B", "qwen14b": "14B"}


def load_runs(root: Path, names: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in names:
        path = Path(root) / name / "results.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def paired_effect(payload: dict, metric: str, a: str, b: str):
    """The per-book paired difference a − b for one metric in one run."""
    by_policy: dict[str, dict[str, float]] = {}
    for row in payload["records"]:
        by_policy.setdefault(row["policy"], {})[row["book"]] = float(row.get(metric, 0.0))
    if a not in by_policy or b not in by_policy:
        return None
    books = sorted(set(by_policy[a]) & set(by_policy[b]))
    if not books:
        return None
    return paired_bootstrap(
        [by_policy[a][bk] for bk in books], [by_policy[b][bk] for bk in books]
    )


def _dot_panel(ax, labels, effects, title, xlabel):
    ys = list(range(len(labels)))
    for y, effect in zip(ys, effects, strict=False):
        if effect is None:
            continue
        colour = FOCAL if effect.excludes_zero else CONTEXT
        ax.plot([effect.ci_low, effect.ci_high], [y, y], color=colour, linewidth=2, zorder=3)
        ax.plot([effect.mean_delta], [y], marker="o", markersize=7, color=colour, zorder=4)
        ax.text(
            effect.ci_high, y - 0.28, f"{effect.mean_delta:+.3g}",
            fontsize=7.5, color=INK, va="center",
        )
    ax.axvline(0, color=INK_SOFT, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    _style(ax, xlabel=xlabel, title=title)
    ax.grid(axis="y", visible=False)


def figure_scale(runs: dict[str, dict], out: Path, a: str = "dsg-full", b: str = "append-only"):
    """Effect of the calculus at each model size."""
    present = [(m, runs[f"pdnc-{m}"]) for m in SCALE_ORDER if f"pdnc-{m}" in runs]
    if len(present) < 2:
        return None
    labels = [SCALE_LABEL[m] for m, _ in present]
    panels = [
        ("violations_per_100w", "Structural violations per 100 windows",
         "Δ (negative favours DSG)"),
        ("conll_f1", "Identity CoNLL F1", "Δ (positive favours DSG)"),
        ("mention_acc", "Mention linking accuracy", "Δ (positive favours DSG)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    for ax, (metric, title, xlabel) in zip(axes, panels, strict=False):
        _dot_panel(ax, labels, [paired_effect(p, metric, a, b) for _, p in present], title, xlabel)
    fig.suptitle(
        f"Effect of the revision calculus ({a} − {b}) by model size, "
        "with 95% bootstrap intervals over books",
        fontsize=10, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out)


def figure_length_dependence(runs: dict[str, dict], out: Path,
                             a: str = "dsg-full", b: str = "append-only"):
    """The same effect on full novels versus 2K-token excerpts."""
    pairs = [
        ("LitBank\n~2K-token excerpts", runs.get("litbank-qwen7b")),
        ("PDNC\nfull novels", runs.get("pdnc-qwen7b")),
    ]
    pairs = [(label, p) for label, p in pairs if p]
    if len(pairs) < 2:
        return None
    labels = [label for label, _ in pairs]
    panels = [
        ("violations_per_100w", "Structural violations per 100 windows",
         "Δ (negative favours DSG)"),
        ("conll_f1", "Identity CoNLL F1", "Δ (positive favours DSG)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
    for ax, (metric, title, xlabel) in zip(axes, panels, strict=False):
        _dot_panel(ax, labels, [paired_effect(p, metric, a, b) for _, p in pairs], title, xlabel)
    fig.suptitle(
        "The gap is a function of reading depth, not of the method alone",
        fontsize=10, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, out)


def _table(runs: dict[str, dict], metrics: tuple[str, ...], a: str, b: str) -> str:
    lines = ["| run | books | metric | mean Δ | 95% CI | excludes 0 | wins |",
             "|---|---|---|---|---|---|---|"]
    for name, payload in runs.items():
        n = payload["meta"]["books"]
        for metric in metrics:
            effect = paired_effect(payload, metric, a, b)
            if effect is None:
                continue
            lines.append(
                f"| {name} | {n} | {metric} | {effect.mean_delta:+.4f} | "
                f"[{effect.ci_low:+.4f}, {effect.ci_high:+.4f}] | "
                f"{'**yes**' if effect.excludes_zero else 'no'} | "
                f"{effect.wins_a}–{effect.wins_b} |"
            )
    return "\n".join(lines)


def _yield_table(runs: dict[str, dict]) -> str:
    lines = ["| run | books | windows | parse failures | parse rate |",
             "|---|---|---|---|---|"]
    for name, payload in sorted(runs.items()):
        rows = [r for r in payload["records"] if r["policy"] == "dsg-full"]
        if not rows:
            continue
        windows = sum(r["windows"] for r in rows)
        failures = sum(int(r.get("parse_failures", 0)) for r in rows)
        rate = 1 - failures / windows if windows else 0.0
        lines.append(
            f"| {name} | {len(rows)} | {windows:,} | {failures:,} | {rate:.3f} |"
        )
    return "\n".join(lines)


def build(results_root: Path, out_dir: Path, names: list[str] | None = None) -> Path:
    names = names or [
        "pdnc-qwen7b", "litbank-qwen7b", "pdnc-qwen3b", "pdnc-qwen1.5b", "pdnc-qwen14b",
    ]
    runs = load_runs(results_root, names)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = out_dir / "figures"
    figure_scale(runs, figs / "fig6-model-scale")
    figure_length_dependence(runs, figs / "fig7-length-dependence")

    metrics = (
        "violations_per_100w", "conll_f1", "mention_acc", "rollback", "monotone_fraction",
    )
    body = [
        "# Cross-run comparison",
        "",
        f"Runs included: {', '.join(sorted(runs))}",
        "",
        "## Extraction yield per run",
        "",
        "A smaller model does not merely extract *worse*, it extracts *less*: "
        "a window whose output cannot be parsed contributes nothing to any "
        "policy. The scale comparison below is therefore partly confounded by "
        "yield, so the parse rate is reported alongside it rather than buried.",
        "",
        _yield_table(runs),
        "",
        "## DSG (`dsg-full`) versus the standard incremental pipeline (`append-only`)",
        "",
        _table(runs, metrics, "dsg-full", "append-only"),
        "",
        "## Deferred versus eager commitment (`dsg-full` − `dsg-eager`)",
        "",
        _table(runs, ("rollback", "monotone_fraction", "conll_f1"), "dsg-full", "dsg-eager"),
        "",
        "## Figures",
        "",
        "![scale](figures/fig6-model-scale.png)",
        "![length](figures/fig7-length-dependence.png)",
        "",
    ]
    path = out_dir / "comparison.md"
    path.write_text("\n".join(body))
    return path
