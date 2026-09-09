"""E1 reporting -- tables and figures for prefix-causal quote attribution.

Reuses the reading study's palette and paired bootstrap so the two studies are
visually and statistically the same object, and so the statistics need no fresh
validation.

    python -m dsg.eval.attribution_report --result artifacts/e1/pilot-qwen7b.json \
        --out artifacts/report/e1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsg.eval.bootstrap import paired_bootstrap
from dsg.experiments.report import CONTEXT, FOCAL, FOIL, GRID, INK, INK_SOFT, ORACLE, _style

# Published non-causal reference points. Both are read off Table 1 of
# Michel, Epure, Hennequin & Cerisara (2024), arXiv:2406.11380, and are shown as
# reference lines only -- they are NOT a like-for-like comparison, because every
# published number sees text after the quote and E1's causal conditions do not.
PUBLISHED = {
    "Llama-3 8B (non-causal)": {"overall": 0.906, "non_explicit": 0.891},
    "BookNLP+ (non-causal)": {"overall": 0.785, "non_explicit": 0.689},
}

CONDITION_ORDER = ["prior", "recency", "text-causal", "state-causal", "oracle-noncausal"]
CONDITION_LABEL = {
    "prior": "prior\n(candidates only)",
    "recency": "recency\n(last speaker)",
    "text-causal": "text\n(causal prefix)",
    "state-causal": "text + state\n(causal)",
    "oracle-noncausal": "oracle\n(non-causal)",
}
COLOR_FOR = {
    "prior": CONTEXT,
    "recency": CONTEXT,
    "text-causal": FOIL,
    "state-causal": FOCAL,
    "oracle-noncausal": ORACLE,
}


def _series(result: dict, condition: str, metric: str) -> list[float]:
    """One value per novel, in a stable novel order, for paired comparison."""
    return [
        float(result["per_novel"][n][condition][metric])
        for n in sorted(result["per_novel"])
        if condition in result["per_novel"][n]
    ]


def _present(result: dict) -> list[str]:
    seen = {c for n in result["per_novel"].values() for c in n}
    return [c for c in CONDITION_ORDER if c in seen]


def _mean(result: dict, condition: str, metric: str) -> float:
    vals = _series(result, condition, metric)
    return sum(vals) / len(vals) if vals else 0.0


def figure_ladder(result: dict, out: Path) -> Path:
    """Accuracy by condition, overall and non-explicit, against published lines."""
    conds = _present(result)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    panels = [("accuracy", "overall", "All quotes"),
              ("accuracy_non_explicit", "non_explicit", "Non-explicit only")]

    for ax, (metric, pubkey, title) in zip(axes, panels, strict=True):
        vals = [_mean(result, c, metric) for c in conds]
        y = range(len(conds))
        ax.barh(list(y), vals, color=[COLOR_FOR[c] for c in conds], height=0.62, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels([CONDITION_LABEL[c] for c in conds], fontsize=8)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
        for name, ref in PUBLISHED.items():
            ax.axvline(ref[pubkey], color=INK_SOFT, linestyle=":", linewidth=1.1, zorder=2)
            ax.text(ref[pubkey], -0.75, name.split(" (")[0], rotation=90,
                    fontsize=6.5, color=INK_SOFT, ha="right", va="bottom")
        ax.set_xlim(0, 1.0)
        _style(ax, xlabel="speaker accuracy", title=title)

    fig.suptitle(
        "Prefix-causal quote attribution on PDNC "
        "(dotted lines are published NON-CAUSAL systems, not a like-for-like comparison)",
        fontsize=8.5, color=INK_SOFT, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def figure_causality_cost(result: dict, out: Path) -> Path:
    """Per-novel oracle-minus-causal gap: what reading forward is worth."""
    novels = sorted(result["per_novel"])
    a = _series(result, "state-causal", "accuracy")
    b = _series(result, "oracle-noncausal", "accuracy")
    if not a or not b or len(a) != len(b):
        return out
    fig, ax = plt.subplots(figsize=(6.6, max(3.0, 0.28 * len(novels) + 1.2)))
    for i, (va, vb) in enumerate(zip(a, b, strict=True)):
        ax.plot([va, vb], [i, i], color=GRID, linewidth=1.6, zorder=2)
        ax.scatter([va], [i], color=FOCAL, s=30, zorder=3)
        ax.scatter([vb], [i], color=ORACLE, s=30, zorder=3)
    ax.set_yticks(range(len(novels)))
    ax.set_yticklabels([n[:26] for n in novels], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    _style(ax, xlabel="speaker accuracy",
           title="What reading forward buys, per novel (blue causal, green oracle)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def comparisons(result: dict) -> list[dict]:
    """The contrasts the design turns on, paired over novels."""
    pairs = [
        ("state-causal", "text-causal", "does the state add anything?"),
        ("text-causal", "recency", "does prefix text beat a trivial heuristic?"),
        ("recency", "prior", "does the last speaker help at all?"),
        ("oracle-noncausal", "state-causal", "the price of causality"),
    ]
    rows = []
    for a, b, question in pairs:
        va, vb = _series(result, a, "accuracy"), _series(result, b, "accuracy")
        if not va or not vb or len(va) != len(vb):
            continue
        r = paired_bootstrap(va, vb)
        rows.append({
            "comparison": f"{a} vs {b}",
            "question": question,
            "mean_delta": r.mean_delta,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "excludes_zero": r.excludes_zero,
            "wins_a": r.wins_a,
            "wins_b": r.wins_b,
            "n": len(va),
        })
    return rows


def write_markdown(result: dict, rows: list[dict], figures: list[Path], out: Path) -> Path:
    conds = _present(result)
    novels = sorted(result["per_novel"])
    n_quotes = int(sum(
        result["per_novel"][n][conds[0]]["n"] for n in novels
    )) if conds else 0

    lines = [
        "# E1 -- prefix-causal quote attribution on PDNC",
        "",
        f"{len(novels)} novel(s), {n_quotes:,} quotes per condition, "
        f"backend `{result.get('backend')}`, window {result.get('window_chars')} chars, "
        f"{result.get('seconds')}s.",
        "",
        "Every causal condition is whitelist-guarded: each evidence block is proven to be "
        "a slice of `text[:quote_start]` or of the spoken words. "
        f"Repeated-text diagnostics (a line the novel says twice, seen in its earlier "
        f"occurrence -- not leakage): {result.get('repeated_text_diagnostics', 0)}.",
        "",
        "## Means over novels",
        "",
        "| condition | n | accuracy | explicit | non-explicit |",
        "|---|---|---|---|---|",
    ]
    for c in conds:
        lines.append(
            f"| {c} | {int(_mean(result, c, 'n'))} | {_mean(result, c, 'accuracy'):.3f} "
            f"| {_mean(result, c, 'accuracy_explicit'):.3f} "
            f"| {_mean(result, c, 'accuracy_non_explicit'):.3f} |"
        )

    lines += [
        "",
        "## Published reference points (NON-CAUSAL -- not a like-for-like comparison)",
        "",
        "| system | overall | non-explicit | setting |",
        "|---|---|---|---|",
        "| BookNLP+ | 0.785 | 0.689 | non-causal |",
        "| Llama-3 8B zero-shot | 0.906 | 0.891 | non-causal |",
        "",
        "Source: Michel, Epure, Hennequin & Cerisara (2024), arXiv:2406.11380, Table 1. "
        "These systems read text *after* each quote; E1's causal conditions do not. "
        "The comparison bounds the task, it does not rank the systems.",
        "",
        "## Paired comparisons (bootstrap over novels)",
        "",
        "| comparison | question | mean D | 95% CI | excludes 0 | wins |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['comparison']} | {r['question']} | {r['mean_delta']:+.4f} "
            f"| [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
            f"| {'**yes**' if r['excludes_zero'] else 'no'} "
            f"| {r['wins_a']}-{r['wins_b']} |"
        )

    if figures:
        lines += ["", "## Figures", ""]
        lines += [f"![{f.stem}]({f.name})" for f in figures]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/report/e1"))
    args = ap.parse_args(argv)

    result = json.loads(args.result.read_text())
    figs = [
        figure_ladder(result, args.out / "e1-ladder"),
        figure_causality_cost(result, args.out / "e1-causality-cost"),
    ]
    rows = comparisons(result)
    md = write_markdown(result, rows, figs, args.out / "attribution.md")
    print(f"wrote {md}")
    for f in figs:
        print(f"wrote {f}")
    for r in rows:
        print(
            f"  {r['comparison']:38} {r['mean_delta']:+.4f} "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
            f"{'*' if r['excludes_zero'] else ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
