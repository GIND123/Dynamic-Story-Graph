"""Figure for the Stage C condition comparison, saved as matched PNG + PDF.

Deliberately plots the bootstrap confidence interval, not a bare bar: the
whole point of the experiment is whether the conditions are *distinguishable*,
and a bar chart without intervals invites reading noise as signal. Per-seed
points are overlaid so the reader can see the actual spread rather than trust
a summary statistic.

Palette matches gnsm/training/plotting.py and DNG_Data_Visualization.ipynb so
every figure in the project reads as one system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gnsm.training.plotting import BASE, GRID, INK, INK2, MUTED, SERIES_1_BLUE, SURFACE

# Slot 2 (orange) and a muted neutral for the controls -- the treatment keeps
# slot 1 so it is the visually primary series.
SERIES_2_ORANGE = "#eb6834"
CONTROL_GREY = "#898781"

_CONDITION_COLORS = {
    "real": SERIES_1_BLUE,
    "shuffled": SERIES_2_ORANGE,
    "zero": CONTROL_GREY,
}


def plot_conditions(summary: dict[str, Any], out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Two panels, because one would mislead.

    Left: per-condition means with CIs. Right: the *paired* per-seed
    differences. Both are needed -- this is a paired design, and marginal CIs
    that visibly overlap do NOT imply the paired difference includes zero
    (seed-level variance is shared between conditions and cancels in the
    difference). Showing only the left panel would hide a real effect; showing
    only the right would hide the absolute scale.
    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": BASE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    conditions = list(summary["per_condition"])
    means = [summary["per_condition"][c]["mean"] for c in conditions]
    lows = [summary["per_condition"][c]["ci_low"] for c in conditions]
    highs = [summary["per_condition"][c]["ci_high"] for c in conditions]
    colors = [_CONDITION_COLORS.get(c, CONTROL_GREY) for c in conditions]

    comparisons = summary.get("comparisons", {})
    n_panels = 2 if comparisons else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(10.4 if comparisons else 6.4, 4.6))
    ax = axes[0] if comparisons else axes
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Dot-and-interval, NOT bars. The y-axis has to be zoomed (these losses
    # cluster within ~0.1 of each other, and a 0-based axis would hide whether
    # the conditions differ at all -- the actual question). Bars on a truncated
    # axis imply a zero baseline that isn't there and visually exaggerate small
    # gaps; a dot with an interval carries no such implication.
    positions = range(len(conditions))
    for pos, mean, low, high, color in zip(positions, means, lows, highs, colors, strict=True):
        ax.plot(
            [pos, pos], [low, high], color=color, linewidth=2.4, zorder=3, solid_capstyle="round"
        )
        ax.scatter([pos], [mean], s=90, color=color, zorder=5, edgecolor=SURFACE, linewidth=1.6)

    # Per-seed points, jittered off-centre so they never hide the mean marker.
    for pos, condition in zip(positions, conditions, strict=True):
        values = summary["per_condition"][condition]["values"]
        ax.scatter(
            [pos + 0.16] * len(values),
            values,
            s=16,
            facecolor=SURFACE,
            edgecolor=MUTED,
            linewidth=0.9,
            zorder=4,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(conditions)
    ax.set_xlim(-0.5, len(conditions) - 0.3)
    ax.set_ylabel("gold-continuation val loss (lower is better)")
    n_seeds = summary["per_condition"][conditions[0]]["n_seeds"]
    ax.set_title(
        f"GNSM Stage C — state conditioning ({n_seeds} seeds)",
        loc="left",
        fontsize=11,
        color=INK,
        pad=14,
    )
    ax.text(
        0.0,
        1.02,
        "dot = mean, bar = 95% bootstrap CI, small points = per-seed runs",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    span = max(highs) - min(lows)
    pad = max(span * 0.35, 0.01)
    ax.set_ylim(min(lows) - pad, max(highs) + pad)

    if comparisons:
        _plot_paired_differences(axes[1], comparisons)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _plot_paired_differences(ax: Any, comparisons: dict[str, Any]) -> None:
    """Paired per-seed differences with their bootstrap CIs.

    A CI that excludes the zero line is the actual evidence of an effect in a
    paired design -- this panel is what the reader should judge, not the
    overlap of the marginal CIs on the left.
    """

    names = list(comparisons)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, linewidth=1.2, linestyle="--", zorder=2)

    for row, name in enumerate(names):
        comparison = comparisons[name]
        low = comparison["delta_ci_low"]
        high = comparison["delta_ci_high"]
        mean = comparison["mean_delta"]
        excludes = comparison["delta_ci_excludes_zero"]
        color = SERIES_1_BLUE if excludes else CONTROL_GREY
        ax.plot(
            [low, high], [row, row], color=color, linewidth=2.4, zorder=3, solid_capstyle="round"
        )
        ax.scatter([mean], [row], s=90, color=color, zorder=5, edgecolor=SURFACE, linewidth=1.6)
        ax.annotate(
            "CI excludes 0" if excludes else "CI includes 0",
            (high, row),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK2 if excludes else MUTED,
            va="center",
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("real_vs_", "real - ") for n in names])
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("paired difference in val loss (negative favours 'real')")
    ax.set_title("Paired per-seed differences", loc="left", fontsize=11, color=INK, pad=14)
    ax.text(
        0.0,
        1.02,
        "the evidence for a paired design -- not the overlap of marginal CIs",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    lows = [comparisons[n]["delta_ci_low"] for n in names]
    highs = [comparisons[n]["delta_ci_high"] for n in names]
    span = max(highs + [0]) - min(lows + [0])
    pad = max(span * 0.45, 0.01)
    ax.set_xlim(min(lows + [0]) - pad, max(highs + [0]) + pad * 2.2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm plot-experiment", description="Plot the Stage C condition comparison."
    )
    parser.add_argument("summary", type=Path, help="summary.json from adapter_experiment")
    parser.add_argument("--out-dir", type=Path, default=Path(".gnsm_checkpoints/plots"))
    parser.add_argument("--stem", default="adapter-experiment")
    args = parser.parse_args(argv)

    summary = json.loads(args.summary.read_text())
    png_path, pdf_path = plot_conditions(summary, args.out_dir, args.stem)
    print(f"wrote {png_path}\nwrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
