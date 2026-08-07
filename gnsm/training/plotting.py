"""Training-curve figures, saved as matched PNG + PDF pairs.

Color/typography follow the same validated palette as
``DNG_Data_Visualization.ipynb`` (categorical slot 1 = blue for train, slot 2 =
orange for val; one shared y-axis; recessive gridlines) so figures from the
data-analysis notebook and the training runs read as one system.
"""

from __future__ import annotations

from pathlib import Path

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASE = "#898781", "#e1e0d9", "#c3c2b7"
SERIES_1_BLUE, SERIES_2_ORANGE = "#2a78d6", "#eb6834"


def plot_loss_curves(
    train_steps: list[int],
    train_losses: list[float],
    out_dir: Path,
    stem: str,
    title: str,
    val_steps: list[int] | None = None,
    val_losses: list[float] | None = None,
    best_step: int | None = None,
    best_val_loss: float | None = None,
) -> tuple[Path, Path]:
    """Save a train[/val] loss-vs-step figure as both PNG and PDF. Returns
    (png_path, pdf_path)."""

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

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(train_steps, train_losses, color=SERIES_1_BLUE, linewidth=2, label="train", zorder=3)
    if val_steps and val_losses:
        ax.plot(val_steps, val_losses, color=SERIES_2_ORANGE, linewidth=2, label="val", zorder=3)
        ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    if best_step is not None:
        # Recessive marker (muted, not a third data color) for the
        # best-val-loss / early-stopping point, per the dataviz palette's
        # "text wears text tokens, never the series color" rule. Anchored at
        # the bottom of the axis, clear of the top-right legend.
        ax.axvline(best_step, color=BASE, linewidth=1.2, linestyle="--", zorder=2)
        label = f"best step={best_step}" + (f"  val={best_val_loss:.3f}" if best_val_loss else "")
        ax.annotate(
            label,
            (best_step, ax.get_ylim()[0]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
            va="bottom",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path
