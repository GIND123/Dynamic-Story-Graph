import importlib.util
from pathlib import Path

import pytest

HAS_MPL = importlib.util.find_spec("matplotlib") is not None

pytestmark = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")

_SUMMARY = {
    "per_condition": {
        "real": {
            "values": [2.61, 2.63, 2.60],
            "mean": 2.62,
            "ci_low": 2.60,
            "ci_high": 2.64,
            "n_seeds": 3,
        },
        "shuffled": {
            "values": [2.64, 2.66, 2.63],
            "mean": 2.65,
            "ci_low": 2.63,
            "ci_high": 2.67,
            "n_seeds": 3,
        },
    }
}


def test_plot_writes_matched_png_and_pdf(tmp_path: Path) -> None:
    from gnsm.eval.plot_experiment import plot_conditions

    png_path, pdf_path = plot_conditions(_SUMMARY, tmp_path, "test-figure")
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert png_path.suffix == ".png"
    assert pdf_path.suffix == ".pdf"


def test_plot_handles_a_single_condition(tmp_path: Path) -> None:
    """Degenerate input must not crash the figure (e.g. a control-only run)."""

    from gnsm.eval.plot_experiment import plot_conditions

    single = {"per_condition": {"real": dict(_SUMMARY["per_condition"]["real"])}}
    png_path, _pdf = plot_conditions(single, tmp_path, "single")
    assert png_path.exists()


def test_plot_handles_identical_conditions_without_zero_span(tmp_path: Path) -> None:
    """Identical means give a zero CI span; the y-limit padding must still
    produce a valid (non-degenerate) axis rather than a matplotlib error."""

    from gnsm.eval.plot_experiment import plot_conditions

    flat = {
        "per_condition": {
            "real": {
                "values": [2.5, 2.5],
                "mean": 2.5,
                "ci_low": 2.5,
                "ci_high": 2.5,
                "n_seeds": 2,
            },
            "zero": {
                "values": [2.5, 2.5],
                "mean": 2.5,
                "ci_low": 2.5,
                "ci_high": 2.5,
                "n_seeds": 2,
            },
        }
    }
    png_path, _pdf = plot_conditions(flat, tmp_path, "flat")
    assert png_path.exists()
