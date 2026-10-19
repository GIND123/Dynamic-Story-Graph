"""Holm correction and the bootstrap p-value behind it."""

from __future__ import annotations

import numpy as np

from dsg.eval import multiplicity as mt


def test_holm_matches_the_textbook_example():
    # p = .01, .04, .03, .005 -> sorted .005,.01,.03,.04 x 4,3,2,1
    adjusted = mt.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_holm_adjusted_values_are_monotone_and_capped():
    adjusted = mt.holm([0.5, 0.4, 0.9])
    assert all(a <= 1.0 for a in adjusted)
    order = np.argsort([0.5, 0.4, 0.9])
    assert list(np.array(adjusted)[order]) == sorted(adjusted)


def test_bootstrap_p_is_small_for_a_clear_effect_and_never_zero():
    _, means = mt._draws([1.0] * 30, [0.0] * 30)
    assert 0 < mt.bootstrap_p(means) < 0.001
    _, means = mt._draws([0.0, 1.0] * 15, [1.0, 0.0] * 15)
    assert mt.bootstrap_p(means) > 0.5


def test_holm_level_interval_widens_with_the_family():
    rows = [{"metric": "m", "p_raw": p, "_means": np.linspace(-1, 1, 1001) + 0.5}
            for p in (0.001, 0.2, 0.5)]
    fam = mt.correct(rows)["m"]["comparisons"]
    first = next(r for r in fam if r["holm_rank"] == 1)
    assert first["holm_level"] == mt.ALPHA / 3
    assert first["holm_rejects"] and not any(r["holm_rejects"] for r in fam if r is not first)
