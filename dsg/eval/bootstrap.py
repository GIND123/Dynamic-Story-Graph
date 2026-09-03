"""Paired bootstrap over books.

Books differ enormously in length, cast size and dialogue density, so unpaired
comparisons are dominated by which books happened to land in the sample. Every
comparison in the paper is therefore paired within book and resampled at the
book level, following the same reasoning recorded in the project's earlier
adapter study -- where a sign test at five seeds could not have cleared 0.05
however consistent the effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    wins_a: int
    wins_b: int
    ties: int
    p_sign: float
    p_sign_floor: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": float(self.n), "mean_a": self.mean_a, "mean_b": self.mean_b,
            "mean_delta": self.mean_delta, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "excludes_zero": float(self.excludes_zero), "wins_a": float(self.wins_a),
            "wins_b": float(self.wins_b), "ties": float(self.ties),
            "p_sign": self.p_sign, "p_sign_floor": self.p_sign_floor,
        }


def _sign_test(wins_a: int, wins_b: int) -> float:
    from math import comb

    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2**n))


def paired_bootstrap(
    a: list[float], b: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> PairedResult:
    """``a`` minus ``b``, resampled over the paired units (books)."""
    if len(a) != len(b):
        raise ValueError(f"paired inputs must match: {len(a)} vs {len(b)}")
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    delta = arr_a - arr_b
    n = len(delta)
    if n == 0:
        return PairedResult(0, 0, 0, 0, 0, 0, False, 0, 0, 0, 1.0, 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    wins_a = int((delta > 0).sum())
    wins_b = int((delta < 0).sum())
    ties = int((delta == 0).sum())
    return PairedResult(
        n=n,
        mean_a=float(arr_a.mean()),
        mean_b=float(arr_b.mean()),
        mean_delta=float(delta.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        excludes_zero=bool(lo > 0 or hi < 0),
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        p_sign=_sign_test(wins_a, wins_b),
        p_sign_floor=2 / (2 ** max(1, wins_a + wins_b)),
    )
