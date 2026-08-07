"""Minimal, honest statistics for comparing two conditions across seeds.

Stdlib-only, and deliberately not over-claiming precision: a sign test is
exact for any sample size, unlike a t-test's normal/t-distribution
assumption, which matters here because these comparisons typically run only
a handful of seeds (cheap, real GPU time is the limiting factor, not
statistical theory).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int


def bootstrap_ci(
    values: list[float], confidence: float = 0.95, n_resamples: int = 2000, seed: int = 0
) -> BootstrapResult:
    """Percentile-bootstrap confidence interval for the mean of `values`."""

    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    resample_means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        resample_means.append(statistics.fmean(resample))
    resample_means.sort()
    alpha = 1 - confidence
    lower_idx = max(0, min(int(round(alpha / 2 * n_resamples)), n_resamples - 1))
    upper_idx = max(0, min(int(round((1 - alpha / 2) * n_resamples)) - 1, n_resamples - 1))
    return BootstrapResult(
        mean=statistics.fmean(values),
        ci_low=resample_means[lower_idx],
        ci_high=resample_means[upper_idx],
        confidence=confidence,
        n_resamples=n_resamples,
    )


@dataclass(frozen=True, slots=True)
class PairedDifferenceResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    confidence: float
    fraction_favouring_a: float
    n_pairs: int

    @property
    def significant(self) -> bool:
        """True when the CI for the mean difference excludes zero."""

        return (self.ci_low > 0) or (self.ci_high < 0)


def bootstrap_paired_difference(
    a: list[float],
    b: list[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> PairedDifferenceResult:
    """Bootstrap CI for the mean paired difference (a - b).

    Preferred over :func:`paired_sign_test` when the number of pairs is small:
    an exact two-sided sign test on n pairs cannot return a p-value below
    2 / 2**n, so at n=5 the smallest attainable p is 0.0625 -- it cannot clear
    a 0.05 threshold even when every pair favours the same side. This keeps
    the sign test's assumption-free spirit (resampling, no normality
    assumption) while actually being able to resolve an effect at small n.

    Resamples pairs jointly, preserving the pairing.
    """

    if len(a) != len(b):
        raise ValueError("bootstrap_paired_difference needs equal-length sequences")
    if not a:
        raise ValueError("bootstrap_paired_difference needs at least one pair")

    differences = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(differences)
    means = []
    for _ in range(n_resamples):
        resample = [differences[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(resample))
    means.sort()
    alpha = 1 - confidence
    lower_idx = max(0, min(int(round(alpha / 2 * n_resamples)), n_resamples - 1))
    upper_idx = max(0, min(int(round((1 - alpha / 2) * n_resamples)) - 1, n_resamples - 1))
    return PairedDifferenceResult(
        mean_difference=statistics.fmean(differences),
        ci_low=means[lower_idx],
        ci_high=means[upper_idx],
        confidence=confidence,
        # "a wins" means a < b here (these are losses; lower is better).
        fraction_favouring_a=sum(1 for d in differences if d < 0) / n,
        n_pairs=n,
    )


@dataclass(frozen=True, slots=True)
class SignTestResult:
    n_pairs: int
    n_positive: int  # count where a < b ("a" improved over "b")
    n_negative: int
    n_ties: int
    p_value: float


def paired_sign_test(a: list[float], b: list[float]) -> SignTestResult:
    """Exact two-sided sign test on paired differences (a - b), testing
    whether positive and negative differences are equally likely (H0).

    Note the small-n floor: the smallest attainable two-sided p-value is
    2 / 2**n_nontied, so n=5 bottoms out at 0.0625. Pair this with
    :func:`bootstrap_paired_difference` rather than reading a non-significant
    result at small n as evidence of no effect.
    """

    if len(a) != len(b):
        raise ValueError("paired_sign_test needs equal-length sequences")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n_positive = sum(1 for d in diffs if d > 0)
    n_negative = sum(1 for d in diffs if d < 0)
    n_ties = sum(1 for d in diffs if d == 0)
    n = n_positive + n_negative
    if n == 0:
        p_value = 1.0
    else:
        k = min(n_positive, n_negative)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p_value = min(1.0, 2 * tail)
    return SignTestResult(
        n_pairs=len(a), n_positive=n_positive, n_negative=n_negative, n_ties=n_ties, p_value=p_value
    )
