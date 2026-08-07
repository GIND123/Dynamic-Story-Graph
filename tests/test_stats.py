import statistics

import pytest

from gnsm.training.stats import bootstrap_ci, bootstrap_paired_difference, paired_sign_test


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = bootstrap_ci(values, confidence=0.95, n_resamples=2000, seed=0)
    assert result.mean == pytest.approx(3.0)
    assert result.ci_low <= result.mean <= result.ci_high
    # With n=5 spread over [1,5], the CI must be strictly inside the data range.
    assert result.ci_low >= 1.0
    assert result.ci_high <= 5.0


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed() -> None:
    values = [0.5, 1.5, 2.5, 3.5]
    first = bootstrap_ci(values, seed=7)
    second = bootstrap_ci(values, seed=7)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_bootstrap_ci_of_identical_values_is_a_point() -> None:
    result = bootstrap_ci([2.0] * 10, seed=0)
    assert result.ci_low == pytest.approx(2.0)
    assert result.ci_high == pytest.approx(2.0)


def test_bootstrap_ci_narrows_as_sample_size_grows() -> None:
    rng_small = [float(x) for x in range(10)]
    rng_large = [float(x % 10) for x in range(200)]
    small = bootstrap_ci(rng_small, seed=0)
    large = bootstrap_ci(rng_large, seed=0)
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_bootstrap_ci_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_sign_test_all_one_direction_is_significant() -> None:
    a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    result = paired_sign_test(a, b)
    assert result.n_negative == 6
    assert result.n_positive == 0
    # Exact two-sided sign test with n=6, k=0: 2 * (1/64) = 0.03125
    assert result.p_value == pytest.approx(2 * (1 / 64))
    assert result.p_value < 0.05


def test_sign_test_even_split_is_not_significant() -> None:
    a = [1.0, 3.0, 1.0, 3.0]
    b = [2.0, 2.0, 2.0, 2.0]
    result = paired_sign_test(a, b)
    assert result.n_positive == 2
    assert result.n_negative == 2
    assert result.p_value == pytest.approx(1.0)


def test_sign_test_all_ties_returns_p_one() -> None:
    values = [1.0, 2.0, 3.0]
    result = paired_sign_test(values, list(values))
    assert result.n_ties == 3
    assert result.p_value == 1.0


def test_sign_test_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        paired_sign_test([1.0, 2.0], [1.0])


def test_sign_test_matches_hand_computed_binomial() -> None:
    # n=5 non-tied pairs, 1 in the minority direction.
    # two-sided p = 2 * (C(5,0) + C(5,1)) / 2^5 = 2 * 6/32 = 0.375
    a = [0.0, 0.0, 0.0, 0.0, 5.0]
    b = [1.0, 1.0, 1.0, 1.0, 1.0]
    result = paired_sign_test(a, b)
    assert result.n_negative == 4
    assert result.n_positive == 1
    assert result.p_value == pytest.approx(0.375)


def test_bootstrap_mean_matches_statistics_fmean() -> None:
    values = [3.2, 1.1, 4.7, 2.2]
    assert bootstrap_ci(values, seed=1).mean == pytest.approx(statistics.fmean(values))


def test_paired_bootstrap_detects_a_consistent_advantage() -> None:
    a = [1.0, 1.1, 0.9, 1.05, 0.95]
    b = [2.0, 2.1, 1.9, 2.05, 1.95]
    result = bootstrap_paired_difference(a, b, seed=0)
    assert result.mean_difference == pytest.approx(-1.0)
    assert result.ci_high < 0  # entirely below zero -> a is reliably better
    assert result.significant is True
    assert result.fraction_favouring_a == 1.0


def test_paired_bootstrap_reports_no_effect_when_there_is_none() -> None:
    a = [1.0, 2.0, 1.0, 2.0, 1.5]
    b = [2.0, 1.0, 2.0, 1.0, 1.5]
    result = bootstrap_paired_difference(a, b, seed=0)
    assert result.ci_low < 0 < result.ci_high  # CI straddles zero
    assert result.significant is False


def test_paired_bootstrap_beats_the_sign_test_p_floor_at_n_five() -> None:
    """The reason this function exists: at n=5 the exact sign test bottoms out
    at p=0.0625 and can never clear 0.05, even on a perfectly consistent
    effect. The paired bootstrap can still resolve it."""

    a = [1.0, 1.02, 0.98, 1.01, 0.99]
    b = [2.0, 2.02, 1.98, 2.01, 1.99]
    sign = paired_sign_test(a, b)
    assert sign.p_value == pytest.approx(0.0625)  # the floor, despite 5/5 wins
    assert sign.p_value > 0.05
    boot = bootstrap_paired_difference(a, b, seed=0)
    assert boot.significant is True  # the same data does resolve here


def test_paired_bootstrap_preserves_pairing() -> None:
    # Differences are all exactly -1, so every resample mean must be -1.
    a = [10.0, 20.0, 30.0, 40.0]
    b = [11.0, 21.0, 31.0, 41.0]
    result = bootstrap_paired_difference(a, b, seed=3)
    assert result.ci_low == pytest.approx(-1.0)
    assert result.ci_high == pytest.approx(-1.0)


def test_paired_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        bootstrap_paired_difference([1.0, 2.0], [1.0])


def test_paired_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_paired_difference([], [])
