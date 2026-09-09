"""Tied ranks in the G3 correlation.

16 of 28 novels score exactly 0.000 on name cloze. Giving tied values distinct
ranks orders them by input order -- alphabetical, here -- and manufactures a
correlation out of novel names.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsg.eval.attribution_report import memorization_check


def _fixture(acc: list[float], cloze: list[float]) -> tuple[dict, dict]:
    names = [f"N{i:02d}" for i in range(len(acc))]
    result = {"per_novel": {n: {"state-causal": {"accuracy": a}} for n, a in zip(names, acc)}}
    cl = {"per_novel": {n: {"cloze_accuracy": c} for n, c in zip(names, cloze)}}
    return result, cl


class TestTiedRanks:
    def test_all_tied_covariate_yields_no_correlation(self) -> None:
        """If every novel ties, there is no ranking information at all."""
        res, cl = _fixture([0.1 * i for i in range(10)], [0.0] * 10)
        out = memorization_check(res, cl)
        assert out.get("note") == "no variance in one series"

    def test_ties_do_not_inherit_alphabetical_order(self) -> None:
        """Accuracy ascending with names, covariate all-tied but for one book.

        An argsort-only rank would rank the ties by name and report a strong
        correlation. With average ranks the association is carried only by the
        single distinct value.
        """
        acc = [0.1 * i for i in range(12)]
        cloze = [0.0] * 11 + [0.5]
        out = memorization_check(*_fixture(acc, cloze))
        assert out["p_permutation"] > 0.05, out

    def test_a_genuine_monotone_relationship_is_still_detected(self) -> None:
        acc = [0.1 * i for i in range(12)]
        cloze = [0.05 * i for i in range(12)]
        out = memorization_check(*_fixture(acc, cloze))
        assert out["spearman_rho"] > 0.95
        assert out["p_permutation"] < 0.01
        assert out["contaminated"] is True

    def test_a_negative_relationship_is_not_flagged_as_contamination(self) -> None:
        """Contamination means accuracy RISES with memorization, not falls."""
        acc = [0.1 * i for i in range(12)]
        cloze = [0.05 * (11 - i) for i in range(12)]
        out = memorization_check(*_fixture(acc, cloze))
        assert out["spearman_rho"] < -0.95
        assert out["contaminated"] is False

    def test_too_few_novels_is_reported_not_computed(self) -> None:
        out = memorization_check(*_fixture([0.1, 0.2, 0.3], [0.0, 0.1, 0.2]))
        assert "note" in out
