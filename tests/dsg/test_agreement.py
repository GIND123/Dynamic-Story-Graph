"""Agreement statistics and grouped splits for the audit labels."""

from __future__ import annotations

import pytest

from dsg.eval import agreement as ag


def test_kappa_collapses_under_skew_while_ac1_does_not():
    # 98 shared "no", 1 shared "yes", 1 disagreement: near-perfect raw agreement.
    a = ["no"] * 98 + ["yes", "yes"]
    b = ["no"] * 98 + ["yes", "no"]
    out = ag.agreement(a, b)
    assert out["observed"] == pytest.approx(0.99)
    assert out["cohen_kappa"] == pytest.approx(0.0196 / 0.0296)
    assert out["gwet_ac1"] > 0.98
    assert out["positive_agreement"] == pytest.approx(2 / 3)
    assert out["negative_agreement"] == pytest.approx(196 / 197)


def test_perfect_agreement_and_not_asked_rows():
    out = ag.agreement(["yes", "no", "n/a"], ["yes", "no", "n/a"])
    assert out["n"] == 2 and out["cohen_kappa"] == pytest.approx(1.0)
    assert out["gwet_ac1"] == pytest.approx(1.0)


def test_agreement_is_reported_per_stratum():
    rows = [
        {"annotator_b": "b", "real_violation": "yes", "real_violation_b": "yes", "s": "x"},
        {"annotator_b": "b", "real_violation": "no", "real_violation_b": "yes", "s": "y"},
        {"annotator_b": "", "real_violation": "no", "real_violation_b": "", "s": "y"},
    ]
    out = ag.agreement_by_stratum(rows, "real_violation", "s")
    assert out["all"]["n"] == 2 and out["x"]["observed"] == 1.0 and out["y"]["observed"] == 0.0


def test_grouped_split_never_divides_a_group_and_is_seeded():
    groups = {f"i{i}": f"g{i % 7}" for i in range(70)}
    split = ag.grouped_split(groups, 0.5, seed=3)
    for g in set(groups.values()):
        assert len({split[i] for i, gg in groups.items() if gg == g}) == 1
    assert split == ag.grouped_split(groups, 0.5, seed=3)
    assert 0.3 < sum(v == "dev" for v in split.values()) / 70 < 0.7


def test_premise_fingerprint_tracks_content_not_story_id():
    p = {"story_id": "story00", "canon": [{"fact_id": "f", "value": "green"}],
         "characters": ["A"], "objects": ["locket"]}
    q = {**p, "story_id": "story99"}
    r = {**p, "canon": [{"fact_id": "f", "value": "blue"}]}
    assert ag.premise_fingerprint(p) == ag.premise_fingerprint(q)
    assert ag.premise_fingerprint(p) != ag.premise_fingerprint(r)
