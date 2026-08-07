"""Tests for the Stage C experiment aggregation.

`run_matrix` is patched at the trainer boundary so the aggregation/statistics
logic is verified without training anything -- what's under test here is that
the comparison is oriented correctly (lower loss = better) and that the
reported p-value comes from the real paired test, not a hand-wave.
"""

import pytest

from gnsm.eval import adapter_experiment


def _fake_runner(losses_by_condition: dict[str, list[float]]):
    """Return a stand-in for train_adapter.run that yields scripted losses."""

    calls = {"n": 0}
    cursor = {condition: 0 for condition in losses_by_condition}

    def fake_run(config, examples, encoder_state_dict, **kwargs):
        calls["n"] += 1
        condition = config.state_mode
        index = cursor[condition]
        cursor[condition] += 1
        return {
            "best_val_loss": losses_by_condition[condition][index],
            "epochs_run": 3,
            "early_stopped": False,
        }

    return fake_run, calls


def _patch(monkeypatch, losses_by_condition):
    fake_run, calls = _fake_runner(losses_by_condition)
    import gnsm.training.train_adapter as train_adapter

    monkeypatch.setattr(train_adapter, "run", fake_run)
    return calls


def test_matrix_trains_one_adapter_per_condition_and_seed(monkeypatch) -> None:
    calls = _patch(monkeypatch, {"real": [1.0, 1.0, 1.0], "shuffled": [2.0, 2.0, 2.0]})
    summary = adapter_experiment.run_matrix(
        examples=[object()] * 10,
        encoder_state_dict={},
        seeds=[0, 1, 2],
        conditions=("real", "shuffled"),
        progress=False,
    )
    assert calls["n"] == 6  # 2 conditions x 3 seeds
    assert summary["per_condition"]["real"]["n_seeds"] == 3
    assert len(summary["runs"]) == 6


def test_negative_mean_delta_favours_real(monkeypatch) -> None:
    """Lower loss is better, so real beating the control must show up as a
    negative delta -- getting this sign backwards would invert the headline."""

    _patch(monkeypatch, {"real": [1.0, 1.1, 0.9], "shuffled": [2.0, 2.1, 1.9]})
    summary = adapter_experiment.run_matrix(
        examples=[object()] * 10,
        encoder_state_dict={},
        seeds=[0, 1, 2],
        conditions=("real", "shuffled"),
        progress=False,
    )
    comparison = summary["comparisons"]["real_vs_shuffled"]
    assert comparison["mean_delta"] < 0
    assert comparison["real_better_on_n_seeds"] == 3
    assert comparison["control_better_on_n_seeds"] == 0


def test_control_winning_is_reported_faithfully(monkeypatch) -> None:
    """If the control wins, the summary must say so rather than hide it."""

    _patch(monkeypatch, {"real": [2.0, 2.0, 2.0], "shuffled": [1.0, 1.0, 1.0]})
    summary = adapter_experiment.run_matrix(
        examples=[object()] * 10,
        encoder_state_dict={},
        seeds=[0, 1, 2],
        conditions=("real", "shuffled"),
        progress=False,
    )
    comparison = summary["comparisons"]["real_vs_shuffled"]
    assert comparison["mean_delta"] > 0
    assert comparison["control_better_on_n_seeds"] == 3
    assert comparison["real_better_on_n_seeds"] == 0


def test_p_value_matches_the_exact_sign_test(monkeypatch) -> None:
    _patch(monkeypatch, {"real": [1.0] * 6, "shuffled": [2.0] * 6})
    summary = adapter_experiment.run_matrix(
        examples=[object()] * 10,
        encoder_state_dict={},
        seeds=list(range(6)),
        conditions=("real", "shuffled"),
        progress=False,
    )
    comparison = summary["comparisons"]["real_vs_shuffled"]
    # 6/6 in one direction -> two-sided exact p = 2 * (1/2^6) = 0.03125
    assert comparison["sign_test_p_value"] == pytest.approx(0.0312, abs=1e-3)
    # The floor is reported so a non-significant result at small n can be read
    # correctly rather than as evidence of no effect.
    assert comparison["sign_test_p_floor"] == pytest.approx(0.0312, abs=1e-3)


def test_sign_test_floor_is_reported_for_the_actual_seed_count(monkeypatch) -> None:
    """At 5 seeds the sign test cannot go below 0.0625, so the summary must
    surface that floor alongside the p-value."""

    _patch(monkeypatch, {"real": [1.0] * 5, "shuffled": [2.0] * 5})
    summary = adapter_experiment.run_matrix(
        examples=[object()] * 10,
        encoder_state_dict={},
        seeds=list(range(5)),
        conditions=("real", "shuffled"),
        progress=False,
    )
    comparison = summary["comparisons"]["real_vs_shuffled"]
    assert comparison["sign_test_p_floor"] == pytest.approx(0.0625, abs=1e-3)
    assert comparison["sign_test_p_value"] == pytest.approx(0.0625, abs=1e-3)
    # ...but the paired bootstrap can still resolve a consistent effect.
    assert comparison["delta_ci_excludes_zero"] is True


def test_paired_bootstrap_ci_is_reported_per_comparison(monkeypatch) -> None:
    _patch(monkeypatch, {"real": [1.0, 1.1, 0.9], "zero": [2.0, 2.1, 1.9]})
    summary = adapter_experiment.run_matrix(
        examples=[object()] * 10,
        encoder_state_dict={},
        seeds=[0, 1, 2],
        conditions=("real", "zero"),
        progress=False,
    )
    comparison = summary["comparisons"]["real_vs_zero"]
    assert comparison["delta_ci_low"] <= comparison["mean_delta"] <= comparison["delta_ci_high"]


def test_confidence_interval_brackets_each_condition_mean(monkeypatch) -> None:
    _patch(monkeypatch, {"real": [1.0, 2.0, 3.0], "zero": [4.0, 5.0, 6.0]})
    summary = adapter_experiment.run_matrix(
        examples=[object()] * 10,
        encoder_state_dict={},
        seeds=[0, 1, 2],
        conditions=("real", "zero"),
        progress=False,
    )
    for condition in ("real", "zero"):
        stats = summary["per_condition"][condition]
        assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


def test_no_comparison_section_without_the_real_condition(monkeypatch) -> None:
    _patch(monkeypatch, {"shuffled": [1.0], "zero": [2.0]})
    summary = adapter_experiment.run_matrix(
        examples=[object()] * 10,
        encoder_state_dict={},
        seeds=[0],
        conditions=("shuffled", "zero"),
        progress=False,
    )
    assert summary["comparisons"] == {}
