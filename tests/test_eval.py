from gnsm.eval.consistency.metrics import classification_metrics
from gnsm.eval.human_study.design import preregistered_cells


def test_consistency_metrics_report_precision_and_recall() -> None:
    metrics = classification_metrics({"a", "b"}, {"b", "c"})
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_human_study_has_powered_cells() -> None:
    cells = preregistered_cells()
    assert len(cells) == 6
    assert all(cell.stories >= 20 for cell in cells)
