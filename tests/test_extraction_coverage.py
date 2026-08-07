from gnsm.eval.extraction_coverage import CoverageReport, measure_coverage


def test_coverage_counts_entities_in_text_the_extractor_understands() -> None:
    # The reference extractor keys off capitalised names and a few verb patterns.
    report = measure_coverage(["Mara waited in the observatory."])
    assert report.n_texts == 1
    assert report.mean_entities > 0


def test_coverage_reports_zero_for_structureless_text() -> None:
    report = measure_coverage(["the the the", "and and and"])
    assert report.n_texts == 2
    assert report.mean_edges == 0
    assert report.mean_attributes == 0
    assert report.texts_with_any_edge == 0
    assert report.texts_with_any_attribute == 0


def test_supports_consistency_checks_is_false_without_edges_or_attributes() -> None:
    """The guard exists so a corpus that can't trigger the verifier is caught
    before a vacuous zero-violation score gets reported as a result."""

    barren = CoverageReport(
        n_texts=50,
        mean_entities=7.0,
        mean_edges=0.0,
        mean_attributes=0.0,
        mean_quotes=0.0,
        texts_with_any_edge=0,
        texts_with_any_attribute=0,
    )
    assert barren.supports_consistency_checks is False


def test_supports_consistency_checks_is_true_when_structure_appears() -> None:
    populated = CoverageReport(
        n_texts=50,
        mean_entities=7.0,
        mean_edges=0.5,
        mean_attributes=0.0,
        mean_quotes=0.0,
        texts_with_any_edge=12,
        texts_with_any_attribute=0,
    )
    assert populated.supports_consistency_checks is True


def test_coverage_handles_empty_input_without_dividing_by_zero() -> None:
    report = measure_coverage([])
    assert report.n_texts == 0
    assert report.mean_entities == 0
