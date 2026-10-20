"""Rendering a value the way the docs cite it, so a changed citation is caught."""

from __future__ import annotations

import pytest

from dsg.eval import study_repro as sr


@pytest.mark.parametrize("value, fmt, cited", [
    (0.2091, "f3", "+0.209"),
    (-0.00347, "f3", "−0.003"),
    (-0.0001, "f3", "0.000"),
    (0.0, "f0", "0"),
    (8.6e-6, "e0", "9×10⁻⁶"),
    (0.20599, "pct1", "20.6%"),
    (0.0725, "pct0", "7%"),
    (3226.0, "int", "3,226"),
    ((28.0, 0.0), "wins", "28–0"),
])
def test_a_value_renders_as_cited(value, fmt, cited):
    assert sr.render(value, fmt) == sr.normalise(cited)


def test_a_third_decimal_change_is_a_text_change():
    assert sr.render(-0.746, "f3") != sr.render(-0.741, "f3")
