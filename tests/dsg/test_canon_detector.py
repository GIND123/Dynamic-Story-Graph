"""What the canon-violation detector must and must not flag.

The cases are kept as tables so every detector version is held to the same
sentences. Under v1 a contradicting vocabulary word anywhere within the window
of a subject mention is a violation, so the innocent cases are expected to fail
there; they are marked strict-xfail so the suite stays green and records the
defect, and a later version that binds the value to the attribute must pass
them outright.
"""

from __future__ import annotations

import pytest

from dsg.generate.canon import DETECTOR_VERSIONS, CanonFact

FACTS: dict[str, CanonFact] = {
    "miriam_eyes": CanonFact("f0", "Miriam", "eyes", "green", "Miriam has green eyes."),
    "tobias_hair": CanonFact("f1", "Tobias", "hair", "auburn", "Tobias has auburn hair."),
    "locket_silver": CanonFact("f2", "locket", "metal", "silver", "The locket is silver."),
    "locket_gold": CanonFact("f3", "locket", "metal", "gold", "The locket is gold."),
    "tobias_trade": CanonFact(
        "f4", "Tobias", "trade", "clockmaker", "Tobias is the village clockmaker."
    ),
    "rowena_home": CanonFact(
        "f5", "Rowena", "home", "Ashgrove",
        "Rowena was born in Ashgrove and has never left it.",
    ),
}

# (fact, text): a vocabulary word sits near the subject but describes something
# else, so the planted fact is not contradicted.
INNOCENT: list[tuple[str, str]] = [
    ("miriam_eyes", "Miriam stepped out into the grey morning."),
    ("miriam_eyes", "Miriam watched the blue water."),
    ("tobias_hair", "Tobias wore his black coat."),
    ("tobias_hair", "Tobias pushed the red door open."),
    ("locket_silver", "She held the locket up to the gold light."),
    ("locket_silver", "The locket hung from an iron nail."),
    ("tobias_trade", "Tobias nodded to the apothecary."),
    ("tobias_trade", "Silas, the apothecary, nodded to Tobias."),
    ("rowena_home", "Rowena rode beside Cormac, who had come down from Ravensmoor."),
    # 'silver' inside the hair value 'silver-white' is not a metal.
    ("locket_gold", "Maud pushed back her silver-white hair and opened the locket."),
]

# (fact, text): the subject's own attribute is given a wrong value.
VIOLATIONS: list[tuple[str, str]] = [
    ("miriam_eyes", "Miriam's blue eyes narrowed."),
    ("miriam_eyes", "Miriam turned. Her blue eyes narrowed."),
    ("miriam_eyes", "Miriam looked up with grey eyes."),
    ("tobias_hair", "Tobias ran a hand through his black hair."),
    ("tobias_hair", "Tobias's black hair fell across his brow."),
    ("locket_silver", "The locket, gold and heavy, swung from her hand."),
    ("locket_silver", "The locket was gold."),
    ("locket_silver", "She opened the gold locket."),
    ("tobias_trade", "Tobias, the apothecary, measured out the powder."),
    ("tobias_trade", "Tobias worked as a blacksmith."),
    ("rowena_home", "Rowena was born in Ravensmoor."),
    ("rowena_home", "Rowena, who came from Thornbury, said nothing."),
]

# (fact, text): the true value appears near the subject but does not describe
# the subject's attribute, so it is not a restatement of the canon.
NOT_RESTATED: list[tuple[str, str]] = [
    ("miriam_eyes", "Miriam crossed the green lawn."),
    ("locket_silver", "The locket lay beside a silver spoon."),
    ("tobias_hair", "Tobias passed the auburn mare in the yard."),
]

RESTATED: list[tuple[str, str]] = [
    ("miriam_eyes", "Miriam's green eyes narrowed."),
    ("locket_silver", "The silver locket caught the light."),
    ("tobias_hair", "Tobias pushed his auburn hair back."),
]

_V1_DEFECT = pytest.mark.xfail(
    strict=True, reason="v1 counts any vocabulary word inside the window"
)
# Removed when v2 lands; strict, so a v2 that passes forces its removal.
_V2_NOT_BUILT = pytest.mark.xfail(
    strict=True, raises=NotImplementedError, reason="detector v2 is built in phase 1c"
)


def _by_version(cases: list[tuple[str, str]], v1_defect: bool) -> list:
    """Every case under every version: v1 may carry the known defect, v2 may not."""
    params = []
    for fact, text in cases:
        for version in DETECTOR_VERSIONS:
            marks = []
            if version == "v1" and v1_defect:
                marks.append(_V1_DEFECT)
            if version == "v2":
                marks.append(_V2_NOT_BUILT)
            params.append(
                pytest.param(version, fact, text, marks=marks, id=f"{version}-{fact}:{text[:32]}")
            )
    return params


@pytest.mark.parametrize(("version", "fact", "text"), _by_version(INNOCENT, v1_defect=True))
def test_innocent_text_is_not_a_violation(version, fact, text):
    violated, evidence = FACTS[fact].check(text, version)
    assert not violated, evidence


@pytest.mark.parametrize(("version", "fact", "text"), _by_version(VIOLATIONS, v1_defect=False))
def test_a_clear_violation_is_flagged(version, fact, text):
    assert FACTS[fact].check(text, version)[0]


@pytest.mark.parametrize(("version", "fact", "text"), _by_version(NOT_RESTATED, v1_defect=True))
def test_an_unrelated_use_of_the_true_value_is_not_a_restatement(version, fact, text):
    assert not FACTS[fact].mentioned(text, version)


@pytest.mark.parametrize(("version", "fact", "text"), _by_version(RESTATED, v1_defect=False))
def test_a_restatement_is_detected(version, fact, text):
    assert FACTS[fact].mentioned(text, version)
