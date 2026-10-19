"""Locating an extractor's evidence string in its source window."""

from __future__ import annotations

from dsg.evidence import EXACT, FUZZY, NOT_FOUND, locate

WINDOW = (
    "Emma sat by the fire. Mr. Knightley came in late, and said nothing. "
    "Emma realised Harriet had gone to London. She was alone again.\n\n"
    "The next morning it rained."
)


def _sentence(loc) -> str:
    return WINDOW[loc.start : loc.end]


def test_exact_match_returns_the_enclosing_sentence():
    loc = locate(WINDOW, "Harriet had gone to London")
    assert loc.method == EXACT
    assert _sentence(loc) == "Emma realised Harriet had gone to London."


def test_exact_match_ignores_case_whitespace_and_curly_quotes():
    loc = locate("He said “no”   twice.", '"No" twice')
    assert loc.method == EXACT


def test_a_paraphrase_is_matched_fuzzily_to_the_right_sentence():
    loc = locate(WINDOW, "Emma realizes that Harriet went off to London")
    assert loc.method == FUZZY
    assert "Harriet had gone to London" in _sentence(loc)


def test_unrelated_evidence_is_not_found_rather_than_guessed():
    assert locate(WINDOW, "the regiment marched to Brighton").method == NOT_FOUND
    assert locate(WINDOW, "").method == NOT_FOUND


def test_location_never_reads_past_the_window_it_is_given():
    cut = WINDOW.index("She was alone")
    loc = locate(WINDOW[:cut], "She was alone again")
    assert loc.method == NOT_FOUND


def test_a_long_sentence_sharing_common_words_is_not_a_fuzzy_match():
    long_sentence = (
        "It is because the hopes of a young man were bound up with him, and the name "
        "of a fair creature, and even now the old man would blush to think of the "
        "years, the house, the garden, the river, the church, the market, the fields, "
        "the road, the bridge and the mill that stood beside the water every morning."
    )
    loc = locate(long_sentence, "young man, even now blush for your unworthiness, the name")
    assert loc.method == NOT_FOUND
