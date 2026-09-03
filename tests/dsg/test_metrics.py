"""Metric behaviour, checked against hand-built states with known answers."""

from __future__ import annotations

from dsg.data.pdnc import GoldCharacter, Novel
from dsg.eval.bootstrap import paired_bootstrap
from dsg.eval.identity import occurring_aliases, score_identity, system_partition
from dsg.schemas import Span
from dsg.store import POLICIES, NarrativeState

TEXT = (
    "Elizabeth walked out. Lizzy was cheerful. Mr. Darcy bowed to Elizabeth. "
    "Darcy said little. Jane smiled at Lizzy."
)

NOVEL = Novel(
    book_id="toy",
    text=TEXT,
    characters=[
        GoldCharacter("1", "Elizabeth", {"Elizabeth", "Lizzy"}),
        GoldCharacter("2", "Mr. Darcy", {"Mr. Darcy", "Darcy"}),
        GoldCharacter("3", "Jane", {"Jane"}),
    ],
    quotes=[],
)


def _state_with(groups: list[list[str]]) -> NarrativeState:
    s = NarrativeState(POLICIES["dsg-full"])
    s.step(0, len(TEXT))
    for group in groups:
        first = s.observe_entity(group[0], Span(0, 1))
        for surface in group[1:]:
            node = s.entities[s.deref(first)]
            node.observe(surface, 0, Span(0, 1), proper=True)
    return s


def test_gold_items_are_the_aliases_that_occur_in_the_text():
    items = occurring_aliases(NOVEL)
    assert set(items) == {"elizabeth", "lizzy", "mr. darcy", "darcy", "jane"}


def test_perfect_grouping_scores_one():
    s = _state_with([["Elizabeth", "Lizzy"], ["Mr. Darcy", "Darcy"], ["Jane"]])
    score = score_identity(s, NOVEL)
    assert score.b3_f1 == 1.0 and score.ceaf_f1 == 1.0
    assert score.fragmentation == 0.0 and score.conflation == 0.0
    assert score.coverage == 1.0


def test_fragmentation_is_penalised():
    s = _state_with([["Elizabeth"], ["Lizzy"], ["Mr. Darcy", "Darcy"], ["Jane"]])
    score = score_identity(s, NOVEL)
    assert score.b3_f1 < 1.0
    assert score.fragmentation > 0.0 and score.conflation == 0.0


def test_conflation_is_penalised():
    s = _state_with([["Elizabeth", "Lizzy", "Jane"], ["Mr. Darcy", "Darcy"]])
    score = score_identity(s, NOVEL)
    assert score.conflation > 0.0


def test_unseen_aliases_stay_in_the_item_set_as_singletons():
    """Extracting less must not raise the score."""
    full = score_identity(
        _state_with([["Elizabeth", "Lizzy"], ["Mr. Darcy", "Darcy"], ["Jane"]]), NOVEL
    )
    partial = score_identity(_state_with([["Elizabeth", "Lizzy"]]), NOVEL)
    assert partial.coverage < full.coverage
    assert partial.b3_f1 < full.b3_f1
    assert partial.n_items == full.n_items


def test_partition_assigns_every_gold_item():
    s = _state_with([["Elizabeth", "Lizzy"]])
    items = occurring_aliases(NOVEL)
    partition = system_partition(s, items)
    assert set(partition) == set(items)
    assert sum(1 for v in partition.values() if v.startswith("__unseen__")) == 3


def test_paired_bootstrap_detects_a_consistent_small_effect():
    a = [0.70, 0.72, 0.68, 0.75, 0.71, 0.69]
    b = [x - 0.03 for x in a]
    r = paired_bootstrap(a, b, n_boot=2000, seed=1)
    assert r.excludes_zero and r.mean_delta > 0 and r.wins_a == 6


def test_paired_bootstrap_reports_the_sign_test_floor():
    a = [0.7, 0.71, 0.72]
    r = paired_bootstrap(a, [x - 0.01 for x in a], n_boot=500, seed=1)
    assert r.p_sign >= r.p_sign_floor
    assert r.p_sign_floor == 2 / 8
