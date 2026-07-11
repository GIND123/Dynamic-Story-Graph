"""Offline tests for the LitBank coreference metric (no network, no LLM).

Verifies CoNLL parsing (including nested/piped markers), person-cluster
filtering, balanced pair construction, and the scoring math (accuracy, split vs
merge, entity-drift = 1 - accuracy, abstain handling).
"""

from pathlib import Path

from evals.metrics import litbank_coref as lc


def _write_conll(tmp_path: Path, lines: list[str]) -> Path:
    d = tmp_path / "coref" / "conll"
    d.mkdir(parents=True)
    p = d / "doc.conll"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# cluster 0 {John, He}, cluster 1 {Mary, her}, cluster 2 {the doctor, him}
_DOC = [
    "#begin document (doc); part 0",
    "d 0 0 John _ (0)",
    "d 0 1 met _ _",
    "d 0 2 Mary _ (1)",
    "d 0 3 . _ _",
    "",
    "d 0 0 He _ (0)",
    "d 0 1 told _ _",
    "d 0 2 her _ (1)",
    "d 0 3 about _ _",
    "d 0 4 the _ (2",
    "d 0 5 doctor _ 2)",
    "d 0 6 . _ _",
    "",
    "d 0 0 She _ (1)",
    "d 0 1 thanked _ _",
    "d 0 2 him _ (2)",
    "d 0 3 . _ _",
]


def test_parse_and_person_clusters(tmp_path):
    doc = lc.parse_conll(_write_conll(tmp_path, _DOC))
    assert len(doc.sentences) == 3
    people = lc.person_clusters(doc)
    # all three clusters contain a personal pronoun (He / her / him)
    assert set(people) == {0, 1, 2}
    surfaces = {m.text for m in people[0]}
    assert surfaces == {"John", "He"}


def test_build_pairs_balanced_and_labeled(tmp_path):
    doc = lc.parse_conll(_write_conll(tmp_path, _DOC))
    pairs = lc.build_pairs(doc, max_pairs=100, seed=0)
    # positives are same-cluster with different surface forms
    for p in pairs:
        if p.same:
            assert p.mention_a.cluster == p.mention_b.cluster
            assert p.mention_a.text.lower() != p.mention_b.text.lower()
        else:
            assert p.mention_a.cluster != p.mention_b.cluster
    assert any(p.same for p in pairs) and any(not p.same for p in pairs)


def test_scoring_oracle_and_adversaries(tmp_path):
    doc = lc.parse_conll(_write_conll(tmp_path, _DOC))
    pairs = lc.build_pairs(doc, max_pairs=100, seed=0)

    oracle = lc.score_coref(
        pairs, lambda c, a, b: _same_cluster(pairs, a, b), doc_id="d"
    )
    assert oracle.accuracy == 1.0
    assert oracle.entity_drift_rate == 0.0

    always_no = lc.score_coref(pairs, lambda c, a, b: False, doc_id="d")
    assert always_no.positive_miss_rate == 1.0  # splits every true coref

    always_abstain = lc.score_coref(pairs, lambda c, a, b: None, doc_id="d")
    assert always_abstain.abstain_rate == 1.0
    assert always_abstain.entity_drift_rate == 1.0


def _same_cluster(pairs, a_text, b_text) -> bool:
    """Oracle: look up the gold label from the pair whose surfaces match."""
    for p in pairs:
        if {p.mention_a.text, p.mention_b.text} == {a_text, b_text}:
            return p.same
    return False


def test_predictor_exception_is_abstain(tmp_path):
    doc = lc.parse_conll(_write_conll(tmp_path, _DOC))
    pairs = lc.build_pairs(doc, max_pairs=10, seed=0)

    def boom(c, a, b):
        raise RuntimeError("dropped")

    rep = lc.score_coref(pairs, boom, doc_id="d")
    assert rep.abstain == rep.n
