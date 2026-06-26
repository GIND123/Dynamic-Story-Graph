"""Offline FAKE-mode tests for the evaluation metrics.

No live Neo4j, no datasets, no LLM. Graph reads are exercised through a tiny
fake driver; PDNC character metadata is read from a committed fixture.
"""

from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures" / "metrics"
PDNC_NOVEL = FIXTURES / "pdnc" / "TinyNovel"


# ---------------------------------------------------------------------------
# Fake Neo4j driver
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, records: list[dict]) -> None:
        self.records = records


class _FakeDriver:
    """Dispatches execute_query by matching a substring of the query."""

    def __init__(self, table: dict[str, list[dict]]) -> None:
        self.table = table

    def execute_query(self, query: str, **params):
        for needle, records in self.table.items():
            if needle in query:
                return _FakeResult(records)
        return _FakeResult([])


def _gold_row(uid, qtype, speaker, addressees, text, seq):
    return {
        "uid": uid,
        "quote_type": qtype,
        "speaker": speaker,
        "addressees": addressees,
        "text": text,
        "seq": seq,
    }


# ---------------------------------------------------------------------------
# Metric 1 — quote attribution
# ---------------------------------------------------------------------------


class TestQuoteAttribution:
    def _gold(self):
        # speakers use alias surface forms to exercise normalization
        return [
            _gold_row("m:Event:quote:Q0", "explicit", "Lizzy", ["Mr. Darcy"], "t0", 0),
            _gold_row("m:Event:quote:Q1", "implicit", "Eliza", [], "t1", 1),
            _gold_row("m:Event:quote:Q2", "anaphoric", "Darcy", ["Lizzy"], "t2", 2),
            _gold_row("m:Event:quote:Q3", "explicit", "Mrs. Bennet", [], "t3", 3),
            _gold_row("m:Event:quote:Q4", "implicit", "Butler", [], "t4", 4),
        ]

    def test_alias_normalization_and_perfect_predictor(self):
        from evals.metrics import quote_attribution as qa

        driver = _FakeDriver({"is_dialogue": self._gold()})

        # Oracle predictor: returns each quote's gold speaker (alias form) by id.
        gold_by_id = {
            q.quote_id: q.speaker for q in qa.fetch_gold_dialogue(driver, "m")
        }

        def oracle(text, ctx):
            return gold_by_id[ctx["quote_id"]]

        result = qa.run_quote_attribution(
            "m", str(PDNC_NOVEL), predict=oracle, driver=driver
        )
        rep = result["filtered"]
        # Butler (minor) is filtered out; Q0..Q3 remain (3 distinct major/intermediate).
        assert rep.filter_rule.startswith("category")
        assert rep.n_quotes == 4
        assert rep.n_characters == 3  # Elizabeth, Darcy, Mrs. Bennet
        assert rep.overall == 1.0  # oracle + alias normalization -> all correct
        assert rep.explicit == 1.0
        assert rep.non_explicit == 1.0

    def test_explicit_vs_nonexplicit_split_reflects_predictor(self):
        """Predictor right on explicit, wrong on non-explicit -> split must show it."""
        from evals.metrics import quote_attribution as qa

        driver = _FakeDriver({"is_dialogue": self._gold()})
        gold = qa.fetch_gold_dialogue(driver, "m")
        # Build predictions: correct on explicit, deliberately wrong on the rest.
        predictions = {}
        for q in gold:
            if q.quote_type == "explicit":
                predictions[q.quote_id] = q.speaker  # correct (alias form)
            else:
                predictions[q.quote_id] = "Wrong Person"

        def predict(text, ctx):
            return predictions[ctx["quote_id"]]

        rep = qa.run_quote_attribution(
            "m", str(PDNC_NOVEL), predict=predict, driver=driver
        )["filtered"]
        assert rep.explicit == 1.0  # both explicit (Q0, Q3) correct
        assert rep.non_explicit == 0.0  # implicit+anaphoric all wrong
        assert rep.implicit == 0.0
        assert rep.anaphoric == 0.0
        assert 0.0 < rep.overall < 1.0  # mixed overall

    def test_category_filter_excludes_minor(self):
        from evals.metrics import quote_attribution as qa

        driver = _FakeDriver({"is_dialogue": self._gold()})
        result = qa.run_quote_attribution(
            "m", str(PDNC_NOVEL), predict=lambda t, c: None, driver=driver
        )
        # filtered excludes Butler(minor); 'all' includes it.
        assert result["filtered"].n_quotes == 4
        assert result["all"].n_quotes == 5

    def test_fallback_to_quote_threshold_when_no_category(self):
        from evals.metrics import quote_attribution as qa
        from evals.metrics.pdnc_meta import PdncChars

        # No Category metadata -> fallback to >=10 quotes spoken.
        chars = PdncChars(alias_map={}, category={}, canonical_names={"A", "B"})
        gold = [
            qa.GoldQuote(f"q{i}", "explicit", "A", [], i, "t") for i in range(10)
        ] + [qa.GoldQuote("qb", "explicit", "B", [], 99, "t")]
        allowed, rule = qa._resolve_allowed(gold, chars)
        assert allowed == {"A"}  # A has 10 quotes, B has 1
        assert ">=10" in rule


# ---------------------------------------------------------------------------
# Metric 2 — name cloze
# ---------------------------------------------------------------------------


def _pad(core: str, n_tokens: int) -> str:
    """Pad `core` with filler words to reach exactly n_tokens whitespace tokens."""
    words = core.split()
    while len(words) < n_tokens:
        words.append("lorem")
    return " ".join(words)


class TestNameCloze:
    def _passages(self):
        # Darcy occurs in p2 and p4 (>=2) so it is a counted entity; "Foot
        # passengers" is generic and never counts.
        return [
            {"number": 1, "text": _pad("In the morning Elizabeth walked alone .", 45)},
            {"number": 2, "text": _pad("Both Elizabeth and Darcy were present .", 45)},
            {"number": 3, "text": _pad("A crowd of Foot passengers gathered .", 45)},
            {"number": 4, "text": "Elizabeth and Darcy left ."},  # too short
            {
                "number": 5,
                "text": _pad("Later that evening Elizabeth read quietly .", 45),
            },
        ]

    def _names(self):
        return {"Elizabeth", "Darcy", "Foot passengers"}

    def test_selection_rejects_multiname_generic_and_short(self):
        from evals.metrics import name_cloze as nc

        selected, n_skipped = nc.select_cloze_passages(
            self._passages(), self._names(), seed=0
        )
        numbers = sorted(p.number for p in selected)
        assert numbers == [1, 5]  # only single-proper-name, in-range passages
        # multi-name(2: Elizabeth+Darcy), generic-only(3: 0 counted entities),
        # short(4); generic "Foot passengers" is NOT counted as an entity.
        assert n_skipped == 3
        for p in selected:
            assert p.answer == "Elizabeth"
            assert "[MASK]" in p.masked_text
            assert "Elizabeth" not in p.masked_text  # masked out

    def test_proper_name_target_filter_excludes_generic_span(self):
        from evals.metrics import name_cloze as nc

        targets = nc.proper_name_targets(self._passages(), self._names())
        assert "Elizabeth" in targets  # proper + recurs
        assert "Foot passengers" not in targets  # generic span filtered

    def test_one_name_plus_generic_spans_is_accepted(self):
        """One proper name + several generic common-noun spans must be ACCEPTED:
        generic spans are not named entities, so they don't break the one-entity
        rule. The proper name becomes the cloze target. (The LitBank fix.)"""
        from evals.metrics import name_cloze as nc

        names = {"Esther", "Foot passengers", "ancient Greenwich pensioners"}
        text = _pad(
            "Among the Foot passengers and the ancient Greenwich pensioners "
            "stood Esther watching the grey river .",
            45,
        )
        passages = [
            {"number": 1, "text": text},
            {"number": 2, "text": _pad("Later Esther walked home alone .", 45)},
        ]
        selected, n_skipped = nc.select_cloze_passages(passages, names, seed=0)

        assert {p.number for p in selected} == {1, 2}  # both accepted
        first = next(p for p in selected if p.number == 1)
        assert first.answer == "Esther"  # the proper name is the target
        assert "[MASK]" in first.masked_text and "Esther" not in first.masked_text
        # the generic spans remain in the (unmasked) context, uncounted
        assert "Foot passengers" in first.masked_text
        assert n_skipped == 0

    def test_scoring_accuracy_and_seed_reproducible(self):
        from evals.metrics import name_cloze as nc

        sel1, _ = nc.select_cloze_passages(self._passages(), self._names(), seed=7)
        sel2, _ = nc.select_cloze_passages(self._passages(), self._names(), seed=7)
        assert [p.number for p in sel1] == [p.number for p in sel2]  # deterministic

        report = nc.score_cloze(
            sel1, lambda masked, mid: "Elizabeth", "m", n_skipped=3, seed=7
        )
        assert report.accuracy == 1.0
        assert report.n_passages == 2
        assert report.n_skipped == 3
        assert report.seed == 7

        wrong = nc.score_cloze(
            sel1, lambda masked, mid: "Nobody", "m", n_skipped=3, seed=7
        )
        assert wrong.accuracy == 0.0

    def test_run_name_cloze_via_fake_driver(self):
        from evals.metrics import name_cloze as nc

        char_rows = [{"name": n} for n in self._names()]
        passage_rows = self._passages()
        driver = _FakeDriver({"(ch:Chapter": passage_rows, "(c:Character": char_rows})
        report = nc.run_name_cloze(
            "m", predict=lambda masked, mid: "Elizabeth", driver=driver, seed=0
        )
        # Full driver->window->select->score path yields usable passages and
        # scores them (exact windowed counts are covered by the short-sentence test).
        assert report.n_passages >= 1
        assert report.accuracy == 1.0

    # --- sentence-windowing (the real-data fix: short LitBank-style sentences) ---
    def _short_passages(self):
        """8 single-sentence passages (~10 tokens each), as LitBank stores them.

        Names at sentence 0 and 4; the rest are lowercase filler. None is in the
        40-token range alone — they MUST be windowed into valid passages.
        """
        name_s = "Aldous crossed the bridge before the morning fog lifted ."  # 10 tok
        filler = "the lantern light flickered against the wet grey stones ."  # 10 tok
        texts = [name_s if i in (0, 4) else filler for i in range(8)]
        return [{"number": i, "text": t} for i, t in enumerate(texts)]

    def test_windowing_combines_short_sentences(self):
        from evals.metrics import name_cloze as nc

        # No single passage qualifies on its own (each is 10 tokens < 40).
        assert nc.select_cloze_passages(self._short_passages(), {"Aldous"})[0] == []

        windows = nc.build_sentence_windows(self._short_passages())
        assert len(windows) == 2  # 8 sentences * 10 tok -> two 40-token windows
        for w in windows:
            assert 40 <= len(w["text"].split()) <= 60
            assert w["text"].split().count("Aldous") == 1  # exactly one name each

    def test_windowed_passages_pass_selection(self):
        from evals.metrics import name_cloze as nc

        windows = nc.build_sentence_windows(self._short_passages())
        selected, _ = nc.select_cloze_passages(windows, {"Aldous"}, seed=0)
        assert len(selected) == 2
        for p in selected:
            assert p.answer == "Aldous"
            assert "[MASK]" in p.masked_text and "Aldous" not in p.masked_text

    def test_run_name_cloze_windows_short_sentences(self):
        from evals.metrics import name_cloze as nc

        driver = _FakeDriver(
            {
                "(ch:Chapter": self._short_passages(),
                "(c:Character": [{"name": "Aldous"}],
            }
        )
        report = nc.run_name_cloze(
            "m", predict=lambda masked, mid: "Aldous", driver=driver, seed=0
        )
        assert report.n_passages == 2  # windowing made short sentences usable
        assert report.accuracy == 1.0

    def test_abbreviation_not_split_midsentence(self):
        from evals.metrics.name_cloze import _split_sentences

        # "Mr." must not end a sentence — the name stays intact in one sentence.
        sents = _split_sentences("Mr. Darcy bowed. Elizabeth turned away.")
        assert sents[0] == "Mr. Darcy bowed."
        assert len(sents) == 2


# ---------------------------------------------------------------------------
# Metric 3 — consistency
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_default_cases_perfect_on_tier1(self):
        from evals.metrics import consistency as cons

        report = cons.evaluate_consistency()
        assert report["n_cases"] == 10
        assert report["precision"] == 1.0
        assert report["recall"] == 1.0
        assert report["f1"] == 1.0
        # each of the 5 classes: one caught contradiction + one clean control
        assert set(report["per_class"]) == {
            "stance",
            "kinship",
            "trait",
            "identity",
            "location",
        }
        for cls in report["per_class"].values():
            assert cls["tp"] == 1 and cls["tn"] == 1
            assert cls["fp"] == 0 and cls["fn"] == 0

    def test_recall_drops_with_blind_checker(self):
        """A checker that never fires -> all contradictions missed (recall 0)."""
        from evals.metrics import consistency as cons

        report = cons.evaluate_consistency(check=lambda extraction, canon: [])
        assert report["recall"] == 0.0
        assert report["fn"] == 5  # the 5 tripping cases all missed
        assert report["fp"] == 0  # nothing wrongly flagged
        for cls in report["per_class"].values():
            assert cls["fn"] == 1 and cls["tn"] == 1

    def test_precision_drops_with_overfiring_checker(self):
        """A checker that always fires -> clean controls wrongly flagged (FP)."""
        from evals.metrics import consistency as cons

        report = cons.evaluate_consistency(check=lambda extraction, canon: ["x"])
        assert report["fp"] == 5  # the 5 clean controls
        assert report["recall"] == 1.0  # all real contradictions also flagged
        assert report["precision"] == 0.5  # 5 TP / (5 TP + 5 FP)
