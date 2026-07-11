"""Offline FAKE-mode tests for the three baselines.

A `FakeGraph` answers every Cypher shape the predictors/harnesses use; the LLM
and retrieval are injected. No live Neo4j, no API, no fastembed.
"""

from collections import Counter
from pathlib import Path

from baselines.config import BaselineConfig
from baselines.flat_long_context import FlatLongContextPredictor
from baselines.graph_method import GraphPredictor, neighborhood_agents
from baselines.vector_rag import VectorRAGPredictor, _exclude_self

PDNC_NOVEL = str(Path(__file__).parent / "fixtures" / "metrics" / "pdnc" / "TinyNovel")
MID = "pdnc:PrideAndPrejudice"
# major+intermediate canonical names in the TinyNovel fixture
CANDIDATES = ["Elizabeth Bennet", "Fitzwilliam Darcy", "Mrs. Bennet"]


class _Result:
    def __init__(self, records):
        self.records = records


class FakeGraph:
    """Answers the predictor/harness query shapes from in-memory data."""

    def __init__(self, chapters: dict[int, str], dialogue: list[dict]):
        self.chapters = chapters  # number -> text
        self.dialogue = dialogue  # [{seq, agent, quote_type, text, uid, addressees}]

    def verify_connectivity(self):
        return True

    def execute_query(self, query, **p):
        if "CONTAINS $snippet" in query:
            snip = p["snippet"]
            nums = sorted(n for n, t in self.chapters.items() if snip and snip in t)
            return _Result([{"number": nums[0]}] if nums else [])
        if "ch.number >= $lo" in query:
            lo, hi = p["lo"], p["hi"]
            return _Result(
                [
                    {"number": n, "text": self.chapters[n]}
                    for n in sorted(self.chapters)
                    if lo <= n <= hi
                ]
            )
        if "sequence_index <>" in query:  # neighborhood_agents (holdout)
            lo, hi, ex = p["lo"], p["hi"], p["exclude"]
            rows = [
                {"seq": d["seq"], "agent": d["agent"]}
                for d in self.dialogue
                if lo <= d["seq"] <= hi and d["seq"] != ex
            ]
            rows.sort(key=lambda r: r["seq"])
            return _Result(rows)
        if "count(r) AS degree" in query:  # character_degrees
            c = Counter(d["agent"] for d in self.dialogue)
            return _Result([{"name": n, "degree": k} for n, k in c.most_common()])
        if "is_dialogue: true" in query and "AS speaker" in query:  # gold fetch
            rows = []
            for d in sorted(self.dialogue, key=lambda x: x["seq"]):
                rows.append(
                    {
                        "uid": d["uid"],
                        "quote_type": d["quote_type"],
                        "speaker": d["agent"],
                        "addressees": d.get("addressees", []),
                        "text": d["text"],
                        "seq": d["seq"],
                    }
                )
            return _Result(rows)
        if "(ch:Chapter" in query and "IS NOT NULL" in query:  # cloze passages
            return _Result(
                [{"number": n, "text": self.chapters[n]} for n in sorted(self.chapters)]
            )
        if "(c:Character" in query and "AS name" in query:  # cloze char names
            return _Result([{"name": a} for a in {d["agent"] for d in self.dialogue}])
        return _Result([])


def _chapters_with_quotes(dialogue, *, size=900, n=25):
    """Build n filler chapters; embed each quote's text in a distinct chapter."""
    chapters = {i: ("filler narrative prose . " * (size // 24)) for i in range(n)}
    for i, d in enumerate(dialogue):
        ch = (i * 3) % n
        chapters[ch] = chapters[ch] + " " + d["text"]
    return chapters


def _q(seq, agent, qtype, text):
    return {
        "seq": seq,
        "agent": agent,
        "quote_type": qtype,
        "text": text,
        "uid": f"{MID}:Event:quote:Q{seq}",
        "addressees": [],
    }


def _context(quote_id, qtype, seq, prev=None):
    return {
        "manuscript_id": MID,
        "quote_id": quote_id,
        "quote_type": qtype,
        "sequence_index": seq,
        "previous_speaker": prev,
        "candidates": CANDIDATES,
    }


# ---------------------------------------------------------------------------


class TestPredictorSignatures:
    def test_each_predictor_returns_candidate_or_none(self):
        dialogue = [_q(0, "Elizabeth Bennet", "explicit", "I could forgive his pride.")]
        graph = FakeGraph(_chapters_with_quotes(dialogue), dialogue)
        ask = lambda prompt, meta: "I think Elizabeth Bennet speaks here."  # noqa: E731
        ctx = _context("Q0", "explicit", 0)

        for cls in (FlatLongContextPredictor, VectorRAGPredictor, GraphPredictor):
            kw = {}
            if cls is VectorRAGPredictor:
                kw["retrieve"] = lambda mid, q, k: [
                    {
                        "chapter": 1,
                        "text": "Elizabeth Bennet replied calmly .",
                        "score": 0.9,
                    }
                ]
            pred = cls(MID, graph, ask, BaselineConfig(), **kw)
            out = pred.predict_quote(dialogue[0]["text"], ctx)
            assert out is None or out in CANDIDATES


class TestEqualBudgetAccounting:
    def _dialogue(self):
        return [
            _q(10, "Elizabeth Bennet", "explicit", "A quote line to attribute here.")
        ]

    def test_long_context_and_rag_near_budget_graph_far_less(self):
        dialogue = self._dialogue()
        graph = FakeGraph(_chapters_with_quotes(dialogue, size=900, n=30), dialogue)
        cfg = BaselineConfig()
        ctx = _context("Q10", "explicit", 10)
        ask = lambda prompt, meta: "Elizabeth Bennet"  # noqa: E731

        lc = FlatLongContextPredictor(MID, graph, ask, cfg)
        lc.predict_quote(dialogue[0]["text"], ctx)
        lc_in = lc.records[-1].input_tokens

        chunk = {"chapter": 1, "text": "prose chunk . " * 70, "score": 0.9}
        rag = VectorRAGPredictor(
            MID,
            graph,
            ask,
            cfg,
            retrieve=lambda mid, q, k: [dict(chunk) for _ in range(k)],
        )
        rag.predict_quote(dialogue[0]["text"], ctx)
        rag_in = rag.records[-1].input_tokens

        gp = GraphPredictor(MID, graph, ask, cfg)
        gp.predict_quote(dialogue[0]["text"], ctx)
        gp_in = gp.records[-1].input_tokens

        # long-context and RAG land in the neighbourhood of the 4000-token budget
        assert 0.6 * cfg.token_budget <= lc_in <= 1.4 * cfg.token_budget
        assert 0.6 * cfg.token_budget <= rag_in <= 1.4 * cfg.token_budget
        # graph reads materially fewer tokens — the efficiency result
        assert gp_in < 0.2 * lc_in


class TestRagSelfExclusion:
    def test_exclude_self_drops_exact_quote_chunk(self):
        quote = "I could easily forgive his pride, if he had not mortified mine."
        chunks = [
            {
                "chapter": 1,
                "text": "Earlier that day the sisters spoke .",
                "score": 0.9,
            },
            {"chapter": 2, "text": "Context: " + quote + " she said .", "score": 0.95},
        ]
        kept = _exclude_self(chunks, quote)
        assert all(quote[:40] not in c["text"] for c in kept)
        assert len(kept) == 1

    def test_predictor_prompt_omits_self_chunk(self):
        quote = "You must allow me to tell you how ardently I admire you."
        dialogue = [_q(5, "Fitzwilliam Darcy", "explicit", quote)]
        graph = FakeGraph(_chapters_with_quotes(dialogue), dialogue)
        captured = {}

        def ask(prompt, meta):
            captured["prompt"] = prompt
            return "Fitzwilliam Darcy"

        retrieve = lambda mid, q, k: [  # noqa: E731
            {"chapter": 9, "text": "SELFMARKER " + quote, "score": 0.99},
            {
                "chapter": 3,
                "text": "Darcy approached and spoke quietly .",
                "score": 0.8,
            },
        ]
        rag = VectorRAGPredictor(MID, graph, ask, BaselineConfig(), retrieve=retrieve)
        rag.predict_quote(quote, _context("Q5", "explicit", 5))
        assert "SELFMARKER" not in captured["prompt"]  # self chunk excluded


class TestGraphHoldout:
    def test_target_gold_agent_never_in_graph_input(self):
        """Q at seq 100, gold agent 'Zara' speaks ONLY Q. After holdout the
        neighborhood the graph predictor reads must not contain Zara."""
        dialogue = [
            _q(98, "Elizabeth Bennet", "implicit", "a"),
            _q(99, "Mrs. Bennet", "implicit", "b"),
            _q(100, "Zara", "explicit", "the target quote"),  # gold, unique speaker
            _q(101, "Elizabeth Bennet", "implicit", "c"),
            _q(102, "Mrs. Bennet", "implicit", "d"),
        ]
        graph = FakeGraph(_chapters_with_quotes(dialogue), dialogue)

        agents = neighborhood_agents(graph, MID, 100, 8, exclude_seq=100)
        assert "Zara" not in [a for _s, a in agents]  # holdout removed it

        gp = GraphPredictor(MID, graph, None, BaselineConfig())
        prediction = gp.predict_quote(
            "the target quote", _context("Q100", "explicit", 100, prev="Mrs. Bennet")
        )
        assert prediction != "Zara"  # cannot predict what it never read
        assert prediction in CANDIDATES or prediction is None


class TestExplicitVsNonExplicitSplit:
    def test_runner_reports_split_per_method(self):
        from baselines.run_baselines import run_baselines

        # explicit quotes easy, non-explicit hard (LLM abstains on non-explicit)
        dialogue = [
            _q(0, "Elizabeth Bennet", "explicit", "Explicit line zero here."),
            _q(1, "Fitzwilliam Darcy", "anaphoric", "Anaphoric line one here."),
            _q(2, "Mrs. Bennet", "explicit", "Explicit line two here."),
            _q(3, "Elizabeth Bennet", "implicit", "Implicit line three here."),
        ]
        graph = FakeGraph(_chapters_with_quotes(dialogue), dialogue)
        gold_map = {f"Q{d['seq']}": d["agent"] for d in dialogue}

        def ask(prompt, meta):
            if meta.get("quote_type") == "explicit":
                return gold_map[meta["quote_id"]]  # correct on explicit
            return ""  # abstain on non-explicit -> wrong

        retrieve = lambda mid, q, k: [  # noqa: E731
            {"chapter": 1, "text": "surrounding prose .", "score": 0.5}
        ]
        results = run_baselines(
            MID,
            novel_dir=PDNC_NOVEL,
            driver=graph,
            ask=ask,
            retrieve=retrieve,
            max_cloze=0,
        )
        qa = results["quote_attribution"]
        for method in ("flat_long_context", "vector_rag"):
            assert qa[method]["explicit"] == 1.0  # explicit all correct
            assert qa[method]["non_explicit"] == 0.0  # non-explicit all wrong
        # telemetry is recorded per method
        assert qa["flat_long_context"]["telemetry"]["n_calls"] >= 1
        assert "graph" in qa  # graph method also reported (heuristic)
