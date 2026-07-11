"""Offline tests for the context-length experiment (no GPU, no network, no Neo4j).

Covers: num_ctx passthrough + prompt-token capture, the needle haystack (depth +
canon-free filler), needle scoring, failure/crossover math, ladder capping, and
an end-to-end Curve A run against a fake driver + the PDNC fixture.
"""

from pathlib import Path

from evals.metrics import long_context as lc
from evals.run_context_experiment import advertised_window, cap_ladder

PDNC_NOVEL = str(Path(__file__).parent / "fixtures" / "metrics" / "pdnc" / "TinyNovel")


# ---------------------------------------------------------------------------
# num_ctx plumbing
# ---------------------------------------------------------------------------
def test_ollama_num_ctx_passthrough_and_token_capture():
    from app.llm import OllamaLLM

    llm = OllamaLLM()  # constructs an httpx client but makes no connection
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "content": '{"verdict":"PASS","contradictions":[],'
                    '"coherence_score":0.9,"reasoning":"ok"}'
                },
                "prompt_eval_count": 4321,
            }

    class _Client:
        def post(self, url, json):
            captured["payload"] = json
            return _Resp()

    llm._client = _Client()
    llm.num_ctx = 8000
    verdict = llm.judge("canon", "draft")
    assert captured["payload"]["options"]["num_ctx"] == 8000
    assert llm.last_prompt_tokens == 4321
    assert verdict.verdict == "PASS"


# ---------------------------------------------------------------------------
# Failure point & crossover
# ---------------------------------------------------------------------------
def test_failure_and_crossover():
    pts = [
        {"actual_tokens": 1000, "overall": 0.80},
        {"actual_tokens": 4000, "overall": 0.70},
        {"actual_tokens": 16000, "overall": 0.30},  # < 50% of peak 0.80
        {"actual_tokens": 64000, "overall": 0.10},
    ]
    assert lc.failure_point(pts) == 16000
    assert lc.crossover_point(pts, 0.55) == 16000
    assert lc.crossover_point(pts, 0.05) is None  # never drops below the graph
    assert lc.failure_point([]) is None


def test_effective_context():
    # peak = 0.90 -> 80% bar = 0.72
    pts = [
        {"actual_tokens": 1000, "overall": 0.90},
        {"actual_tokens": 4000, "overall": 0.80},  # still within 20% of peak
        {"actual_tokens": 8000, "overall": 0.60},  # drops below the bar here
        {"actual_tokens": 16000, "overall": 0.85},  # noisy recovery must be ignored
    ]
    tokens, is_lower_bound = lc.effective_context(pts, frac=0.8)
    assert tokens == 4000 and is_lower_bound is False  # last good BEFORE the drop
    # never drops in range -> the effective length is a lower bound at the deepest rung
    tokens2, lb2 = lc.effective_context(pts[:2], frac=0.8)
    assert tokens2 == 4000 and lb2 is True
    assert lc.effective_context([]) == (None, False)


# ---------------------------------------------------------------------------
# Needle haystack + scoring
# ---------------------------------------------------------------------------
def test_build_haystack_depth_and_canon_free():
    from baselines.config import count_tokens

    fact = "CANON FACT: the captain is dead."
    h = lc.build_haystack(fact, 3000, 0.5)
    # built to target/density (len//4 ~2300) so REAL tokens ~= the 3000 target
    assert 1900 < count_tokens(h) < 2800
    assert 0.35 < h.index("CANON FACT") / len(h) < 0.65
    # filler must not smuggle in canon names
    for name in ("Brann", "Mara", "Sera", "Duskwall", "Citadel"):
        assert name not in lc._FILLER_SEED

    deep = lc.build_haystack(fact, 3000, 0.9)
    assert deep.index("CANON FACT") / len(deep) > 0.8


def test_needle_curve_scoring_and_truncation_flag():
    def judge(canon, draft, num_ctx):
        contradict = any(
            w in draft
            for w in (
                "drew his sword",
                "read the long letter",
                "hauling nets",
                "swung wide",
            )
        )
        # report a real length well under target -> truncated flag should fire
        return ("FAIL" if contradict else "PASS"), 100

    rep = lc.run_needle_curve(judge, lengths=[8000], depths=[0.0, 0.9])
    for p in rep["points"]:
        assert p["detection_recall"] == 1.0
        assert p["over_flag_rate"] == 0.0
        assert p["malformed_rate"] == 0.0
        assert p["truncated"] is True  # 100 << 8000


def test_needle_malformed_counts():
    rep = lc.run_needle_curve(
        lambda c, d, n: (None, 8000), lengths=[8000], depths=[0.5]
    )
    assert rep["points"][0]["malformed_rate"] == 1.0
    assert rep["points"][0]["detection_recall"] == 0.0  # None != FAIL


# ---------------------------------------------------------------------------
# Ladder capping / advertised windows
# ---------------------------------------------------------------------------
def test_advertised_and_ladder_cap():
    L = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 131072]
    assert advertised_window("gemma2:9b") == 8192
    assert advertised_window("qwen2.5:72b") == 131072
    # Rungs fit within min(window, hardware) - margin; top rung sits at the ceiling.
    # Gemma (8K window): tops at 8192-1024=7168, never past the window.
    assert cap_ladder(L, 8192, 16000, margin=1024) == [1000, 2000, 4000, 7168]
    # Qwen (128K window) on a 16K machine: hardware-limited to 16000-1024=14976.
    assert cap_ladder(L, 131072, 16000, margin=1024) == [1000, 2000, 4000, 8000, 14976]
    # Yi (4K window): only ~3K of prose fits.
    assert cap_ladder(L, 4096, 16000, margin=1024) == [1000, 2000, 3072]
    # No rung ever exceeds the window (the overflow/garbage regime).
    assert max(cap_ladder(L, 8192, 16000)) < 8192


# ---------------------------------------------------------------------------
# Curve A end to end (fake driver + PDNC fixture + oracle ask)
# ---------------------------------------------------------------------------
class _Result:
    def __init__(self, records):
        self.records = records


class _FakeDriver:
    """Dispatch execute_query by query substring (gold / locate / window)."""

    _GOLD = [
        {
            "uid": "m:Event:quote:Q0",
            "quote_type": "explicit",
            "speaker": "Lizzy",
            "addressees": ["Mr. Darcy"],
            "text": "truth universally acknowledged",
            "seq": 0,
        },
        {
            "uid": "m:Event:quote:Q1",
            "quote_type": "anaphoric",
            "speaker": "Darcy",
            "addressees": [],
            "text": "I could easily forgive his pride",
            "seq": 1,
        },
        {
            "uid": "m:Event:quote:Q2",
            "quote_type": "explicit",
            "speaker": "Mama",
            "addressees": [],
            "text": "Oh my dear Mr Bennet",
            "seq": 2,
        },
    ]

    def execute_query(self, query, **params):
        if "is_dialogue" in query:
            return _Result(self._GOLD)
        if "CONTAINS $snippet" in query:
            return _Result([{"number": 1}])
        if "ch.number >=" in query:
            return _Result([{"number": n, "text": f"prose {n}"} for n in (0, 1, 2)])
        return _Result([])


def test_quote_length_curve_oracle_and_wrong(tmp_path):
    driver = _FakeDriver()
    # gold speaker (alias) -> canonical Main Name the oracle should answer
    gold_to_canon = {
        "truth universally acknowledged": "Elizabeth Bennet",
        "I could easily forgive his pride": "Fitzwilliam Darcy",
        "Oh my dear Mr Bennet": "Mrs. Bennet",
    }

    def oracle(prompt, num_ctx):
        answer = next((c for q, c in gold_to_canon.items() if q in prompt), "")
        return lc.AskResult(text=answer, prompt_tokens=num_ctx - lc._NUM_CTX_MARGIN)

    rep = lc.run_quote_length_curve(
        driver,
        "pdnc:Tiny",
        PDNC_NOVEL,
        oracle,
        lengths=[1000, 4000],
        items_per_length=3,
        seed=0,
    )
    assert rep["n_items"] == 3
    for p in rep["points"]:
        assert p["overall"] == 1.0  # oracle gets every quote
        assert p["no_answer_rate"] == 0.0
        assert p["truncated"] is False

    # a predictor that always answers the wrong character -> 0 accuracy, all wrong
    def wrong(prompt, num_ctx):
        return lc.AskResult(text="Butler", prompt_tokens=num_ctx)

    rep2 = lc.run_quote_length_curve(
        driver,
        "pdnc:Tiny",
        PDNC_NOVEL,
        wrong,
        lengths=[1000],
        items_per_length=3,
        seed=0,
    )
    p = rep2["points"][0]
    assert p["overall"] == 0.0
    assert p["wrong_rate"] + p["no_answer_rate"] == 1.0


# ---------------------------------------------------------------------------
# Location needle + cloze/coref length curves (reuse of the existing scorers)
# ---------------------------------------------------------------------------
def test_location_needle_instances():
    insts = lc.location_needle_instances()
    assert len(insts) == 8
    assert sum(i.expected == "FAIL" for i in insts) == 4
    assert sum(i.expected == "PASS" for i in insts) == 4
    assert all("CANON FACT" in i.fact for i in insts)


def _ctx_ask(answer_for):
    """A ContextAsk returning answer_for(prompt); tokens = num_ctx - margin (= target)."""

    def ask(prompt, num_ctx):
        return lc.AskResult(
            text=answer_for(prompt), prompt_tokens=num_ctx - lc._NUM_CTX_MARGIN
        )

    return ask


def test_cloze_length_curve_oracle_and_wrong():
    from evals.metrics.gpt4_books_cloze import ClozeRow

    rows = [
        ClozeRow("x", "Elizabeth", "MARK_A: [MASK] walked into the room."),
        ClozeRow("x", "Darcy", "MARK_B: [MASK] bowed stiffly."),
    ]

    def oracle(prompt):
        if "MARK_A" in prompt:
            return "Elizabeth"
        return "Darcy" if "MARK_B" in prompt else ""

    rep = lc.run_cloze_length_curve(
        _ctx_ask(oracle), rows, lengths=[1000, 4000], book_id="test", depth=0.5
    )
    assert rep["curve"] == "name_cloze"
    for p in rep["points"]:
        assert p["accuracy"] == 1.0
        assert p["abstain_rate"] == 0.0
        assert p["truncated"] is False  # actual tokens ~ target

    # naming a non-character -> all miss, accuracy 0
    rep2 = lc.run_cloze_length_curve(
        _ctx_ask(lambda p: "Nobody"), rows, lengths=[1000], book_id="test"
    )
    assert rep2["points"][0]["accuracy"] == 0.0
    assert rep2["points"][0]["miss_rate"] == 1.0


def test_coref_length_curve_oracle():
    from evals.metrics.litbank_coref import CorefPair, Mention

    pairs = [
        CorefPair(
            Mention(0, 0, "Pip"), Mention(0, 1, "he"), True, "SAME_MARK Pip ... he"
        ),
        CorefPair(
            Mention(1, 0, "Estella"),
            Mention(2, 0, "Joe"),
            False,
            "DIFF_MARK Estella ... Joe",
        ),
    ]

    def oracle(prompt):
        if "SAME_MARK" in prompt:
            return "yes"
        return "no" if "DIFF_MARK" in prompt else "maybe"

    rep = lc.run_coref_length_curve(
        _ctx_ask(oracle), pairs, lengths=[1000, 4000], doc_id="test", depth=0.5
    )
    assert rep["curve"] == "coreference"
    for p in rep["points"]:
        assert p["accuracy"] == 1.0
        assert p["entity_drift_rate"] == 0.0


# ---------------------------------------------------------------------------
# Per-window matrix banding (plot_context_curves) — bands by TARGET, not actual
# ---------------------------------------------------------------------------
def test_per_window_matrix_banding():
    from evals.plot_context_curves import _acc_matrix_lines, _target_band

    # each rung sits under the window it was AIMED at (actual tokens run lower)
    assert _target_band(1000) == 0
    assert _target_band(2000) == 1
    assert _target_band(3072) == 2  # yi's "4K" rung
    assert _target_band(7168) == 3  # gemma's "8K" rung
    assert _target_band(14976) == 4  # a "16K" rung
    assert _target_band(30976) is None  # 32K rung -> beyond the 16K grid

    prof = {
        "model": "m",
        "advertised_window": 8192,
        "curves": {
            "quote_attribution": {
                "points": [
                    {"target_tokens": 1000, "actual_tokens": 900, "overall": 0.9},
                    {"target_tokens": 8000, "actual_tokens": 6800, "overall": 0.6},
                ]
            }
        },
    }
    row = next(
        ln
        for ln in _acc_matrix_lines([prof], "quote_attribution", "overall", "Q")
        if ln.startswith("| m |")
    )
    cells = [c.strip() for c in row.strip("|").split("|")]  # model,1K,2K,4K,8K,16K
    assert cells[1] == "0.90"  # 1K band
    assert cells[4] == "0.60"  # 8K band (target 8000, actual 6800 -> still 8K)
    assert cells[2] == cells[3] == cells[5] == "—"  # untested bands
