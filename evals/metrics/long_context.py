"""Context-length failure-point experiment.

Curve A — quote attribution vs input length: feed each PDNC gold quote with a
prose window of ~L tokens (the quote centered), set Ollama's `num_ctx` to cover
it, and score attribution accuracy at each L. The point is to watch accuracy
degrade as L grows toward (and past) each model's advertised context window.

Reports overall / explicit / non-explicit accuracy per length (non-explicit =
anaphoric/implicit quotes that require looking back = the true long-range
signal), plus a wrong-answer vs no-answer split, and derives a failure point and
a crossover point (where the model drops below the length-independent graph line).

The `num_ctx` fix matters: without it Ollama silently truncates every prompt to
~2048 tokens, which is exactly why the earlier fixed-context runs were all at ~2K.

Curve B (consistency needle) lives alongside this in a later addition.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Protocol

from baselines.base import BasePredictor, fetch_chapter_window, locate_quote_chapter
from baselines.config import AVG_CHUNK_TOKENS
from baselines.flat_long_context import _QUOTE_PROMPT
from evals.metrics.pdnc_meta import PdncChars, read_pdnc_characters
from evals.metrics.quote_attribution import (
    GoldQuote,
    _resolve_allowed,
    fetch_gold_dialogue,
)

_NON_EXPLICIT = ("implicit", "anaphoric")
# Headroom over the prose budget for the prompt template, candidate list, and the
# short answer, so the whole prompt fits inside num_ctx (no context-shift). Kept
# modest so small-window models (Gemma 8K, Yi 4K) still get usable prose room.
_NUM_CTX_MARGIN = 1024
# The len//4 heuristic UNDER-counts real tokens for dense literary prose (measured
# ~1.3x: a len//4 "4000" built ~5200 real tokens). We divide targets by this so
# the built prose's REAL length matches the intended target and the prompt fits
# num_ctx (otherwise it overflows the window and Ollama context-shifts to garbage).
_TOKEN_DENSITY = 1.3


@dataclass
class AskResult:
    text: str
    prompt_tokens: int  # real prompt length Ollama evaluated (prompt_eval_count)
    error: bool = False  # server error / timeout — model couldn't run at this length


# ask(prompt, num_ctx) -> AskResult. Injected so tests need no live model.
ContextAsk = Callable[[str, int], AskResult]


class _Driver(Protocol):
    def execute_query(self, query: str, **params): ...


@dataclass
class LengthPoint:
    target_tokens: int
    actual_tokens: float  # mean real prompt length at this rung
    n: int
    overall: float
    explicit: float
    non_explicit: float
    wrong_rate: float  # answered, but wrong (comprehension failure)
    no_answer_rate: float  # empty / unparseable (format collapse)
    error_rate: float  # server error / OOM — couldn't run at this length (hardware)
    truncated: bool  # actual << target -> model/hardware capped the context

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Live Ollama context-ask (sets num_ctx, records the real prompt length)
# ---------------------------------------------------------------------------
def make_ollama_context_ask(max_tokens: int = 32) -> ContextAsk:
    """A `(prompt, num_ctx) -> AskResult` ask against the configured Ollama."""
    import httpx

    from app.config import settings

    client = httpx.Client(base_url=settings.ollama_base_url, timeout=600)
    model = settings.ollama_model

    def ask(prompt: str, num_ctx: int) -> AskResult:
        # A server error (commonly OOM at a large num_ctx / prompt on limited VRAM)
        # is a real result — the model can't run at this length on this hardware —
        # so we record it and let the run continue rather than crash the sweep.
        try:
            resp = client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Answer in English. Be terse."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_tokens,
                        "num_ctx": num_ctx,
                    },
                },
            )
            resp.raise_for_status()
            body = resp.json()
            return AskResult(
                text=body["message"]["content"],
                prompt_tokens=body.get("prompt_eval_count", 0),
            )
        except Exception:
            return AskResult(text="", prompt_tokens=0, error=True)

    return ask


# ---------------------------------------------------------------------------
# Sample preparation (fixed, locatable quotes reused across every length)
# ---------------------------------------------------------------------------
def prepare_quote_sample(
    driver: _Driver,
    manuscript_id: str,
    gold: list[GoldQuote],
    chars: PdncChars,
    allowed: set[str],
    n: int,
    seed: int,
    cache: dict,
) -> list[tuple[GoldQuote, int]]:
    """Deterministically pick `n` scorable, prose-locatable quotes (quote, center).

    The SAME quotes are scored at every length so the curve isolates length.
    """
    scorable = [q for q in gold if chars.normalize(q.speaker) in allowed and q.text]
    random.Random(seed).shuffle(scorable)
    out: list[tuple[GoldQuote, int]] = []
    for q in scorable:
        center = locate_quote_chapter(driver, manuscript_id, q.text, cache)
        if center is not None:
            out.append((q, center))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Curve A — quote attribution vs length
# ---------------------------------------------------------------------------
def _centered_prose(driver: _Driver, mid: str, center: int, target_tokens: int) -> str:
    """~target_tokens of REAL prose centered on the quote's chapter (quote middle).

    AVG_CHUNK_TOKENS is a len//4 estimate that under-counts real tokens, so we
    divide the target by _TOKEN_DENSITY: the built prose then measures ~target_tokens
    in real tokens and the prompt fits num_ctx (no window overflow / context-shift).
    """
    real_target = target_tokens / _TOKEN_DENSITY
    each_side = max(1, round(real_target / AVG_CHUNK_TOKENS / 2))
    chapters = fetch_chapter_window(driver, mid, center - each_side, center + each_side)
    return "\n\n".join(c["text"] for c in chapters)


def score_length(
    sample: list[tuple[GoldQuote, int]],
    target_tokens: int,
    driver: _Driver,
    manuscript_id: str,
    candidates: list[str],
    chars: PdncChars,
    ask: ContextAsk,
    num_ctx_cap: "int | None" = None,
) -> LengthPoint:
    """Score the fixed sample at one context length.

    `num_ctx_cap` (= the model's trained window) prevents requesting a context
    larger than the model supports: doing so triggers positional-encoding
    breakdown (garbage output), which would masquerade as a comprehension failure.
    The prompt is still allowed to exceed the window — Ollama truncates it, which
    is the honest "prompt bigger than the window" result.
    """
    correct = wrong = no_answer = errors = 0
    per_type = {"explicit": [0, 0], "non_explicit": [0, 0]}
    tok_sum = ok_calls = 0
    num_ctx = target_tokens + _NUM_CTX_MARGIN
    if num_ctx_cap:
        num_ctx = min(num_ctx, num_ctx_cap)

    for q, center in sample:
        prose = _centered_prose(driver, manuscript_id, center, target_tokens)
        prompt = _QUOTE_PROMPT.format(
            prose=prose, quote=q.text, candidates=", ".join(candidates)
        )
        res = ask(prompt, num_ctx)
        bucket = "non_explicit" if q.quote_type in _NON_EXPLICIT else "explicit"
        per_type[bucket][1] += 1
        if res.error:
            errors += 1  # server/OOM error -> counts as a miss (can't run here)
            continue
        tok_sum += res.prompt_tokens
        ok_calls += 1
        pred = BasePredictor.pick_candidate(res.text, candidates)
        if pred is None:
            no_answer += 1
        elif chars.normalize(pred) == chars.normalize(q.speaker):
            correct += 1
            per_type[bucket][0] += 1
        else:
            wrong += 1

    n = len(sample) or 1
    # actual length is the mean over calls that actually ran (0 if all errored)
    actual = tok_sum / ok_calls if ok_calls else 0.0
    return LengthPoint(
        target_tokens=target_tokens,
        actual_tokens=actual,
        n=len(sample),
        overall=correct / n,
        explicit=_safe_ratio(per_type["explicit"]),
        non_explicit=_safe_ratio(per_type["non_explicit"]),
        wrong_rate=wrong / n,
        no_answer_rate=no_answer / n,
        error_rate=errors / n,
        # If Ollama capped num_ctx below what we asked, the real prompt is far
        # short of target — a hardware/model context ceiling, itself a result.
        truncated=bool(ok_calls) and actual < 0.75 * target_tokens,
    )


def _safe_ratio(pair: list[int]) -> float:
    caught, total = pair
    return caught / total if total else 0.0


def run_quote_length_curve(
    driver: _Driver,
    manuscript_id: str,
    novel_dir: str,
    ask: ContextAsk,
    *,
    lengths: list[int],
    items_per_length: int,
    seed: int = 0,
    num_ctx_cap: "int | None" = None,
) -> dict:
    """End-to-end Curve A: fixed sample scored across the length ladder."""
    chars = read_pdnc_characters(novel_dir)
    gold = fetch_gold_dialogue(driver, manuscript_id)
    allowed, filter_rule = _resolve_allowed(gold, chars)
    candidates = sorted(allowed)
    cache: dict = {}
    sample = prepare_quote_sample(
        driver, manuscript_id, gold, chars, allowed, items_per_length, seed, cache
    )
    points = [
        score_length(
            sample, L, driver, manuscript_id, candidates, chars, ask, num_ctx_cap
        ).to_dict()
        for L in lengths
    ]
    return {
        "curve": "quote_attribution",
        "filter_rule": filter_rule,
        "n_candidates": len(candidates),
        "n_items": len(sample),
        "points": points,
    }


# ---------------------------------------------------------------------------
# Failure point & crossover (pure; operate on a list of point dicts)
# ---------------------------------------------------------------------------
def failure_point(points: list[dict], frac: float = 0.5, key: str = "overall"):
    """First actual length where accuracy drops below `frac` of the model's peak.

    Returns the actual_tokens at that rung, or None if it never fails in range.
    """
    if not points:
        return None
    peak = max(p[key] for p in points)
    threshold = frac * peak
    for p in sorted(points, key=lambda p: p["actual_tokens"]):
        if p[key] < threshold:
            return p["actual_tokens"]
    return None


def crossover_point(points: list[dict], graph_accuracy: float, key: str = "overall"):
    """First actual length where the model drops BELOW the flat graph line."""
    for p in sorted(points, key=lambda p: p["actual_tokens"]):
        if p[key] < graph_accuracy:
            return p["actual_tokens"]
    return None


def effective_context(
    points: list[dict], frac: float = 0.8, key: str = "overall"
) -> tuple[float | None, bool]:
    """The reliably-usable context length — the honest answer to "how many tokens
    can this model take in and still track the story."

    It is the largest tested length at which accuracy is still >= `frac` of the
    model's own peak, with no earlier drop below that bar. `frac=0.8` (still
    within 20% of its best) follows the threshold convention for effective context
    length in the long-context literature (RULER / NoLiMa); the 50%-of-peak
    `failure_point` above is the far more severe "it has fully collapsed" bound.

    Returns (tokens, is_lower_bound). `is_lower_bound=True` means accuracy never
    fell below the bar in the tested range, so the true effective length is only
    known to be AT LEAST `tokens` — we hit the hardware/window ceiling before
    finding where it breaks. `(None, False)` means it was already below the bar at
    the very first (shortest) rung.
    """
    if not points:
        return None, False
    ordered = sorted(points, key=lambda p: p["actual_tokens"])
    peak = max(p[key] for p in ordered)
    threshold = frac * peak
    last_good: float | None = None
    for p in ordered:
        if p[key] < threshold:
            return last_good, False  # found where it drops below the bar
        last_good = p["actual_tokens"]
    return last_good, True  # never dropped in range -> lower bound


# ---------------------------------------------------------------------------
# Curve B — consistency needle vs length (buried canon fact + depth axis)
# ---------------------------------------------------------------------------
# Canon-free neutral filler (no Duskwall names); repeated to any length. Repeated
# haystack filler is the standard needle-test convention (cf. RULER).
_FILLER_SEED = (
    "The harbour town woke slowly under a grey and even sky. Gulls turned above "
    "the tide line while carts came down from the hills with sacks of grain and "
    "bolts of cloth. In the market the stalls opened one by one, and the smell of "
    "bread and salt drifted between the awnings. A clerk counted coins beneath a "
    "faded sign; two fishermen argued mildly over the price of nets; a child chased "
    "a rolling hoop across the wet stones. Nothing of consequence happened, and the "
    "hours passed as such hours do, unremarkable and calm. "
)

# judge_ask(canon, draft, num_ctx) -> (verdict "PASS"/"FAIL"/None, prompt_tokens)
NeedleJudge = Callable[[str, str, int], "tuple[str | None, int]"]


@dataclass
class NeedleInstance:
    fact: str  # the canon fact buried in the filler
    draft: str  # the passage to judge against it
    expected: str  # "FAIL" (draft contradicts fact) | "PASS" (consistent)


def default_needle_instances() -> list[NeedleInstance]:
    """Buried-fact contradictions + clean controls (own fact each, no shared cue)."""
    return [
        NeedleInstance(
            "CANON FACT: Captain Brann was killed at the Citadel gate; he is dead, "
            "and in Duskwall the dead do not return.",
            "Brann strode into the hall, drew his sword, and barked an order at "
            "the assembled guards.",
            "FAIL",
        ),
        NeedleInstance(
            "CANON FACT: Mara cannot read or write; she is illiterate.",
            "Mara broke the seal and read the long letter twice, her eyes moving "
            "quickly down the page.",
            "FAIL",
        ),
        NeedleInstance(
            "CANON FACT: Sera lives and works at the Old Library, across the city "
            "from the docks.",
            "Sera spent the whole day at the docks, hauling nets and tallying "
            "cargo for the harbourmaster.",
            "FAIL",
        ),
        NeedleInstance(
            "CANON FACT: The city gates are sealed at dusk and cannot open again "
            "until dawn.",
            "A little past midnight the eastern gate swung wide to let the wagons "
            "roll through into the city.",
            "FAIL",
        ),
        NeedleInstance(
            "CANON FACT: Captain Brann was killed at the Citadel gate; he is dead.",
            "Mara laid a single flower where Brann had fallen. He was gone, and she "
            "turned back toward the docks.",
            "PASS",
        ),
        NeedleInstance(
            "CANON FACT: Mara cannot read or write; she is illiterate.",
            'Mara pushed the letter across the table. "Read it to me," she said, '
            "and Sera read it aloud.",
            "PASS",
        ),
        NeedleInstance(
            "CANON FACT: Sera lives and works at the Old Library.",
            "A dockhand carried Mara's note up to the Old Library, where Sera read "
            "it among her shelves and sent a reply.",
            "PASS",
        ),
        NeedleInstance(
            "CANON FACT: The city gates are sealed at dusk until dawn.",
            "They waited by the eastern gate until the first grey light, when the "
            "seals were lifted and it could open.",
            "PASS",
        ),
    ]


def build_haystack(fact: str, target_tokens: int, depth: float) -> str:
    """~target_tokens of REAL neutral filler with `fact` inserted at `depth` (0..1).

    Divided by _TOKEN_DENSITY so the real length matches the target and the prompt
    fits num_ctx (same reason as _centered_prose).
    """
    from baselines.config import count_tokens

    real_target = int(target_tokens / _TOKEN_DENSITY)
    fact_toks = count_tokens(fact)
    filler_budget = max(0, real_target - fact_toks)
    reps = max(1, filler_budget // count_tokens(_FILLER_SEED) + 1)
    filler = (_FILLER_SEED * reps).strip()
    # Split the filler at the depth fraction and insert the fact there.
    cut = int(len(filler) * max(0.0, min(1.0, depth)))
    space = filler.rfind(" ", 0, cut) if cut else 0
    cut = space if space > 0 else cut
    return f"{filler[:cut]}\n\n{fact}\n\n{filler[cut:]}".strip()


def run_needle_curve(
    judge_ask: NeedleJudge,
    *,
    lengths: list[int],
    depths: list[float] = (0.0, 0.5, 0.9),
    instances: list[NeedleInstance] | None = None,
    num_ctx_cap: "int | None" = None,
) -> dict:
    """Curve B: can the model still catch a contradiction of a fact buried at
    depth `d` under `L` tokens of filler? Detection recall + over-flag per (L, d).

    `num_ctx_cap` (= the model's trained window) avoids requesting a context
    beyond what the model supports (which produces garbage, not a real miss).
    """
    instances = instances or default_needle_instances()
    points: list[dict] = []
    for target in lengths:
        num_ctx = target + _NUM_CTX_MARGIN
        if num_ctx_cap:
            num_ctx = min(num_ctx, num_ctx_cap)
        for depth in depths:
            tp = fn = fp = tn = malformed = 0
            tok_sum = 0
            for inst in instances:
                canon = build_haystack(inst.fact, target, depth)
                verdict, prompt_tokens = judge_ask(canon, inst.draft, num_ctx)
                tok_sum += prompt_tokens
                if verdict is None:
                    malformed += 1
                flagged = verdict == "FAIL"
                if inst.expected == "FAIL":
                    tp += flagged
                    fn += not flagged
                else:
                    fp += flagged
                    tn += not flagged
            n = len(instances) or 1
            n_pos = tp + fn
            n_neg = fp + tn
            points.append(
                {
                    "target_tokens": target,
                    "depth": depth,
                    "actual_tokens": tok_sum / n,
                    "n": len(instances),
                    "detection_recall": tp / n_pos if n_pos else 0.0,
                    "over_flag_rate": fp / n_neg if n_neg else 0.0,
                    "malformed_rate": malformed / n,
                    "truncated": (tok_sum / n) < 0.75 * target,
                }
            )
    return {"curve": "consistency_needle", "depths": list(depths), "points": points}


# ---------------------------------------------------------------------------
# Needle sanity check — proves the model actually attends to N tokens
# (and demonstrates the old ~2K truncation: retrieval collapses when num_ctx<L)
# ---------------------------------------------------------------------------
_MAGIC_PROMPT = (
    "{haystack}\n\nQuestion: what is the special access code mentioned above? "
    "Reply with only the number.\nAnswer:"
)


def needle_retrieval_check(
    ask: ContextAsk,
    *,
    lengths: list[int],
    num_ctx: "int | None" = None,
    depth: float = 0.5,
) -> list[dict]:
    """Plant a magic code in filler at `depth`, ask the model to read it back.

    If `num_ctx` is None it is set to cover each length (the honest test); pass a
    small fixed value (e.g. 2048) to reproduce the silent-truncation failure.
    """
    out = []
    for target in lengths:
        code = 40000 + target  # deterministic, length-specific
        fact = f"The special access code is {code}."
        haystack = build_haystack(fact, target, depth)
        nc = num_ctx if num_ctx is not None else target + _NUM_CTX_MARGIN
        res = ask(_MAGIC_PROMPT.format(haystack=haystack), nc)
        out.append(
            {
                "target_tokens": target,
                "actual_tokens": res.prompt_tokens,
                "retrieved": str(code) in res.text,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Curve D — location-inconsistency needle (a location-specific contradiction set)
# ---------------------------------------------------------------------------
def location_needle_instances() -> list[NeedleInstance]:
    """Location contradictions (wrong place / two places at once / impossible move)
    + clean 'bait' controls that respect the geography. Feeds run_needle_curve, so
    it yields the location-inconsistency rate across length (the failure profile's
    `location` metric, now on the context-length axis)."""
    return [
        NeedleInstance(
            "CANON FACT: Sera lives and works at the Old Library, across the city "
            "from the docks.",
            "All morning Sera worked the fishing boats down at the docks, far from "
            "any book or shelf.",
            "FAIL",
        ),
        NeedleInstance(
            "CANON FACT: The Old Library and the Citadel stand on opposite sides of "
            "Duskwall, an hour's walk apart.",
            "Sera stepped out of the Old Library and, in the same breath, was inside "
            "the Citadel's great hall.",
            "FAIL",
        ),
        NeedleInstance(
            "CANON FACT: Mara works the Duskwall docks and has never been inside the "
            "royal Citadel.",
            "Mara paced the private royal chambers deep inside the Citadel as if she "
            "had lived there for years.",
            "FAIL",
        ),
        NeedleInstance(
            "CANON FACT: Duskwall sits far inland, three days' ride from the nearest "
            "coast.",
            "By evening they strolled along Duskwall's seafront, watching the waves "
            "break on the city's own shore.",
            "FAIL",
        ),
        NeedleInstance(
            "CANON FACT: Sera lives and works at the Old Library.",
            "Sera spent the day among the shelves of the Old Library, cataloguing a "
            "crate of old maps.",
            "PASS",
        ),
        NeedleInstance(
            "CANON FACT: Mara works the Duskwall docks.",
            "Mara loaded crates on the docks before dawn, then haggled with a ship's "
            "captain over the fare.",
            "PASS",
        ),
        NeedleInstance(
            "CANON FACT: The Old Library and the Citadel are an hour's walk apart.",
            "Sera left the Old Library and, an hour later, arrived footsore at the "
            "Citadel gate.",
            "PASS",
        ),
        NeedleInstance(
            "CANON FACT: Duskwall sits far inland, three days' ride from the coast.",
            "It took the messenger three hard days of riding to carry the letter from "
            "Duskwall down to the coast.",
            "PASS",
        ),
    ]


# ---------------------------------------------------------------------------
# Curve E — name cloze vs length (mask a name, bury the passage, still recall it?)
# ---------------------------------------------------------------------------
def run_cloze_length_curve(
    ask: ContextAsk,
    rows: list,
    *,
    lengths: list[int],
    book_id: str,
    depth: float = 0.5,
    num_ctx_cap: "int | None" = None,
) -> dict:
    """Bury each masked cloze passage under L tokens of neutral filler at `depth`,
    then score name recall by reusing gpt4_books_cloze.score_cloze unchanged (a
    haystacking predictor is all that's new). Accuracy + entity-drift/miss/abstain
    per length — the failure profile's cloze metric on the context-length axis."""
    from evals.metrics.gpt4_books_cloze import _CLOZE_PROMPT, score_cloze

    points: list[dict] = []
    for target in lengths:
        num_ctx = target + _NUM_CTX_MARGIN
        if num_ctx_cap:
            num_ctx = min(num_ctx, num_ctx_cap)
        toks: list[int] = []

        def predict(passage, _book, target=target, num_ctx=num_ctx, toks=toks):
            res = ask(
                _CLOZE_PROMPT.format(masked=build_haystack(passage, target, depth)),
                num_ctx,
            )
            if res.error:
                return None  # server error/OOM -> abstain (score_cloze counts it)
            toks.append(res.prompt_tokens)
            first = res.text.strip().splitlines()[0].strip() if res.text.strip() else ""
            return first or None

        rep = score_cloze(
            rows, predict, book_id=book_id, n_total=len(rows), sample_indices=[]
        )
        actual = sum(toks) / len(toks) if toks else 0.0
        points.append(
            {
                "target_tokens": target,
                "actual_tokens": actual,
                "n": rep.n,
                "accuracy": rep.accuracy,
                "entity_drift_rate": rep.entity_drift_rate,
                "miss_rate": rep.miss_rate,
                "abstain_rate": rep.abstain_rate,
                "truncated": bool(toks) and actual < 0.75 * target,
            }
        )
    return {"curve": "name_cloze", "depth": depth, "book_id": book_id, "points": points}


# ---------------------------------------------------------------------------
# Curve F — coreference vs length (two mentions buried together, "same person?")
# ---------------------------------------------------------------------------
def run_coref_length_curve(
    ask: ContextAsk,
    pairs: list,
    *,
    lengths: list[int],
    doc_id: str,
    depth: float = 0.5,
    num_ctx_cap: "int | None" = None,
) -> dict:
    """Bury each coref pair's resolving context under L tokens of filler at `depth`,
    then score 'same person?' by reusing litbank_coref.score_coref unchanged.
    Accuracy + entity-drift/split/merge per length — the failure profile's coref /
    entity-drift metric on the context-length axis."""
    from evals.metrics.litbank_coref import _COREF_PROMPT, score_coref

    points: list[dict] = []
    for target in lengths:
        num_ctx = target + _NUM_CTX_MARGIN
        if num_ctx_cap:
            num_ctx = min(num_ctx, num_ctx_cap)
        toks: list[int] = []

        def predict(context, a, b, target=target, num_ctx=num_ctx, toks=toks):
            res = ask(
                _COREF_PROMPT.format(
                    context=build_haystack(context, target, depth), a=a, b=b
                ),
                num_ctx,
            )
            if res.error:
                return None
            toks.append(res.prompt_tokens)
            low = res.text.strip().lower()
            if low.startswith(("yes", "same")):
                return True
            if low.startswith(("no", "diff")):
                return False
            return None

        rep = score_coref(pairs, predict, doc_id=doc_id)
        actual = sum(toks) / len(toks) if toks else 0.0
        points.append(
            {
                "target_tokens": target,
                "actual_tokens": actual,
                "n": rep.n,
                "accuracy": rep.accuracy,
                "entity_drift_rate": rep.entity_drift_rate,
                "positive_miss_rate": rep.positive_miss_rate,
                "false_merge_rate": rep.false_merge_rate,
                "abstain_rate": rep.abstain_rate,
                "truncated": bool(toks) and actual < 0.75 * target,
            }
        )
    return {"curve": "coreference", "depth": depth, "doc_id": doc_id, "points": points}
