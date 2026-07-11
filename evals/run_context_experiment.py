"""Per-model context-length experiment runner — ONE model at a time (no loop).

Sweeps a single Ollama-served model up a context-length ladder and records where
its story-tracking accuracy collapses:

    curves:
      quote_attribution : accuracy vs input length (overall/explicit/non-explicit)
                          + the flat GraphPredictor reference line       [needs Neo4j]
      consistency_needle: catch a buried contradiction vs length & depth [no Neo4j]
    derived:
      effective_context = largest length still within 20% of the model's peak
                          (>=80%) — the reliably-usable window (the headline)
      failure_point     = first length where accuracy < 50% of the peak (collapse)
      crossover_point   = first length where the model drops below the graph line
      advertised vs effective context window

WORKFLOW (Colab compute, local codebase): identical to run_model_profile.py —
put the tunnel URL in OLLAMA_BASE_URL, then:
    python evals/run_context_experiment.py --model qwen2.5:7b
The ladder is auto-capped so every rung's whole prompt fits INSIDE the model's
advertised window and the --max-length hardware ceiling (~16K on a T4, ~32K on an
A100). We deliberately do not test past the window — there Ollama context-shifts
the prompt into garbage, a non-result. Resumable per-curve; detached-friendly.

Sets OLLAMA_MODEL from --model BEFORE importing app.config (settings singleton).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULT_DIR = Path(__file__).parent / "results" / "context"
DEFAULT_LADDER = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 131072]
DEFAULT_PDNC = "pdnc:PrideAndPrejudice"
DEFAULT_CLOZE_BOOK = "1342_pride_and_prejudice"  # gpt4-books; same novel as quotes
DEFAULT_COREF_DOC = "1400_great_expectations_brat"  # litbank; good person-cluster yield

# Marketed context windows (tokens), by Ollama family prefix. Effective (the
# failure point) is what the experiment measures against these.
_ADVERTISED = {
    "qwen2.5": 131072,
    "llama3.1": 131072,
    "llama3.2": 131072,
    "phi3.5": 131072,
    "mistral": 32768,
    "gemma2": 8192,
    "yi": 4096,
}


def advertised_window(model: str) -> int:
    return _ADVERTISED.get(model.split(":")[0], 32768)


def cap_ladder(
    ladder: list[int], advertised: int, max_length: int, margin: int = 1024
) -> list[int]:
    """Rungs whose whole prompt (prose target + `margin` for question/answer) fits
    inside the model's window AND the hardware ceiling — so the model sees the
    entire prompt (no context-shift, no positional-encoding breakdown). Tops out
    with a rung right at that fitting ceiling. We deliberately do NOT test beyond
    the window: past it the prompt overflows and Ollama mangles it, which is a
    non-result, not "degradation"."""
    ceiling = min(advertised, max_length) - margin
    if ceiling < 1000:
        return [ceiling] if ceiling >= 256 else []
    rungs = sorted(x for x in ladder if x <= ceiling)
    if not rungs or rungs[-1] < ceiling - 500:
        rungs.append(ceiling)
    return rungs


def safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Curve orchestration
# ---------------------------------------------------------------------------
def run_quote_curve(
    driver, manuscript_id, novel_dir, ask, ladder, items, seed, num_ctx_cap
) -> dict:
    """Curve A over a shared fixed sample + the flat GraphPredictor reference."""
    from baselines.config import BaselineConfig
    from baselines.graph_method import GraphPredictor
    from evals.metrics import long_context as lc
    from evals.metrics.pdnc_meta import read_pdnc_characters
    from evals.metrics.quote_attribution import (
        _resolve_allowed,
        fetch_gold_dialogue,
        score_quote_attribution,
    )

    chars = read_pdnc_characters(novel_dir)
    gold = fetch_gold_dialogue(driver, manuscript_id)
    allowed, filter_rule = _resolve_allowed(gold, chars)
    candidates = sorted(allowed)
    cache: dict = {}
    sample = lc.prepare_quote_sample(
        driver, manuscript_id, gold, chars, allowed, items, seed, cache
    )
    points = [
        lc.score_length(
            sample, L, driver, manuscript_id, candidates, chars, ask, num_ctx_cap
        ).to_dict()
        for L in ladder
    ]
    # Flat graph reference on the SAME sample (length-independent by construction).
    graph_pred = GraphPredictor(manuscript_id, driver, None, BaselineConfig())
    graph_report = score_quote_attribution(
        [q for q, _ in sample],
        graph_pred.predict_quote,
        chars,
        allowed=allowed,
        filter_rule=filter_rule,
        manuscript_id=manuscript_id,
    )
    graph_acc = graph_report.overall
    eff_tokens, eff_is_lower_bound = lc.effective_context(points, frac=0.8)
    return {
        "curve": "quote_attribution",
        "filter_rule": filter_rule,
        "n_items": len(sample),
        "graph_reference_accuracy": graph_acc,
        "points": points,
        # Reliably-usable length: largest rung still within 20% of the model's
        # peak. The headline "effective window" (vs the advertised one).
        "effective_context": eff_tokens,
        "effective_frac": 0.8,
        "effective_is_lower_bound": eff_is_lower_bound,
        # Severe bounds: full collapse (< 50% of peak) / below the graph line.
        # Both are None here — the degradation is steady erosion, not a cliff.
        "failure_point": lc.failure_point(points),
        "crossover_point": lc.crossover_point(points, graph_acc),
    }


def make_judge_ask(num_ctx_holder):
    """Wrap OllamaLLM.judge into (canon, draft, num_ctx) -> (verdict|None, tokens)."""
    from app.llm import OllamaLLM

    llm = OllamaLLM()

    def judge_ask(canon: str, draft: str, num_ctx: int):
        llm.num_ctx = num_ctx
        try:
            verdict = llm.judge(canon, draft).verdict
        except Exception:
            verdict = None
        return verdict, llm.last_prompt_tokens

    return judge_ask


def load_cloze_rows(book: str, items: int, seed: int) -> list:
    """Deterministic sample of masked cloze passages from a gpt4-books file."""
    from app.config import settings
    from evals.metrics.gpt4_books_cloze import book_path, read_book, select_rows

    rows = read_book(book_path(book, settings.gpt4books_root))
    selected, _ = select_rows(rows, items, seed)
    return selected


def load_coref_pairs(doc: str, items: int, seed: int) -> list:
    """Deterministic balanced person-mention pairs from a LitBank conll doc."""
    from app.config import settings
    from evals.metrics.litbank_coref import build_pairs, conll_path, parse_conll

    return build_pairs(parse_conll(conll_path(doc, settings.litbank_root)), items, seed)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _assemble(model, base_url, advertised, ladder, seed, curves) -> dict:
    return {
        "model": model,
        "base_url": base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "advertised_window": advertised,
        "ladder": ladder,
        "seed": seed,
        "curves": curves,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_context_experiment", description=__doc__)
    parser.add_argument(
        "--model", default=None, help="Ollama tag; overrides OLLAMA_MODEL"
    )
    parser.add_argument("--lengths", default=None, help="comma list; default ladder")
    parser.add_argument(
        "--max-length", type=int, default=16000, help="hardware ceiling"
    )
    parser.add_argument("--items-per-length", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pdnc-manuscript", default=DEFAULT_PDNC)
    parser.add_argument("--cloze-book", default=DEFAULT_CLOZE_BOOK)
    parser.add_argument("--coref-doc", default=DEFAULT_COREF_DOC)
    parser.add_argument(
        "--curves",
        default="quote,needle,cloze,coref,location",
        help="any of: quote,needle,cloze,coref,location",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    from app.config import settings

    if settings.llm_mode != "ollama":
        print(
            "Set LLM_MODE=ollama and OLLAMA_BASE_URL (the Colab tunnel), then re-run."
        )
        return 2
    model = args.model or settings.ollama_model
    advertised = advertised_window(model)
    ladder_in = (
        [int(x) for x in args.lengths.split(",")] if args.lengths else DEFAULT_LADDER
    )
    ladder = cap_ladder(ladder_in, advertised, args.max_length)
    want = set(args.curves.split(","))

    from evals.metrics import long_context as lc

    ask = lc.make_ollama_context_ask()

    driver = None
    novel_dir = None
    if "quote" in want:
        try:
            from app import graph

            driver = graph.init_driver()
            driver.verify_connectivity()
            folder = args.pdnc_manuscript.split(":", 1)[1]
            novel_dir = str(Path(settings.pdnc_root) / "data" / folder)
        except Exception as exc:  # noqa: BLE001
            print(f"Neo4j not reachable ({exc}); quote curve skipped.")
            driver = None

    out_path = RESULT_DIR / f"{safe_model_name(model)}.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else None
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prev = {} if args.force else (existing or {}).get("curves", {})
    # Seed from prev so a partial --curves run keeps the curves it isn't recomputing
    # (each requested curve below overwrites its entry; others are preserved).
    curves: dict = dict(prev)

    def checkpoint():
        out_path.write_text(
            json.dumps(
                _assemble(
                    model,
                    settings.ollama_base_url,
                    advertised,
                    ladder,
                    args.seed,
                    curves,
                ),
                indent=2,
            )
        )

    print(f"Context experiment: {model}  advertised={advertised}  ladder={ladder}\n")

    # Curve A — quote attribution vs length (needs Neo4j)
    if "quote" in want:
        if "quote_attribution" in prev:
            curves["quote_attribution"] = prev["quote_attribution"]
            print("quote_attribution: cached")
        elif driver is None:
            curves["quote_attribution"] = {"skipped": "no Neo4j"}
        else:
            print("quote_attribution: running...")
            curves["quote_attribution"] = run_quote_curve(
                driver,
                args.pdnc_manuscript,
                novel_dir,
                ask,
                ladder,
                args.items_per_length,
                args.seed,
                advertised,
            )
            checkpoint()

    # Curve B — consistency needle vs length (no Neo4j)
    if "needle" in want:
        if "consistency_needle" in prev:
            curves["consistency_needle"] = prev["consistency_needle"]
            print("consistency_needle: cached")
        else:
            print("consistency_needle: running...")
            curves["consistency_needle"] = lc.run_needle_curve(
                make_judge_ask(None), lengths=ladder, num_ctx_cap=advertised
            )
            checkpoint()

    # Curve C — name cloze vs length (no Neo4j; reads a gpt4-books file)
    if "cloze" in want:
        if "name_cloze" in prev:
            curves["name_cloze"] = prev["name_cloze"]
            print("name_cloze: cached")
        else:
            try:
                rows = load_cloze_rows(
                    args.cloze_book, args.items_per_length, args.seed
                )
            except Exception as exc:  # noqa: BLE001
                print(f"cloze data unavailable ({exc}); cloze curve skipped.")
                rows = []
            if rows:
                print("name_cloze: running...")
                curves["name_cloze"] = lc.run_cloze_length_curve(
                    ask,
                    rows,
                    lengths=ladder,
                    book_id=args.cloze_book,
                    num_ctx_cap=advertised,
                )
                checkpoint()
            else:
                curves["name_cloze"] = {"skipped": "no cloze data"}

    # Curve D — coreference / entity drift vs length (no Neo4j; reads litbank conll)
    if "coref" in want:
        if "coreference" in prev:
            curves["coreference"] = prev["coreference"]
            print("coreference: cached")
        else:
            try:
                pairs = load_coref_pairs(
                    args.coref_doc, args.items_per_length, args.seed
                )
            except Exception as exc:  # noqa: BLE001
                print(f"coref data unavailable ({exc}); coref curve skipped.")
                pairs = []
            if pairs:
                print("coreference: running...")
                curves["coreference"] = lc.run_coref_length_curve(
                    ask,
                    pairs,
                    lengths=ladder,
                    doc_id=args.coref_doc,
                    num_ctx_cap=advertised,
                )
                checkpoint()
            else:
                curves["coreference"] = {"skipped": "no coref data"}

    # Curve E — location-inconsistency needle vs length (no Neo4j)
    if "location" in want:
        if "location_needle" in prev:
            curves["location_needle"] = prev["location_needle"]
            print("location_needle: cached")
        else:
            print("location_needle: running...")
            curves["location_needle"] = lc.run_needle_curve(
                make_judge_ask(None),
                lengths=ladder,
                instances=lc.location_needle_instances(),
                num_ctx_cap=advertised,
            )
            checkpoint()

    out_path.write_text(
        json.dumps(
            _assemble(
                model, settings.ollama_base_url, advertised, ladder, args.seed, curves
            ),
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")
    _print_summary(model, advertised, curves)
    return 0


def _print_summary(model: str, advertised: int, curves: dict) -> None:
    print(f"\n{'─' * 52}\nContext profile: {model}  (advertised {advertised} tok)")
    qa = curves.get("quote_attribution") or {}
    if "points" in qa:
        print(f"  graph reference accuracy: {qa['graph_reference_accuracy']:.3f}")
        print("  quote attribution vs length (actual_tok: overall / non-explicit):")
        for p in qa["points"]:
            flag = "  <TRUNCATED" if p["truncated"] else ""
            print(
                f"    {p['actual_tokens']:>7.0f}: {p['overall']:.3f} / "
                f"{p['non_explicit']:.3f}{flag}"
            )
        eff = qa.get("effective_context")
        lb = "≥" if qa.get("effective_is_lower_bound") else ""
        eff_str = f"{lb}{eff:.0f} tok" if eff is not None else "below first rung"
        print(f"  effective window (≥80% of peak): {eff_str}")
        print(f"  hard-failure point (<50% peak) : {qa['failure_point']}")
        print(
            f"  crossover pt  : {qa['crossover_point']} (graph beats model past here)"
        )
    nd = curves.get("consistency_needle") or {}
    if "points" in nd:
        print("  needle detection recall (actual_tok @ depth):")
        for p in nd["points"]:
            print(
                f"    {p['actual_tokens']:>7.0f} @ {p['depth']:.0%}: "
                f"recall={p['detection_recall']:.2f} over_flag={p['over_flag_rate']:.2f}"
            )
    cl = curves.get("name_cloze") or {}
    if "points" in cl:
        print("  name cloze vs length (actual_tok: accuracy / entity_drift):")
        for p in cl["points"]:
            print(
                f"    {p['actual_tokens']:>7.0f}: {p['accuracy']:.3f} / "
                f"{p['entity_drift_rate']:.3f}"
            )
    co = curves.get("coreference") or {}
    if "points" in co:
        print("  coreference vs length (actual_tok: accuracy / entity_drift):")
        for p in co["points"]:
            print(
                f"    {p['actual_tokens']:>7.0f}: {p['accuracy']:.3f} / "
                f"{p['entity_drift_rate']:.3f}"
            )
    loc = curves.get("location_needle") or {}
    if "points" in loc:
        print("  location-inconsistency detection (actual_tok @ depth):")
        for p in loc["points"]:
            print(
                f"    {p['actual_tokens']:>7.0f} @ {p['depth']:.0%}: "
                f"recall={p['detection_recall']:.2f} over_flag={p['over_flag_rate']:.2f}"
            )


if __name__ == "__main__":
    sys.exit(main())
