"""Race the three baselines through the Change-3 metric harnesses.

Same question set, same candidates, same scoring (the real
score_quote_attribution / run_name_cloze). For each (method, task) we record
accuracy (overall + explicit vs non-explicit split for quotes), avg input/output
tokens, cost (tokens x price), and per-query wall-clock latency.

Real mode is guarded by Neo4j + dataset + API key like evals/run_eval.py;
everything is offline-testable by injecting a driver/ask/retrieve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.config import BaselineConfig  # noqa: E402
from baselines.flat_long_context import FlatLongContextPredictor  # noqa: E402
from baselines.graph_method import GraphPredictor  # noqa: E402
from baselines.vector_rag import VectorRAGPredictor  # noqa: E402

_PREDICTORS = (FlatLongContextPredictor, VectorRAGPredictor, GraphPredictor)


def _telemetry_dict(t) -> dict:
    return {
        "n_calls": t.n_calls,
        "avg_input_tokens": t.avg_input_tokens,
        "avg_output_tokens": t.avg_output_tokens,
        "avg_latency_seconds": t.avg_latency_seconds,
        "total_cost": t.total_cost,
    }


def run_baselines(
    manuscript_id: str,
    *,
    novel_dir: str | None = None,
    driver=None,
    ask=None,
    retrieve=None,
    config: BaselineConfig | None = None,
    sample_quotes: int | None = None,
    sample_offset: int = 0,
    max_cloze: int = 50,
    seed: int = 0,
) -> dict:
    """Run quote-attribution (pdnc only) and name-cloze for all three methods."""
    config = config or BaselineConfig()
    results: dict = {
        "manuscript_id": manuscript_id,
        "config": {
            "token_budget": config.token_budget,
            "rag_k": config.rag_k,
            "graph_window": config.graph_window,
            "prose_chapters_each_side": config.prose_chapters_each_side,
            "price_per_token": config.price_per_token,
        },
        "quote_attribution": {},
        "name_cloze": {},
    }

    def _new(cls):
        kw = {"retrieve": retrieve} if cls is VectorRAGPredictor else {}
        return cls(manuscript_id, driver, ask, config, **kw)

    # -- quote attribution (PDNC) -----------------------------------------
    if manuscript_id.startswith("pdnc:") and novel_dir:
        from evals.metrics.quote_attribution import (
            _resolve_allowed,
            fetch_gold_dialogue,
            score_quote_attribution,
        )
        from evals.metrics.pdnc_meta import read_pdnc_characters

        chars = read_pdnc_characters(novel_dir)
        gold = fetch_gold_dialogue(driver, manuscript_id)
        if sample_quotes is not None:
            gold = gold[sample_offset : sample_offset + sample_quotes]
        allowed, filter_rule = _resolve_allowed(gold, chars)

        for cls in _PREDICTORS:
            predictor = _new(cls)
            report = score_quote_attribution(
                gold,
                predictor.predict_quote,
                chars,
                allowed=allowed,
                filter_rule=filter_rule,
                manuscript_id=manuscript_id,
            )
            results["quote_attribution"][cls.method_name] = {
                "overall": report.overall,
                "explicit": report.explicit,
                "non_explicit": report.non_explicit,
                "implicit": report.implicit,
                "anaphoric": report.anaphoric,
                "n_quotes": report.n_quotes,
                "telemetry": _telemetry_dict(predictor.telemetry()),
            }

    # -- name cloze --------------------------------------------------------
    from evals.metrics.name_cloze import run_name_cloze

    for cls in _PREDICTORS:
        predictor = _new(cls)
        report = run_name_cloze(
            manuscript_id,
            predict=predictor.predict_cloze,
            driver=driver,
            max_passages=max_cloze,
            seed=seed,
        )
        results["name_cloze"][cls.method_name] = {
            "accuracy": report.accuracy,
            "n_passages": report.n_passages,
            "telemetry": _telemetry_dict(predictor.telemetry()),
        }

    return results


def format_table(results: dict) -> str:
    lines: list[str] = []
    cfg = results["config"]
    lines.append(f"Baselines for {results['manuscript_id']}")
    lines.append(
        f"  budget={cfg['token_budget']} tok  rag_k={cfg['rag_k']} "
        f"(budget-matched)  graph_window={cfg['graph_window']}  "
        f"price={cfg['price_per_token']:.2e}/tok"
    )

    qa = results.get("quote_attribution") or {}
    if qa:
        lines.append("\n  QUOTE ATTRIBUTION")
        lines.append(
            f"    {'method':<18}{'overall':>8}{'explicit':>9}{'non-exp':>8}"
            f"{'n':>5}{'in_tok':>8}{'out_tok':>8}{'cost$':>10}{'lat_s':>8}"
        )
        for method, r in qa.items():
            t = r["telemetry"]
            lines.append(
                f"    {method:<18}{r['overall']:>8.3f}{r['explicit']:>9.3f}"
                f"{r['non_explicit']:>8.3f}{r['n_quotes']:>5}"
                f"{t['avg_input_tokens']:>8.0f}{t['avg_output_tokens']:>8.0f}"
                f"{t['total_cost']:>10.4f}{t['avg_latency_seconds']:>8.3f}"
            )

    nc = results.get("name_cloze") or {}
    if nc:
        lines.append("\n  NAME CLOZE")
        lines.append(
            f"    {'method':<18}{'accuracy':>9}{'n':>5}{'in_tok':>8}"
            f"{'cost$':>10}{'lat_s':>8}"
        )
        for method, r in nc.items():
            t = r["telemetry"]
            lines.append(
                f"    {method:<18}{r['accuracy']:>9.3f}{r['n_passages']:>5}"
                f"{t['avg_input_tokens']:>8.0f}{t['total_cost']:>10.4f}"
                f"{t['avg_latency_seconds']:>8.3f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_baselines", description=__doc__)
    parser.add_argument("manuscript_id")
    parser.add_argument("--sample-quotes", type=int, default=30)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-cloze", type=int, default=50)
    args = parser.parse_args(argv)

    # Real-mode guards (like run_eval.py): need Neo4j; LLM methods need a key.
    try:
        from app import graph

        driver = graph.init_driver()
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        print(f"Neo4j not reachable ({exc}); baselines need a live graph. Skipped.")
        return 0

    from app.config import settings
    from baselines.llm_ask import make_llm_ask

    ask = make_llm_ask()
    if ask is None:
        print(
            "No ANTHROPIC_API_KEY / LLM_MODE!=anthropic: long-context & RAG will "
            "abstain (graph baseline still runs)."
        )

    novel_dir = None
    if args.manuscript_id.startswith("pdnc:"):
        folder = args.manuscript_id.split(":", 1)[1]
        novel_dir = str(Path(settings.pdnc_root) / "data" / folder)

    results = run_baselines(
        args.manuscript_id,
        novel_dir=novel_dir,
        driver=driver,
        ask=ask,
        sample_quotes=args.sample_quotes,
        sample_offset=args.sample_offset,
        max_cloze=args.max_cloze,
    )
    print(format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
