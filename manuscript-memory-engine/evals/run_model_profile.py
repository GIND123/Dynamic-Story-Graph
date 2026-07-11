"""Per-model failure-profile runner — ONE vanilla LLM at a time (no loop).

Runs a single Ollama-served model through four probe tasks and rolls the errors
into three headline failure rates, writing one resumable JSON per model:

    tasks:
      name_cloze        (GPT4-Books)  -> accuracy + entity-drift proxy   [no Neo4j]
      coreference       (LitBank)     -> identity-tracking accuracy       [no Neo4j]
      consistency       (cases.json)  -> contradiction / location rates  [no Neo4j]
      quote_attribution (PDNC)        -> speaker accuracy                 [needs Neo4j]
    rates:
      contradiction, location_inconsistency,
      entity_drift (from coreference; falls back to the cloze proxy if coref
                    did not run — the source is recorded in the profile)

WORKFLOW (Colab compute, local codebase):
  1. Start the model on Colab, get the cloudflared URL.
  2. Put it in .env:  LLM_MODE=ollama  OLLAMA_BASE_URL=<url>
  3. python evals/run_model_profile.py --model qwen2.5:7b
  4. Repeat per model. Re-running the same --model resumes (finished tasks are
     cached; pass --force to recompute).

ROBUSTNESS: every model call is wrapped — a refusal / invalid JSON / dropped
connection is counted as a failure and surfaced (malformed / abstain), never
crashes the task. Neo4j gates ONLY quote-attribution; the other two run without it.

The runner sets OLLAMA_MODEL from --model BEFORE importing app.config, so the
`settings` singleton (built at import) picks up the override.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

PROFILE_DIR = Path(__file__).parent / "results" / "profiles"
# Tightened "final run" defaults: more books/docs/samples for lower-noise numbers.
DEFAULT_BOOKS = ["1342_pride_and_prejudice", "2489_moby_dick", "158_emma"]
DEFAULT_COREF_DOCS = [
    "1400_great_expectations_brat",
    "74_the_adventures_of_tom_sawyer_brat",
    "351_of_human_bondage_brat",
]
DEFAULT_PDNC = "pdnc:PrideAndPrejudice"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_model_name(model: str) -> str:
    """Filesystem-safe profile filename stem for a model tag."""
    return model.replace(":", "_").replace("/", "_")


def safe_ask(raw_ask: Callable | None) -> Callable | None:
    """Wrap an ask so any transport/refusal error becomes '' instead of raising."""
    if raw_ask is None:
        return None

    def ask(prompt: str, meta: dict) -> str:
        try:
            return raw_ask(prompt, meta) or ""
        except Exception:
            return ""

    return ask


# ---------------------------------------------------------------------------
# Individual tasks
# ---------------------------------------------------------------------------
def run_cloze_task(ask, books: list[str], root: str, max_cloze: int, seed: int) -> dict:
    """Name cloze over GPT4-Books; aggregate counts across books (exact)."""
    from evals.metrics.gpt4_books_cloze import make_ask_cloze_predictor, run_book_cloze

    predict = make_ask_cloze_predictor(ask)
    per_book: dict = {}
    sample_ids: dict = {}
    tot = {"correct": 0, "entity_drift": 0, "miss": 0, "abstain": 0, "n": 0, "cg": 0}

    for book in books:
        try:
            rep = run_book_cloze(
                book, predict=predict, root=root, max_rows=max_cloze, seed=seed
            )
        except FileNotFoundError:
            per_book[book] = {"skipped": "file not found"}
            continue
        per_book[book] = rep.to_dict()
        sample_ids[book] = rep.sample_indices
        tot["correct"] += rep.correct
        tot["entity_drift"] += rep.entity_drift
        tot["miss"] += rep.miss
        tot["abstain"] += rep.abstain
        tot["n"] += rep.n
        tot["cg"] += rep.chatgpt_correct

    n = tot["n"] or 1
    aggregate = {
        "n": tot["n"],
        "accuracy": tot["correct"] / n,
        "entity_drift_rate": tot["entity_drift"] / n,
        "miss_rate": tot["miss"] / n,
        "abstain_rate": tot["abstain"] / n,
        "chatgpt_reference_accuracy": tot["cg"] / n,
    }
    return {"aggregate": aggregate, "per_book": per_book, "sample_ids": sample_ids}


def run_consistency_task(judge_llm) -> dict:
    """Model-as-judge over planted/clean cases -> contradiction/location rates."""
    from evals.metrics.contradiction import (
        load_cases,
        make_judge_fn,
        score_contradiction,
    )

    return score_contradiction(load_cases(), make_judge_fn(judge_llm))


def run_coref_task(ask, docs: list[str], root: str, max_pairs: int, seed: int) -> dict:
    """LitBank pairwise coreference; aggregate over documents (exact counts)."""
    from evals.metrics.litbank_coref import make_ask_coref_predictor, run_document_coref

    predict = make_ask_coref_predictor(ask)
    per_doc: dict = {}
    tot = {
        "correct": 0,
        "n": 0,
        "pos_miss": 0,
        "n_pos": 0,
        "merge": 0,
        "n_neg": 0,
        "abstain": 0,
    }
    for doc in docs:
        try:
            rep = run_document_coref(
                doc, predict=predict, root=root, max_pairs=max_pairs, seed=seed
            )
        except FileNotFoundError:
            per_doc[doc] = {"skipped": "conll file not found"}
            continue
        per_doc[doc] = rep.to_dict()
        tot["correct"] += rep.correct
        tot["n"] += rep.n
        tot["pos_miss"] += round(rep.positive_miss_rate * rep.n_positive)
        tot["n_pos"] += rep.n_positive
        tot["merge"] += round(rep.false_merge_rate * rep.n_negative)
        tot["n_neg"] += rep.n_negative
        tot["abstain"] += round(rep.abstain_rate * rep.n)

    n = tot["n"] or 1
    aggregate = {
        "n": tot["n"],
        "accuracy": tot["correct"] / n,
        "entity_drift_rate": 1.0 - tot["correct"] / n,
        "positive_miss_rate": tot["pos_miss"] / (tot["n_pos"] or 1),
        "false_merge_rate": tot["merge"] / (tot["n_neg"] or 1),
        "abstain_rate": tot["abstain"] / n,
    }
    return {"aggregate": aggregate, "per_doc": per_doc}


def run_quote_task(
    ask, driver, manuscript_id: str, novel_dir: str, sample_quotes: int
) -> dict:
    """Quote attribution over the first `sample_quotes` PDNC gold quotes."""
    from baselines.config import BaselineConfig
    from baselines.flat_long_context import FlatLongContextPredictor
    from evals.metrics.pdnc_meta import read_pdnc_characters
    from evals.metrics.quote_attribution import (
        _resolve_allowed,
        fetch_gold_dialogue,
        score_quote_attribution,
    )

    chars = read_pdnc_characters(novel_dir)
    gold = fetch_gold_dialogue(driver, manuscript_id)
    if sample_quotes:
        gold = gold[:sample_quotes]  # fixed prefix = same quotes for every model
    allowed, filter_rule = _resolve_allowed(gold, chars)
    predictor = FlatLongContextPredictor(manuscript_id, driver, ask, BaselineConfig())
    report = score_quote_attribution(
        gold,
        predictor.predict_quote,
        chars,
        allowed=allowed,
        filter_rule=filter_rule,
        manuscript_id=manuscript_id,
    )
    return {
        "overall": report.overall,
        "explicit": report.explicit,
        "non_explicit": report.non_explicit,
        "n_quotes": report.n_quotes,
        "filter_rule": report.filter_rule,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def compute_rates(tasks: dict) -> dict:
    """Roll task outputs into the headline failure rates.

    entity_drift prefers the LitBank coreference signal (true identity tracking);
    if coref didn't run it falls back to the cloze wrong-but-valid proxy. The
    source is recorded so the number is never read out of context.
    """
    cons = tasks.get("consistency") or {}
    cloze_agg = (tasks.get("name_cloze") or {}).get("aggregate", {})
    coref_agg = (tasks.get("coreference") or {}).get("aggregate", {})

    if coref_agg.get("n"):  # coref ran and produced pairs -> the real signal
        entity_drift, source = coref_agg["entity_drift_rate"], "coreference"
    else:
        entity_drift, source = cloze_agg.get("entity_drift_rate"), "cloze_proxy"

    return {
        "contradiction": cons.get("contradiction_rate"),
        "location_inconsistency": cons.get("location_inconsistency_rate"),
        "entity_drift": entity_drift,
        "entity_drift_source": source,
    }


def _assemble(model, base_url, seed, cfg, tasks) -> dict:
    return {
        "model": model,
        "base_url": base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": cfg,
        "tasks": tasks,
        "rates": compute_rates(tasks),
    }


def run_profile(
    model: str,
    *,
    ask,
    judge_llm,
    driver,
    novel_dir: str | None,
    manuscript_id: str,
    books: list[str],
    cloze_root: str,
    coref_docs: list[str],
    litbank_root: str,
    sample_quotes: int,
    max_cloze: int,
    max_pairs: int,
    seed: int,
    base_url: str,
    existing: dict | None = None,
    force: bool = False,
    checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    """Run the four tasks, checkpointing after each so a disconnect is survivable."""
    prev = {} if force else (existing or {}).get("tasks", {})
    cfg = {
        "sample_quotes": sample_quotes,
        "max_cloze": max_cloze,
        "max_pairs": max_pairs,
        "books": books,
        "coref_docs": coref_docs,
        "pdnc_manuscript": manuscript_id,
    }
    tasks: dict = {}

    def is_done(name: str) -> bool:
        t = prev.get(name)
        return t is not None and "skipped" not in t

    def finish(name: str, value: dict) -> None:
        tasks[name] = value
        if checkpoint:
            checkpoint(_assemble(model, base_url, seed, cfg, tasks))

    # 1. name cloze (no Neo4j)
    if is_done("name_cloze"):
        tasks["name_cloze"] = prev["name_cloze"]
        print("name_cloze: cached")
    else:
        print("name_cloze: running...")
        finish("name_cloze", run_cloze_task(ask, books, cloze_root, max_cloze, seed))

    # 2. coreference (no Neo4j) — the true entity-drift signal
    if is_done("coreference"):
        tasks["coreference"] = prev["coreference"]
        print("coreference: cached")
    else:
        print("coreference: running...")
        finish(
            "coreference",
            run_coref_task(ask, coref_docs, litbank_root, max_pairs, seed),
        )

    # 3. consistency (no Neo4j)
    if is_done("consistency"):
        tasks["consistency"] = prev["consistency"]
        print("consistency: cached")
    else:
        print("consistency: running...")
        finish("consistency", run_consistency_task(judge_llm))

    # 4. quote attribution (needs Neo4j + PDNC)
    if is_done("quote_attribution"):
        tasks["quote_attribution"] = prev["quote_attribution"]
        print("quote_attribution: cached")
    elif driver is None or novel_dir is None:
        finish("quote_attribution", {"skipped": "no Neo4j driver / PDNC novel_dir"})
        print("quote_attribution: skipped (no graph)")
    else:
        print("quote_attribution: running...")
        try:
            value = run_quote_task(ask, driver, manuscript_id, novel_dir, sample_quotes)
        except Exception as exc:  # dataset/graph problem — record, don't crash
            value = {"skipped": f"error: {exc}"}
            print(f"quote_attribution: skipped ({exc})")
        finish("quote_attribution", value)

    return _assemble(model, base_url, seed, cfg, tasks)


def _print_summary(profile: dict) -> None:
    t = profile["tasks"]
    r = profile["rates"]
    print(f"\n{'─' * 52}\nProfile: {profile['model']}")
    qa = t.get("quote_attribution") or {}
    if "overall" in qa:
        print(
            f"  quote_attribution accuracy : {qa['overall']:.3f} "
            f"(explicit {qa['explicit']:.3f} / non-exp {qa['non_explicit']:.3f})"
        )
    else:
        print(f"  quote_attribution          : {qa.get('skipped', 'n/a')}")
    nc = (t.get("name_cloze") or {}).get("aggregate", {})
    if nc:
        print(
            f"  name_cloze accuracy        : {nc['accuracy']:.3f} "
            f"(chatgpt ref {nc['chatgpt_reference_accuracy']:.3f}, n={nc['n']})"
        )
    co = (t.get("coreference") or {}).get("aggregate", {})
    if co:
        print(
            f"  coreference accuracy       : {co['accuracy']:.3f} "
            f"(split {co['positive_miss_rate']:.3f} / merge {co['false_merge_rate']:.3f}, "
            f"n={co['n']})"
        )
    cons = t.get("consistency") or {}
    if cons:
        print(
            f"  consistency recall         : {cons.get('recall', 0):.3f} "
            f"(malformed {cons.get('malformed', 0)}/{cons.get('n', 0)})"
        )
    print("  RATES:")
    print(f"    contradiction            = {_fmt(r['contradiction'])}")
    print(f"    location_inconsistency   = {_fmt(r['location_inconsistency'])}")
    print(
        f"    entity_drift             = {_fmt(r['entity_drift'])} "
        f"(source: {r.get('entity_drift_source', 'n/a')})"
    )


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_model_profile", description=__doc__)
    parser.add_argument(
        "--model", default=None, help="Ollama tag; overrides OLLAMA_MODEL"
    )
    parser.add_argument("--sample-quotes", type=int, default=100)
    parser.add_argument("--max-cloze", type=int, default=100, help="cloze rows/book")
    parser.add_argument("--max-pairs", type=int, default=50, help="coref pairs/doc")
    parser.add_argument("--books", nargs="*", default=None)
    parser.add_argument("--coref-docs", nargs="*", default=None)
    parser.add_argument("--pdnc-manuscript", default=DEFAULT_PDNC)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="recompute cached tasks")
    parser.add_argument("--no-quote", action="store_true", help="skip the Neo4j task")
    args = parser.parse_args(argv)

    # CRITICAL: override the model BEFORE app.config builds its settings singleton.
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    from app.config import settings

    if settings.llm_mode not in ("anthropic", "ollama"):
        print(
            f"LLM_MODE={settings.llm_mode!r}: set LLM_MODE=ollama and OLLAMA_BASE_URL "
            "in .env (point it at the Colab tunnel), then re-run."
        )
        return 2
    model = args.model or settings.ollama_model

    from app.llm import get_llm
    from baselines.llm_ask import make_llm_ask

    ask = safe_ask(make_llm_ask(max_tokens=64))
    judge_llm = get_llm()

    driver = None
    novel_dir = None
    if not args.no_quote:
        try:
            from app import graph

            driver = graph.init_driver()
            driver.verify_connectivity()
            if args.pdnc_manuscript.startswith("pdnc:"):
                folder = args.pdnc_manuscript.split(":", 1)[1]
                novel_dir = str(Path(settings.pdnc_root) / "data" / folder)
        except Exception as exc:  # noqa: BLE001
            print(f"Neo4j not reachable ({exc}); quote-attribution will be skipped.")
            driver = None

    out_path = PROFILE_DIR / f"{safe_model_name(model)}.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else None
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    def checkpoint(partial: dict) -> None:
        out_path.write_text(json.dumps(partial, indent=2))

    print(f"Profiling {model} via {settings.ollama_base_url}\n")
    profile = run_profile(
        model,
        ask=ask,
        judge_llm=judge_llm,
        driver=driver,
        novel_dir=novel_dir,
        manuscript_id=args.pdnc_manuscript,
        books=args.books or DEFAULT_BOOKS,
        cloze_root=settings.gpt4books_root,
        coref_docs=args.coref_docs or DEFAULT_COREF_DOCS,
        litbank_root=settings.litbank_root,
        sample_quotes=args.sample_quotes,
        max_cloze=args.max_cloze,
        max_pairs=args.max_pairs,
        seed=args.seed,
        base_url=settings.ollama_base_url,
        existing=existing,
        force=args.force,
        checkpoint=checkpoint,
    )
    out_path.write_text(json.dumps(profile, indent=2))
    print(f"\nWrote {out_path}")
    _print_summary(profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
