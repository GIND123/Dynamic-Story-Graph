"""Judge precision/recall evaluation against hand-crafted contradiction cases.

Runs a model's judge against evals/cases.json (planted contradictions + clean
controls) and reports precision, recall, F1, per-class recall, and the two
headline rates (contradiction, location inconsistency). The scoring core lives in
evals.metrics.contradiction.score_contradiction so the per-model profiler reuses
it unchanged.

Usage (from repo root), with a real backend:
    LLM_MODE=anthropic ANTHROPIC_API_KEY=sk-ant-... python evals/run_eval.py
    LLM_MODE=ollama OLLAMA_BASE_URL=... python evals/run_eval.py

Exits 0 with a skip message if no real backend is configured (LLM_MODE=fake).
"""

import sys
from pathlib import Path

# Allow running as a script from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402
from evals.metrics.contradiction import (  # noqa: E402
    load_cases,
    make_judge_fn,
    score_contradiction,
)


def main() -> None:
    if settings.llm_mode not in ("anthropic", "ollama"):
        print(
            f"LLM_MODE={settings.llm_mode!r} — judge eval skipped (real backend "
            "required). Set LLM_MODE=anthropic or LLM_MODE=ollama and re-run."
        )
        sys.exit(0)

    from app.llm import get_llm  # noqa: PLC0415 (deferred import by design)

    cases = load_cases()
    judge_fn = make_judge_fn(get_llm())
    print(f"Evaluating {len(cases)} cases with LLM_MODE={settings.llm_mode}...\n")

    report = score_contradiction(cases, judge_fn)

    print(f"{'─' * 52}")
    print(f"Precision: {report['precision']:.2f}   (of FAIL predictions, real FAILs)")
    print(f"Recall:    {report['recall']:.2f}   (of real FAILs, how many caught)")
    print(f"F1:        {report['f1']:.2f}")
    print(
        f"TP={report['tp']}  FP={report['fp']}  FN={report['fn']}  TN={report['tn']}  "
        f"malformed={report['malformed']}"
    )
    print("\nPer class (recall on planted issues):")
    for cls, c in report["per_class"].items():
        print(f"  {cls:14s} recall={c['recall']:.2f}  ({c['caught']}/{c['n_planted']})")
    print("Per difficulty (recall on planted issues):")
    for diff in ("blatant", "moderate", "subtle"):
        c = report["per_difficulty"].get(diff)
        if c:
            print(
                f"  {diff:14s} recall={c['recall']:.2f}  ({c['caught']}/{c['n_planted']})"
            )
    loc = report["location_inconsistency_rate"]
    print("\nHeadline rates:")
    print(f"  contradiction_rate          = {report['contradiction_rate']:.2f}")
    print(
        "  location_inconsistency_rate = "
        + (f"{loc:.2f}" if loc is not None else "n/a")
    )


if __name__ == "__main__":
    main()
