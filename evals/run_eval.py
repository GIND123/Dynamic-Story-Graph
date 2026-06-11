"""Judge precision/recall evaluation against hand-crafted test cases.

Runs the AnthropicLLM judge (claude-haiku-4-5) against 12 cases:
  - 6 planted contradictions (expected: FAIL)
  - 6 clean controls      (expected: PASS)

Reports precision, recall, and F1 for contradiction detection.

Usage (from repo root):
    LLM_MODE=anthropic ANTHROPIC_API_KEY=sk-ant-... python evals/run_eval.py

Exits 0 with a skip message if ANTHROPIC_API_KEY is not set.
"""

import json
import os
import sys
from pathlib import Path

# Allow running as a script from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

CASES_PATH = Path(__file__).parent / "cases.json"


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — judge eval skipped (real mode required).")
        print("Set LLM_MODE=anthropic and ANTHROPIC_API_KEY, then re-run.")
        sys.exit(0)

    # Force anthropic mode before importing the LLM factory
    os.environ["LLM_MODE"] = "anthropic"

    from app.llm import AnthropicLLM  # noqa: PLC0415 (deferred import by design)

    cases = json.loads(CASES_PATH.read_text())
    llm = AnthropicLLM()

    tp = fp = fn = tn = 0
    print(
        f"Evaluating {len(cases)} cases with {os.environ.get('JUDGE_MODEL', 'claude-haiku-4-5')}...\n"
    )

    for case in cases:
        verdict = llm.judge(case["canon"], case["draft"])
        predicted = verdict.verdict
        expected = case["expected_verdict"]
        correct = predicted == expected

        marker = "✓" if correct else "✗"
        print(
            f"{marker}  [{case['id']}]  expected={expected}  got={predicted}  score={verdict.coherence_score:.2f}"
        )

        if not correct:
            if verdict.contradictions:
                for c in verdict.contradictions:
                    print(f"       contradiction: {c.violated_fact}")
            else:
                print(f"       reasoning: {verdict.reasoning[:120]}")

        if expected == "FAIL" and predicted == "FAIL":
            tp += 1
        elif expected == "PASS" and predicted == "FAIL":
            fp += 1
        elif expected == "FAIL" and predicted == "PASS":
            fn += 1
        else:
            tn += 1

    total = len(cases)
    correct_count = tp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    print(f"\n{'─' * 50}")
    print(f"Accuracy:  {correct_count}/{total} ({100 * correct_count / total:.0f}%)")
    print(
        f"Precision: {precision:.2f}   (of FAIL predictions, how many were real FAILs)"
    )
    print(f"Recall:    {recall:.2f}   (of real FAILs, how many were caught)")
    print(f"F1:        {f1:.2f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")


if __name__ == "__main__":
    main()
