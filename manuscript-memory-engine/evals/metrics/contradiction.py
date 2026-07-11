"""METRIC 3 — Consistency stress test (model as contradiction detector).

Where consistency.py scores OUR deterministic Tier-1 gate, this scores a *model*:
give it (canon, draft) pairs — planted contradictions + clean controls from
evals/cases.json — and ask it to flag contradictions. It yields:

  - overall precision / recall / F1 (TP/FP/FN/TN) for contradiction detection,
  - per-`cls` recall (contradiction vs location), and
  - the two headline rates:
      contradiction_rate          = 1 - overall recall  (planted misses)
      location_inconsistency_rate = 1 - recall on `location` cases

A judge that errors or refuses on a case (returns None) is treated as "no
contradiction flagged" for the confusion matrix and counted in `malformed` — a
non-crashing outcome that is itself a finding for weaker models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

# judge_fn(canon, draft) -> "PASS" | "FAIL" | None (None = malformed/abstain).
JudgeFn = Callable[[str, str], "str | None"]

CASES_PATH = Path(__file__).parent.parent / "cases.json"


def load_cases(path: str | Path = CASES_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_judge_fn(llm) -> JudgeFn:
    """Adapt an LLMClient into a judge_fn, mapping any judge failure to None.

    LLMOutputIncomplete (refusal/invalid JSON) and transport errors all become
    None so one bad case never aborts the profile run.
    """

    def judge_fn(canon: str, draft: str) -> str | None:
        try:
            return llm.judge(canon, draft).verdict
        except Exception:
            return None

    return judge_fn


def score_contradiction(cases: list[dict], judge_fn: JudgeFn) -> dict:
    """Score a judge over planted/clean cases; return metrics + the two rates."""
    tp = fp = fn = tn = malformed = 0
    # per FAIL class: [caught, planted]
    per_class: dict[str, list[int]] = {}
    per_diff: dict[str, list[int]] = {}  # by difficulty: blatant/moderate/subtle

    for case in cases:
        expected_fail = case["expected_verdict"] == "FAIL"
        result = judge_fn(case["canon"], case["draft"])
        if result is None:
            malformed += 1
        predicted_fail = result == "FAIL"

        if expected_fail:
            cls = case.get("cls", "contradiction")
            diff = case.get("difficulty", "unspecified")
            caught_planted = per_class.setdefault(cls, [0, 0])
            caught_diff = per_diff.setdefault(diff, [0, 0])
            caught_planted[1] += 1
            caught_diff[1] += 1
            if predicted_fail:
                tp += 1
                caught_planted[0] += 1
                caught_diff[0] += 1
            else:
                fn += 1
        else:
            if predicted_fail:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    def _breakdown(counts: dict[str, list[int]]) -> dict:
        return {
            key: {
                "n_planted": planted,
                "caught": caught,
                "missed": planted - caught,
                "recall": (caught / planted) if planted else 0.0,
            }
            for key, (caught, planted) in sorted(counts.items())
        }

    per_class_out = _breakdown(per_class)
    loc = per_class_out.get("location")
    return {
        "n": len(cases),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "malformed": malformed,
        "malformed_rate": malformed / len(cases) if cases else 0.0,
        # Share of CLEAN controls wrongly flagged — exposes 'flag-everything'
        # models whose (1 - recall) contradiction rate looks deceptively perfect.
        "over_flag_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "per_class": per_class_out,
        # Per-difficulty recall shows blatant (all models catch) vs subtle (the
        # discriminating tier) — this is what makes the rates separate models.
        "per_difficulty": _breakdown(per_diff),
        # Headline failure rates (1 - recall = share of planted issues missed).
        "contradiction_rate": 1.0 - recall,
        "location_inconsistency_rate": (1.0 - loc["recall"]) if loc else None,
    }
