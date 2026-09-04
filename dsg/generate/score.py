"""Scoring generated stories against their planted canon.

The primary quantity is **cumulative canon violation**: by chapter c, what
fraction of the facts established in chapter 1 has the story contradicted at
least once? A fact once contradicted stays contradicted -- continuity, once
broken, is broken -- so the curve is monotone in c and its slope is the rate at
which a condition forgets.

Everything here is deterministic string matching against facts we planted, so
no model and no human judgement enters the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dsg.eval.bootstrap import paired_bootstrap
from dsg.generate.canon import Premise


@dataclass(slots=True)
class RunScore:
    story_id: str
    condition: str
    n_canon: int
    chapters: int
    total_chars: int
    violated: set[str] = field(default_factory=set)
    first_violation: dict[str, int] = field(default_factory=dict)
    restated_ever: set[str] = field(default_factory=set)
    cumulative: list[float] = field(default_factory=list)   # per chapter
    per_chapter: list[int] = field(default_factory=list)
    prompt_tokens_total: int = 0
    prompt_tokens_last: int = 0
    state_facts_final: int = 0
    state_entities_final: int = 0
    canon_in_state_final: int = 0
    rollbacks_final: int = 0

    @property
    def violation_rate(self) -> float:
        return len(self.violated) / self.n_canon if self.n_canon else 0.0

    @property
    def retention(self) -> float:
        return 1.0 - self.violation_rate

    @property
    def restatement_rate(self) -> float:
        return len(self.restated_ever) / self.n_canon if self.n_canon else 0.0

    def as_dict(self) -> dict:
        return {
            "story_id": self.story_id, "condition": self.condition,
            "n_canon": self.n_canon, "chapters": self.chapters,
            "total_chars": self.total_chars,
            "violation_rate": self.violation_rate, "retention": self.retention,
            "restatement_rate": self.restatement_rate,
            "n_violated": len(self.violated),
            "mean_first_violation": (
                sum(self.first_violation.values()) / len(self.first_violation)
                if self.first_violation else float(self.chapters + 1)
            ),
            "prompt_tokens_total": float(self.prompt_tokens_total),
            "prompt_tokens_last": float(self.prompt_tokens_last),
            "canon_capture": (
                self.canon_in_state_final / self.n_canon if self.n_canon else 0.0
            ),
            "state_facts_final": float(self.state_facts_final),
            "state_entities_final": float(self.state_entities_final),
            "rollbacks_final": float(self.rollbacks_final),
        }


def score_runs(payload: dict) -> list[RunScore]:
    premises = {p["story_id"]: Premise.from_json(p) for p in payload["premises"]}
    n_chapters = payload["meta"]["chapters"]

    scores: dict[tuple[str, str], RunScore] = {}
    for rec in sorted(payload["records"], key=lambda r: r["chapter"]):
        key = (rec["story_id"], rec["condition"])
        score = scores.get(key)
        if score is None:
            score = RunScore(
                story_id=rec["story_id"], condition=rec["condition"],
                n_canon=len(premises[rec["story_id"]].canon), chapters=n_chapters,
                total_chars=0,
            )
            scores[key] = score
        score.total_chars += rec["chars"]
        score.prompt_tokens_total += int(rec.get("prompt_tokens", 0))
        score.prompt_tokens_last = int(rec.get("prompt_tokens", 0))
        for fact_id in rec["violations"]:
            if fact_id not in score.violated:
                score.first_violation[fact_id] = rec["chapter"]
            score.violated.add(fact_id)
        score.restated_ever.update(rec["restated"])
        score.per_chapter.append(len(rec["violations"]))
        score.cumulative.append(
            len(score.violated) / score.n_canon if score.n_canon else 0.0
        )
        score.state_facts_final = rec.get("state_facts", 0)
        score.canon_in_state_final = len(rec.get("canon_in_state", []))
        score.state_entities_final = rec.get("state_entities", 0)
        score.rollbacks_final = rec.get("rollbacks", 0)
    return list(scores.values())


def curves(scores: list[RunScore]) -> dict[str, list[float]]:
    """Mean cumulative violation rate by chapter, per condition."""
    by_condition: dict[str, list[list[float]]] = {}
    for s in scores:
        by_condition.setdefault(s.condition, []).append(s.cumulative)
    out: dict[str, list[float]] = {}
    for condition, series in by_condition.items():
        depth = min(len(x) for x in series)
        out[condition] = [
            sum(x[i] for x in series) / len(series) for i in range(depth)
        ]
    return out


COMPARISONS = (
    # Does any memory help at all?
    ("base:rolling-summary", "base:none"),
    ("base:full-context", "base:rolling-summary"),
    # What does a maintained state buy over the practical alternatives?
    ("base:dsg-state", "base:none"),
    ("base:dsg-state", "base:rolling-summary"),
    ("base:dsg-state", "base:full-context"),
    # Does the state need to be revisable?
    ("base:dsg-state", "base:append-only-state"),
    # What does training the writer to use the state buy?
    ("tuned:dsg-state", "base:dsg-state"),
    ("tuned:full-context", "base:full-context"),
    # And the full system against the strongest practical baseline.
    ("tuned:dsg-state", "tuned:full-context"),
    ("tuned:dsg-repair", "tuned:dsg-state"),
    ("tuned:dsg-repair", "base:full-context"),
)

METRICS = ("violation_rate", "retention", "restatement_rate", "canon_capture",
           "mean_first_violation", "total_chars",
           "prompt_tokens_total", "prompt_tokens_last")


def compare(scores: list[RunScore]) -> dict:
    by_condition: dict[str, dict[str, dict]] = {}
    for s in scores:
        by_condition.setdefault(s.condition, {})[s.story_id] = s.as_dict()
    out: dict[str, dict] = {}
    for a, b in COMPARISONS:
        if a not in by_condition or b not in by_condition:
            continue
        stories = sorted(set(by_condition[a]) & set(by_condition[b]))
        if not stories:
            continue
        entry: dict[str, object] = {"n_stories": len(stories)}
        for metric in METRICS:
            xs = [float(by_condition[a][s][metric]) for s in stories]
            ys = [float(by_condition[b][s][metric]) for s in stories]
            entry[metric] = paired_bootstrap(xs, ys).as_dict()
        out[f"{a}_vs_{b}"] = entry
    return out


def summarize(scores: list[RunScore], order: tuple[str, ...]) -> str:
    cols = ("violation_rate", "retention", "restatement_rate", "canon_capture",
            "mean_first_violation", "total_chars", "prompt_tokens_last")
    lines = ["| condition | n | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for condition in order:
        rows = [s.as_dict() for s in scores if s.condition == condition]
        if not rows:
            continue
        cells = []
        for c in cols:
            mean = sum(r[c] for r in rows) / len(rows)
            cells.append(
                f"{mean:,.0f}" if c in ("total_chars", "prompt_tokens_last")
                else f"{mean:.3f}"
            )
        lines.append(f"| {condition} | {len(rows)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
