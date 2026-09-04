"""The chapter loop, factored so it can be driven in lockstep batches.

Generation within a story is strictly sequential, so batching happens *across*
stories and conditions: at chapter t every run submits its prompt at once. Each
chapter costs up to three batched rounds -- write the chapter, update the
rolling summary where that condition needs one, and read the new chapter into
the state where that condition keeps one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dsg.generate.canon import Premise, check_chapter
from dsg.generate.conditions import (
    CONDITIONS,
    STATE_CONDITIONS,
    StoryRun,
    build_chapter_prompt,
    build_summary_prompt,
)
from dsg.policies.runner import apply_window
from dsg.proposals import build_prompt, parse_proposal
from dsg.schemas import Span, Window
from dsg.store import POLICIES, NarrativeState

# Which reading policy backs each state-carrying condition.
POLICY_FOR = {"append-only-state": "append-only", "dsg-state": "dsg-full"}


@dataclass(slots=True)
class ChapterRecord:
    story_id: str
    condition: str
    chapter: int
    text: str
    chars: int
    violations: list[str] = field(default_factory=list)
    restated: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    state_entities: int = 0
    state_facts: int = 0
    state_violations: int = 0
    rollbacks: int = 0

    def to_json(self) -> dict:
        return {
            "story_id": self.story_id, "condition": self.condition,
            "chapter": self.chapter, "chars": self.chars,
            "violations": self.violations, "restated": self.restated,
            "evidence": self.evidence, "state_entities": self.state_entities,
            "state_facts": self.state_facts,
            "state_violations": self.state_violations, "rollbacks": self.rollbacks,
        }


def init_runs(
    premises: list[Premise], conditions: tuple[str, ...] = CONDITIONS
) -> list[StoryRun]:
    runs: list[StoryRun] = []
    for premise in premises:
        for condition in conditions:
            state = (
                NarrativeState(POLICIES[POLICY_FOR[condition]])
                if condition in STATE_CONDITIONS
                else None
            )
            runs.append(
                StoryRun(story_id=premise.story_id, condition=condition, state=state)
            )
    return runs


def chapter_prompts(
    runs: list[StoryRun], premises: dict[str, Premise], chapter: int,
    total: int, words: int = 500,
) -> list[str]:
    return [
        build_chapter_prompt(premises[r.story_id], r, chapter, total, words)
        for r in runs
    ]


def summary_targets(runs: list[StoryRun]) -> list[StoryRun]:
    return [r for r in runs if r.condition == "rolling-summary"]


def state_targets(runs: list[StoryRun]) -> list[StoryRun]:
    return [r for r in runs if r.condition in STATE_CONDITIONS]


def extraction_prompt(run: StoryRun, chapter_text: str) -> str:
    """Read the newly written chapter with the same extractor used for novels."""
    start = sum(len(c) + 2 for c in run.chapters[:-1])
    window = Window(
        index=len(run.chapters) - 1, start=start,
        end=start + len(chapter_text), text=chapter_text,
    )
    digest = run.state.digest() if run.state is not None else ""
    return build_prompt(window, "", digest or "(none yet)")


def apply_extraction(run: StoryRun, raw: str, chapter_text: str) -> None:
    if run.state is None:
        return
    start = sum(len(c) + 2 for c in run.chapters[:-1])
    window = Window(
        index=len(run.chapters) - 1, start=start,
        end=start + len(chapter_text), text=chapter_text,
    )
    proposal = parse_proposal(raw, window)
    run.state.step(window.index, window.end)
    apply_window(run.state, proposal, Span(window.start, window.end))
    run.state.close_step()


def record(run: StoryRun, premise: Premise, text: str, chapter: int) -> ChapterRecord:
    report = check_chapter(premise, text, chapter)
    rec = ChapterRecord(
        story_id=run.story_id, condition=run.condition, chapter=chapter,
        text=text, chars=len(text), violations=report.violations,
        restated=report.restated, evidence=report.evidence,
    )
    if run.state is not None:
        rec.state_entities = len(run.state.live_entities())
        rec.state_facts = sum(1 for a in run.state.assertions.values() if a.live)
        rec.state_violations = len(run.state.violations)
        rec.rollbacks = run.state.op_counts()["rollback"]
    return rec


def clean_chapter(raw: str) -> str:
    """Strip the scaffolding models add around requested prose."""
    text = (raw or "").strip()
    for marker in ("CHAPTER", "Chapter"):
        if text.startswith(marker):
            newline = text.find("\n")
            if 0 < newline < 60:
                text = text[newline + 1 :].strip()
    for lead in ("Here is", "Sure,", "Certainly"):
        if text.startswith(lead):
            newline = text.find("\n")
            if newline > 0:
                text = text[newline + 1 :].strip()
    return text


__all__ = [
    "ChapterRecord", "POLICY_FOR", "apply_extraction", "chapter_prompts",
    "clean_chapter", "extraction_prompt", "init_runs", "record",
    "state_targets", "summary_targets", "build_summary_prompt",
]
