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
    GUARDED_MEMORIES,
    STATE_MEMORIES,
    StoryRun,
    build_chapter_prompt,
    build_summary_prompt,
    memory_of,
    variant_of,
)
from dsg.lexicon import is_underspecified, is_valid_object, normalize_predicate
from dsg.matching import classify_surface
from dsg.policies.runner import apply_window
from dsg.proposals import FactProposal, WindowProposal, is_plausible_entity
from dsg.schemas import Span, Window
from dsg.store import POLICIES, NarrativeState

# Which reading policy backs each state-carrying condition.
POLICY_FOR = {
    "append-only-state": "append-only",
    "dsg-state": "dsg-full",
    "dsg-repair": "dsg-full",
}


@dataclass(slots=True)
class ChapterRecord:
    story_id: str
    condition: str
    chapter: int
    text: str
    chars: int
    prompt_chars: int = 0
    prompt_tokens: int = 0
    violations: list[str] = field(default_factory=list)
    restated: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    state_entities: int = 0
    state_facts: int = 0
    state_violations: int = 0
    rollbacks: int = 0
    canon_in_state: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "story_id": self.story_id, "condition": self.condition,
            "chapter": self.chapter, "chars": self.chars,
            "prompt_chars": self.prompt_chars,
            "prompt_tokens": self.prompt_tokens,
            "violations": self.violations, "restated": self.restated,
            "evidence": self.evidence, "state_entities": self.state_entities,
            "state_facts": self.state_facts,
            "state_violations": self.state_violations, "rollbacks": self.rollbacks,
            "canon_in_state": self.canon_in_state,
        }


def init_runs(
    premises: list[Premise], conditions: tuple[str, ...] = CONDITIONS
) -> list[StoryRun]:
    runs: list[StoryRun] = []
    for premise in premises:
        for condition in conditions:
            memory = memory_of(condition)
            state = (
                NarrativeState(POLICIES[POLICY_FOR[memory]])
                if memory in STATE_MEMORIES
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
    return [r for r in runs if memory_of(r.condition) == "rolling-summary"]


def guarded_targets(runs: list[StoryRun]) -> list[StoryRun]:
    """Runs whose chapter must clear the graph before it is accepted."""
    return [r for r in runs if memory_of(r.condition) in GUARDED_MEMORIES]


def check_chapter_against_state(run: StoryRun, raw_extraction: str, chapter_text: str):
    """Conflicts a freshly written chapter would introduce, without applying it."""
    from dsg.generate.repair import detect_conflicts

    if run.state is None:
        return []
    proposal = parse_write_extraction(raw_extraction, _window_for(run, chapter_text))
    return detect_conflicts(run.state, proposal)


def tuned_targets(runs: list[StoryRun]) -> list[bool]:
    """Which runs should be served by the fine-tuned adapter."""
    return [variant_of(r.condition) == "tuned" for r in runs]


def state_targets(runs: list[StoryRun]) -> list[StoryRun]:
    return [r for r in runs if memory_of(r.condition) in STATE_MEMORIES]


# The novel-reading prompt asks for four things at once (mentions, identity
# links, facts, speech) because a novel needs all four. Reading back a chapter
# just written needs one: what this chapter establishes. A smaller ask is
# answered far more reliably -- the general prompt made a 3B put attributes in
# the entity slot and invent a place name. Both state-carrying conditions use
# this identically, so it does not favour either; the extractor is part of the
# method, exactly as keeping the transcript is the method for full-context.
WRITE_EXTRACT_PROMPT = """Read this chapter and list what it establishes about each named
character and object.

Output one line per fact, in exactly this form, and nothing else:

FACT | <name exactly as written> | <property> | <value>

Properties you may use:
eye_colour, hair_colour, occupation, material, birthplace, resides_in, location,
alive, married_to, parent_of, child_of, sibling_of, possesses, knows, member_of,
stance_toward, emotion

Rules:
- Only what THIS chapter states. Do not guess and do not carry anything over.
- If a property is not given in the chapter, LEAVE THE LINE OUT. Never write
  "not specified", "unknown" or similar -- omit it instead.
- Use the name exactly as it appears in the chapter.
- Include physical objects too, not only people: what an object is made of, who
  holds it, where it is.
- Do not give the same value to two different names unless the chapter says so
  of each of them separately.
- Keep values short: one or two words where possible.
- At most 16 lines.

Chapter:
<<<TEXT>>>
{chapter}
<<<END>>>

FACTS:"""


def _window_for(run: StoryRun, chapter_text: str) -> Window:
    start = sum(len(c) + 2 for c in run.chapters[:-1])
    return Window(
        index=len(run.chapters) - 1, start=start,
        end=start + len(chapter_text), text=chapter_text,
    )


def extraction_prompt(run: StoryRun, chapter_text: str) -> str:
    return WRITE_EXTRACT_PROMPT.format(chapter=chapter_text)


def parse_write_extraction(raw: str, window: Window) -> WindowProposal:
    """Parse the four-field FACT lines into the same proposal type."""
    out = WindowProposal(index=window.index, start=window.start, end=window.end)
    seen: set[str] = set()
    for line in (raw or "").splitlines():
        cells = [c.strip().strip("*-` ") for c in line.split("|")]
        if len(cells) < 4 or cells[0].upper().lstrip("- ").strip() != "FACT":
            continue
        subject, predicate, value = cells[1], cells[2], cells[3]
        if not (subject and predicate and value):
            continue
        if not is_plausible_entity(subject):
            continue
        if not is_valid_object(normalize_predicate(predicate), value):
            continue
        if is_underspecified(value):
            continue
        key = f"{subject}|{predicate}|{value}".lower()
        if key in seen or len(out.facts) >= 14:
            continue
        seen.add(key)
        if subject.lower() not in {e["surface"].lower() for e in out.entities}:
            out.entities.append(
                {"surface": subject, "kind": classify_surface(subject)}
            )
        out.facts.append(
            FactProposal(subject=subject, predicate=predicate, object=value,
                         certainty="narrated", evidence="")
        )
    out.parse_ok = bool(out.facts)
    return out


def apply_extraction(run: StoryRun, raw: str, chapter_text: str) -> None:
    if run.state is None:
        return
    window = _window_for(run, chapter_text)
    proposal = parse_write_extraction(raw, window)
    run.state.step(window.index, window.end)
    apply_window(run.state, proposal, Span(window.start, window.end))
    run.state.close_step()


def replay_extraction(state, chapters, cache, book_id, upto, offset=0) -> int:
    """Rebuild a state from cached extractions, without touching a GPU.

    This is what makes an interrupted dataset build cheap to resume: the
    expensive step is the extraction, and it is cached, so recovery replays it
    on the CPU instead of paying for it twice. Returns the new character offset.
    """
    for step in range(min(upto, len(chapters))):
        text = chapters[step]
        raw = cache.get(f"{book_id}:{step}", "")
        window = Window(index=step, start=offset, end=offset + len(text), text=text)
        proposal = parse_write_extraction(raw, window)
        state.step(step, window.end)
        apply_window(state, proposal, Span(window.start, window.end))
        state.close_step()
        offset = window.end + 2
    return offset


def _canon_held(run: StoryRun, premise: Premise) -> list[str]:
    """Which planted facts the state actually holds as live beliefs.

    The diagnostic that explains any result on the state conditions: a digest
    can only protect a fact it captured. Extraction is the ceiling on the
    method, and this makes the ceiling visible rather than leaving a null to be
    misread as the representation failing.
    """
    if run.state is None:
        return []
    held: list[str] = []
    live: set[tuple[str, str]] = set()
    for a in run.state.assertions.values():
        if not a.live:
            continue
        node = run.state.entities.get(a.subject)
        if node is None:
            continue
        for surface in node.surfaces:
            live.add((surface.strip().lower(), a.object.strip().lower()))
    for fact in premise.canon:
        subject = fact.subject.strip().lower()
        for held_subject, value in live:
            if held_subject != subject:
                continue
            if fact.value.lower() in value or value in fact.value.lower():
                held.append(fact.fact_id)
                break
    return held


def record(
    run: StoryRun, premise: Premise, text: str, chapter: int,
    prompt_chars: int = 0, prompt_tokens: int = 0,
) -> ChapterRecord:
    report = check_chapter(premise, text, chapter)
    rec = ChapterRecord(
        story_id=run.story_id, condition=run.condition, chapter=chapter,
        text=text, chars=len(text), violations=report.violations,
        restated=report.restated, evidence=report.evidence,
        prompt_chars=prompt_chars, prompt_tokens=prompt_tokens,
    )
    if run.state is not None:
        rec.canon_in_state = _canon_held(run, premise)
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
    "state_targets", "summary_targets", "build_summary_prompt", "tuned_targets",
    "guarded_targets", "check_chapter_against_state", "replay_extraction",
    "parse_write_extraction", "WRITE_EXTRACT_PROMPT",
]
