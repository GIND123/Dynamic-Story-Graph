"""What the writer is allowed to remember.

Every condition sees the same premise, the same outline beat, and the same
planted canon **in chapter 1 only**. From chapter 2 the canon block is withheld,
so what survives is whatever that condition's memory carries forward. Repeating
the canon in every prompt would test prompt-following, not memory.

The conditions form a ladder of what production systems actually do, ending at
the two that differ only in whether the carried state may be revised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dsg.generate.canon import Premise
from dsg.generate.prompt import state_memory, writer_prompt
from dsg.store import NarrativeState

# Memory kinds: what the writer is given about everything before this chapter.
MEMORIES = (
    "none",             # premise + beat only: the floor
    "last-chapter",     # + the previous chapter verbatim
    "rolling-summary",  # + a running summary the model maintains
    "full-context",     # + every previous chapter, truncated to the window
    "append-only-state",# + a state digest that can never be revised
    "dsg-state",        # + a state digest under the full revision calculus
    "dsg-repair",       # + the same digest, and the graph may reject a chapter
    "dsg-hybrid",       # + the digest AND the previous chapter
    "dsg-hybrid-repair",# + both, and the graph may reject a chapter
    "beat-retrieval",   # + only the canon the *next beat* implicates
    "beat-hybrid",      # + that, and the previous chapter
    "beat-hybrid-repair",# + both, and the graph may reject a chapter
)

STATE_MEMORIES = (
    "append-only-state", "dsg-state", "dsg-repair",
    "dsg-hybrid", "dsg-hybrid-repair",
    "beat-retrieval", "beat-hybrid", "beat-hybrid-repair",
)

# Memories whose chapter is checked against the graph before it is accepted.
GUARDED_MEMORIES = ("dsg-repair", "dsg-hybrid-repair", "beat-hybrid-repair")

# A condition is a (backbone variant, memory) pair, written "variant:memory".
# Pairing them inside one run keeps every comparison within-story: the effect of
# memory is read down a variant, the effect of training across variants at the
# same memory.
CONDITIONS = (
    "base:none",
    "base:last-chapter",
    "base:rolling-summary",
    "base:full-context",
    "base:append-only-state",
    "base:dsg-state",
    "tuned:full-context",
    "tuned:dsg-state",
    "tuned:dsg-repair",
)

# The base-only ladder used after the fine-tuned writer degenerated into
# corpus pastiche: its numbers were not comparable, so it is excluded until
# retrained. These six isolate what the state is worth against the practical
# alternatives, at matched premise adherence.
BASE_LADDER = (
    "base:none",
    "base:last-chapter",
    "base:full-context",
    "base:dsg-state",
    "base:dsg-hybrid",
    "base:dsg-hybrid-repair",
)

# The retrieval ladder: the untested cell that both this project's evidence
# and the prior literature point at. `last-chapter` (0.308) is the incumbent
# every new condition has to beat; `dsg-state` (0.588) is the known-failing
# serialisation, kept as a negative control.
# Test time selection: draw k candidates for the same chapter and let the graph
# rank them. The random arm is the control that separates "the graph helped"
# from "four samples helped"; both draw the same k at the same budget.
SELECTION_LADDER = (
    "base:last-chapter",                  # incumbent, one sample
    "base:last-chapter@bo4-random",       # same budget, chosen blind
    "base:last-chapter@bo4-graph",        # same budget, ranked by the graph
    "base:full-context",                  # strong baseline, five times the context
)

RETRIEVAL_LADDER = (
    "base:none",
    "base:last-chapter",
    "base:full-context",
    "base:beat-retrieval",
    "base:beat-hybrid",
    "base:beat-hybrid-repair",
)


def parse_selection(condition: str) -> tuple[int, str]:
    """How many candidates to draw for this condition, and how to pick one.

    Encoded as ``variant:memory@k-strategy``; a plain condition means one
    candidate taken as written.
    """
    if "@" not in condition:
        return 1, "first"
    _, _, spec = condition.partition("@")
    k_text, _, strategy = spec.partition("-")
    try:
        k = max(1, int(k_text.lstrip("bo") or 1))
    except ValueError:
        k = 1
    return k, (strategy or "first")


def split_condition(condition: str) -> tuple[str, str]:
    base = condition.split("@", 1)[0]
    variant, _, memory = base.partition(":")
    return (variant, memory) if memory else ("base", variant)


def memory_of(condition: str) -> str:
    return split_condition(condition)[1]


def variant_of(condition: str) -> str:
    return split_condition(condition)[0]


# Backwards-compatible alias used by the state-policy map.
STATE_CONDITIONS = STATE_MEMORIES



SUMMARY_PROMPT = """Update the running story summary.

Previous summary:
{summary}

New chapter:
{chapter}

Write an updated summary of the whole story so far in at most {words} words.
Keep concrete established details -- names, appearances, materials, occupations,
family relations, places -- because later chapters depend on them. Output only
the summary.

SUMMARY:"""


@dataclass(slots=True)
class StoryRun:
    """Per-(story, condition) generation state."""

    story_id: str
    condition: str
    chapters: list[str] = field(default_factory=list)
    summary: str = ""
    state: NarrativeState | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.chapters)


def _truncate_head_and_tail(chapters: list[str], budget_chars: int) -> str:
    """Keep the opening (where canon was established) and the most recent text.

    This is what a careful long-context user does when the window runs out, so
    it is the fair form of the baseline rather than a naive tail truncation that
    would drop the canon and hand us the result.
    """
    if not chapters:
        return ""
    joined = "\n\n".join(chapters)
    if len(joined) <= budget_chars:
        return joined
    head = chapters[0][: budget_chars // 3]
    tail_budget = budget_chars - len(head) - 40
    tail: list[str] = []
    for chapter in reversed(chapters[1:]):
        if sum(len(c) for c in tail) + len(chapter) > tail_budget:
            break
        tail.append(chapter)
    return head + "\n\n[...]\n\n" + "\n\n".join(reversed(tail))


def build_memory(
    run: StoryRun, budget_chars: int = 24_000, beat: str = ""
) -> str:
    """The memory block for the next chapter, under this run's condition."""
    if not run.chapters:
        return ""
    condition = memory_of(run.condition)
    if condition == "none":
        return ""
    if condition == "last-chapter":
        return f"The previous chapter:\n{run.chapters[-1]}"
    if condition == "rolling-summary":
        return f"The story so far:\n{run.summary}"
    if condition == "full-context":
        return (
            "The story so far:\n"
            + _truncate_head_and_tail(run.chapters, budget_chars)
        )
    if condition in ("beat-retrieval", "beat-hybrid", "beat-hybrid-repair"):
        # Query the state with the beat rather than serialising all of it. A
        # beat names two or three people; retrieving only their canon leaves
        # room for the recent text instead of displacing it.
        digest = state_memory(
            run.state.beat_digest(beat) if run.state is not None else ""
        )
        if condition == "beat-retrieval" or not run.chapters:
            return digest
        return f"{digest}\n\nThe previous chapter:\n{run.chapters[-1]}"
    if condition in ("dsg-hybrid", "dsg-hybrid-repair"):
        # A digest and the recent text are not rivals. Forcing a choice between
        # them is an artefact of the ablation, not something a real writer would
        # do: the digest carries what must stay true, the previous chapter
        # carries voice and immediate continuity.
        digest = state_memory(run.state.fact_digest() if run.state is not None else "")
        return f"{digest}\n\nThe previous chapter:\n{run.chapters[-1]}"
    if condition in STATE_MEMORIES:
        return state_memory(run.state.fact_digest() if run.state is not None else "")
    raise ValueError(f"unknown condition {condition!r}")


def build_chapter_prompt(
    premise: Premise, run: StoryRun, chapter: int, total: int, words: int = 500
) -> str:
    """Canon is supplied for chapter 1 only; after that memory has to carry it."""
    header = (
        f"Setting: {premise.setting}\n"
        f"Characters: {', '.join(premise.characters)}"
    )
    if chapter == 1:
        header += (
            "\n\nEstablished details that must hold for the whole novel "
            "(state them naturally in this chapter):\n"
            + premise.canon_block()
        )
    return writer_prompt(
        title=premise.title,
        header=header,
        memory=build_memory(run, beat=premise.beats[(chapter - 1) % len(premise.beats)]),
        beat=premise.beats[(chapter - 1) % len(premise.beats)],
        chapter=chapter,
        words=words,
    )


def build_summary_prompt(run: StoryRun, chapter_text: str, words: int = 220) -> str:
    return SUMMARY_PROMPT.format(
        summary=run.summary or "(nothing yet)", chapter=chapter_text, words=words
    )
