"""The writer prompts, shared by training and every inference condition.

Training and inference must agree exactly on the shape of the ask, or a
fine-tuned writer is evaluated out of distribution and the comparison measures
formatting drift rather than method. So every template lives here, in one
place, and within a template every condition differs only in what goes into
``memory``.

There are two templates. ``classic`` is the one every published result was
produced with and is frozen. ``prose`` is an opt-in treatment aimed at
readability: a system voice, an optional example passage, canon framed as
background, the end of the previous chapter, and a budget sized to the beat.
It is a bundle, so a difference between them is the bundle's, not any one part's.
"""

from __future__ import annotations

import re
from pathlib import Path

WRITER_PROMPT = """You are writing chapter {chapter} of a novel. Write prose only.

Title: {title}
{header}{memory}
What happens in this chapter:
{beat}

Write chapter {chapter} in about {words} words. Keep every established detail
consistent. Do not summarise, do not use headings, write only the chapter prose.

CHAPTER {chapter}:"""


def writer_prompt(
    *,
    title: str,
    beat: str,
    chapter: int,
    memory: str = "",
    header: str = "",
    words: int = 500,
) -> str:
    """``memory`` is the only thing that varies between conditions."""
    return WRITER_PROMPT.format(
        title=title.strip() or "Untitled",
        header=(header.rstrip() + "\n") if header.strip() else "",
        memory=("\n" + memory.strip() + "\n") if memory.strip() else "",
        beat=beat.strip() or "Continue the story.",
        chapter=chapter,
        words=words,
    )


def state_memory(digest: str) -> str:
    """How a graph state is presented to the writer, in training and inference."""
    return (
        "Established facts you must keep consistent (the current canon):\n"
        + (digest.strip() or "(nothing established yet)")
    )


# --------------------------------------------------------------------- prose

TEMPLATES = ("classic", "prose")

PROSE_SYSTEM = """You are a novelist writing modern, plain fiction prose. Write the way a good
contemporary novel reads: concrete, specific, easy to read aloud. Show what
happens through what people do and say, and through small physical details.
Trust the reader to infer feelings and history. Never explain the story,
never sound like an encyclopedia, a summary, or an assistant."""

EXAMPLE_BLOCK = """

Here is a short passage in the voice to write in. It is from a different
story: take its voice and rhythm, not its content, names, or events.
<<<
{example}
>>>"""

PROSE_PROMPT = """This is chapter {chapter} of {total} of a novel called "{title}".

{header}{background}{memory}{tail}What happens in this chapter:
{beat}
{next_hint}
How to write it:
- About {words} words. {size_note}
- Tell it as scenes: people doing things and talking to each other.
- Do not name or summarise feelings ("she felt afraid", "he was angry"); let
  them show in what people do and say.
- Do not list a character's traits, appearance, or history. Mention a detail
  only when the scene uses it.
- {canon_rule}
- No headings, no summary, no commentary. Prose only.

Begin chapter {chapter}."""

# Chapter 1 must get the planted canon onto the page; later chapters must not
# drift into restating it. One rule each, so the two never contradict.
CANON_RULES = {
    "introduce": "Bring each of the facts above in once, inside the action, and do not "
    "repeat them. Keep each fact attached to its own person or object.",
    "background": "Never contradict what earlier chapters established. Do not restate "
    "it unless this scene needs it.",
}

SIZE_NOTES = {
    "small": "This is a small moment: keep it brief and end the chapter when it is done.",
    "medium": "Give it the space it needs and no more.",
    "large": "This is a turning point: let the scene play out fully.",
}

# A passage outside this range is either too thin to carry a voice or long
# enough to crowd the prompt and invite copying.
EXAMPLE_WORDS = (150, 300)
PLACEHOLDER_MARK = "[[PLACEHOLDER"


def _block(text: str) -> str:
    return text.strip() + "\n\n" if text.strip() else ""


def prose_system(example: str = "") -> str:
    """The voice, plus the example passage when one is supplied."""
    if not example.strip():
        return PROSE_SYSTEM
    return PROSE_SYSTEM + EXAMPLE_BLOCK.format(example=example.strip())


def chapter_one_background(canon_block: str) -> str:
    """Canon in chapter 1: shown once, inside the action, never as a list.

    The facts still have to reach the page, or the state cannot capture them
    and the scorer has nothing to hold later chapters to.
    """
    if not canon_block.strip():
        return ""
    return (
        "What is true in this world. Each of these must come through once in this\n"
        "chapter, inside the action (something a character does, handles, or says),\n"
        "never as a description or a list:\n" + canon_block.strip()
    )


def background_memory(digest: str) -> str:
    """How a graph state is presented to the writer under the prose template."""
    return (
        "Background (true in this story; do not contradict it, and do not restate it\n"
        "unless the scene needs it):\n" + (digest.strip() or "(nothing established yet)")
    )


# How each predicate reads as a phrase. A "{x}" is the object, verbatim; "{a}"
# is the object with an article where a bare common noun would read as a label.
_PHRASES = {
    "eye_colour": "{x} eyes",
    "hair_colour": "{x} hair",
    "material": "made of {x}",
    "birthplace": "born in {x}",
    "parent_of": "parent of {x}",
    "child_of": "child of {x}",
    "sibling_of": "sibling of {x}",
    "relative_of": "related to {x}",
    "identity_of": "really {x}",
    "true_name": "real name {x}",
    "gender": "{x}",
    "species": "{x}",
    "location": "at {a}",
    "possesses": "has {a}",
    "married_to": "married to {x}",
    "engaged_to": "engaged to {x}",
    "stance_toward": "attitude toward {x}",
    "emotion": "feeling {x}",
    "occupation": "works as {an}",
    "resides_in": "lives in {x}",
    "travelling_to": "travelling to {x}",
    "knows": "knows {x}",
    "member_of": "member of {a}",
    "health": "health: {x}",
    "wealth": "wealth: {x}",
}
_NEGATED = {"eye_colour": "eyes not {x}", "hair_colour": "hair not {x}"}
_ALIVE_TRUE = frozenset({"true", "yes", "alive", "living"})
_DETERMINERS = ("a ", "an ", "the ", "his ", "her ", "their ", "its ", "my ", "our ",
                "your ", "some ")
_DIGEST_LINE = re.compile(r"^- (?P<head>[^:(]+(?: \(also: [^)]*\))?): (?P<facts>.+)$")


def _with_article(obj: str, indefinite: bool) -> str:
    lowered = obj.lower()
    if not obj[:1].islower() or lowered.startswith(_DETERMINERS):
        return obj
    if indefinite:
        return ("an " if lowered[0] in "aeiou" else "a ") + obj
    return "the " + obj


def plain_fact(predicate: str, obj: str, negated: bool = False) -> str:
    """One digest fact as a short phrase. Deterministic: a table, not a model."""
    if predicate == "alive":
        alive = obj.strip().lower().rstrip(".") in _ALIVE_TRUE
        return "alive" if alive != negated else "dead"
    if negated and predicate in _NEGATED:
        return _NEGATED[predicate].format(x=obj)
    template = _PHRASES.get(predicate, predicate.replace("_", " ") + " {x}")
    phrase = template.format(
        x=obj, a=_with_article(obj, False), an=_with_article(obj, True)
    )
    return "not " + phrase if negated else phrase


def plain_digest(digest: str) -> str:
    """Reword a state digest as plain phrases, keeping every fact and its order.

    The facts and their selection are the state's; only the labels change, so
    "hair_colour auburn" reaches the writer as "auburn hair". Lines that are
    not facts pass through untouched.
    """
    out: list[str] = []
    for line in (digest or "").splitlines():
        match = _DIGEST_LINE.match(line)
        if not match:
            out.append(line)
            continue
        phrases = []
        for fact in match["facts"].split("; "):
            predicate, _, rest = fact.partition(" ")
            negated = rest.startswith("NOT ")
            obj = rest[4:] if negated else rest
            phrases.append(plain_fact(predicate, obj, negated) if obj else fact)
        out.append(f"- {match['head']}: " + "; ".join(phrases))
    return "\n".join(out)


_SENTENCE_START = re.compile(r"(?<=[.!?])[\"'”’]?\s+")


def previous_tail(text: str, max_paragraphs: int = 3, max_chars: int = 1200) -> str:
    """The last few paragraphs of the previous chapter, whole where possible.

    Paragraphs are taken from the end until the next would break either limit.
    A final paragraph longer than the budget on its own is cut at a sentence
    start, so the writer never continues from the middle of a sentence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paragraphs) == 1 and "\n" in paragraphs[0]:
        paragraphs = [p.strip() for p in paragraphs[0].splitlines() if p.strip()]
    if not paragraphs:
        return ""
    kept: list[str] = []
    for paragraph in reversed(paragraphs):
        if len(kept) >= max_paragraphs:
            break
        if sum(len(p) + 2 for p in kept) + len(paragraph) > max_chars:
            break
        kept.append(paragraph)
    if kept:
        return "\n\n".join(reversed(kept))
    last = paragraphs[-1][-max_chars:]
    match = _SENTENCE_START.search(last)
    return last[match.end():].strip() if match else last.strip()


TAIL_HEADER = "The previous chapter ended:"
# The smoke run showed "pick up where it ends" read as "restage the ending":
# chapter 2 reopened on chapter 1's last exchange, mostly verbatim.
CONTINUE_CUE = (
    "This chapter starts after the previous chapter's final line. Open on a new "
    "moment (later in time, a new place, or the next action). Do not reuse any "
    "sentence or line of dialogue from the previous chapter."
)
# The same instruction after a quoted tail, pointing at the quote instead.
TAIL_CUE = (
    "This chapter starts after the final line quoted above. Open on a new moment "
    "(later in time, a new place, or the next action). Do not reuse any sentence "
    "or line of dialogue from the previous chapter."
)


def tail_block(previous: str, memory: str = "") -> str:
    """Where the writer picks up from.

    When the memory already carries the end of the previous chapter, quoting it
    again would only duplicate it, so a one-line cue points back at it instead.
    The memory itself is never trimmed: it must stay identical between arms.
    """
    tail = previous_tail(previous)
    if not tail:
        return ""
    if tail in memory:
        return CONTINUE_CUE
    return (
        TAIL_HEADER + '\n"""\n' + tail + '\n"""\n' + TAIL_CUE
    )


def continuity_of(messages: list[dict]) -> str:
    """Which continuity device a rendered prompt used: "tail", "cue" or "none"."""
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if CONTINUE_CUE in user:
        return "cue"
    if TAIL_HEADER in user:
        return "tail"
    return "none"


def prose_writer_prompt(
    *,
    title: str,
    beat: str,
    chapter: int,
    total: int,
    header: str = "",
    canon: str = "",
    memory: str = "",
    previous: str = "",
    next_beat: str = "",
    words: int = 450,
    size: str = "medium",
) -> str:
    """The prose user turn. ``canon`` is the chapter-1 block, ``previous`` the
    prior chapter's full text (only its end is shown)."""
    hint = ""
    if next_beat.strip() and chapter < total:
        hint = (
            "\nLater in the story (do not write this yet; let this chapter lean "
            "toward it): " + next_beat.strip() + "\n"
        )
    return PROSE_PROMPT.format(
        title=title.strip() or "Untitled",
        chapter=chapter,
        total=total,
        header=_block(header),
        background=_block(chapter_one_background(canon)),
        memory=_block(memory),
        tail=_block(tail_block(previous, memory)),
        beat=beat.strip() or "Continue the story.",
        next_hint=hint,
        words=words,
        size_note=SIZE_NOTES[size],
        canon_rule=CANON_RULES["introduce" if canon.strip() else "background"],
    )


def writer_messages(template: str, *, example: str = "", **fields) -> list[dict]:
    """The chat turns for one chapter, for training and inference alike.

    ``classic`` is a single user turn holding exactly ``writer_prompt``, so the
    published path is unchanged. ``prose`` adds a system turn, which also
    displaces the backbone's default "helpful assistant" system prompt.
    """
    if template == "classic":
        if example.strip():
            raise ValueError("the classic template takes no example passage")
        return [{"role": "user", "content": writer_prompt(**fields)}]
    if template == "prose":
        return [
            {"role": "system", "content": prose_system(example)},
            {"role": "user", "content": prose_writer_prompt(**fields)},
        ]
    raise ValueError(f"unknown template {template!r}; expected one of {TEMPLATES}")


def with_note(messages: list[dict], note: str) -> list[dict]:
    """Append a note (the repair instruction) to the last user turn, as a copy."""
    out = [dict(m) for m in messages]
    for message in reversed(out):
        if message["role"] == "user":
            message["content"] += note
            return out
    raise ValueError("no user turn to append to")


def load_example(path: str | Path | None) -> str:
    """The example passage, or "" when none is configured.

    The shipped file is a marked placeholder and is treated as no example; a
    path that was asked for but does not exist is an error, not a silent skip.
    """
    if not path:
        return ""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text or PLACEHOLDER_MARK in text:
        return ""
    n = len(text.split())
    lo, hi = EXAMPLE_WORDS
    if not lo <= n <= hi:
        raise ValueError(f"example passage is {n} words; it must be {lo}-{hi}")
    return text
