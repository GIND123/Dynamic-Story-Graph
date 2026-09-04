"""The single writer prompt, shared by training and every inference condition.

Training and inference must agree exactly on the shape of the ask, or a
fine-tuned writer is evaluated out of distribution and the comparison measures
formatting drift rather than method. So there is one template, one place, and
every condition differs only in what goes into ``memory``.
"""

from __future__ import annotations

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
