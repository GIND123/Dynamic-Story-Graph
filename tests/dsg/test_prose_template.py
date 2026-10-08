"""The opt-in prose template, and the guarantee that the classic one never moves."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dsg.generate.canon import build_premises
from dsg.generate.conditions import StoryRun, build_chapter_prompt
from dsg.generate.prompt import (
    CONTINUE_CUE,
    WRITER_PROMPT,
    background_memory,
    load_example,
    previous_tail,
    prose_system,
    state_memory,
    with_note,
    writer_messages,
    writer_prompt,
)
from dsg.generate.repair import Conflict, repair_instruction
from dsg.generate.run import init_runs


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Recorded from the classic template as published. A change here is a change to
# the prompt every reported generation result was produced with.
_FROZEN = {
    "WRITER_PROMPT": "067661325e2ffbdf2481b706403d1cf4cfe1074d00161d76085ec815cb29a9e2",
    "state_memory": "da0b881698ab543082c042a1bec260034065fa32735f90dec887d7c9ad4c5df2",
    "ch1": "eda7d2d02fbf148850efc480f42a1d7044bc8792db22f387584413de2998c174",
    "ch2": "f4788d6d19f78c8355f46aa1204e33e154efa5da8d83b2d35c9c5ea3a21fe61c",
    "hybrid_ch2": "c65bdc06821d9298eb354cb1688e9051f591f58cbea2e0a58e660caa274f22a2",
}


def test_classic_template_is_frozen():
    """The published results depend on these exact prompts, byte for byte."""
    premise = build_premises(1)[0]
    run = StoryRun(story_id=premise.story_id, condition="base:last-chapter")
    seen = {
        "WRITER_PROMPT": _sha(WRITER_PROMPT),
        "state_memory": _sha(state_memory("x")),
        "ch1": _sha(build_chapter_prompt(premise, run, 1, 8)),
    }
    run.chapters.append("The first chapter.")
    seen["ch2"] = _sha(build_chapter_prompt(premise, run, 2, 8))
    hybrid = init_runs([premise], ("base:dsg-hybrid",))[0]
    hybrid.chapters.append("The first chapter.")
    seen["hybrid_ch2"] = _sha(build_chapter_prompt(premise, hybrid, 2, 8))
    assert seen == _FROZEN, "the classic writer prompt changed"


def _prose(**kw):
    fields = dict(title="The Locket", beat="An alliance shifts.", chapter=2, total=8,
                  header="Setting: a canal port\nCharacters: Hesper, Jarrow")
    fields.update(kw)
    return writer_messages("prose", **fields)


def test_classic_messages_are_exactly_the_classic_prompt():
    fields = dict(title="T", beat="B", chapter=3, memory="m", header="h", words=500)
    assert writer_messages("classic", **fields) == [
        {"role": "user", "content": writer_prompt(**fields)}
    ]


def test_classic_refuses_an_example_and_unknown_templates_fail():
    with pytest.raises(ValueError):
        writer_messages("classic", example="voice", title="T", beat="B", chapter=1)
    with pytest.raises(ValueError):
        writer_messages("poetry", title="T", beat="B", chapter=1)


def test_prose_has_a_system_voice_and_a_user_turn():
    messages = _prose()
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "novelist" in messages[0]["content"]
    assert "<<<" not in messages[0]["content"], "no example unless one is supplied"
    user = messages[1]["content"]
    assert "chapter 2 of 8" in user
    assert "must keep consistent" not in user.lower()
    assert user.rstrip().endswith("Begin chapter 2.")


def test_the_example_goes_in_the_system_turn():
    passage = "word " * 200
    system = _prose(example=passage)[0]["content"]
    assert "<<<" in system and passage.strip() in system


def test_canon_is_rendered_only_when_given():
    block = "- Hesper has grey eyes."
    assert "Hesper has grey eyes." in _prose(chapter=1, canon=block)[1]["content"]
    assert "What is true in this world" not in _prose()[1]["content"]


def test_chapter_one_is_never_told_both_to_show_and_not_to_restate():
    first = _prose(chapter=1, canon="- Hesper has grey eyes.")[1]["content"]
    later = _prose()[1]["content"]
    assert "Bring each of the facts above in once" in first
    assert "Do not restate" not in first
    assert "Never contradict what earlier chapters established" in later
    assert "facts above" not in later


def test_state_is_framed_as_background_not_a_checklist():
    memory = background_memory("Hesper: eye_colour grey")
    assert "do not contradict" in memory and "do not restate" in memory
    assert "must keep consistent" not in memory.lower()


def test_the_next_beat_is_hinted_except_in_the_last_chapter():
    hint = "Later in the story"
    assert hint in _prose(next_beat="A choice is made.")[1]["content"]
    assert hint not in _prose(chapter=8, next_beat="A choice is made.")[1]["content"]


def test_size_sets_the_note_and_unknown_sizes_fail():
    assert "small moment" in _prose(size="small", words=300)[1]["content"]
    assert "About 300 words" in _prose(size="small", words=300)[1]["content"]
    with pytest.raises(KeyError):
        _prose(size="huge")


def test_the_tail_appears_only_when_there_is_a_previous_chapter():
    previous = "First.\n\nSecond.\n\nThird.\n\nFourth, the last."
    user = _prose(previous=previous)[1]["content"]
    assert "The previous chapter ended:" in user and "Fourth, the last." in user
    assert "First." not in user
    assert "previous chapter ended" not in _prose()[1]["content"]


def test_the_tail_is_never_shown_twice():
    """A memory that carries the previous chapter gets a cue, not a second copy."""
    previous = "Hesper locked the door.\n\nThe bell rang twice.\n\nJarrow did not come."
    for memory in (
        "The previous chapter:\n" + previous,
        "Background (true in this story):\n- Hesper: grey eyes\n\n"
        "The previous chapter:\n" + previous,
        "The story so far:\nOPENING\n\n[...]\n\n" + previous,
    ):
        user = _prose(memory=memory, previous=previous)[1]["content"]
        for paragraph in previous.split("\n\n"):
            assert user.count(paragraph) == 1, paragraph
        assert CONTINUE_CUE in user and "The previous chapter ended:" not in user
        assert memory.strip() in user, "the memory itself is not trimmed"

    for memory in ("", "Background (true in this story):\n- Hesper: grey eyes"):
        user = _prose(memory=memory, previous=previous)[1]["content"]
        assert "The previous chapter ended:" in user and CONTINUE_CUE not in user
        for paragraph in previous.split("\n\n"):
            assert user.count(paragraph) == 1, paragraph


def test_tail_keeps_whole_paragraphs_within_both_limits():
    paragraphs = [f"Paragraph {i}. " + "x" * 300 for i in range(6)]
    tail = previous_tail("\n\n".join(paragraphs))
    kept = tail.split("\n\n")
    assert len(kept) <= 3 and len(tail) <= 1200
    assert kept == paragraphs[-len(kept):], "cut at paragraph boundaries"
    assert len(previous_tail("\n\n".join(paragraphs), max_paragraphs=2).split("\n\n")) == 2


def test_an_oversized_last_paragraph_is_cut_at_a_sentence_start():
    long = " ".join(f"Sentence number {i} ends here." for i in range(200))
    tail = previous_tail(long)
    assert len(tail) <= 1200
    assert tail.startswith("Sentence number") and tail.endswith("ends here.")


def test_tail_of_nothing_is_empty():
    assert previous_tail("") == "" and previous_tail("   \n\n ") == ""


def test_example_loader(tmp_path):
    assert load_example(None) == ""
    shipped = Path(__file__).parents[2] / "dsg" / "generate" / "voice" / "example.txt"
    assert load_example(shipped) == "", "the shipped placeholder is skipped"
    good = tmp_path / "good.txt"
    good.write_text("word " * 200)
    assert load_example(good).split() == ["word"] * 200
    short = tmp_path / "short.txt"
    short.write_text("word " * 40)
    with pytest.raises(ValueError):
        load_example(short)
    with pytest.raises(FileNotFoundError):
        load_example(tmp_path / "missing.txt")


def test_repair_note_lands_in_the_last_user_turn():
    note = repair_instruction([Conflict("Hesper", "eye_colour", "grey", "blue")])
    fields = dict(title="T", beat="B", chapter=2)
    classic = writer_messages("classic", **fields)
    assert with_note(classic, note)[0]["content"] == writer_prompt(**fields) + note

    prose = _prose()
    repaired = with_note(prose, note)
    assert repaired[0] == prose[0], "the system turn is untouched"
    assert repaired[1]["content"].endswith(note)
    assert not prose[1]["content"].endswith(note), "the original is not mutated"


def test_prose_system_is_stable_without_an_example():
    assert prose_system() == prose_system("   ")
