"""The opt-in prose template, and the guarantee that the classic one never moves."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dsg.generate.canon import build_premises
from dsg.generate.conditions import StoryRun, build_chapter_prompt
from dsg.generate.prompt import (
    CONTINUE_CUE,
    EXAMPLE_WORDS,
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

_SHIPPED = Path(__file__).parents[2] / "dsg" / "generate" / "voice" / "example.txt"


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
    assert "Keep each fact attached to its own person or object." in first
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
    placeholder = tmp_path / "placeholder.txt"
    placeholder.write_text("[[PLACEHOLDER: paste a passage here]]")
    assert load_example(placeholder) == "", "a placeholder is never an example"
    lo, hi = EXAMPLE_WORDS
    assert lo <= len(load_example(_SHIPPED).split()) <= hi, "the shipped passage loads"
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


# ------------------------------------------------------------ step 3: plumbing

def _story(condition, chapters=()):
    from dsg.generate.run import apply_extraction

    premise = build_premises(1)[0]
    run = init_runs([premise], (condition,))[0]
    for text in chapters:
        run.chapters.append(text)
        apply_extraction(
            run, "FACT | Hesper | eye_colour | grey\nFACT | seal | material | iron\n"
            "FACT | Jarrow | hair_colour | auburn", text,
        )
    return premise, run


_CHAPTER = (
    "Hesper set the iron seal on the table.\n\n"
    '"Three ships this month," she said.\n\n'
    "Jarrow pushed his auburn hair back and did not answer.\n\n"
    "Outside, the fog came down over the water."
)


def test_template_tags_parse_without_disturbing_memory_or_variant():
    from dsg.generate.conditions import (
        BASE_LADDER,
        CONDITIONS,
        RETRIEVAL_LADDER,
        SELECTION_LADDER,
        memory_of,
        template_of,
        variant_of,
    )

    tagged = "base:dsg-hybrid~prose@bo4-graph"
    assert (variant_of(tagged), memory_of(tagged), template_of(tagged)) == (
        "base", "dsg-hybrid", "prose"
    )
    for condition in CONDITIONS + BASE_LADDER + RETRIEVAL_LADDER + SELECTION_LADDER:
        assert template_of(condition) == "classic"
        assert "~" not in memory_of(condition) + variant_of(condition)
    with pytest.raises(ValueError):
        template_of("base:none~poetry")


def test_tagged_conditions_get_the_same_state_as_their_twin():
    premise = build_premises(1)[0]
    runs = {r.condition: r for r in init_runs(
        [premise], ("base:dsg-hybrid", "base:dsg-hybrid~prose", "base:last-chapter~prose")
    )}
    assert runs["base:dsg-hybrid~prose"].state.policy.name == "dsg-full"
    assert runs["base:dsg-hybrid"].state.policy.name == "dsg-full"
    assert runs["base:last-chapter~prose"].state is None


def test_classic_chapter_messages_are_the_classic_prompt():
    from dsg.generate.conditions import build_chapter_messages
    from dsg.generate.run import chapter_messages, chapter_prompts

    premise, run = _story("base:dsg-hybrid", [_CHAPTER])
    assert build_chapter_messages(premise, run, 2, 5) == [
        {"role": "user", "content": build_chapter_prompt(premise, run, 2, 5)}
    ]
    by_id = {premise.story_id: premise}
    assert [m[0]["content"] for m in chapter_messages([run], by_id, 2, 5)] == (
        chapter_prompts([run], by_id, 2, 5)
    )


def test_prose_gives_canon_in_chapter_one_only():
    from dsg.generate.conditions import build_chapter_messages

    premise, run = _story("base:none~prose")
    first = build_chapter_messages(premise, run, 1, 5)[1]["content"]
    assert all(f.statement in first for f in premise.canon)
    run.chapters.append(_CHAPTER)
    later = build_chapter_messages(premise, run, 2, 5)[1]["content"]
    assert not any(f.statement in later for f in premise.canon), "memory, not prompt"


def test_prose_shows_the_same_memory_as_classic_apart_from_digest_framing():
    from dsg.generate.conditions import build_chapter_messages, build_memory

    premise, run = _story("base:last-chapter~prose", [_CHAPTER])
    user = build_chapter_messages(premise, run, 2, 5)[1]["content"]
    assert build_memory(run, beat=premise.beats[1]) in user, "memory is not trimmed"

    premise, run = _story("base:dsg-hybrid~prose", [_CHAPTER])
    user = build_chapter_messages(premise, run, 2, 5)[1]["content"]
    assert "do not contradict" in user and "must keep consistent" not in user.lower()
    assert "auburn hair" in user and "hair_colour" not in user


def test_a_real_prose_prompt_never_shows_the_previous_chapter_twice():
    from dsg.generate.conditions import build_chapter_messages
    from dsg.generate.prompt import continuity_of

    for condition, quoted in (
        ("base:last-chapter~prose", False), ("base:dsg-hybrid~prose", False),
        ("base:beat-hybrid~prose", False), ("base:none~prose", True),
        ("base:dsg-state~prose", True),
    ):
        premise, run = _story(condition, [_CHAPTER])
        user = build_chapter_messages(premise, run, 2, 5)[1]["content"]
        paragraphs = _CHAPTER.split("\n\n")
        for paragraph in paragraphs:
            assert user.count(paragraph) <= 1, (condition, paragraph)
        assert user.count(paragraphs[-1]) == 1, "the writer always sees where it left off"
        assert ("The previous chapter ended:" in user) is quoted, condition
        messages = build_chapter_messages(premise, run, 2, 5)
        assert continuity_of(messages) == ("tail" if quoted else "cue"), condition


def test_classic_digest_keeps_its_labels():
    from dsg.generate.conditions import build_memory

    _, run = _story("base:dsg-state", [_CHAPTER])
    memory = build_memory(run)
    assert memory.startswith("Established facts you must keep consistent")
    assert "hair_colour auburn" in memory


def test_beat_budgets_scale_with_the_size_of_the_event():
    from dsg.generate.canon import _BEATS, beat_budget, beat_size

    sizes = {beat_size(b) for b in _BEATS}
    assert sizes == {"small", "medium", "large"}
    small = beat_budget("An alliance shifts.", 500)
    medium = beat_budget("An accusation is made in public.", 500)
    large = beat_budget("A last reckoning between the two who began it.", 500)
    assert small < medium < large
    assert beat_size("Something no outline mentions.") == "medium"
    mean = sum(beat_budget(b, 500) for b in _BEATS) / len(_BEATS)
    assert abs(mean - 500) / 500 <= 0.10, "total length stays matched to classic"


def test_the_budget_is_flat_for_classic_and_per_beat_for_prose():
    from dsg.generate.conditions import build_chapter_messages, chapter_budget

    premise, classic = _story("base:none")
    _, prose = _story("base:none~prose")
    assert chapter_budget(premise, classic, 1, 500) == 500
    assert chapter_budget(premise, prose, 1, 500) == 600       # beat 1 is large
    user = build_chapter_messages(premise, prose, 1, 5)[1]["content"]
    assert "About 600 words" in user


def test_chapter_one_is_never_given_less_than_the_medium_budget():
    """Chapter 1 plants every canon fact, so a small opening beat still gets room."""
    from dsg.generate.canon import Premise
    from dsg.generate.conditions import build_chapter_messages, chapter_budget

    base, run = _story("base:none~prose")
    small_first = Premise(**{**{f: getattr(base, f) for f in base.__slots__},
                             "beats": ["An alliance shifts."] + base.beats[1:]})
    assert chapter_budget(small_first, run, 1, 500) == 450
    user = build_chapter_messages(small_first, run, 1, 5)[1]["content"]
    assert "About 450 words" in user and "small moment" not in user
    run.chapters.append(_CHAPTER)
    later = Premise(**{**{f: getattr(base, f) for f in base.__slots__},
                       "beats": base.beats[:1] + ["An alliance shifts."] + base.beats[2:]})
    assert chapter_budget(later, run, 2, 500) == 300, "only chapter 1 is lifted"
    assert chapter_budget(base, run, 1, 500) == 600, "a large opening stays large"


def test_records_carry_template_budget_and_truncation():
    from dsg.generate.conditions import build_chapter_messages
    from dsg.generate.prompt import continuity_of
    from dsg.generate.run import record

    premise, run = _story("base:last-chapter~prose")
    payload = record(run, premise, _CHAPTER, 2, word_budget=600, truncated=True,
                     continuity="cue").to_json()
    assert (payload["template"], payload["word_budget"], payload["truncated"],
            payload["continuity"]) == ("prose", 600, True, "cue")
    premise, run = _story("base:none")
    plain = record(run, premise, _CHAPTER, 1)
    assert (plain.template, plain.continuity) == ("classic", "none")
    assert continuity_of(build_chapter_messages(premise, run, 1, 5)) == "none"


def _facts(line):
    head, _, facts = line.partition(": ")
    return head, facts.split("; ") if facts else []


def test_rewording_keeps_every_fact_of_a_real_digest():
    """Only labels change: each head, fact count, object and negation survives."""
    from dsg.generate.prompt import plain_digest
    from dsg.generate.run import apply_extraction

    premise = build_premises(1)[0]
    run = init_runs([premise], ("base:dsg-state",))[0]
    text = "Hesper, Jarrow and Halloran met at the quay with the seal."
    run.chapters.append(text)
    apply_extraction(run, "\n".join([
        "FACT | Hesper | eye_colour | grey", "FACT | Hesper | possesses | seal",
        "FACT | Hesper | location | the quay", "FACT | Hesper | occupation | cartographer",
        "FACT | Jarrow | hair_colour | auburn", "FACT | Jarrow | sibling_of | Hesper",
        "FACT | Jarrow | birthplace | Ravensmoor", "FACT | seal | material | iron",
        "FACT | Halloran | alive | false", "FACT | Halloran | emotion | uneasy",
    ]), text)
    digest = run.state.fact_digest()
    reworded = plain_digest(digest)
    assert len(digest.splitlines()) == len(reworded.splitlines())
    n_facts = 0
    for before, after in zip(digest.splitlines(), reworded.splitlines(), strict=True):
        head_b, facts_b = _facts(before)
        head_a, facts_a = _facts(after)
        assert head_a == head_b and len(facts_a) == len(facts_b)
        for fact, phrase in zip(facts_b, facts_a, strict=True):
            n_facts += 1
            predicate, _, obj = fact.partition(" ")
            assert "_" not in phrase, phrase
            if predicate == "alive":
                assert phrase in ("alive", "dead")
            else:
                assert obj.removeprefix("NOT ") in phrase
    assert n_facts >= 8, digest


def test_rewording_covers_every_predicate_and_negation():
    from dsg.generate.prompt import _PHRASES, plain_digest, plain_fact
    from dsg.lexicon import CANONICAL

    assert set(_PHRASES) | {"alive"} >= CANONICAL, "every known predicate has a phrase"
    for predicate in sorted(set(_PHRASES)):
        line = f"- Hesper: {predicate} Ravensmoor; {predicate} NOT Ravensmoor"
        phrase, negated = plain_digest(line).partition(": ")[2].split("; ")
        assert "Ravensmoor" in phrase and "Ravensmoor" in negated
        assert "not" in negated and "not" not in phrase.replace("Ravensmoor", "")
    assert plain_fact("alive", "true") == "alive" and plain_fact("alive", "dead") == "dead"
    assert plain_fact("alive", "true", negated=True) == "dead"
    assert plain_fact("wears", "a coat") == "wears a coat", "unknown predicates still read"
    assert plain_fact("possesses", "seal") == "has the seal"
    assert plain_fact("possesses", "Hesper's map") == "has Hesper's map"
    assert plain_fact("occupation", "apothecary") == "works as an apothecary"
    passthrough = "- Odell\n(nothing established yet)"
    assert plain_digest(passthrough) == passthrough


def test_rotation_starts_each_story_further_along_the_outline():
    from dsg.generate.canon import _BEATS, rotate_beats

    premises = rotate_beats(build_premises(n=11), 2)
    assert [p.beats[0] for p in premises[:3]] == [_BEATS[0], _BEATS[2], _BEATS[4]]
    assert premises[10].beats[0] == _BEATS[0], "wraps around the outline"
    assert all(sorted(p.beats) == sorted(_BEATS) for p in premises)
    stock = build_premises(n=3)
    assert [p.to_json() for p in rotate_beats(build_premises(n=3), 0)] == [
        p.to_json() for p in stock
    ]


def test_tuned_conditions_stay_classic():
    from dsg.generate.conditions import CONDITIONS, PROSE_LADDER, check_conditions

    check_conditions(CONDITIONS + PROSE_LADDER)
    with pytest.raises(ValueError):
        check_conditions(("tuned:full-context~prose",))


def test_a_prose_run_never_loses_its_example_by_accident(tmp_path):
    from dsg.generate.conditions import BASE_LADDER, PROSE_LADDER, resolve_example

    placeholder = tmp_path / "placeholder.txt"
    placeholder.write_text("[[PLACEHOLDER: paste a passage here]]")
    voice = tmp_path / "voice.txt"
    voice.write_text("word " * 200)

    assert resolve_example(PROSE_LADDER, str(voice)).split() == ["word"] * 200
    for path in ("", str(placeholder)):
        with pytest.raises(ValueError, match="--no-example"):
            resolve_example(PROSE_LADDER, path)
    assert resolve_example(PROSE_LADDER, no_example=True) == ""
    with pytest.raises(ValueError, match="not both"):
        resolve_example(PROSE_LADDER, str(voice), no_example=True)
    assert resolve_example(BASE_LADDER) == "", "classic runs need nothing"
    with pytest.raises(ValueError, match="only applies"):
        resolve_example(BASE_LADDER, str(voice))


def test_the_continuity_cue_asks_for_a_new_moment_not_a_restaged_ending():
    assert "starts after the previous chapter's final line" in CONTINUE_CUE
    assert "Do not reuse any sentence or line of dialogue" in CONTINUE_CUE
    user = _prose(memory="The previous chapter:\nA.\n\nB.", previous="A.\n\nB.")[1]["content"]
    assert CONTINUE_CUE in user


def test_only_the_prose_ladder_changes_chapter_sampling():
    from dsg.generate.conditions import CHAPTER_SAMPLING, chapter_sampling

    assert CHAPTER_SAMPLING == {"max_tokens": 900}, "published ladders are unchanged"
    for ladder in ("base", "retrieval", "selection", "all", "unknown"):
        assert chapter_sampling(ladder) == {"max_tokens": 900}
    assert chapter_sampling("prose") == {"max_tokens": 1400, "repetition_penalty": 1.1}
    chapter_sampling("prose")["max_tokens"] = 1
    assert chapter_sampling("prose")["max_tokens"] == 1400, "callers get a copy"


def test_the_quoted_tail_carries_the_same_new_moment_instruction():
    from dsg.generate.prompt import TAIL_CUE

    user = _prose(previous="A.\n\nB.")[1]["content"]
    assert "The previous chapter ended:" in user and TAIL_CUE in user
    assert "Pick up from here" not in user
    shared = "Open on a new moment (later in time, a new place, or the next action). "
    shared += "Do not reuse any sentence or line of dialogue from the previous chapter."
    assert shared in TAIL_CUE and shared in CONTINUE_CUE
