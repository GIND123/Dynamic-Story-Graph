"""The opt-in prose template, and the guarantee that the classic one never moves."""

from __future__ import annotations

import hashlib

from dsg.generate.canon import build_premises
from dsg.generate.conditions import StoryRun, build_chapter_prompt
from dsg.generate.prompt import WRITER_PROMPT, state_memory
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
