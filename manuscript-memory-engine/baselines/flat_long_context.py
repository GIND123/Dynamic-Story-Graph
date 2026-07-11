"""PREDICTOR 1 — Flat long-context.

Reads a window of surrounding PROSE (~TOKEN_BUDGET) and asks the LLM the question.
It reads prose only — never a graph speaker label — so it is automatically honest:
the answer is implicit in the narrative ("said Elizabeth"), not handed to it.
"""

from __future__ import annotations

from baselines.base import (
    BasePredictor,
    fetch_chapter_window,
    join_to_budget,
    locate_quote_chapter,
)

_QUOTE_PROMPT = """\
You are attributing dialogue in a novel. Using ONLY the prose below, decide who \
speaks the quoted line.

PROSE:
{prose}

QUOTED LINE: "{quote}"

Answer with exactly one name from this list: {candidates}.
Name:"""

_CLOZE_PROMPT = """\
A character name in the PASSAGE has been hidden as [MASK]. Using the PASSAGE and \
the surrounding CONTEXT prose, name the single character that fills [MASK].

PASSAGE:
{masked}

CONTEXT:
{context}

Answer with just the name.
Name:"""


class FlatLongContextPredictor(BasePredictor):
    method_name = "flat_long_context"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._chapter_cache: dict = {}

    # -- quote attribution -------------------------------------------------
    def predict_quote(self, quote_text: str, context: dict) -> str | None:
        mid = context.get("manuscript_id") or self.manuscript_id
        candidates = list(context.get("candidates", []))
        center = locate_quote_chapter(self.driver, mid, quote_text, self._chapter_cache)
        if center is None:
            self._ask_llm("", {"task": "quote", "located": False})
            return None
        each = self.config.prose_chapters_each_side
        chapters = fetch_chapter_window(self.driver, mid, center - each, center + each)
        prose = join_to_budget([c["text"] for c in chapters], self.config.token_budget)
        prompt = _QUOTE_PROMPT.format(
            prose=prose, quote=quote_text, candidates=", ".join(candidates)
        )
        answer = self._ask_llm(
            prompt,
            {
                "task": "quote",
                "quote_id": context.get("quote_id"),
                "quote_type": context.get("quote_type"),
            },
        )
        return self.pick_candidate(answer, candidates)

    # -- cloze -------------------------------------------------------------
    def predict_cloze(self, masked_passage: str, manuscript_id: str) -> str | None:
        before = masked_passage.split("[MASK]", 1)[0].strip()
        snippet_source = before or masked_passage
        center = locate_quote_chapter(
            self.driver, manuscript_id, snippet_source, self._chapter_cache
        )
        context_prose = ""
        if center is not None:
            each = self.config.prose_chapters_each_side
            chapters = fetch_chapter_window(
                self.driver, manuscript_id, center - each, center + each
            )
            context_prose = join_to_budget(
                [c["text"] for c in chapters], self.config.token_budget
            )
        prompt = _CLOZE_PROMPT.format(masked=masked_passage, context=context_prose)
        answer = self._ask_llm(prompt, {"task": "cloze"})
        return answer.strip().splitlines()[0].strip() if answer.strip() else None
