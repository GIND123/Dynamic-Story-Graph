"""Prompt-hardening tests.

Both the judge and the extract prompts feed untrusted draft text to the LLM,
so both must fence that text as data, not instructions.
"""

from app.llm import _EXTRACT_PROMPT, _JUDGE_PROMPT


class TestPromptInjectionFence:
    def test_judge_prompt_fences_draft_as_untrusted(self) -> None:
        assert "untrusted story text" in _JUDGE_PROMPT

    def test_extract_prompt_fences_draft_as_untrusted(self) -> None:
        assert "untrusted story text" in _EXTRACT_PROMPT
