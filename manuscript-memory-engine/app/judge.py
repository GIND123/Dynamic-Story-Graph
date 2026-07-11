"""Tier-2 LLM judge — called only when Tier-1 passes.

Thin wrapper: formats the canon dict for the judge prompt and delegates to
the LLMClient implementation (FakeLLM in tests, AnthropicLLM in production).
Keeping this separate from llm.py means the judge orchestration can change
(e.g. add a second-opinion judge, log verdicts) without touching the protocol.
"""

from app.llm import LLMClient
from app.models import JudgeVerdict
from app.retrieval import format_canon


def evaluate(canon: dict, draft: str, llm: LLMClient) -> JudgeVerdict:
    """Run the LLM judge against the draft. Called only after tier1_check passes."""
    return llm.judge(format_canon(canon), draft)
