"""PREDICTOR 2 — Vector RAG.

Retrieves budget-matched k PROSE chunks via the existing
app.vectors.similar_passages (manuscript-scoped, reused — not the index
directly), EXCLUDES any retrieved chunk that contains the target quote line (so
it can't trivially read its own answer), concatenates the rest up to the budget,
and asks the LLM. Reads prose chunks only — automatically honest.
"""

from __future__ import annotations

from typing import Callable

from baselines.base import BasePredictor, join_to_budget
from baselines.flat_long_context import _CLOZE_PROMPT, _QUOTE_PROMPT

# (manuscript_id, query, k) -> list[{"chapter","text","score"}]
Retrieve = Callable[[str, str, int], list]


def _default_retrieve(manuscript_id: str, query: str, k: int) -> list:
    from app.vectors import similar_passages

    return similar_passages(manuscript_id, query, k)


def _exclude_self(chunks: list, target_text: str) -> list:
    """Drop chunks that contain the target line (guards self-retrieval)."""
    snippet = target_text.strip()[:40]
    return [c for c in chunks if snippet and snippet not in c.get("text", "")]


class VectorRAGPredictor(BasePredictor):
    method_name = "vector_rag"

    def __init__(self, *args, retrieve: Retrieve | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Budget-matched k; similar_passages over-fetches k*4 globally then
        # filters to this manuscript, so requesting rag_k yields ~rag_k chunks.
        self.retrieve = retrieve or _default_retrieve

    def predict_quote(self, quote_text: str, context: dict) -> str | None:
        mid = context.get("manuscript_id") or self.manuscript_id
        candidates = list(context.get("candidates", []))
        chunks = self.retrieve(mid, quote_text, self.config.rag_k)
        chunks = _exclude_self(chunks, quote_text)[: self.config.rag_k]
        prose = join_to_budget([c["text"] for c in chunks], self.config.token_budget)
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

    def predict_cloze(self, masked_passage: str, manuscript_id: str) -> str | None:
        query = masked_passage.replace("[MASK]", " ").strip()
        chunks = self.retrieve(manuscript_id, query, self.config.rag_k)
        # Exclude the source passage itself (match its pre-mask fragment).
        before = masked_passage.split("[MASK]", 1)[0].strip()
        chunks = _exclude_self(chunks, before or masked_passage)[: self.config.rag_k]
        prose = join_to_budget([c["text"] for c in chunks], self.config.token_budget)
        prompt = _CLOZE_PROMPT.format(masked=masked_passage, context=prose)
        answer = self._ask_llm(prompt, {"task": "cloze"})
        return answer.strip().splitlines()[0].strip() if answer.strip() else None
