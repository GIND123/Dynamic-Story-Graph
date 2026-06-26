"""Shared, configurable EQUAL-BUDGET config for the three baselines.

All three methods get the same TOKEN_BUDGET of information so the comparison is
fair. Long-context fills it with a prose window; RAG fills it with k retrieved
chunks where k is BUDGET-MATCHED (k * avg_chunk_tokens ~= budget), NOT a guessed
7; the graph method deliberately reads a small structural neighborhood — using
far fewer tokens than the budget is exactly its advantage, so it is not padded.
"""

from __future__ import annotations

from dataclasses import dataclass

# PDNC prose Chapter chunks are ~206 tokens each (verified).
AVG_CHUNK_TOKENS = 200
# Equal information budget every method receives.
TOKEN_BUDGET = 4000
# Single named per-token price for cost accounting (USD/token). Placeholder near
# Claude Sonnet input pricing (~$3 / 1M tokens); override via BaselineConfig.
PRICE_PER_TOKEN = 3.0e-6


def count_tokens(text: str) -> int:
    """Token estimate, reusing retrieval.py's len // 4 heuristic for consistency."""
    return len(text) // 4


@dataclass
class BaselineConfig:
    token_budget: int = TOKEN_BUDGET
    avg_chunk_tokens: int = AVG_CHUNK_TOKENS
    price_per_token: float = PRICE_PER_TOKEN
    # Graph structural neighborhood half-width (quotes each side). Small on
    # purpose: the graph reads names+order, not prose, so it stays well under
    # the budget — that token efficiency is the result we are measuring.
    graph_window: int = 8

    @property
    def rag_k(self) -> int:
        """Budget-matched retrieval depth: k * avg_chunk_tokens ~= token_budget."""
        return max(1, round(self.token_budget / self.avg_chunk_tokens))

    @property
    def prose_chapters_each_side(self) -> int:
        """Chapters per side so the window ~= token_budget (chunks ~avg_chunk)."""
        return max(1, round(self.token_budget / self.avg_chunk_tokens / 2))

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens + output_tokens) * self.price_per_token
