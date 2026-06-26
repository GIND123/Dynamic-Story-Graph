"""Injectable LLM-ask adapter: `(prompt, meta) -> answer_text`.

Anthropic mode mirrors AnthropicLLM's client construction (without modifying
llm.py) and does a short messages.create for the question. Ollama mode calls the
local server for a free, no-key comparison. Any other mode returns None so tests
inject their own deterministic ask and the runner stays runnable with no key.
The signature passes only non-gold `meta` (quote_id/quote_type/task).
"""

from __future__ import annotations

from baselines.base import LLMAsk


def make_llm_ask(max_tokens: int = 64) -> LLMAsk | None:
    """Return a live ask for the configured LLM_MODE, or None when none applies.

    anthropic -> Anthropic messages.create; ollama -> local Ollama /api/chat;
    anything else (fake/unset) -> None so the baselines abstain gracefully.
    """
    from app.config import settings

    if settings.llm_mode == "anthropic" and settings.anthropic_api_key:
        return _anthropic_ask(max_tokens)
    if settings.llm_mode == "ollama":
        return _ollama_ask(max_tokens)
    return None


def _anthropic_ask(max_tokens: int) -> LLMAsk:
    from app.config import settings

    import anthropic

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key, max_retries=4, timeout=120
    )
    model = settings.generation_model

    def ask(prompt: str, meta: dict) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    return ask


def _ollama_ask(max_tokens: int) -> LLMAsk:
    """Free local backend, matching OllamaLLM's config (base_url + model).

    A short, deterministic (temperature 0) completion. `num_predict` caps output
    so the baselines' answers stay terse and token accounting mirrors Anthropic's
    `max_tokens`. English is pinned for the same code-switch reason as OllamaLLM.
    """
    from app.config import settings

    import httpx

    client = httpx.Client(base_url=settings.ollama_base_url, timeout=180)
    model = settings.ollama_model

    def ask(prompt: str, meta: dict) -> str:
        resp = client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Answer in English. Be terse."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    return ask
