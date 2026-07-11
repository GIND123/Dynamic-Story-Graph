"""Injectable LLM-ask adapter: `(prompt, meta) -> answer_text`.

Anthropic mode mirrors AnthropicLLM's client construction (without modifying
llm.py) and does a short messages.create for the question. Ollama mode calls the
local server for a free, no-key comparison. Any other mode returns None so tests
inject their own deterministic ask and the runner stays runnable with no key.
The signature passes only non-gold `meta` (quote_id/quote_type/task).
"""

from __future__ import annotations

from baselines.base import LLMAsk


def make_llm_ask(max_tokens: int = 64, num_ctx: int | None = None) -> LLMAsk | None:
    """Return a live ask for the configured LLM_MODE, or None when none applies.

    anthropic -> Anthropic messages.create; ollama -> local Ollama /api/chat;
    anything else (fake/unset) -> None so the baselines abstain gracefully.

    `num_ctx` (ollama only) sets how many tokens the model actually keeps; without
    it Ollama silently truncates to ~2048. The ollama ask also writes the real
    prompt-token count into `meta["prompt_tokens"]` for context-length accounting.
    """
    from app.config import settings

    if settings.llm_mode == "anthropic" and settings.anthropic_api_key:
        return _anthropic_ask(max_tokens)
    if settings.llm_mode == "ollama":
        return _ollama_ask(max_tokens, num_ctx)
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


def _ollama_ask(max_tokens: int, num_ctx: int | None = None) -> LLMAsk:
    """Free local backend, matching OllamaLLM's config (base_url + model).

    A short, deterministic (temperature 0) completion. `num_predict` caps output
    so the baselines' answers stay terse and token accounting mirrors Anthropic's
    `max_tokens`. `num_ctx`, when set, controls how many input tokens the model
    keeps (default ~2048 truncates). English is pinned for the same code-switch
    reason as OllamaLLM. The real prompt-token count is written to
    `meta["prompt_tokens"]` so callers can plot the true context length.
    """
    from app.config import settings

    import httpx

    client = httpx.Client(base_url=settings.ollama_base_url, timeout=600)
    model = settings.ollama_model
    options: dict = {"temperature": 0.0, "num_predict": max_tokens}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

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
                "options": options,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        meta["prompt_tokens"] = body.get("prompt_eval_count", 0)
        return body["message"]["content"]

    return ask
