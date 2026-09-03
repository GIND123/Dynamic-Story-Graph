"""Local Apple-Silicon inference via MLX. No API, no network at run time."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _load(model_id: str):
    from mlx_lm import load

    return load(model_id)


class MLXBackend:
    def __init__(self, model_id: str = "mlx-community/Qwen2.5-3B-Instruct-4bit") -> None:
        self.model_id = model_id
        self.name = model_id
        self._model = None
        self._tokenizer = None

    def _ensure(self) -> None:
        if self._model is None:
            self._model, self._tokenizer = _load(self.model_id)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        self._ensure()
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        sampler = make_sampler(temp=temperature)
        return mlx_generate(
            self._model, self._tokenizer, prompt=text,
            max_tokens=max_tokens, sampler=sampler, verbose=False,
        )
