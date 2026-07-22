"""Generate a context-size matrix for an external model runner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextSweepCase:
    model: str
    context_tokens: int
    seed: int


def build_sweep(
    models: list[str],
    context_sizes: list[int] | None = None,
    seeds: range = range(3),
) -> list[ContextSweepCase]:
    sizes = context_sizes or [2_048, 4_096, 8_192, 16_384, 32_768]
    return [
        ContextSweepCase(model, size, seed) for model in models for size in sizes for seed in seeds
    ]
