"""Shuffle/split helpers shared by every trainer (train_state.py,
train_adapter.py, ...). Deliberately dataset-agnostic -- operates on any
list of examples.
"""

from __future__ import annotations

import random
from typing import Any


def train_val_split(
    examples: list[Any], val_fraction: float, seed: int
) -> tuple[list[Any], list[Any]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def shuffled_batches(examples: list[Any], batch_size: int, seed: int) -> list[list[Any]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]
