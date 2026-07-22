"""Shared training utilities that do not commit to a tracking provider."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class TrainingRun:
    name: str
    output_dir: Path
    seed: int = 42

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch
        except ImportError:
            return
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
