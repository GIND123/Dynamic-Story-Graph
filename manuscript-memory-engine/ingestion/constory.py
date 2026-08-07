"""ConStory-Bench loader — Consistency evaluation benchmark.

Repo: huggingface.co/datasets/jayden8888/ConStory-Bench
Format: Parquet file (prompts.parquet) with evaluation prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from ingestion.base import DatasetIR, DatasetLoader, chunk_text

logger = logging.getLogger(__name__)


class ConStoryLoader(DatasetLoader):
    """Load one ConStory-Bench prompt into the normalized IR."""

    def __init__(self, parquet_path: str | Path, prompt_id: int | str) -> None:
        self.parquet_path = Path(parquet_path)
        self.prompt_id = int(prompt_id) if str(prompt_id).isdigit() else prompt_id
        if not self.parquet_path.is_file():
            raise FileNotFoundError(f"ConStory file not found: {self.parquet_path}")

    def namespace(self) -> str:
        return f"constory:{self.prompt_id}"

    def load(self) -> DatasetIR:
        df = pd.read_parquet(self.parquet_path)
        row = df[df["id"] == self.prompt_id]
        
        if len(row) == 0:
            raise ValueError(f"prompt_id {self.prompt_id} not found in {self.parquet_path}")
        
        prompt_text = str(row.iloc[0]["prompt"])
        segments = chunk_text(prompt_text)

        logger.info(
            "ConStory: loaded prompt %s with %d segments",
            self.prompt_id, len(segments)
        )

        return DatasetIR(
            entities=[],
            events=[],
            quotations=[],
            segments=segments,
        )
