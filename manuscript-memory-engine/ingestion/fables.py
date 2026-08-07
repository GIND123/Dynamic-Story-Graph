"""FABLES loader — Faithfulness evaluation dataset.

Repo: github.com/mungg/FABLES
Format: JSON file containing faithfulness claims across books and summarizers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ingestion.base import DatasetIR, DatasetLoader, Event, chunk_text

logger = logging.getLogger(__name__)


class FablesLoader(DatasetLoader):
    """Load one FABLES summary and its claims into the normalized IR."""

    def __init__(self, json_path: str | Path, book_name: str, summarizer_name: str) -> None:
        self.json_path = Path(json_path)
        self.book_name = book_name
        self.summarizer_name = summarizer_name
        if not self.json_path.is_file():
            raise FileNotFoundError(f"FABLES file not found: {self.json_path}")

    def namespace(self) -> str:
        return f"fables:{self.book_name}:{self.summarizer_name}"

    def load(self) -> DatasetIR:
        with self.json_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        
        fab = raw.get("FABLES", {})
        book_data = fab.get(self.book_name)
        if not book_data:
            raise ValueError(f"book {self.book_name} not found in {self.json_path}")
            
        summary_data = book_data.get(self.summarizer_name)
        if not summary_data:
            raise ValueError(f"summarizer {self.summarizer_name} not found for book {self.book_name}")
            
        summary_text = summary_data.get("summary", "")
        segments = chunk_text(summary_text)

        events: list[Event] = []
        claims = summary_data.get("claims") or {}
        
        # FABLES claims are a dict keyed by string indices ("0", "1", ...)
        # Parse them as Events where the claim text is the summary.
        for cid_str, claim_data in claims.items():
            if isinstance(claim_data, dict):
                claim_text = claim_data.get("claim", "")
                events.append(Event(
                    summary=claim_text,
                    participant_names=[],
                    order=int(cid_str) if cid_str.isdigit() else 0
                ))

        logger.info(
            "FABLES: loaded book %s (summarizer: %s) with %d claims and %d segments",
            self.book_name, self.summarizer_name, len(events), len(segments)
        )

        return DatasetIR(
            entities=[],
            events=events,
            quotations=[],
            segments=segments,
        )
