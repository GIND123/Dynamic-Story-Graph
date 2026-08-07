"""EvolvTrip / LitCharToM loader — Temporal ToM and narrative state corpus.

Repo: huggingface.co/datasets/yangbh217/EvolvTrip
Format: JSON array of records mapping characters and plot indices to state triples.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ingestion.base import DatasetIR, DatasetLoader, Entity, Event, chunk_text

logger = logging.getLogger(__name__)

# Longest prefix matches first
_BASE_RELATIONS = [
    "FeelsTowards", "FeelsAbout", "BelievesAbout", "BelievesThat", "BelievesIn",
    "DesiresToKnow", "DesiresTo", "IntendsTo", "Intends",
    "Feels", "Believes", "Desires"
]

class EvolvTripLoader(DatasetLoader):
    """Load one EvolvTrip book into the normalized IR."""

    def __init__(self, json_path: str | Path, book_name: str) -> None:
        self.json_path = Path(json_path)
        self.book_name = book_name
        if not self.json_path.is_file():
            raise FileNotFoundError(f"EvolvTrip file not found: {self.json_path}")

    def namespace(self) -> str:
        return f"evolvtrip:{self.book_name}"

    def load(self) -> DatasetIR:
        with self.json_path.open(encoding="utf-8") as fh:
            records = json.load(fh)
        
        book_records = [r for r in records if r.get("book_name") == self.book_name]
        if not book_records:
            raise ValueError(f"book_name {self.book_name} not found in {self.json_path}")

        entities_dict: dict[str, Entity] = {}
        events: list[Event] = []
        plot_summaries: list[str] = []

        # Sort records by plot_index
        book_records.sort(key=lambda r: int(r.get("plot_index", 0)))

        for rec in book_records:
            target_char = rec.get("character")
            if target_char and target_char not in entities_dict:
                entities_dict[target_char] = Entity(name=target_char, type="CHARACTER")

            # Collect segments from plot_summary
            plot_idx = int(rec.get("plot_index", 0))
            summary = rec.get("plot_summary")
            if summary and summary not in plot_summaries:
                plot_summaries.append(summary)
            
            # Parse Triples
            triples_data = rec.get("triples") or {}
            for target, strings in triples_data.items():
                for s in strings:
                    s = s.strip()
                    if s.startswith("(") and s.endswith(")"):
                        parts = s[1:-1].split(",", 2)
                        if len(parts) >= 3:
                            rel = parts[1].strip()
                            val = parts[2].strip()
                            
                            # Clean the relation
                            base_rel = rel
                            obj = None
                            for base in _BASE_RELATIONS:
                                if rel.startswith(base):
                                    base_rel = base
                                    rem = rel[len(base):].strip()
                                    if rem:
                                        obj = rem
                                    break
                            
                            participants = [target_char]
                            if obj:
                                participants.append(obj)
                                if obj not in entities_dict:
                                    entities_dict[obj] = Entity(name=obj, type="CHARACTER")
                            
                            events.append(Event(
                                summary=f"{base_rel}({target_char}{', ' + obj if obj else ''}) -> {val}",
                                participant_names=participants,
                                order=plot_idx
                            ))

        full_text = "\n\n".join(plot_summaries)
        segments = chunk_text(full_text)

        entities = list(entities_dict.values())
        logger.info(
            "EvolvTrip: loaded book %s with %d entities, %d events, %d segments",
            self.book_name, len(entities), len(events), len(segments)
        )

        return DatasetIR(
            entities=entities,
            events=events,
            quotations=[],
            segments=segments,
        )
