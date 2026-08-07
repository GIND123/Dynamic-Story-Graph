"""BookCoref loader — Long-range character coreference corpus.

Repo: huggingface.co/datasets/sapienzanlp/bookcoref
Format: JSONL files (train/validation/test) with pre-tokenized sentences and character clusters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ingestion.base import DatasetIR, DatasetLoader, Entity, chunk_text

logger = logging.getLogger(__name__)


class BookCorefLoader(DatasetLoader):
    """Load one BookCoref document into the normalized IR."""

    def __init__(self, jsonl_path: str | Path, doc_key: str) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.doc_key = doc_key
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"BookCoref file not found: {self.jsonl_path}")

    def namespace(self) -> str:
        return f"bookcoref:{self.doc_key}"

    def load(self) -> DatasetIR:
        doc = None
        with self.jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("doc_key") == self.doc_key:
                    doc = row
                    break

        if doc is None:
            raise ValueError(f"doc_key {self.doc_key} not found in {self.jsonl_path}")

        entities: list[Entity] = []
        for char_info in doc.get("characters", []):
            name = char_info.get("name")
            if name:
                entities.append(Entity(name=name, type="CHARACTER"))

        # Reconstruct text from tokenized sentences.
        # This is a naive detokenization, but sufficient for creating segments.
        sentences = doc.get("sentences", [])
        paragraphs = []
        current_para = []
        for sent in sentences:
            sent_text = " ".join(sent).replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            current_para.append(sent_text)
            if len(current_para) >= 5:  # Arbitrary paragraph grouping
                paragraphs.append(" ".join(current_para))
                current_para = []
        if current_para:
            paragraphs.append(" ".join(current_para))

        full_text = "\n\n".join(paragraphs)
        segments = chunk_text(full_text)

        logger.info(
            "BookCoref: loaded doc %s with %d entities and %d segments",
            self.doc_key, len(entities), len(segments)
        )

        return DatasetIR(
            entities=entities,
            events=[],
            quotations=[],
            segments=segments,
        )
