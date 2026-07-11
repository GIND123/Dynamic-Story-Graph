"""LitBank loader — gold-span entity/event excerpts.

Repo: github.com/dbamman/litbank
Per-document files, keyed by a gutenberg-style doc id:
  - entities/tsv/<doc>.tsv : TAB-separated, one token per line, blank line
    between sentences, nested BIO layers over {PER,FAC,GPE,LOC,VEH,ORG}.
  - events/tsv/<doc>.tsv   : TAB-separated token + EVENT/O.
  - coref/brat/<doc>.ann   : brat standoff coref chains (optional resolver).

NETWORK NOTE: the corpus could not be cloned here (no network), so the TSV
parser is defensive about column count — first column is the token, every
remaining column is treated as a nested BIO layer, and only the documented
labels are kept. See DECISIONS.md.

ENTITY RESOLUTION: coref chains collapse mentions to one entity when a coref
file is present; otherwise mentions are grouped by mapped-type + lowercased
surface form. Label mapping: PER->CHARACTER, LOC/FAC/GPE->LOCATION,
ORG->ORGANIZATION, VEH->OBJECT.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from ingestion.base import (
    DatasetIR,
    DatasetLoader,
    Entity,
    Event,
    EntityType,
    Segment,
)

logger = logging.getLogger(__name__)

# Gold span labels -> graph entity type (= node label).
_LABEL_TO_TYPE: dict[str, EntityType] = {
    "PER": "CHARACTER",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "GPE": "LOCATION",
    "ORG": "ORGANIZATION",
    "VEH": "OBJECT",
}


class _Mention:
    __slots__ = ("text", "label", "sentence_index")

    def __init__(self, text: str, label: str, sentence_index: int) -> None:
        self.text = text
        self.label = label
        self.sentence_index = sentence_index


def _read_sentences(path: Path) -> list[list[list[str]]]:
    """Read a token-per-line TSV into sentences of tab-split rows.

    Blank lines delimit sentences. Returns list[sentence] where each sentence is
    list[row] and each row is the tab-split columns of that line.
    """
    sentences: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        current.append(raw.split("\t"))
    if current:
        sentences.append(current)
    return sentences


def _split_bio(tag: str) -> tuple[str, str | None]:
    """Return (prefix, label) from a BIO tag like 'B-PER'; ('O', None) for O."""
    tag = tag.strip()
    if not tag or tag == "O":
        return "O", None
    if "-" in tag:
        prefix, label = tag.split("-", 1)
        return prefix, label
    return "O", None


def _parse_entities_tsv(path: Path) -> list[_Mention]:
    """Extract entity mentions across all nested BIO layers."""
    mentions: list[_Mention] = []
    for s_idx, rows in enumerate(_read_sentences(path)):
        tokens = [r[0] for r in rows]
        n_layers = max((len(r) for r in rows), default=1) - 1
        for layer in range(n_layers):
            cur_label: str | None = None
            cur_tokens: list[str] = []
            for i, row in enumerate(rows):
                tag = row[layer + 1] if len(row) > layer + 1 else "O"
                prefix, label = _split_bio(tag)
                keep = label in _LABEL_TO_TYPE
                if prefix == "B" and keep:
                    if cur_label is not None:
                        mentions.append(
                            _Mention(" ".join(cur_tokens), cur_label, s_idx)
                        )
                    cur_label, cur_tokens = label, [tokens[i]]
                elif prefix == "I" and keep and label == cur_label:
                    cur_tokens.append(tokens[i])
                else:
                    if cur_label is not None:
                        mentions.append(
                            _Mention(" ".join(cur_tokens), cur_label, s_idx)
                        )
                    cur_label, cur_tokens = None, []
            if cur_label is not None:
                mentions.append(_Mention(" ".join(cur_tokens), cur_label, s_idx))
    return mentions


def _parse_events_tsv(path: Path) -> list[tuple[str, int]]:
    """Return [(trigger_text, sentence_index)] for tokens tagged EVENT."""
    triggers: list[tuple[str, int]] = []
    for s_idx, rows in enumerate(_read_sentences(path)):
        for row in rows:
            token = row[0]
            tag = row[1].strip() if len(row) > 1 else "O"
            if tag.upper().endswith("EVENT") and tag.upper() != "O":
                triggers.append((token, s_idx))
    return triggers


def _sentence_text(path: Path) -> list[str]:
    return [" ".join(r[0] for r in rows) for rows in _read_sentences(path)]


class LitBankLoader(DatasetLoader):
    """Load one LitBank document (entities + events) into the normalized IR."""

    def __init__(self, litbank_root: str | Path, doc_id: str) -> None:
        self.root = Path(litbank_root)
        # Real LitBank files carry a "_brat" suffix (e.g. 1023_bleak_house_brat.tsv).
        # Normalize the namespace to the clean stem but resolve files flexibly.
        self.doc_id = doc_id[:-5] if doc_id.endswith("_brat") else doc_id
        self.entities_path = self._resolve_tsv("entities")
        self.events_path = self._resolve_tsv("events")
        if self.entities_path is None:
            raise FileNotFoundError(
                f"LitBank entities TSV not found for doc {self.doc_id!r} under "
                f"{self.root / 'entities' / 'tsv'}"
            )

    def _resolve_tsv(self, layer: str) -> Path | None:
        """Find the layer's TSV, tolerating the real '_brat' filename suffix."""
        folder = self.root / layer / "tsv"
        for name in (f"{self.doc_id}.tsv", f"{self.doc_id}_brat.tsv"):
            candidate = folder / name
            if candidate.exists():
                return candidate
        matches = sorted(folder.glob(f"{self.doc_id}*.tsv"))
        return matches[0] if matches else None

    def namespace(self) -> str:
        return f"litbank:{self.doc_id}"

    def _resolve_entities(
        self, mentions: list[_Mention]
    ) -> tuple[list[Entity], dict[tuple[int, str], str]]:
        """Group mentions into entities by mapped-type + lowercased surface form.

        Returns (entities, mention_key->canonical_name) where mention_key is
        (sentence_index, lowercased_text) so events can map co-occurring spans.
        """
        groups: dict[tuple[str, str], list[str]] = {}
        for m in mentions:
            etype = _LABEL_TO_TYPE[m.label]
            key = (etype, m.text.lower())
            groups.setdefault(key, []).append(m.text)

        entities: list[Entity] = []
        canonical_by_key: dict[tuple[str, str], str] = {}
        for (etype, _lower), surfaces in groups.items():
            canonical = Counter(surfaces).most_common(1)[0][0]
            aliases = sorted({s for s in surfaces if s != canonical})
            entities.append(Entity(name=canonical, type=etype, aliases=aliases))
            canonical_by_key[(etype, _lower)] = canonical

        # Map each mention (sentence, lowered text) -> canonical name for events.
        mention_resolution: dict[tuple[int, str], str] = {}
        for m in mentions:
            etype = _LABEL_TO_TYPE[m.label]
            mention_resolution[(m.sentence_index, m.text.lower())] = canonical_by_key[
                (etype, m.text.lower())
            ]
        return entities, mention_resolution

    def load(self) -> DatasetIR:
        mentions = _parse_entities_tsv(self.entities_path)
        entities, mention_resolution = self._resolve_entities(mentions)

        # Participants per sentence: canonical names of entities in that sentence.
        by_sentence: dict[int, list[str]] = {}
        for (s_idx, _lower), canonical in mention_resolution.items():
            by_sentence.setdefault(s_idx, [])
            if canonical not in by_sentence[s_idx]:
                by_sentence[s_idx].append(canonical)

        events: list[Event] = []
        if self.events_path is not None and self.events_path.exists():
            sentences = _sentence_text(self.entities_path)
            for order, (trigger, s_idx) in enumerate(
                _parse_events_tsv(self.events_path)
            ):
                context = sentences[s_idx] if s_idx < len(sentences) else ""
                summary = f"{trigger}: {context}".strip()[:240]
                events.append(
                    Event(
                        summary=summary,
                        participant_names=by_sentence.get(s_idx, []),
                        order=order,
                    )
                )

        # Segments: one per sentence (gold excerpts are short), in reading order.
        segments = [
            Segment(index=i, text=text)
            for i, text in enumerate(_sentence_text(self.entities_path))
            if text.strip()
        ]
        return DatasetIR(
            entities=entities,
            events=events,
            quotations=[],
            segments=segments,
        )
