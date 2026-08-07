"""Turn raw EvolvTrip records into the tensors gnsm.state.neural expects.

EvolvTrip has no explicit multi-entity relation graph like LitBank/BookCoref;
each record is one (book, character, plot_index) "timestep" holding a free-text
`triples` dict of the form ``(subject, Relation, object)`` — e.g.
``"(King Lear, BelievesAboutCordelia, Cordelia's profession of love will
outshine her sisters')"``. The parsing regex and relation-canonicalization
vocabulary below are ported from ``DNG_Data_Visualization.ipynb`` (Blocks
14-15), which validated them against the full corpus (1,169 raw relation
strings collapse to ~12 base relations once the target-character name suffix
is stripped).

Per-record graph construction (a deliberate v1, not a final design):

- node 0 is the character; nodes 1..K are the record's distinct triple
  *objects* (hashed text), padded to ``nodes`` with zero vectors.
- one edge per triple, character -> object node, labelled with the base
  relation (``edge_types`` classes, feature-hashed text is never used for the
  label itself) and, on the same row, the coarser ToM dimension
  (``attribute_classes`` = Belief/Desire/Intention/Emotion/other).
- ``emotion_classes`` = 2 (does this scene contain a Feels/emotion triple).
- a training example pairs consecutive plot points for the same
  (book, character): predicting the record at ``t+1`` from the record at
  ``t`` is the supervised transition; ``delta_labels`` picks whichever ToM
  dimension gained the most new triples between the two, or "no_change".

Node/edge features come from feature hashing (the Stage-0 baseline named in
gnsm/DECISIONS.md), not a real text encoder — no model download required to
prove this pipeline end-to-end. Swapping in ModernBERT embeddings later only
touches ``_hash_text``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnsm.training.batch_config import BatchConfig  # noqa: F401 (re-exported for existing callers)

_TRIPLE_RE = re.compile(r"^\(\s*([^,]+?)\s*,\s*([A-Za-z]+)\s*,\s*(.+?)\s*\)$")

# Longest-prefix-first so e.g. "BelievesAbout" wins over "Believes".
_BASE_RELATIONS = [
    "FeelsTowards",
    "FeelsAbout",
    "BelievesAbout",
    "BelievesThat",
    "BelievesIn",
    "DesiresToKnow",
    "DesiresTo",
    "IntendsTo",
    "Intends",
    "Feels",
    "Believes",
    "Desires",
]
_BASE_RELATIONS.sort(key=len, reverse=True)
_OTHER_RELATION = "other"
RELATION_VOCAB = [*_BASE_RELATIONS, _OTHER_RELATION]
EDGE_TYPES = len(RELATION_VOCAB)  # 13

_DIMENSION_BY_PREFIX = [
    ("Believes", "belief"),
    ("Desires", "desire"),
    ("Intends", "intention"),
    ("Feels", "emotion"),
]
_OTHER_DIMENSION = "other"
DIMENSION_VOCAB = [name for _, name in _DIMENSION_BY_PREFIX] + [_OTHER_DIMENSION]
ATTRIBUTE_CLASSES = len(DIMENSION_VOCAB)  # 5
DELTA_VOCAB = [name for _, name in _DIMENSION_BY_PREFIX] + ["no_change"]
DELTA_CLASSES = len(DELTA_VOCAB)  # 5
EMOTION_CLASSES = 2


def _base_relation(raw_relation: str) -> str:
    for base in _BASE_RELATIONS:
        if raw_relation.startswith(base):
            return base
    return _OTHER_RELATION


def _dimension(base_relation: str) -> str:
    for prefix, name in _DIMENSION_BY_PREFIX:
        if base_relation.startswith(prefix):
            return name
    return _OTHER_DIMENSION


@dataclass(frozen=True, slots=True)
class ParsedTriple:
    subject: str
    base_relation: str
    dimension: str
    obj: str


def parse_triples(record: dict) -> list[ParsedTriple]:
    """Parse and canonicalize every ``(subject, Relation, object)`` string in
    a record's ``triples`` dict (values are lists of raw strings, keyed by an
    unused role label such as "Target Character")."""

    parsed: list[ParsedTriple] = []
    for raw_list in (record.get("triples") or {}).values():
        for raw in raw_list:
            match = _TRIPLE_RE.match(raw.strip())
            if not match:
                continue
            subject, raw_relation, obj = match.groups()
            base = _base_relation(raw_relation)
            parsed.append(ParsedTriple(subject.strip(), base, _dimension(base), obj.strip()))
    return parsed


def _hash_text(text: str, dim: int, seed: int = 0) -> np.ndarray:
    """Feature-hashing trick: hash each token into a signed bucket, then
    L2-normalize. Deterministic, no vocabulary, no model download."""

    vector = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if not tokens:
        return vector
    for token in tokens:
        digest = hash((seed, token))
        bucket = digest % dim
        sign = 1.0 if (digest // dim) % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


@dataclass(frozen=True, slots=True)
class EvolvTripExample:
    book: str
    character: str
    step_from: int
    step_to: int
    object_texts: tuple[str, ...]  # node 1..K at t, aligned with edge_targets
    edge_targets: tuple[int, ...]  # index into object_texts (0-based) per triple
    edge_relations: tuple[int, ...]  # RELATION_VOCAB index per triple
    edge_dimensions: tuple[int, ...]  # DIMENSION_VOCAB index per triple
    has_emotion: bool
    next_object_texts: tuple[str, ...]
    action_text: str  # the t+1 scenario/plot_summary driving the transition
    delta_label: int  # DELTA_VOCAB index


def _record_key(record: dict) -> tuple[str, str]:
    return record["book_name"], record["character"]


def load_examples(path: Path) -> list[EvolvTripExample]:
    """Pair consecutive plot points for the same (book, character) in
    ``all_books_current.json`` into supervised transition examples."""

    records = json.loads(Path(path).read_text())
    by_character: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        by_character.setdefault(_record_key(record), []).append(record)

    examples: list[EvolvTripExample] = []
    for (book, character), recs in by_character.items():
        recs = sorted(recs, key=lambda r: r["plot_index"])
        for current, following in zip(recs, recs[1:], strict=False):
            current_triples = parse_triples(current)
            next_triples = parse_triples(following)

            objects = list(dict.fromkeys(t.obj for t in current_triples))
            object_index = {obj: i for i, obj in enumerate(objects)}
            edge_targets = tuple(object_index[t.obj] for t in current_triples)
            edge_relations = tuple(RELATION_VOCAB.index(t.base_relation) for t in current_triples)
            edge_dimensions = tuple(DIMENSION_VOCAB.index(t.dimension) for t in current_triples)
            has_emotion = any(t.dimension == "emotion" for t in current_triples)

            current_counts = {name: 0 for name in DELTA_VOCAB[:-1]}
            next_counts = {name: 0 for name in DELTA_VOCAB[:-1]}
            for t in current_triples:
                if t.dimension in current_counts:
                    current_counts[t.dimension] += 1
            for t in next_triples:
                if t.dimension in next_counts:
                    next_counts[t.dimension] += 1
            gains = {name: next_counts[name] - current_counts[name] for name in current_counts}
            best_dim = max(gains, key=lambda name: gains[name])
            delta_label = (
                DELTA_VOCAB.index(best_dim) if gains[best_dim] > 0 else len(DELTA_VOCAB) - 1
            )

            examples.append(
                EvolvTripExample(
                    book=book,
                    character=character,
                    step_from=current["plot_index"],
                    step_to=following["plot_index"],
                    object_texts=tuple(objects),
                    edge_targets=edge_targets,
                    edge_relations=edge_relations,
                    edge_dimensions=edge_dimensions,
                    has_emotion=has_emotion,
                    next_object_texts=tuple(dict.fromkeys(t.obj for t in next_triples)),
                    action_text=following.get("scenario") or following.get("plot_summary") or "",
                    delta_label=delta_label,
                )
            )
    return examples


def collate_batch(examples: list[EvolvTripExample], config: BatchConfig, seed: int = 0):
    """Build the exact tensor shapes gnsm.training.smoke.run's forward() uses,
    from real EvolvTrip examples instead of synthetic random data."""

    import torch

    max_objects = config.nodes - 1
    max_edges = config.edges_per_graph
    pad_relation = RELATION_VOCAB.index(_OTHER_RELATION)
    pad_dimension = DIMENSION_VOCAB.index(_OTHER_DIMENSION)

    node_features = np.zeros((len(examples), config.nodes, config.input_dim), dtype=np.float32)
    next_node_features = np.zeros_like(node_features)
    action_features = np.zeros((len(examples), config.hidden_dim), dtype=np.float32)
    edge_pairs = np.zeros((len(examples), max_edges, 2), dtype=np.int64)
    edge_labels = np.full((len(examples), max_edges), pad_relation, dtype=np.int64)
    attribute_labels = np.full((len(examples), max_edges), pad_dimension, dtype=np.int64)
    emotion_labels = np.zeros(len(examples), dtype=np.int64)
    delta_labels = np.zeros(len(examples), dtype=np.int64)

    for row, example in enumerate(examples):
        node_features[row, 0] = _hash_text(
            f"{example.book} {example.character}", config.input_dim, seed
        )
        for col, text in enumerate(example.object_texts[:max_objects], start=1):
            node_features[row, col] = _hash_text(text, config.input_dim, seed)

        next_node_features[row, 0] = _hash_text(
            f"{example.book} {example.character}", config.input_dim, seed
        )
        for col, text in enumerate(example.next_object_texts[:max_objects], start=1):
            next_node_features[row, col] = _hash_text(text, config.input_dim, seed)

        action_features[row] = _hash_text(example.action_text, config.hidden_dim, seed)

        n_edges = min(len(example.edge_targets), max_edges)
        for e in range(n_edges):
            target_node = (
                min(example.edge_targets[e], max_objects - 1) + 1
            )  # +1: node 0 is the character
            edge_pairs[row, e] = (0, target_node)
            edge_labels[row, e] = example.edge_relations[e]
            attribute_labels[row, e] = example.edge_dimensions[e]
        for e in range(n_edges, max_edges):
            edge_pairs[row, e] = (0, 0)  # pad edges self-loop on the character node

        emotion_labels[row] = int(example.has_emotion)
        delta_labels[row] = example.delta_label

    return {
        "node_features": torch.from_numpy(node_features),
        "next_node_features": torch.from_numpy(next_node_features),
        "action_features": torch.from_numpy(action_features),
        "edge_pairs": torch.from_numpy(edge_pairs),
        "edge_labels": torch.from_numpy(edge_labels).reshape(-1),
        "attribute_labels": torch.from_numpy(attribute_labels).reshape(-1),
        "emotion_labels": torch.from_numpy(emotion_labels),
        "delta_labels": torch.from_numpy(delta_labels),
    }
