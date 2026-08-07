"""Turn PDNC (quote/speaker/addressee) novels into the tensors
gnsm.state.neural expects. The PDNC counterpart of evolvtrip_adapter.py --
see that module's docstring for the general design rationale (feature
hashing, fine/coarse label split, delta-as-"what changed").

Reuses ``ingestion.pdnc.PDNCLoader`` from manuscript-memory-engine (stdlib-only
dependencies -- csv/ast/pathlib -- so it's safe to import standalone without
that project's Neo4j/FastAPI stack) rather than re-parsing PDNC's CSVs here,
per gnsm/DECISIONS.md's "reuse validated loaders instead of copying and
letting implementations drift."

Per-record graph construction (v1, not final):

- One training example per pair of *consecutive quotes by the same speaker*
  within a novel (reading order = the loader's quotation list order).
- node 0 is the speaker; nodes 1..K are the current quote's addressees
  (hashed name), padded to ``nodes``. A quote with zero addressees
  (monologue/exclamation) gets a single self-loop edge on node 0.
- one edge per addressee, speaker -> addressee, labelled with the quote type
  (``edge_types`` = explicit/implicit/anaphoric/pad) and, on the same row,
  whether the quote is directed at all (``attribute_classes`` =
  has_addressee/no_addressee) -- the fine/coarse split mirrors EvolvTrip's
  relation/dimension pair.
- no emotion signal in PDNC -- omitted from the batch entirely (the shared
  trainer treats it as optional).
- ``delta_labels``: which quote type the *next* quote by this speaker became,
  or "no_change" if the type didn't change -- DELTA_VOCAB =
  explicit/implicit/anaphoric/no_change.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnsm.training.batch_config import BatchConfig

QUOTE_TYPES = ["explicit", "implicit", "anaphoric"]
_PAD_TYPE = "pad"
EDGE_TYPE_VOCAB = [*QUOTE_TYPES, _PAD_TYPE]
EDGE_TYPES = len(EDGE_TYPE_VOCAB)  # 4

ATTRIBUTE_VOCAB = ["no_addressee", "has_addressee", "pad"]
ATTRIBUTE_CLASSES = len(ATTRIBUTE_VOCAB)  # 3

DELTA_VOCAB = [*QUOTE_TYPES, "no_change"]
DELTA_CLASSES = len(DELTA_VOCAB)  # 4

EMOTION_CLASSES = 2  # unused -- PDNC has no emotion signal; kept for head construction only


def _ingestion_path() -> Path:
    return Path(__file__).resolve().parents[2] / "manuscript-memory-engine"


def _pdnc_loader_cls():
    """Import ingestion.pdnc.PDNCLoader, adding manuscript-memory-engine to
    sys.path first if it isn't already importable (same pattern the top-level
    README's Colab Step 6 uses)."""

    try:
        from ingestion.pdnc import PDNCLoader
    except ImportError:
        sys.path.insert(0, str(_ingestion_path()))
        from ingestion.pdnc import PDNCLoader
    return PDNCLoader


def _hash_text(text: str, dim: int, seed: int = 0) -> np.ndarray:
    """Same feature-hashing trick as evolvtrip_adapter._hash_text."""

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
class PDNCExample:
    novel: str
    speaker: str
    addressee_names: tuple[str, ...]
    quote_type: str
    next_addressee_names: tuple[str, ...]
    next_text: str
    delta_label: int


def load_examples(pdnc_data_root: Path, novels: list[str] | None = None) -> list[PDNCExample]:
    """Pair consecutive same-speaker quotes across every novel folder under
    ``pdnc_data_root`` (or just ``novels`` if given)."""

    loader_cls = _pdnc_loader_cls()
    pdnc_data_root = Path(pdnc_data_root)
    novel_names = novels or sorted(p.name for p in pdnc_data_root.iterdir() if p.is_dir())

    examples: list[PDNCExample] = []
    for novel in novel_names:
        ir = loader_cls(pdnc_data_root / novel).load()
        by_speaker: dict[str, list] = defaultdict(list)
        for quote in ir.quotations:
            by_speaker[quote.speaker_name].append(quote)

        for quotes in by_speaker.values():
            for current, following in zip(quotes, quotes[1:], strict=False):
                if following.quote_type != current.quote_type:
                    delta_label = DELTA_VOCAB.index(following.quote_type)
                else:
                    delta_label = len(DELTA_VOCAB) - 1  # no_change
                examples.append(
                    PDNCExample(
                        novel=novel,
                        speaker=current.speaker_name,
                        addressee_names=tuple(current.addressee_names),
                        quote_type=current.quote_type,
                        next_addressee_names=tuple(following.addressee_names),
                        next_text=following.text,
                        delta_label=delta_label,
                    )
                )
    return examples


def collate_batch(examples: list[PDNCExample], config: BatchConfig, seed: int = 0) -> dict:
    import torch

    max_addressees = config.nodes - 1
    max_edges = config.edges_per_graph
    pad_edge_type = EDGE_TYPE_VOCAB.index(_PAD_TYPE)
    pad_attribute = ATTRIBUTE_VOCAB.index("pad")

    node_features = np.zeros((len(examples), config.nodes, config.input_dim), dtype=np.float32)
    next_node_features = np.zeros_like(node_features)
    action_features = np.zeros((len(examples), config.hidden_dim), dtype=np.float32)
    edge_pairs = np.zeros((len(examples), max_edges, 2), dtype=np.int64)
    edge_labels = np.full((len(examples), max_edges), pad_edge_type, dtype=np.int64)
    attribute_labels = np.full((len(examples), max_edges), pad_attribute, dtype=np.int64)
    delta_labels = np.zeros(len(examples), dtype=np.int64)

    for row, example in enumerate(examples):
        node_features[row, 0] = _hash_text(
            f"{example.novel} {example.speaker}", config.input_dim, seed
        )
        for col, name in enumerate(example.addressee_names[:max_addressees], start=1):
            node_features[row, col] = _hash_text(name, config.input_dim, seed)

        next_node_features[row, 0] = _hash_text(
            f"{example.novel} {example.speaker}", config.input_dim, seed
        )
        for col, name in enumerate(example.next_addressee_names[:max_addressees], start=1):
            next_node_features[row, col] = _hash_text(name, config.input_dim, seed)

        action_features[row] = _hash_text(example.next_text, config.hidden_dim, seed)

        edge_type_idx = QUOTE_TYPES.index(example.quote_type)
        attribute_idx = ATTRIBUTE_VOCAB.index(
            "has_addressee" if example.addressee_names else "no_addressee"
        )
        n_addressees = min(len(example.addressee_names), max_addressees)
        if n_addressees == 0:
            # Monologue/exclamation: a self-loop still carries the quote type.
            edge_pairs[row, 0] = (0, 0)
            edge_labels[row, 0] = edge_type_idx
            attribute_labels[row, 0] = attribute_idx
            for e in range(1, max_edges):
                edge_pairs[row, e] = (0, 0)
        else:
            for e in range(n_addressees):
                edge_pairs[row, e] = (0, e + 1)
                edge_labels[row, e] = edge_type_idx
                attribute_labels[row, e] = attribute_idx
            for e in range(n_addressees, max_edges):
                edge_pairs[row, e] = (0, 0)

        delta_labels[row] = example.delta_label

    return {
        "node_features": torch.from_numpy(node_features),
        "next_node_features": torch.from_numpy(next_node_features),
        "action_features": torch.from_numpy(action_features),
        "edge_pairs": torch.from_numpy(edge_pairs),
        "edge_labels": torch.from_numpy(edge_labels).reshape(-1),
        "attribute_labels": torch.from_numpy(attribute_labels).reshape(-1),
        "delta_labels": torch.from_numpy(delta_labels),
        # no "emotion_labels" key -- PDNC has no emotion signal; the shared
        # trainer (gnsm.training.train_state.run) treats this as optional.
    }
