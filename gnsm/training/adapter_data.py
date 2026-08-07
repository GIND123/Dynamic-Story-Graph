"""Turn EvolvTrip examples into (node_features, target_text) pairs for Stage
C adapter training.

Reuses ``evolvtrip_adapter.load_examples``/``collate_batch`` (already real,
already verified against the real EvolvTrip corpus) rather than writing any
new dataset code. The only new idea here is *which field is the training
target*: ``EvolvTripExample.action_text`` is the next scene's
scenario/plot_summary -- "what happens next" -- exactly the continuation
StatePrefixAdapter is trained to help predict. The edge/attribute/delta/
emotion label tensors `collate_batch` also produces are for the state
encoder's own training (`train_state.py`) and are unused here; Stage C only
needs the encoder's `global_state` output, not its auxiliary heads.
"""

from __future__ import annotations

from dataclasses import dataclass

from gnsm.training.batch_config import BatchConfig
from gnsm.training.evolvtrip_adapter import EvolvTripExample, collate_batch


@dataclass(frozen=True, slots=True)
class AdapterBatch:
    node_features: object  # torch.Tensor, shape (batch, nodes, input_dim)
    target_texts: tuple[str, ...]


def collate_for_adapter(
    examples: list[EvolvTripExample], config: BatchConfig, seed: int = 0
) -> AdapterBatch:
    batch = collate_batch(examples, config, seed)
    target_texts = tuple(example.action_text for example in examples)
    return AdapterBatch(node_features=batch["node_features"], target_texts=target_texts)
