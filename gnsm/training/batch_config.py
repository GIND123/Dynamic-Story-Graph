"""Batch-shape config shared by every dataset adapter (evolvtrip_adapter.py,
pdnc_adapter.py, ...). Genuinely dataset-agnostic: it only describes tensor
shapes the gnsm.state.neural modules expect, not what the values mean.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchConfig:
    nodes: int = 8
    edges_per_graph: int = 8
    input_dim: int = 64
    hidden_dim: int = 128
