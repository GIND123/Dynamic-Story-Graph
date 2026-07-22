# GNSM — Grounded Narrative State Model

GNSM is a runnable research scaffold for consistent long-form story generation.
It anchors a continuous narrative state to an interpretable scene graph, rolls
that state forward using supervised graph deltas, conditions a frozen LLM, and
checks every draft before accepting it.

## Quick start

Python 3.11+ is required.

```bash
python -m pip install -e ".[dev]"
python -m gnsm demo
python -m pytest
```

The demo uses deterministic reference components and downloads nothing. Neural
experiments use the optional training stack:

```bash
python -m pip install -e ".[training]"
```

## Three planes

1. `extraction/` turns text into canonical entities, coreference mentions,
   dialogue edges, relations, and world-state attributes.
2. `state/` produces node embeddings and global `z_t`, reconstructs grounded
   graph labels, and predicts the supervised next-scene delta.
3. `generation/` conditions a frozen generator through a soft-prefix or
   cross-attention adapter; `verifier/` checks and controls acceptance.

The default `GNSMSystem.reference()` wires conservative rule-based extractors,
a graph-aware feature-hashing encoder, rule transition model, template frozen
generator, and deterministic verifier. Replace one interface at a time as model
weights and normalized datasets become available.

See [architecture](docs/architecture.md), [dataset roles](data/README.md), and
the [decision log](DECISIONS.md).

## Build phases

- P0: create cached state vectors and labels, then run
  `python -m gnsm.training.stage0_probe features.npy labels.npy`.
- P1: implement/version the Maverick or BookNLP, PDNC, and structured-relation
  adapters behind `gnsm.extraction.base`.
- P2: train `GraphStateEncoder`, grounded decode heads, and
  `NeuralTransitionModel` with normalized graph/delta batches.
- P3: run the pre-registered kinetic/introspective human study.
- P4: execute model and context scale sweeps.

`gnsm/configs/ablations` contains the initial experiment matrix. Raw datasets,
checkpoints, and artifacts are ignored by Git.
