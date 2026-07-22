# Evaluation

- `consistency/`: contradiction precision, recall, F1, coreference/entity drift,
  name cloze, and quote attribution adapters.
- `context_sweep/`: the context-length failure-point ladder.
- `human_study/`: pre-registered kinetic/introspective cells with at least 20
  stories per cell.

Reuse score implementations from `manuscript-memory-engine/evals/metrics` via
thin adapters; do not duplicate validated metric code here.
