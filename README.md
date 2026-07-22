# Dynamic Narration Graph

Research workspace for long-context consistent story generation.

## Projects

- [`gnsm/`](gnsm/) — Grounded Narrative State Model: symbolic extraction,
  learned narrative state, supervised transitions, conditioned generation, and
  deterministic consistency verification.
- [`manuscript-memory-engine/`](manuscript-memory-engine/) — the existing graph
  memory baseline and evaluation harness.
- [`Dataset Generation/`](Dataset%20Generation/) and
  [`Long Model Comparison/`](Long%20Model%20Comparison/) — research notebooks.

Start with `python -m gnsm demo` for a dependency-light end-to-end GNSM run.
See [`gnsm/README.md`](gnsm/README.md) for setup, architecture, and extension
points.

## Run on Colab (clone → paste → run)

Open [`gnsm/colab/GNSM_Colab.ipynb`](gnsm/colab/GNSM_Colab.ipynb) in Colab, or
paste this into a fresh **GPU** Colab cell (Runtime → Change runtime type → GPU):

```python
!git clone https://github.com/GIND123/Dynamic-Narration-Graph.git
%cd Dynamic-Narration-Graph
!python -m gnsm.colab.bootstrap   # keeps Colab's CUDA torch, installs the rest, prints a report
!python -m gnsm demo              # deterministic end-to-end loop (no downloads)
!python -m gnsm smoke --json      # trains the neural state stack on the GPU
```

CUDA target is **cu121** (compatible with Colab and CUDA 12.1–12.4 hosts). Full
notes: [`gnsm/colab/README.md`](gnsm/colab/README.md).
