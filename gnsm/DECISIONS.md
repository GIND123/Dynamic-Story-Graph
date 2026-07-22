# Decision log

- Use a self-contained top-level `gnsm` Python package so the existing
  `manuscript-memory-engine` remains an untouched baseline.
- Keep the default path dependency-light and deterministic; heavy neural
  dependencies are optional so extraction, verification, and CI run without a
  GPU or model download.
- Put all cross-plane data in explicit dataclasses rather than leaking model or
  framework objects across boundaries.
- Include feature hashing as an executable Stage-0 baseline, while keeping
  ModernBERT/Graph Transformer modules as the primary learned architecture.
- Make graph reconstruction and supervised next-delta classification explicit
  losses; latent regression remains secondary and stop-gradient grounded.
- Reuse validated metrics from the existing baseline through adapters instead
  of copying them and allowing implementations to drift.
- Store local datasets outside Git and track only manifests and provenance.
