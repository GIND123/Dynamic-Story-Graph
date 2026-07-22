# GNSM architecture

```text
scene text ──> extraction ──> G_t ──> state encoder ──> z_t + node states
                    ^                    │
                    │                    v
accepted scene <── verifier <── draft <── generator adapter
                    ^                    ^
                    └── predicted ΔG <── transition(z_t, plot action)
```

Hard facts remain symbolic and enforceable. Graded internal state remains in
the continuous representation. The primary state objective reconstructs real
graph labels, and the transition target is the real next-scene graph delta—not
another learned embedding.

## Contract boundaries

All planes exchange types from `gnsm.schemas`. Model-specific objects stop at
their adapters. This lets the project substitute BookNLP for Maverick, Llama
for Qwen, or soft prompts for cross-attention without changing the controller.

## Acceptance loop

1. Encode the accepted `G_t`.
2. Predict `ΔG_(t+1)` from `z_t` and the outline action.
3. Draft with the frozen generator and trainable state adapter.
4. Extract realized `G_(t+1)` from the draft.
5. Compare realized state with current canon and the predicted delta.
6. Accept or regenerate with precise corrective constraints.
