# Data layout

Downloaded corpora are intentionally excluded from Git. Dataset adapters should
normalize every source into scene-level JSONL with text, canonical entities,
typed edges, attributes, speaker/addressee links, and next-scene deltas.

```text
raw/          original licensed/downloaded corpora
processed/    normalized scene graphs and train/dev/test splits
cache/        offline text/node encodings
manifests/    roles, expected fields, and provenance notes
```

Never mix evaluation-only stories into generator or adapter training.
