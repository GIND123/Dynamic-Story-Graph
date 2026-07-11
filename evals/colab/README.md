# Vanilla-LLM failure profile — Colab workflow

Profile ~8 open-source LLMs on narrative-coherence tasks, **one model at a time**
(no automated loop). The codebase runs locally; each model runs on **Colab Pro**
via Ollama + a cloudflared tunnel. Only LLM calls cross the network.

## What gets measured

Per model, three probe tasks roll up into three failure rates:

| Task | Metric | Feeds rate | Needs Neo4j? |
|---|---|---|---|
| Quote Attribution (PDNC) | speaker accuracy | — | yes |
| Name Cloze (GPT4-Books) | cloze accuracy | entity drift (fallback proxy) | no |
| Coreference (LitBank) | pairwise identity accuracy | **entity drift (primary)** | no |
| Consistency (cases.json) | detection recall | contradiction + location | no |

`contradiction` / `location_inconsistency` = 1 − recall on planted cases;
`entity_drift` = coreference error rate (1 − accuracy, capturing identity splits
and merges), falling back to the cloze wrong-but-valid-character proxy if coref
didn't run. The `entity_drift_source` is recorded in each profile.

## One-time local setup

```bash
# .env
LLM_MODE=ollama
OLLAMA_BASE_URL=<paste per session>

# Only quote-attribution needs the graph; the other two run without it.
make up
make ingest ARGS="pdnc PrideAndPrejudice"
```

## Per model (the loop-free routine)

1. Open [`model_server.ipynb`](model_server.ipynb) in Colab, pick a GPU runtime,
   run cells 1–4. Set `MODEL` in cell 2 to the model you're testing.
2. Copy the printed `https://….trycloudflare.com` URL into `.env` as
   `OLLAMA_BASE_URL` (the URL changes every Colab session).
3. On your laptop:
   ```bash
   python evals/run_model_profile.py --model qwen2.5:7b
   ```
   Writes `evals/results/profiles/qwen2_5_7b.json`. Re-running the same `--model`
   resumes (finished tasks are cached; `--force` recomputes). A dropped tunnel
   never loses a completed task.
4. Repeat for the next model (new Colab session → new URL → new `--model`).

Then combine everything:

```bash
python evals/aggregate_profiles.py   # -> evals/results/failure_profile.md + .csv
```

## Ollama tags (verify before pulling)

`qwen2.5:7b`, `qwen2.5:72b`, `llama3.1:8b`, `phi3.5`, `mistral:7b`, `yi:9b`,
`gemma2:9b`. Confirm each exists / the instruct-chat variant on ollama.com/library
first — paper names differ from tags.

## Notes / caveats

- **72B needs an A100** (T4/L4 too small) and runs slow with CPU offload even
  there; A100 access on Pro isn't guaranteed and burns compute units. The seven
  ≤9B models work on any Colab GPU.
- Fairness: a fixed `--seed` + fixed quote/cloze prefixes mean every model is
  scored on identical items.
- Robustness: refusals / invalid JSON / dropped calls are counted (malformed /
  abstain), never crash a run — that failure is part of the profile.
- Small n (per-class consistency, 50 cloze/book by default) → rates are
  directional. Widen with `--max-cloze` and `--books`.
