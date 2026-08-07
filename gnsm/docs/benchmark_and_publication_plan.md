# Benchmark & journal-figure plan

Planning document, not implementation — maps GNSM's claim ("consistent
long-form story generation") to the benchmarks, baselines, ablations, and
figures a journal reviewer would expect, and to the code that already exists
versus what still needs to be written. Nothing here runs until real
training exists (see [Loader status](../../README.md#loader-status) and
`gnsm/DECISIONS.md` for current build phase — P0/P1; this plan targets P3/P4).

No figure below is real until `gnsm/training/train_state.py` /
`train_adapter.py` produce an actual trained checkpoint to evaluate — the
Modal + Hugging Face checkpoint harness (`gnsm/infra/modal_app.py`,
`gnsm/training/checkpointing.py`) exists to make that checkpoint durable and
resumable once it does.

## Benchmark matrix

| Capability | Dataset | Metric | Existing code to reuse | Status |
| --- | --- | --- | --- | --- |
| Consistency (contradiction / repetition) | ConStory-Bench — 2,000 prompts, 5 categories / 19 subtypes (CED / GRR) | Category- and subtype-level precision/recall/F1 against the taxonomy | `gnsm/eval/consistency/metrics.py:classification_metrics` (generic P/R/F1, not benchmark-specific yet) | Needs a ConStory-Bench adapter under `gnsm/eval/consistency/` that maps generations → the 19 subtypes; `constory_bench` is `trainable: false` in `gnsm/data/manifests/datasets.yaml` — eval-only, never in training data |
| Faithfulness | FABLES — 26 books × 5 summarizers, claim-level judgements | Claim-level precision/recall vs. gold `general_comment`/`claims` | None in `gnsm/eval/` yet | Needs a new adapter; must preserve the top-level `canary` string per the top-level README's contamination-check rule |
| Memorization guardrail | GPT4-Books — 571 books × 100 name-cloze rows | Cloze accuracy (high accuracy = suspected memorization, not a capability to maximize) | `manuscript-memory-engine/evals/metrics/gpt4_books_cloze.py`, `name_cloze.py` — reuse via adapter per `gnsm/eval/README.md`'s stated reuse-not-copy rule | Loader exists (`manuscript-memory-engine/evals/metrics/gpt4_books_cloze.py`); no `gnsm/eval/` adapter yet |
| Temporal ToM (belief/desire/intention/emotion) | EvolvTrip / LitCharToM — 638 records + 779 OOD test records | Triple-level accuracy + QA accuracy on `ood_test_book.json` | None in `gnsm/eval/` yet | Needs a new adapter; also the direct supervision signal for the soft internal-state heads in `gnsm/state/neural.py`, so this doubles as a training-diagnostic metric |
| Coreference — within-context | LitBank (100 docs) | CoNLL F1 / B³ / CEAF via `maverick-mes-litbank` | `manuscript-memory-engine/evals/metrics/litbank_coref.py` | Reuse via adapter |
| Coreference — book-length | BookCoref (53 books, 30K–67K tokens) | Same coref metrics at book scale via `xcore-litbank` | `manuscript-memory-engine/evals/metrics/litbank_coref.py` (extend for book-length inputs) | Reuse via adapter; this is the dataset that actually stresses long-range entity tracking |
| Quote attribution | PDNC (28 novels) | Speaker-attribution accuracy | `manuscript-memory-engine/evals/metrics/quote_attribution.py`, `pdnc_meta.py` | Reuse via adapter |
| Long-context degradation | Synthetic context-length sweep | Score vs. context length curve | `manuscript-memory-engine/evals/metrics/long_context.py` (704 lines, existing) + `gnsm/eval/context_sweep/run.py:build_sweep` (builds the `(model, context_tokens, seed)` grid — default sizes `[2048, 4096, 8192, 16384, 32768]` × 3 seeds — but does not execute it) | Grid defined, execution missing |
| Internal consistency gate | Synthetic contradiction/clean pairs | Precision/recall/F1 by relation type (stance/kinship/trait/identity/location) | `manuscript-memory-engine/evals/metrics/consistency.py`, `contradiction.py` (Tier-1 gate eval, reusable pattern for `gnsm/verifier/`) | Pattern exists for the baseline system; GNSM's own verifier needs the equivalent |

## Baselines (head-to-head comparison)

Already implemented in `manuscript-memory-engine/baselines/`:
`flat_long_context.py`, `vector_rag.py`, `graph_method.py`, `llm_ask.py`,
orchestrated by `run_baselines.py`. These should be run against the same
benchmark matrix above so GNSM's numbers have a same-generator, same-eval
comparison point rather than only literature numbers.

## Ablations

`gnsm/configs/ablations/{full,no_graph_anchor,no_latent}.yaml` are already
defined (not yet wired to a runner). Minimum ablation set for a journal
submission: full model vs. no-graph-anchor (does the symbolic scene graph
matter?) vs. no-latent (does the continuous state matter, or is the graph
alone sufficient?) — directly tests the paper's central claim that grounding
a continuous state to an interpretable graph is what buys consistency.

## Human evaluation

`gnsm/eval/human_study/design.py:preregistered_cells` already defines the
pre-registered design: 3 systems (`base_llm`, `symbolic_kg`, `gnsm`) ×
2 narrative types (kinetic, introspective) × ≥20 stories/cell — a fixed,
already-decided study design, just not executed. This is the qualitative
complement to the automatic metrics above and typically what reviewers weight
most heavily for a "does this actually read better" claim.

## Figure list (mapped to a typical results section)

1. **Data section** — per-corpus statistics (entity/event counts, novel
   lengths, coref cluster sizes, consistency-subtype distribution, etc.).
   Already produced by `DNG_Data_Visualization.ipynb` for all seven corpora —
   lift those figures directly rather than regenerating them.
2. **Main results table/bar chart** — GNSM vs. the four baselines above,
   scored on every row of the benchmark matrix (consistency, faithfulness,
   coref, ToM, quote attribution), single figure or table per benchmark.
3. **Long-context degradation curve** — score vs. context length
   (2K→32K tokens) for GNSM vs. baselines, from `context_sweep`. This is the
   figure that most directly supports a "consistent at long context" claim.
4. **Ablation deltas** — full vs. no-graph-anchor vs. no-latent, per
   benchmark, as a grouped bar chart or forest plot.
5. **Human study results** — win rates / Likert scores by system ×
   narrative type, with the pre-registered cell structure from
   `human_study/design.py` shown directly (reviewers respond well to seeing
   the design was fixed in advance, not post-hoc).
6. **Memorization guardrail** — GPT4-Books cloze accuracy distribution,
   flagging the high-scoring tail as a contamination signal rather than
   reporting it as a capability score (per the dataset's stated role in
   `README.md`'s dataset table).
7. **Training/infra appendix figure (optional)** — loss curve + wall-clock
   from the Modal-run smoke/training logs, if a reviewer asks for compute
   transparency; not central to the narrative-consistency claim.

## Sequencing / gaps

Nothing in items 2–6 above is real until: (a) a real training loop replaces
`training/smoke.py`'s synthetic-batch wiring check, (b) each benchmark row's
"needs adapter" cell is filled in under `gnsm/eval/`, and (c) baselines and
ablations are actually run end-to-end against the same generations. Figure 1
(data statistics) is the only one already fully supported by existing code
(`DNG_Data_Visualization.ipynb`) and can be finalized independently of
training progress.
