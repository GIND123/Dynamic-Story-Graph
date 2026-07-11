# Manuscript Memory Engine — Evaluation Report

**Date:** 2026-06-26
**Run by:** local end-to-end execution on real literary datasets
**LLM backend:** local open-source model (Ollama `qwen2.5:14b`) — **zero API cost, no API key**

---

## 1. Executive summary

We loaded two real literary-NLP datasets (PDNC and LitBank) into the system's
Neo4j knowledge graph, evaluated our system with the project's own metrics
(Change 3), and ran a controlled three-way comparison of retrieval strategies —
Flat Long-Context vs Vector RAG vs Graph — measuring accuracy, token usage,
cost, and speed (Change 4). All language-model calls used a **free local model**.

**Headline findings:**

1. On **Quote Attribution** (a relationship/structure task — "who said this line?"),
   the **Graph method wins outright**: highest accuracy **and** ~20–30× fewer
   tokens, ~20–30× cheaper, and dramatically faster than the LLM-based baselines.
2. On **Name Cloze** (a prose-language task — "predict the masked name"), the
   **LLM baselines win on accuracy**, as expected — but the Graph still answers at
   ~30× lower token/cost.
3. The **Tier-1 consistency gate** scores **F1 = 1.00** across all five relation
   classes on its labeled synthetic set.
4. **Token efficiency, cost, and speed favor the graph decisively and are
   model-agnostic** — they hold no matter which LLM powers the baselines.

---

## 2. What we set up (plain English)

- **Datasets loaded:** Two published literary-NLP corpora were ingested into the
  graph as the source of truth — entities, events, quotations, and prose passages.
  - **PDNC** (Project Dialogism Novel Corpus) — *Pride and Prejudice*: gold
    speaker labels for dialogue, used for Quote Attribution.
  - **LitBank** — *Bleak House*: entity/event annotations, used for Name Cloze.
- **Three retrieval approaches** were each run **in isolation** over the same
  questions, same candidate answers, and same scoring, on an **equal token budget**:
  - **Flat Long-Context** — feed the LLM a window of raw surrounding prose.
  - **Vector RAG** — retrieve semantically similar passages from the vector index,
    feed them to the LLM.
  - **Graph** — read a small structured neighborhood from the Neo4j graph;
    **no LLM, no vector search.**
- **Metrics computed:** Quote Attribution accuracy, Name Cloze accuracy, and the
  Tier-1 Consistency gate (precision/recall/F1), plus per-method token/cost/latency.

### Important clarification: what "Graph" means here

In the comparison tables, **"Graph" is the graph used *alone*** — a deliberate
*ablation* to isolate what the structured graph contributes by itself.

This is **not** the full product. The Manuscript Memory Engine in production is a
**hybrid**: it uses the **graph** for canon/relationship facts **and** **Vector
RAG** for prose continuity **and** an **LLM judge** for validation, together. The
baseline comparison exists to *justify* that design — it shows the graph carries
the relational questions at a fraction of the cost, which is why the product puts
the graph at its center.

---

## 3. Environment

| Component | Value |
|---|---|
| Host | macOS (Apple Silicon, arm64), 24 GB RAM |
| Container runtime | colima (CLI Docker, no Docker Desktop) |
| Graph DB | Neo4j 5.26 Community |
| Broker / cache | Redis 7 |
| App | FastAPI API + Celery worker (Dockerized) |
| LLM (baselines) | **Ollama `qwen2.5:14b`**, local, running on GPU (Metal) |
| Embeddings | fastembed `BAAI/bge-small-en-v1.5`, dim 384 |

---

## 4. Datasets ingested

| Manuscript | Entities | Events | Quotations | Passages |
|---|---|---|---|---|
| `pdnc:PrideAndPrejudice` | 74 | 0 | 1270 | 666 |
| `litbank:1023_bleak_house` | 154 | 61 | 0 | 60 |

(PDNC carries dialogue quotations → Quote Attribution. LitBank carries
entity/event annotations → Name Cloze. Each metric runs where its gold data exists.)

---

## 5. Change-3 metrics — evaluating OUR system

### 5.1 Consistency / Tier-1 gate (synthetic labeled set; deterministic, no LLM)

Identical for both corpora (the gate is dataset-independent):

```
precision=1.00  recall=1.00  f1=1.00   (TP=5 FP=0 FN=0 TN=5, n=10)
  per class:
    identity  P=1.00 R=1.00 F1=1.00
    kinship   P=1.00 R=1.00 F1=1.00
    location  P=1.00 R=1.00 F1=1.00
    stance    P=1.00 R=1.00 F1=1.00
    trait     P=1.00 R=1.00 F1=1.00
```

The deterministic Cypher-based gate correctly flags every planted contradiction
(dead-character action, kinship inversion, trait conflict, stance conflict,
identity collision, two-places-at-once) and passes every clean control.

### 5.2 Quote Attribution — PDNC (gold = PDNC speaker labels)

Reference predictor (majority-class floor), full corpus:

| Scope | Filter | Overall | Explicit | Non-explicit | n quotes | n chars |
|---|---|---|---|---|---|---|
| filtered | major+intermediate | 0.334 | 0.338 | 0.333 | 1200 | 14 |
| all | all characters | 0.316 | 0.312 | 0.317 | 1270 | 27 |

### 5.3 Name Cloze

| Manuscript | Accuracy | n passages | n skipped | seed |
|---|---|---|---|---|
| `pdnc:PrideAndPrejudice` | 0.310 | 100 | 1074 | 0 |
| `litbank:1023_bleak_house` | 1.000 | 3 | 15 | 0 |

(LitBank yields only 3 usable proper-name passages from this single ~2000-word
excerpt — a known property of the small sample, documented in DECISIONS.md.)

---

## 6. Change-4 baseline comparison — Graph vs RAG vs Long-Context

**Config:** token budget = 4000, RAG k = 20 (budget-matched), graph window = 8,
reference price = $3.00e-06 / token. Sample = 49 quotes / 50 cloze passages.
LLM = local Ollama `qwen2.5:14b`.

### 6.1 Quote Attribution (relational task: "who spoke this line?")

| Method | Accuracy | Explicit | Non-exp | Input tokens | Cost $ | Latency (s) |
|---|---|---|---|---|---|---|
| Flat Long-Context | 0.653 | 0.333 | 0.698 | 2825 | 0.4157 | 1.619 |
| Vector RAG | 0.469 | 0.500 | 0.465 | 3943 | 0.5800 | 24.204 |
| **Graph** | **0.816** | **0.667** | **0.837** | **130** | **0.0195** | **0.024** |

**Graph wins on accuracy AND efficiency:** highest overall accuracy (0.816), while
using **~22–30× fewer input tokens**, costing **~21–30× less**, and running
**67–1000× faster** than the LLM baselines.

### 6.2 Name Cloze (prose-language task: "predict the masked name")

| Method | Accuracy | Input tokens | Cost $ | Latency (s) |
|---|---|---|---|---|
| **Flat Long-Context** | **0.500** | 3411 | 0.5118 | 21.486 |
| Vector RAG | 0.300 | 3946 | 0.5920 | 25.117 |
| Graph | 0.260 | **109** | **0.0167** | **0.007** |

**LLM baselines win on accuracy here** (Long-Context 0.500), as expected — guessing
a masked name from prose is not a graph-shaped question. The Graph is weakest on
accuracy (0.260) but still answers at **~30× lower token/cost**.

### 6.3 Efficiency summary (Graph vs best LLM baseline)

| Task | Graph accuracy | Best LLM accuracy | Graph token saving | Graph cost saving | Graph speedup |
|---|---|---|---|---|---|
| Quote Attribution | **0.816 (best)** | 0.653 | ~22× fewer | ~21× cheaper | ~67× faster |
| Name Cloze | 0.260 | 0.500 | ~31× fewer | ~31× cheaper | ~3000× faster |

---

## 7. What this means

- The graph is the **right tool for relational/structural questions** (who said
  what, who relates to whom): it is both **more accurate and radically cheaper**.
- The graph is **not** the right tool for **free-form prose prediction** — there
  the LLM-over-prose baselines lead on accuracy. This is the expected division of
  labor and is exactly why the real product is a **hybrid** (graph + RAG + LLM),
  not graph-only.
- **Token / cost / speed advantages are model-agnostic.** Using a stronger model
  (e.g. Claude) would raise the Long-Context / RAG *accuracy* numbers, but would
  not change the graph's ~20–30× token-and-cost advantage — that is a property of
  the retrieval strategy, not the model.

---

## 8. Caveats (so the numbers are read fairly)

1. **Local model.** Baseline LLM accuracy reflects `qwen2.5:14b`. A frontier model
   would lift Long-Context / RAG accuracy (not the graph's efficiency lead).
2. **Sample size.** 49 quotes / 50 cloze passages, one novel per corpus —
   illustrative and directionally solid, not publication-scale.
3. **Cost column is modeled,** at a reference $3/million-token price, to show
   *relative* economics. The actual run cost **$0** (local inference).
4. **Name Cloze on LitBank** uses only 3 usable passages from a single excerpt;
   the 1.000 there is correct but on a tiny n.

---

## 9. How to reproduce

```bash
# 1. Runtime + stack
colima start --cpu 4 --memory 8
make up                                   # Neo4j + Redis + api + worker

# 2. Ingest the real datasets
make ingest ARGS="pdnc PrideAndPrejudice"
make ingest ARGS="litbank 1023_bleak_house"

# 3. Change-3 metrics (our system)
docker compose exec api python evals/run_metrics.py pdnc:PrideAndPrejudice
docker compose exec api python evals/run_metrics.py litbank:1023_bleak_house

# 4. Change-4 baseline comparison (LLM_MODE=ollama in .env)
docker compose exec api python baselines/run_baselines.py \
    pdnc:PrideAndPrejudice --sample-quotes 50 --max-cloze 50
```
