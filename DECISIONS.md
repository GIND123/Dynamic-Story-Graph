# DECISIONS.md — Manuscript Memory Engine

Every non-trivial architectural choice is recorded here. Format: **what**, why, alternative rejected.

---

## Embeddings: fastembed (local ONNX) instead of a cloud API

fastembed with `BAAI/bge-small-en-v1.5` (dim=384) runs fully in-process, requiring no second API key. Production swap is Voyage AI, which Anthropic's docs recommend as the embedding partner. Rejected: calling the Anthropic API for embeddings — Anthropic has no embedding model (verified June 2026).

## EMBED_DIM defined once in config.py

`EMBED_DIM = 384` is a module constant in `app/config.py`. Both the Neo4j vector index DDL and the fastembed wrapper import it from this single location. A mismatch between index dimension and embedder dimension is a silent correctness bug caught only at query time; one source of truth prevents it.

## k_overfetch = k * 4 for vector similarity search

The Neo4j `db.index.vector.queryNodes` call cannot pre-filter by `manuscript_id` — it returns results from all manuscripts. We fetch `k*4` candidates and then filter by `manuscript_id` in the WHERE clause after YIELD. Rejected: separate vector DB with native filtering — adds infrastructure complexity and a second data store to keep in sync.

## LLMClient Protocol (ports-and-adapters)

`app/llm.py` defines a `Protocol` with `draft_chapter`, `extract`, and `judge`. Two implementations: `AnthropicLLM` (real API) and `FakeLLM` (deterministic, zero-cost). This means the full pipeline, gate, retry loop, idempotency, and Neo4j transactions are all exercised in CI without an API key. Dependency inversion is also a strong interview talking point. Rejected: always-real LLM calls in tests — flaky, expensive, and non-deterministic.

## OllamaLLM: a third LLMClient adapter for a free local backend

A third implementation of the `LLMClient` Protocol, `OllamaLLM`, calls a local Ollama server (`LLM_MODE=ollama`, default model `qwen2.5:14b`), giving a zero-cost, no-API-key option alongside `AnthropicLLM` and `FakeLLM`. (The smaller `qwen2.5:7b-instruct` drafts and extracts fine but is a noisy continuity judge — it false-positives on legitimate character movement — so the default is the 14B, which fits comfortably in 16 GB+ RAM and judges far more reliably; `OLLAMA_MODEL` swaps it with no code change.) This is purely additive: the pipeline, two-tier gate, retry loop, idempotency, and Neo4j transactions are untouched — only the factory gained a branch. It reuses the existing `httpx` dependency (no new package) and the same `_DRAFT_PROMPT`/`_EXTRACT_PROMPT`/`_JUDGE_PROMPT` constants. `draft_chapter` uses plain text generation; `extract`/`judge` use Ollama's structured-output `format` set to the Pydantic model's `model_json_schema()`, then validate the response against that model and retry once on a validation miss — mirroring AnthropicLLM's `messages.parse` guarantee without Anthropic's server-side grammar. A system turn pins English output (multilingual open models occasionally code-switch — observed once on a cold first call) and temperature is 0 for the structured steps. Docker note: the worker reaches the host server at `host.docker.internal:11434`, not `localhost`. Rejected: the `ollama` Python package (an extra dependency for what two `httpx` POSTs do); bare `format:"json"` without a schema (less reliable for the relation-heavy extraction shape).

**Numeric bounds are not enforced by Ollama structured outputs.** Anthropic's grammar enforces Pydantic field constraints like `coherence_score: float = Field(ge=0, le=1)` server-side; Ollama's `format`-as-schema mode constrains only JSON *type and structure*, NOT a number's `minimum`/`maximum`. Observed live: `qwen2.5:14b` returned `coherence_score=100` (a 0-100 reading), which `model_validate_json` rejected — so the adapter raised `LLMOutputIncomplete` rather than committing an out-of-contract value (validate-then-retry doing its job). Fix is adapter-local and does NOT touch the shared `_JUDGE_PROMPT`: the Ollama-specific JSON system turn instructs the model to honor schema min/max and keep a max-1 field within 0.0-1.0. Rejected: silently rescaling an out-of-range score in code (a fragile heuristic — a stray `2` is ambiguous between a bug and a 0-100 value); editing the core judge prompt (would alter the Anthropic path's verified behavior).

## Token budget heuristic: len(text) // 4

Characters divided by 4 approximates tokens without a network call. The Anthropic token-counting API (`client.messages.count_tokens`) is the precise alternative but adds latency per job. Heuristic is good enough for a soft budget; over-budget passages are truncated at sentence boundaries before the exact limit matters.

## All chapter-commit writes in one Neo4j transaction (execute_write)

`graph.commit_chapter` runs a single `session.execute_write(tx_fn, ...)`. This is the "canon is never half-updated" guarantee: prose, character updates, events, embeddings, and the PRECEDES edge all commit atomically or not at all. Rejected: multiple separate writes — partial failure leaves canon in an inconsistent state.

## Sync Anthropic client + sync Neo4j driver inside Celery tasks

Celery workers are synchronous Python processes. Mixing asyncio into a Celery task requires a per-task event loop (`asyncio.run(...)`), which is fragile and obscures the code. Both the Anthropic SDK and neo4j Python driver have full sync APIs. Rejected: async clients with asyncio.run — added complexity for no benefit in a worker context.

## uid pattern: `{manuscript_id}:{Label}:{natural_key}`

Neo4j Community edition supports uniqueness constraints only on a single property. Using a composite `uid` string encodes both the manuscript scope and the entity's natural key into one field, giving us effective scoped uniqueness without Enterprise-only node-key constraints. Rejected: composite constraints — Enterprise only.

## Cypher f-strings for constraint/index DDL only

Cypher does not support parameterized label names or constraint names. The `CREATE CONSTRAINT` and `CREATE VECTOR INDEX` statements use Python f-strings where the label/index name is a hardcoded constant (not user input). All data-carrying queries use `$param` placeholders exclusively. This is documented as an explicit exception to the "no f-strings in queries" rule.

## Idempotency: Redis SET NX + Chapter.status

Two-layer idempotency: (1) Redis lock prevents concurrent duplicate execution; (2) checking `Chapter.status == COMMITTED` before running catches re-deliveries after a worker crash where the lock already expired. The lock TTL (900s) is a backstop. Release is an atomic compare-and-delete Lua script (`_LOCK_RELEASE_LUA`) so a slow task that holds the lock past TTL can never delete a lock another worker just acquired.

## Result backend carries status only, not prose

Chapter prose lives in Neo4j. The Celery result payload contains only `{status, chapter_number, attempts, coherence_score}`. Fat Redis results are an anti-pattern: Redis is not a document store, result TTL is short, and prose retrieval belongs to the API's Neo4j read path.

## fastembed model warmup via worker_process_init signal

`@worker_process_init.connect` in `celery_app.py` calls `vectors.warmup()` in each Celery worker child process before the first task runs. The deferred import (`from app.vectors import warmup` inside the handler) keeps fastembed out of the API process (which also imports `celery_app`). Rejected: warmup inside the first task — adds 10-30s of latency to the first real chapter generation and makes it non-deterministic.

## similar_passages over-fetches k*4 due to index pre-filter limitation

`db.index.vector.queryNodes` returns results from all manuscripts; it cannot pre-filter by `manuscript_id`. We request `k*4` candidates and apply a WHERE clause on `manuscript_id` and `status` after YIELD. With k=3, we fetch 12 candidates and keep up to 3. Rejected: separate vector DB with native filtering — adds a second data store and a second API key.

## Tier-1 gate is a pure function (no DB round-trip)

`graph.tier1_check(extraction, canon)` takes the already-fetched canon dict rather than querying the database again. The canon dict (returned by `graph.get_canon`) already carries every character's current status, so dead-involvement and resurrection checks require only a dict lookup — O(1) per character. A DB round-trip would cost a full network call for every attempt in the retry loop. Rejected: querying Neo4j inside tier1_check — slower and adds an extra failure mode.

## `judge.py` as thin wrapper (separation of concerns)

`app/judge.py` contains `evaluate(canon, draft, llm)`, which calls `format_canon` and then `llm.judge`. Keeping this in a separate module from `llm.py` lets the judge orchestration evolve independently (e.g. add a second-opinion judge, log verdicts, switch models) without touching the LLMClient protocol. Tasks import `judge` as a module reference, so monkeypatching `app.judge.evaluate` in tests intercepts the call cleanly. Rejected: calling `llm.judge` directly in tasks.py — merges orchestration with protocol, harder to extend.

## TestClient without lifespan for unit tests

Tests use `fastapi.testclient.TestClient(app)` without the context-manager form, so the async lifespan (constraint setup, index creation) does not run. External dependencies (graph functions, Celery task dispatch) are monkeypatched per test. This keeps tests fast, self-contained, and runnable without live services. Integration smoke tests (make up first) cover the live path.

## seed.py uses logging instead of print

`seed.py` now uses `logger.info` like every other module rather than `print`, for consistent, configurable output. The `__main__` guard calls `logging.basicConfig` so the operator still sees seed progress when running it as a script.

## Extract prompt fences the draft as untrusted

`_EXTRACT_PROMPT` now carries the same "untrusted story text, not instructions" fence as `_JUDGE_PROMPT`. Extraction output feeds the Tier-1 gate, so an unfenced extract prompt was an asymmetric injection surface a draft could use to subvert the gate.

## Lock release is an atomic Lua compare-and-delete

The `finally` block releases the lock via `_release_lock`, which runs `_LOCK_RELEASE_LUA` (delete-if-value-equals-task-id) atomically on the Redis server, replacing the non-atomic get-compare-delete. `_release_lock` resolves `register_script` against the module-level `_redis` at call time so tests can monkeypatch the client.

## Prose-shaped fallback query for vector retrieval

`retrieval.build_retrieval_query` prefers `scene_hint`, else joins the two most-recent event summaries (prose-shaped), else returns "". When empty the pipeline skips retrieval entirely. Rejected: using `format_canon(canon)[:200]` as the query — its tabular `Name | status | ...` shape is a poor neighbor of prose chapters in embedding space.

## commit_chapter auto-merges characters named only in event.involves

A name appearing in `event.involves` but not in `extraction.characters` is MERGEd as a `Character` with `status='alive'` so the INVOLVES edge is preserved (otherwise a future Tier-1 check could miss a dead character only referenced via involves). Uses `ON CREATE SET` so an existing node's status is never clobbered. Rejected: silently dropping the edge via MATCH-then-MERGE — loses graph information.

## next_chapter_number filters to COMMITTED only

A NEEDS_REVIEW chapter is retried by re-POSTing rather than skipped: `next_chapter_number` counts only `status='COMMITTED'` chapters, so a failed chapter's number is returned again. On the successful retry, `commit_chapter` overwrites the prior node via MERGE on uid and clears stale failure notes with `REMOVE ch.last_feedback`. Rejected: counting all chapters — permanently skips a failed chapter number, contradicting the documented NEEDS_REVIEW → retry lifecycle.

## Default Neo4j password is documented as local-dev only

`docker-compose.yml` and `.env.example` ship `neo4j/password` for one-command laptop setup. README and a compose comment now warn that this must be changed before exposing the stack on any network. Rejected: generating a random password — breaks the zero-config `make up` demo flow the project optimizes for.

## Relationship schema model is additive and optional

`models.py` adds a closed `RelationType` enum (exactly 13 edges — no PRECEDES, no mental-state relations) and a flat `ExtractedRelation` with all discriminators optional. `ChapterExtraction.relations` defaults to `[]`, so every existing extraction stays valid and generation behavior is unchanged. Rejected: a per-type subclass hierarchy — large unions complicate the structured-output schema (CLAUDE.md §3.1) for no gain.

## merge_relation interpolates only validated rel-type/labels; all data is parameterized

Cypher cannot parameterize relationship types or node labels, so `graph.merge_relation` f-strings them — but only after validating the type against the `RelationType` enum and labels against the `_NODE_LABELS` allowlist (raising otherwise), the same documented exception used for constraint DDL. Every value (uids, names, the property bundle) is a `$parameter`. Edges carry a universal bundle (source/source_ref/seq_introduced/seq_invalidated + discriminators) set via `ON CREATE SET` so re-projection never overwrites. `None` values are dropped because Neo4j cannot store null; an absent `seq_invalidated` reads back as null = "currently active".

## Dedicated Attribute / Alias / Fact (+ Organization / Object) nodes

Traits, aliases, and facts are first-class nodes (HAS_TRAIT→Attribute, IDENTIFIED_AS→Alias, KNOWS_ABOUT→Fact) rather than node properties, so a shared or divergent trait between two characters is a one-hop graph match instead of a property scan. Each gets a single-property `uid` uniqueness constraint like every other label (Community-safe). Rejected: storing traits/aliases as list properties — can't be matched relationally in one hop.

## PDNC dialogue acts are modeled as EVENT + INVOLVES, not a SPEAKS_TO edge

A speaker→addressee dialogue edge is not in the closed 13, and forcing it onto `RELATES_TO stance='neutral'` would corrupt real ally/enemy semantics. Instead each PDNC quotation becomes an `Event` node (with `quote_type` and `is_dialogue` properties), linked via `INVOLVES role='agent'` to the speaker and `INVOLVES role='patient'` to each addressee. This reuses the existing event/participant machinery and keeps the relation set closed.

## Entity resolution: PDNC aliases, LitBank coref/surface-form

PDNC `character_info` aliases are the canonical resolver — every speaker/addressee surface form is collapsed to its Main Name *before* nodes are created, so one character == one node, and the collapsed aliases become IDENTIFIED_AS edges. LitBank uses coref chains when present, else groups mentions by mapped-type + lowercased surface form. Label mapping: PER→CHARACTER, LOC/FAC/GPE→LOCATION, ORG→ORGANIZATION, VEH→OBJECT.

## Eval data reuses the existing passage_index (no parallel index)

Ingested passages are written as `Chapter` nodes with an `embedding` (same `EMBED_DIM`, computed by `vectors.embed_one`) and `status='COMMITTED'`, so the existing `passage_index` and `similar_passages` query retrieve them with no new index. Eval manuscripts are namespaced (`pdnc:{folder}` / `litbank:{gid}`), and `similar_passages` already filters by `manuscript_id`, so eval passages can never leak into a generation manuscript's retrieval.

## Eval-guard: source='eval' on the Manuscript, checked at the chapter-request entrypoint

`project_to_graph` tags the Manuscript node `source='eval'`. `graph.is_eval_manuscript` reads it, and the `POST /manuscripts/{mid}/chapters` route returns 403 for eval manuscripts. The generation pipeline has no manuscript enumeration, so the request entrypoint is the single place generation "selects" a manuscript — the only guard needed, added without changing the route's behavior for normal manuscripts.

## PDNC/LitBank formats: verified against the real corpora (2026-06)

Both datasets were cloned (`Priya22/project-dialogism-novel-corpus`, `dbamman/litbank`) and the parsers verified against real files. What the loaders assumed vs. what the real data contains, and the fixes made:

**PDNC** (verified on `data/PrideAndPrejudice`):
- Filenames matched assumptions: `character_info.csv`, `quotation_info.csv`, `novel_text.txt`.
- `quotation_info.csv` columns (`quoteID, quoteText, ..., speaker, addressees, quoteType, ...`) and `character_info.csv` columns (`Character ID, Main Name, Aliases, Gender, Category`) matched the case-insensitive header lookups. `quoteType` values are capitalized (`Explicit/Implicit/Anaphoric`) and are lowercased as assumed; addressees use list literals (`['Mr. Bennet']`).
- DIVERGENCE FIXED: the `Aliases` column is **mixed** — most rows are Python *set* literals (`{'Mrs. Collins', 'Miss Lucas'}`) and some are *list* literals (`['Colonel Forster']`). The original `_parse_list_cell` only accepted `[...]`. Fixed to accept `{...}` set literals too (items sorted for determinism). DIVERGENCE FIXED: a character's Main Name appears inside its own alias set — now dropped so no Alias node duplicates the canonical Character.

**LitBank** (verified on `1023_bleak_house`):
- DIVERGENCE FIXED: real files carry a `_brat` suffix (`1023_bleak_house_brat.tsv`), not `{doc_id}.tsv`. `LitBankLoader` now resolves `{doc_id}.tsv` → `{doc_id}_brat.tsv` → `{doc_id}*.tsv` glob, and strips a trailing `_brat` from the namespace stem.
- Layout matched assumptions: entities TSV is token + **4** nested BIO layers over `{PER,FAC,GPE,LOC,VEH,ORG}` (parser already handled arbitrary layer count); events TSV is token + `EVENT`/`O`; coref is brat standoff under `coref/brat`. The surface-form resolver is used (coref left optional); nested spans like `Lord` / `Lord Chancellor` remain distinct without coref, which is acceptable for eval.

Live ingestion of one book from each into Neo4j succeeded and was sanity-checked (alias resolution collapses one person to one node; 0 alias/character name collisions; dialogue events resolve speaker→agent / addressee→patient; passages reuse the single `passage_index`; re-run is idempotent). Regression tests for the set-literal aliases and the `_brat` suffix are in `tests/test_ingestion.py`.

## SPEAKS_TO added as a thin 14th relation (additive, not a replacement)

`SPEAKS_TO` (CHARACTER→CHARACTER, carrying `quote_type` + `source_ref`) is a single-hop speaker→addressee view for first-class quote-attribution queries. It coexists with the EVENT+INVOLVES dialogue model that ingestion already built and verified; neither replaces the other. Wiring is minimal: a new enum member, an entry in `_RELATION_ENDPOINTS` (which IS the merge_relation allowlist), and `quote_type` added to `_RELATION_DISCRIMINATORS`. Rejected: retrofitting the existing eval dialogue edges onto SPEAKS_TO in this step — ingestion is verified and out of scope; a later task can backfill.

## Generation now emits, commits, and validates relations

Extraction returns `relations: list[ExtractedRelation]` (prompt extended; `FakeLLM.extract` emits a deterministic benign set + a self-loop under FAKE_VIOLATION). `commit_chapter` writes them via `merge_relation` inside the SAME execute_write transaction with `source='gen'`, `source_ref=<chapter uid>`, `seq_introduced=<chapter number>`, relying on merge_relation's per-type endpoint defaults and ON CREATE SET (idempotent re-commit, never clobbers status). Relations stay optional — extractions with none commit exactly as before.

## Canon surfaces only ACTIVE relations; tier1_check stays pure

`get_canon` gained an additive `relations` key (existing keys unchanged) populated by `_get_active_relations`, which runs one query per relation kind filtered to `seq_invalidated IS NULL`. The read lives in the canon-assembly function (which already hits the graph); `tier1_check` remains a pure function over `extraction` + the canon dict. Backward-compatible: a canon without a `relations` key makes every new check a no-op.

## Tier-1 relation checks are deliberately conservative (favor NEEDS_REVIEW over false positives)

Five new pure checks, each returning a violation string that routes to the existing regenerate→NEEDS_REVIEW path (no new control flow, no hard reject):
- **Stance**: flags only ally↔enemy contradiction for the same unordered pair; `neutral` never conflicts.
- **Two-places-at-once**: flags only a character asserted at >1 location WITHIN this chapter's extraction. Cross-chapter movement is intentionally NOT flagged, because the pipeline does not yet invalidate prior `LOCATED_AT` edges — flagging canon-vs-new location would false-positive on every legitimate move.
- **Kinship**: self-loop (from extraction alone); parent/child inversion vs canon; spouse-of-dead is flagged "for review" (a violation string, matching existing dead-handling) rather than hard-rejected.
- **Trait** (flagship): compares only `category='physical'` traits via explicit structured `trait_key`/`trait_value` fields (see the dedicated entry below).
- **Identity collision**: flags an `IDENTIFIED_AS` alias that canon already binds to a different character.

## HAS_TRAIT contradiction is structural: explicit trait_key/trait_value, null-key defers to Tier-2

`ExtractedRelation` carries `trait_key` (normalized snake_case attribute, e.g. `eye_color`) and `trait_value` (e.g. `blue`). The prior representation — a `key: value` string convention parsed by `_trait_key_value` — was insufficient: nothing produced that format (the extract prompt asked only for `category`), so a real LLM emitted `target="blue eyes"` with no delimiter, the parser returned `(None, …)`, and the check was silently skipped — dead on real input while still passing unit tests that hand-fed `"eye_color: blue"`. The fix makes the representation explicit end to end: the prompt requires the `trait_key`/`trait_value` split; `merge_relation` keys the **Attribute node by the dimension (`trait_key`)** so one `eye_color` node holds all values rather than fragmenting into one node per value-string, with the specific value carried on the edge; `get_canon` returns the structured fields; and the check fires ONLY on same character + exact-equal `trait_key` + exact-unequal `trait_value` for `category='physical'` traits (physical attributes don't change; personality/skill can) — no substring/fuzzy logic, so paraphrases under different keys (height=tall vs build=towering) never trip. Null/missing `trait_key` makes NO deterministic decision and emits no Tier-1 violation, deferring to the Tier-2 judge / NEEDS_REVIEW path. `FakeLLM` now emits a structured physical trait and a regression test feeds a real-shaped extraction (free-text `target` + structured fields) so the check can't silently regress to dead-on-real-input. The old `_trait_key_value` helper was removed.

## Metric 1 — Quote attribution: gold from the graph, alias-normalized, Category-filtered

PDNC speaker labels are ground truth, pulled directly from the graph (dialogue `Event {is_dialogue}` + `INVOLVES role='agent'/'patient'`), NOT via `get_canon` — `get_canon` caps events at LIMIT 10 and does not surface dialogue, so it is useless for metrics. Predicted and gold names are normalized through PDNC aliases (read from `character_info.csv`, since the graph stores no aliases/Category on Character nodes) so one character = one identity before matching. Accuracy is reported per quote_type and as the standard hard split — explicit vs non-explicit (implicit+anaphoric), the discriminating case. Scoring is filtered to major+intermediate characters via `character_info.csv` Category; when Category is unavailable the rule falls back to "≥10 quotes spoken" and records which rule was used. Both filtered and all-character numbers are reported. The shipped reference predictor is the majority-class floor (predicts the most frequent speaker); graph/vector/long-context predictors drop into the same `predict(quote_text, context)` signature later.

## Metric 2 — Name cloze: sentence-windowed passages, proper-name PER filter, fixed seed

The 40–60 token window is correct but raw Chapter nodes are the wrong granularity (LitBank = one sentence each, far under 40; PDNC = ~1200-char chunks, far over 60). Passage construction therefore **windows sentences**: the ordered Chapter text is concatenated into a sentence stream (`_split_sentences`, with a title-abbreviation guard so "Mr. Bennet" is never split), and consecutive sentences are accumulated into NON-OVERLAPPING windows that land in [40,60] tokens (greedy from the front; a window that cannot land in range advances its start by one sentence — deterministic, so the seed governs only the final selection shuffle).

**Entity counting (corrected):** "named entity" for the one-entity / no-other-entities rules is defined ONLY by the proper-name target set (`proper_name_targets`: passes `is_proper_name` — every token capitalized, not ALL-CAPS — AND recurs ≥2 times). Generic common-noun spans the LitBank loader surfaces as Character nodes ("Foot passengers", "ancient Greenwich pensioners") are NOT named entities under the Name Cloze task, so they are not counted — counting them was incorrect, not conservative. The proper-name filter is the single source of truth for "is this a name"; the token window, masking, scoring, and the filter definition itself are unchanged.

Real post-fix counts: **PDNC `pdnc:PrideAndPrejudice` → 100 usable** (capped; unchanged — it was already entity-correct). **LitBank `litbank:1023_bleak_house` → 3 usable** (was 0). The fix lifts LitBank off zero; the remaining small count is a genuine property of a single ~2000-word excerpt: of 93 Character nodes only 12 are proper names and only **3 recur ≥2 times** (Chizzle, Jarndyce, Mizzle), and those co-occur in 2 of the 18 windows, leaving exactly 3 windows that isolate one. Not relaxed further — dropping the recurrence requirement would weaken the proper-name filter the task fixed in place.

**FUTURE (rigorous version):** carry LitBank's gold entity labels (the PER/FAC/GPE/… spans already in `entities/tsv`) through ingestion onto the nodes, instead of inferring "is this a name" from the capitalization + recurrence heuristic. Gold PER labels would (a) stop the loader surfacing ~81 generic spans as Character nodes at all, and (b) let the recurrence requirement relax, both raising LitBank yield with higher precision than the heuristic.

## Metric 3 — Consistency: synthetic labeled set evaluates tier1_check (not the datasets)

The eval datasets contain no planted contradictions, so — following the existing `evals/cases.json` planted-contradiction + clean-control pattern — consistency is measured on a SYNTHETIC labeled set of (canon, extraction) pairs (one tripping + one clean per relation class: stance/kinship/trait/identity/location). It reports precision/recall/F1 with the same TP/FP/FN/TN conventions as the judge eval, plus a per-class breakdown, evaluating the Change-2 Tier-1 gate itself — deterministic, offline, no graph/LLM. The checker is injectable so tests can drive precision/recall down with a deliberately over-firing or blind checker and confirm the metric responds.

## Baselines: equal-budget comparison of long-context vs RAG vs graph

Three READ-ONLY predictors race through the existing Change-3 harnesses (same gold, same candidates, same `score_quote_attribution` / `run_name_cloze`) on an EQUAL information budget (`baselines/config.py`, default `TOKEN_BUDGET=4000`). Long-context fills the budget with a surrounding-prose window; RAG fills it with **budget-matched k** chunks where `k = round(TOKEN_BUDGET / avg_chunk_tokens) ≈ 20` (documented as budget-matched, NOT a guessed 7); the graph method deliberately reads a small structural neighborhood (`graph_window` quotes each side) using far fewer tokens than the budget — that token efficiency is the result being measured, so it is not padded. Token counting reuses retrieval's `len // 4` heuristic; cost = `(in+out) tokens × price_per_token` (one named constant); latency is per-query wall-clock. The LLM is an injected `ask(prompt, meta)` where `meta` carries only non-gold routing info (quote_id/quote_type), so tests vary behaviour by quote_type without ever exposing the gold.

**Quote→prose mapping:** quote Events carry no edge to their prose Chapter and use a different ordering, so a quote is located in prose by text-match (`ch.text CONTAINS left(quote,40)`, verified Q500→206) and the mapping is cached. Long-context windows ±`prose_chapters_each_side` chapters around it; RAG reuses `app.vectors.similar_passages` (manuscript-scoped) and **excludes any retrieved chunk containing the quote line** so it can't read its own answer.

**Graph holdout (experiment integrity):** `graph_method.neighborhood_agents(..., exclude_seq=center)` returns the sequence_index neighborhood with the target quote's OWN `INVOLVES role='agent'` row removed — the graph predictor never reads the target's gold agent. A test asserts that for a target whose gold speaker appears nowhere else in the neighborhood, that speaker is absent from the predictor's input (and the prediction is therefore never that held-out name). The graph quote heuristic is documented: dialogue alternation away from `previous_speaker` among scene speakers. Every query is manuscript-scoped.
