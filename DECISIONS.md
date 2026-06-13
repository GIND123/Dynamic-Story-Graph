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

Two-layer idempotency: (1) Redis lock prevents concurrent duplicate execution; (2) checking `Chapter.status == COMMITTED` before running catches re-deliveries after a worker crash where the lock already expired. The lock TTL (900s) is a backstop; the get-compare-delete release prevents a race where a slow task holds the lock past TTL and another worker acquires it. A Lua script would make the compare-delete atomic; the current Python implementation has a narrow TOCTOU window documented in a code comment.

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
