# Dynamic Story Graph — Research Plan

**Working title.** *Premature Commitment, Not Incrementality: Revision-Aware
Construction of Narrative State in Book-Length Fiction*

**Target venue.** TACL (primary), with a humanities-facing companion suitable
for the *Journal of Cultural Analytics*.

**Hard constraints (fixed by the project).** No paid API. All inference runs on
a local open-weight model on a single Apple M4 (16 GB). Every primary metric is
scored against **human gold annotation**, never an LLM judge.

---

## 1. The problem

A reader builds a model of a story *as it arrives*. At discourse position `t`
they know only what has been told. Fiction exploits this: identity is withheld
and later revealed, beliefs are planted and later corrected, states change in
the story world while earlier states remain historically true.

Automatic narrative-IE pipelines instead read the finished text as a static
bag and emit a **retrospective** graph. The moment you make them incremental —
which is what any reading-time, streaming, or generation-support system needs —
the structure *breaks*:

- the same person becomes two nodes (met as "the stranger", named 20k tokens later),
- a superseded fact and its replacement both sit in the graph as live assertions,
- a belief the text later corrects is never retracted,
- errors compound, because nothing that was written down can ever be unwritten.

The reflex diagnosis is that incrementality is lossy. **Our claim is that the
culprit is premature commitment**, and that it is fixable with a small amount of
explicit structure rather than a larger model.

## 2. Contributions

1. **A revision calculus for narrative state.** A versioned, provenance-bearing
   assertion store with an explicit three-way update typology:
   - `ELABORATE` — monotone refinement of deliberately **underspecified**
     structure (the "stranger" node acquires the name *Magwitch*). Nothing is
     retracted; earlier state stays a valid substructure of later state.
   - `SUPERSEDE` — **story-world** change. Both facts were true, at different
     story times; the earlier gets a closed validity interval, not a deletion.
   - `REVISE` — non-monotone **reader-model correction**. The earlier belief was
     never true, and must be retracted along with what was derived from it.
   The `SUPERSEDE`/`REVISE` split is the operational form of the narratological
   distinction between *fabula* change and *syuzhet* revelation, and it is what
   lets the store stay consistent without losing history.

2. **An O(1)-memory online construction algorithm.** A small local LLM sees one
   window plus a bounded, entity-focused digest, and *proposes* typed updates;
   it never writes to the store. A deterministic reconciler resolves identity,
   detects contradiction, classifies each update, and enforces structural
   invariants after every step. Neural proposal, symbolic commitment.

3. **A gold-scored evaluation protocol for time-indexed state**, with three
   planes: identity, extrinsic query, and process integrity (§4).

4. **A measurement instrument.** The revision trace is itself a signal: *where*
   in a book the reader's model gets elaborated vs. corrected. This turns the
   system into an instrument for narrative information release, testable
   against text with known structure.

## 3. Position relative to prior work

| Prior work | What it establishes | What it leaves open — our entry |
| --- | --- | --- |
| Chen, *Distinguishing Revision and Delayed Elaboration in Incremental Narrative Interpretation* (CMN'26, arXiv 2608.21364) | The revision/elaboration distinction, conceptually. | Explicitly a position paper: one worked 7-panel manga example, **no implementation, no corpus, no metrics**. Its own Future Work asks for algorithmic update classification, corpus-scale measurement of revision vs. elaboration frequency, hybrid neural-propose/symbolic-commit architectures, and process-oriented metrics (rollback frequency, proportion of monotone refinements). **We build and measure exactly that program.** |
| *Narrative World Model* (arXiv 2607.05577) | Typed narratological memory + query-conditioned retrieval beats generic temporal KGs on multi-hop writer QA. | Frontier-API extraction and an LLM reader as the metric; proprietary corpus; the win is shown to come from **retrieval conditioning**, not the representation (their untyping ablation raises accuracy). We score the **state itself against human gold annotation**, with no LLM in the metric, on a public corpus, with a 3B–7B local model. |
| Graphiti/Zep, GraphRAG, Mem0 | Bi-temporal / graph agent memory. | Generic entity-edge facts, append-dominant, no reader-model correction, evaluated by downstream QA. |
| BookCoref (2507.12075), LitBank, PDNC | Gold annotation at book scale. | Used **offline and non-causally**. We re-purpose them as gold for *online, prefix-causal* state. |
| Entity tracking in sub-1B LMs (2608.18083) | Small LMs do track entities. | Short, synthetic-ish narratives; no persistent external state. |

**The one-sentence novelty.** No prior system builds narrative state under a
strict prefix-causal constraint with an explicit, implemented revision calculus,
and no prior work scores such a state against human gold annotation as a
function of discourse position.

## 4. Experimental design

### Task — Online Narrative State Tracking (ONST)
A book is streamed as windows `w_1..w_N`. At step `t` a system sees `w_t` and
its own bounded state `S_{t-1}` only — never `w_{>t}`. It emits `S_t`.

### Controlled comparison
Extraction is **held constant across policies**: one cached LLM proposal pass
per (book, window, model), conditioned on a policy-agnostic digest. Policies
then consume the identical proposal stream. This isolates the representation
from extractor quality — the same control NWM had to run post hoc — and cuts
compute ~4x.

| Policy | Carries state | Identity merges | Supersede | Revise | Causal |
| --- | --- | --- | --- | --- | --- |
| `window-only` | no | — | — | — | yes |
| `append-only` | yes | no | no | no | yes |
| `dsg-commit` (ablation) | yes | eager | yes | no | yes |
| `dsg-full` (**ours**) | yes | deferred | yes | yes | yes |
| `retrospective` (oracle) | yes | deferred | yes | yes | **no** |

The oracle consumes the same proposals with full lookahead. It isolates the cost
of *causality*, not of context length.

### Evaluation planes
- **A — Identity (gold).** The entity registry induces a partition over
  character mentions; score MUC / B³ / CEAF_e / CoNLL F1 against BookCoref and
  LitBank gold clusters, reported **as a curve over discourse position**.
- **B — Extrinsic query at position t (gold).** PDNC speaker attribution
  answered from state built only from the prefix; accuracy vs. position.
- **C — Process integrity (no gold needed).** Structural-invariant violations,
  entity fragmentation index, revision/elaboration rates, state growth,
  contradiction density. These are the metrics CMN'26 asks for.
- **D — Scale ablation.** Qwen2.5 1.5B / 3B / 7B. Prediction: the benefit of the
  revision calculus *grows as the model shrinks* — structure substitutes for
  scale. A cheap result with a strong claim if it holds.

### Statistics
Paired bootstrap over books (reusing `gnsm/training/stats.py`), effect sizes
with 95% CIs, per-book paired deltas. No claim rests on a sign test at n=5 —
a lesson already learned and documented in `gnsm/docs/adapter_results.md`.

## 5. Falsifiers — stated in advance

The study is informative either way, and these are the outcomes that would
sink the central claim:

1. `dsg-full` ≈ `append-only` on gold identity → premature commitment is not
   the bottleneck; extraction noise dominates.
2. `dsg-full` ≪ `retrospective` with no closing of the gap → causality, not
   commitment policy, is the binding constraint.
3. Invariant violations near zero for `append-only` → the "breakage" premise is
   wrong on real text, and the calculus solves a problem that does not occur.
4. Revision events distribute uniformly over discourse position → the
   measurement instrument has no signal to measure.

## 6. Build order

- **P0** environment, loaders, local MLX backend, deterministic mock backend ✅
- **P1** `dsg/` core: assertion store, revision calculus, invariants, reconciler
- **P2** LLM proposal pass + caching; the five policies
- **P3** metrics (coref scorers, speaker eval, process metrics) + bootstrap
- **P4** run the grid; figures; results doc
- **P5** paper draft

Raw corpora and model weights stay out of Git; only manifests, code, cached
proposals, and results are tracked.
