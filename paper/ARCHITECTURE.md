# Dynamic Story Graph — architecture for the paper

The system as a paper needs to describe it: components, contracts, data flow,
complexity, and the mapping from each claim to the code that supports it.
The formal update rules live in [`docs/METHOD.md`](../docs/METHOD.md); this
document is the engineering view.

---

## 1. The one-figure summary

```
                    ┌─────────────────── prefix-causal boundary ───────────────────┐
                    │  at step t the system may read w_t and state from w_1..w_t-1 │
                    └──────────────────────────────────────────────────────────────┘

  novel text
      │
      ▼
  ┌─────────┐   w_t + lead-in      ┌──────────────┐   typed proposals   ┌──────────────┐
  │ WINDOWS │ ───────────────────► │  LLM (frozen)│ ──────────────────► │  RECONCILER  │
  └─────────┘   + digest of names  │   PROPOSES   │  CHAR/LINK/FACT/    │   DISPOSES   │
       ▲          seen so far      └──────────────┘  SAID lines         └──────┬───────┘
       │                                  ▲                                    │
       │                                  │ policy-agnostic digest             │
       │                           ┌──────┴───────┐                            │
       │                           │SurfaceLedger │◄───────────────────────────┤
       │                           └──────────────┘                            │
       │                                                                       ▼
       │                                                        ┌──────────────────────────┐
       │                                                        │   NARRATIVE STATE S_t    │
       │                                                        │  entities  · assertions  │
       │                                                        │  revision log · two clocks│
       │                                                        └────────────┬─────────────┘
       │                                                                     │
       │                                                        ┌────────────▼─────────────┐
       └────────────────────────── next window ─────────────────│  7 STRUCTURAL INVARIANTS │
                                                                │   checked every step     │
                                                                └──────────────────────────┘
```

**The load-bearing design decision.** The language model has *no write access*.
It emits proposals; a deterministic reconciler classifies each one, resolves
identity, applies the revision calculus, and checks invariants. Two consequences
the paper should state explicitly:

1. The reasoning under test is not a black box — whether a conflict is a world
   change or a reader error is decided by an inspectable table
   (`dsg/lexicon.py`), not by the model.
2. Extraction can be **cached and replayed**, so every policy in the ablation
   consumes a byte-identical proposal stream. Differences are attributable to
   the representation, not to extractor variance. (NWM had to run an
   extractor-fairness control *post hoc*; here it is exact by construction.)

## 2. Component inventory

| Plane | Module | Responsibility | Paper section |
| --- | --- | --- | --- |
| Contracts | `dsg/schemas.py` | `Assertion` with two clocks — discourse time `[d_open, d_close]` (while the reader held it) and story time `[valid_from, valid_to]` (while it held in the world); `EntityNode` with commitment level; `UpdateRecord`; `Violation`. | §3 Representation |
| Contracts | `dsg/lexicon.py` | Predicate mutability (mutable / immutable / multi-valued), revelation markers, underspecified objects, person-denoting nouns, closed-range predicates. The arbiter of `SUPERSEDE` vs `REVISE`. | §3 Calculus |
| Contracts | `dsg/matching.py` | Conservative surface matching. Title-aware: a shared surname never merges anyone, and two conflicting honorifics never match. | §4 Identity |
| Reading | `dsg/windows.py` | Prefix-causal windowing; boundary snapping; lead-in that is always already-read text. | §4 System |
| Extraction | `dsg/proposals.py` | Prompt, tolerant line parser, schema checks, `SurfaceLedger` (policy-agnostic digest), focused identity-resolution second pass. | §4 System |
| Extraction | `dsg/llm/` | Backends: MLX (local Apple Silicon), vLLM on Modal, deterministic mock for CI. On-disk generation cache. | §4 / Appendix |
| State | `dsg/store.py` | `NarrativeState`: the calculus, the policy ladder, merge guards, the revision log. Two mutating entry points only. | §3 Calculus |
| State | `dsg/invariants.py` | Seven structural invariants + the bounded `inconsistent_slot_rate`. | §3 / §5 metrics |
| Policies | `dsg/policies/runner.py` | Replays one cached proposal stream under any policy; answers position-indexed probes. | §5 Design |
| Evaluation | `dsg/eval/identity.py` | Alias-partition scoring: MUC / B³ / CEAF-e / CoNLL, fragmentation, conflation, coverage. | §5 Plane A |
| Evaluation | `dsg/eval/mention.py` | Prefix-causal mention-linking probes at the reading position of each gold mention. | §5 Plane B |
| Evaluation | `dsg/eval/speaker.py` | Speaker attribution over gold quotations. | §5 Plane B′ |
| Evaluation | `dsg/eval/process.py` | Process integrity: violations, monotone fraction, rollbacks, revision profile. | §5 Plane C |
| Evaluation | `dsg/eval/instrument.py` | Identity-resolution latency; position trends with the shuffled-window control. | §7 Instrument |
| Evaluation | `dsg/eval/bootstrap.py` | Paired bootstrap over books. | §5 Statistics |
| Data | `dsg/data/{pdnc,litbank,registry}.py` | Gold loaders. PDNC: text, alias sets, quotations, offset-anchored mentions, bibliographic metadata. LitBank: gold coreference as a short-context control. | §5 Data |
| Infra | `dsg/infra/modal_extract.py` | GPU extraction, batched **across books in lockstep** so causality holds within each book while vLLM gets a full batch. | Appendix |

## 3. Data flow, precisely

1. **Window.** `iter_windows` tiles the novel into contiguous, boundary-snapped
   windows with no overlap. Each carries a `lead_in` of already-read text.
2. **Propose.** The model sees `w_t`, the lead-in, and a bounded digest of names
   and unnamed figures met so far. It returns line-oriented proposals — chosen
   over JSON because a truncated line costs one item while an unclosed brace
   costs the whole window.
3. **Resolve identity (second pass).** Open references that denote *people* are
   put to the model as a separate, focused question. Asked as one of five things
   in a larger schema, a 7B model never emits identity links at all; asked alone
   against an explicit candidate list, it answers.
4. **Reconcile.** For each proposal: resolve the subject to a node (creating a
   provisional node if the reference is a description or contested), normalise
   the predicate, find live assertions on the slot, classify the update, apply
   it, log it.
5. **Check.** Seven invariants over the whole state, including prefix-causality.
6. **Probe.** Any gold mention falling in this window is answered from the state
   as it now stands — before the next window is read.

## 4. The policy ladder

Each rung adds exactly one mechanism, so each is separately priced.

| Policy | State | Merge identity | Elaborate / supersede / revise | Commitment | Causal |
| --- | --- | --- | --- | --- | --- |
| `window-only` | — | — | — | — | yes |
| `append-only` | yes | — | — | eager | yes |
| `dsg-merge` | yes | yes | — | eager | yes |
| `dsg-eager` | yes | yes | yes | eager | yes |
| `dsg-full` | yes | yes | yes | **deferred** | yes |
| `retrospective` | yes | yes | yes | deferred | **no** |

`window-only` retains what it extracted but cannot *match* against it, so its
final graph is an honest union of per-window extractions rather than an empty
state. `retrospective` gets one advantage only — it may see every mention and
identity link before committing to an entity inventory — which prices causality
while holding extraction fixed.

## 5. Complexity

Per step the system touches only the entities mentioned in the current window
and the assertions on their slots. Memory is bounded by cast size, not book
length; the model's context is one window plus a fixed-size digest. Cost is
**linear in book length with constant context**, against the quadratic
attention cost of re-reading a growing prefix.

The one super-linear term is invariant I1, which is pairwise within a slot.
That is why the bounded `inconsistent_slot_rate` is reported as the headline
and the raw count as a companion.

## 6. Claim → evidence map

| Claim | Evidence | Where |
| --- | --- | --- |
| Persistent state buys accuracy | `append-only` − `window-only`: +0.088 CoNLL, +0.310 mention linking | `artifacts/report/pdnc-qwen3b/results.md` |
| …and manufactures contradiction | same contrast: +0.323 inconsistent slots | ″ |
| Fact revision is the antidote | `dsg-eager` − `dsg-merge`: −0.433 inconsistent slots, 0–28 books | ″ |
| …at no accuracy cost | same contrast: 0.000 on both accuracy metrics | ″ |
| Merging without revision is harmful | `dsg-merge` − `append-only`: −0.016 mention linking | ″ |
| Deferral converts rollbacks to monotone updates | `dsg-full` − `dsg-eager`: −2.57 rollbacks (3B), −5.50 (1.5B, 0–28), −0.60 (LitBank) | all three reports |
| Causality has a price | `retrospective` − `dsg-full`: +0.031 mention linking (3B), +0.050 (1.5B) | ″ |
| No identity-accuracy gain (**negative result**) | `dsg-full` − `append-only`: CI spans zero at every model size | `docs/FINDINGS.md` §2 F1 |
| Elaboration is front-loaded in narrative order | ρ = −0.59 in narrative order, +0.08 shuffled | `docs/FINDINGS.md` §2 F4 |
| The revision trend is mostly state growth | ρ = +0.82 narrative vs +0.69 shuffled | ″ |

## 7. The writing system

Reading is half the system. The other half uses the state as a constraint on
generation, and it is where the project's claim lives: **a story can get longer
without breaking its own graph.**

### 7.1 Four stages

```
 Gutenberg fiction ──► state-conditioned dataset ──► LoRA ──► guarded inference
   215 novels             (state_{<t}, beat_t)       writer      write · read back
   chapters               -> chapter_t               learns      · ask the graph
                          built by the calculus      to use it   · rewrite if broken
```

1. **Corpus.** English Gutenberg filtered to fiction at novel length by
   catalogue subject, cut at real chapter headings with a paragraph-block
   fallback (`dsg/data/gutenberg.py`).
2. **Dataset.** One example is *(the graph state a reader would hold after
   chapters 1..t-1, a one-line brief for chapter t) → chapter t*. The state is
   built causally, by the same calculus the reading study measured, so the model
   is trained on exactly the object it is conditioned on at inference
   (`dsg/infra/modal_traindata.py`).
3. **Training.** LoRA with the prompt masked and the split taken **by book**, so
   no novel appears on both sides (`dsg/infra/modal_train.py`).
4. **Guarded inference.** Write the chapter, read it back, put every fact it
   proposes to the state as a dry run, and reject the chapter if it would break
   an established immutable fact (`dsg/generate/repair.py`).

### 7.2 The guard

The guard is the part that makes "without breaking the graph" a mechanism
rather than a hope. After a chapter is written it is read back and each
proposed fact is classified by `classify_only` — the *same* classifier the
update path uses, so the guard cannot drift from the calculus it guards.

A proposal that would force a `REVISE` on an **immutable** predicate is, by the
calculus's own definition, not the world changing but the text contradicting
what it already established: eye colour, material, kinship and birthplace do
not change within a story. That is the rejection signal. A `SUPERSEDE` on a
mutable predicate — a character moving, a mood shifting — passes untouched,
because that is the story advancing.

The chapter goes back once with the contradiction named ("already established
as eye colour grey, but this chapter says blue"). Attempts and acceptances are
counted separately, so the guard's hit rate is reported rather than folded into
the headline.

### 7.3 One prompt

Training and every inference condition share `writer_prompt`
(`dsg/generate/prompt.py`). If they disagreed on the shape of the ask, a
fine-tuned writer would be evaluated out of distribution and the comparison
would measure formatting drift rather than method.

### 7.4 The evaluation

Generated stories have no gold, so the gold is planted: each premise fixes
canon facts from closed vocabularies whose contradictions are enumerable, and a
violation is a deterministic string test near a mention of its subject. Canon
is stated in chapter 1 only, so the experiment tests memory rather than
prompt-following.

Conditions are **(backbone variant × memory)** pairs served from one vLLM
instance via `LoRARequest`, so base and tuned answer identical prompts within
the same story. The memory effect reads down a variant, the training effect
across variants at the same memory, neither confounding the other.

| | none | last-chapter | rolling-summary | full-context | append-only-state | dsg-state | dsg-repair |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **base** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **tuned** | | | | ✓ | ✓ | ✓ | ✓ |

Reported next to accuracy: **context tokens per chapter**. A digest does not
grow with the story and a transcript does, which is what makes the comparison
against `full-context` a question of scaling rather than only of accuracy.

### 7.5 Operating on an unreliable connection

Jobs are deliberately **not** detached: losing the client kills the container
and stops the billing. Every stage checkpoints as it goes — the dataset build
caches each extraction so a resume replays it on CPU rather than re-billing the
GPU, the trainer writes the adapter every `save_every` steps, generation
checkpoints after every chapter — and everything is mirrored to the Hugging
Face Hub. `scripts/modal_control.sh` gives status, stop and fetch.

## 8. What is missing for a top-venue submission

Ordered by how likely a reviewer is to demand it.

1. **Mention-level identity gold.** The alias-partition metric is defensible but
   indirect. BookCoref supplies real book-scale mention clusters; the loader
   needs the Gutenberg-text reconstruction step (`data/bookcoref/bookcoref.py`
   + `gutenberg_delta.json`). This is the single highest-value addition.
2. **An evaluation of the `SUPERSEDE`/`REVISE` decision itself.** The split is
   the paper's conceptual centre and is currently unvalidated: we measure what
   the calculus *does*, not whether its classifications are right. Needs a small
   annotated set of conflict pairs — a few hundred, feasible by hand.
3. **A stronger extractor.** Several planes are extraction-limited. A 14B run is
   scripted and unrun; an alternative is to feed gold mentions and isolate the
   representation entirely.
4. **A second model family.** Everything is Qwen2.5. Llama or Mistral at one
   size would establish the results are not family-specific.
5. **Downstream utility.** The state is evaluated for correctness but never
   *used*. Conditioning generation on it, or answering a multi-hop QA set from
   it, would connect this line to the narrative-memory literature (§7 of
   `RELATED_WORK.md`) on that literature's own terms.
6. **Human evaluation of revision events.** Do readers agree that a flagged
   `REVISE` is a correction rather than a change? Cheap, and it is what turns
   the instrument section from suggestive into evidence.
7. **Non-English text.** PDNC and LitBank are English. The title-aware matching
   rules are explicitly English/19th-century-fiction shaped and would need
   rethinking elsewhere — worth stating as a limitation regardless.
