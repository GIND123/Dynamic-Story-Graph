# A methodology for dynamic narrative state in long-form generation

What the evidence supports, what it rules out, and the one design the evidence
points at that has not yet been tested.

---

## 1. The situation, stated honestly

Two things are established here, on 9,000 generated chapters and 28 annotated
novels:

**Established positive.** A revision-aware narrative state, maintained under an
explicit update calculus, holds **0.000** self-contradictory slots against
**0.370–0.423** for an append-only store — replicated at 1.5B, 3B and 7B,
scored against human annotation, with no LLM in the metric.

**Established negative.** Serialising that state into the writer's prompt does
not help generation. Alone it is *worse than no memory at all* (+0.083
violation rate, CI excludes zero); added to recent text it still loses to
simply pasting the previous chapter (+0.071, CI excludes zero).

The negative is not an isolated accident. It **independently replicates a known
result**: the Narrative World Model paper's own ablation found that serialising
current state scored 0.358 while *query-conditioned retrieval over the same
state* scored 0.898. Their "State Memory" condition is, structurally, exactly
our `dsg-state`. We reproduced their failure mode without having tested their
success mode.

That is the opening.

## 2. The claim worth making

> **A narrative state is not a context to carry. It is an index to query and a
> constraint to check.**

Three consequences, each an independent contribution:

### C1 — The state as a measurement instrument (proven)

The calculus already does something no LLM judge can: it decides, from an
inspectable lexicon rather than a model's opinion, whether two assertions
conflict — and distinguishes *the world changed* (`SUPERSEDE`) from *the text
contradicted itself* (`REVISE`). Immutable predicates cannot change within a
story, so a `REVISE` on one **is** a continuity error, by definition rather
than by judgement.

This yields a consistency metric for *any* long text, generated or written,
that needs no annotation and no judge. That is the contribution the evidence
most strongly supports, and it is a measurement contribution rather than a
leaderboard one.

### C2 — Beat-conditioned retrieval (untested; prior work predicts it wins)

Instead of serialising the whole state, retrieve only what the *next beat*
implicates: the entities it names, their immutable canon, and their current
mutable state. Everything else is withheld.

Why this should work where serialisation failed:
- A serialised digest crowds the prompt with facts irrelevant to the scene
  being written, and our own data shows the cost is real — the digest carries a
  third of the canon while occupying the space that recent text would use.
- A beat mentions two or three characters. Their canon is perhaps ten facts.
  That fits in ~80 tokens against the digest's ~360 and the transcript's ~5,400.
- It is the one cell of the design space that both this project's evidence and
  the prior literature point at, and that neither has run.

### C3 — A benchmark with no annotation and no judge

Planted canon drawn from closed vocabularies makes a continuity violation a
deterministic string test near a mention of its subject. Paired with
`on_premise` — does the chapter write about the story it was asked for — it
resists the failure mode that nearly produced a false positive here: a
degenerate model scored a 7× "improvement" purely by ceasing to mention the
characters.

The corpus is synthetic by necessity (controlled gold), which is the honest
limitation; the *method* transfers to any premise with checkable attributes.

## 3. Experimental design

One backbone, one GPU, conditions paired within story so every comparison is
like-for-like.

| condition | memory given for chapter *t* |
| --- | --- |
| `none` | premise + beat |
| `previous-chapter` | + the previous chapter (**the incumbent to beat: 0.308**) |
| `full-transcript` | + everything so far, truncated (0.329) |
| `state-digest` | + the serialised state (0.588 — the known-failing cell) |
| `digest + previous` | + both (0.379) |
| **`beat-retrieval`** | **+ only the canon of entities the beat implicates** |
| **`beat-retrieval + previous`** | **+ that, and the previous chapter** |
| **`… + guard`** | **+ the graph may reject and demand a rewrite** |

**Primary metric.** Cumulative canon violation by chapter — monotone, since
continuity once broken stays broken.
**Guard metrics.** `on_premise` (a condition that stops writing about the story
cannot be credited), `restatement_rate`, `canon_capture`.
**Cost metric.** Prompt tokens per chapter, which is where a retrieved slice
should dominate a growing transcript.
**Statistics.** Paired bootstrap over stories, effect sizes with intervals, no
reliance on p-values at n=30.

### Pre-registered falsifiers

1. `beat-retrieval` ≈ `state-digest` → the failure is the *state*, not its
   serialisation, and the conditioning line is finished.
2. `beat-retrieval + previous` ≤ `previous-chapter` → the graph adds nothing a
   writer needs; report it and stop.
3. The guard remains null → detection is too sparse to control generation, and
   extraction is the only remaining lever.
4. Any condition wins while `on_premise` falls → an artefact, not a result.

## 4. The binding constraint, named

Extraction. Capture by canon type at chapter 20:

| type | captured |
| --- | --- |
| eye colour | 0.50 |
| trade | 0.38 |
| hair | 0.33 |
| birthplace | 0.23 |
| **object material** | **0.12** |

The extractor is character-centric and largely ignores objects, so half the
planted canon never enters the state. No amount of work on the calculus
recovers that. If C2 fails, this is the only honest place left to work, and it
is a smaller, better-posed problem than anything downstream.

## 5. Why this is publishable either way

If beat-retrieval wins, the finding is that **narrative state must be queried,
not carried** — replicating a frontier-model result with a 3B open model, under
a strict prefix-causal constraint, with no LLM judge anywhere.

If it loses, the finding is that **maintained narrative state is a measurement
instrument, not a conditioning signal** — with a clean negative across three
conditioning designs, a diagnosed cause, and a reusable no-annotation benchmark
that catches the degenerate-model artefact.

The measurement contribution (C1) and the benchmark (C3) hold in both branches.
That is the part of this work that does not depend on a result going our way,
and it is the part I would build a submission around.
