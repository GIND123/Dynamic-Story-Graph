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

### C2 — Beat-conditioned retrieval — **TESTED, AND IT FAILS**

Instead of serialising the whole state, retrieve only what the *next beat*
implicates: the entities it names, their immutable canon, and their current
mutable state. Everything else is withheld.

The prediction was that querying the state would succeed where serialising it
failed. It did not. Beat-conditioned retrieval scores **0.571** — worse than
giving the writer *no memory at all* (0.487; Δ +0.083, CI [+0.038, +0.125],
losing on 17 of 30 stories) and statistically indistinguishable from the
serialised digest it was meant to fix (0.588).

Retrieval plus the previous chapter (0.317) is indistinguishable from the
previous chapter alone (0.312; Δ +0.004, CI spans zero, 8–7). The graph
contributes nothing a writer needs.

**This kills the conditioning line, and it does so cleanly.** Pre-registered
falsifier 1 fired: the failure is the state, not its serialisation.

The decisive detail is that it is **not** an extraction ceiling. In this run the
state held **0.492** of the planted canon — half of it, correctly, at the moment
of writing — and conditioning on it still added nothing. Fixing extraction would
not have rescued this.

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

## 3a. Outcome: four conditioning designs, three runs, 12,600 chapters

| memory | violations ↓ | tokens | verdict |
| --- | --- | --- | --- |
| none | 0.487 | 112 | floor |
| previous chapter | 0.312 | 1,017 | **strong, cheap incumbent** |
| full transcript | 0.254 | 5,362 | best, at 5× the context |
| serialised digest | 0.588 | 357 | worse than nothing |
| digest + previous | 0.379 | 1,194 | loses to previous alone |
| beat retrieval | 0.571 | 212 | worse than nothing |
| beat retrieval + previous | 0.317 | 1,098 | ties previous alone |
| + graph guard | 0.317 | 1,095 | no effect (ns) |

Every design that puts narrative state into the writer's prompt either loses to
a baseline or ties one that is cheaper. The guard never moves a number. The
conclusion is not "needs more work" — it is that this is the wrong use of the
object.

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

The branch resolved to the second one, and it resolved cleanly:

> **A maintained narrative state is a measurement instrument, not a
> conditioning signal.**

That is a defensible finding rather than a consolation. It rests on four
conditioning designs across three runs and 12,600 generated chapters, each
falsifier stated before the run; on a diagnosis that rules out the obvious
escape (capture was 0.492 and it still did not help); and on an independent
replication of a published negative — the Narrative World Model's serialised
`State Memory` condition — obtained with a 3B open model instead of a frontier
API.

What survives, and what a submission should be built on:

- **C1, the measurement instrument.** 0.000 self-contradictory slots against
  0.370–0.423, at three model sizes, scored against human annotation with no
  LLM in the metric.
- **C3, the benchmark.** Planted canon plus premise adherence, no annotation and
  no judge, already demonstrated to catch a degenerate model that scored a
  spurious 7× win by ceasing to write about the story.
- **The negative itself**, which is worth reporting because the field is
  actively building state-conditioned writers and this is evidence, with
  intervals, that the obvious version does not work.

What is *not* supported, and should not be claimed: that a dynamic story graph
improves long-form generation. It does not, in any form tested here.
