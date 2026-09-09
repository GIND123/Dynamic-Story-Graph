# Structured narrative state: what it can and cannot do

A methodology built from what the evidence survived, not from what was hoped
for. Everything below rests on 28 human-annotated novels, 215 public-domain
novels, and 16,200 machine-generated chapters, with no LLM in any metric.

---

## 1. The thesis

> A revision-aware narrative state is a **representational** result, not a
> conditioning signal and not a text-level instrument. It keeps a maintained
> state coherent; it does not help a model write, and its contradiction count
> does not measure a text.

Three claims, each with its scope stated, and three pre-registered negatives
that bound them. The negatives are load-bearing: they are what make the
positive claim narrow enough to be true.

## 2. C1 — The revision calculus (positive, well evidenced)

**Claim.** Given a fixed stream of extracted assertions, an append-only store
turns **0.107–0.423** of its own single-valued slots self-contradictory — at
1.5B, 3B and 7B extractors, over 28 novels — and equipping the store with an
explicit update calculus removes that at **no measurable cost in accuracy**.

**What is measured, and what is not.** The append-only rate is the empirical
quantity. The calculus's own **0.000** is a *conformance check*: invariant I1 is
defined as two live conflicting values in one single-valued slot, and
`_close_and_link` in `dsg/store.py` closes every conflicting assertion before
adding a new one, so I1 = 0 is a postcondition enforced by construction. It
holds on all 184 book-runs, which is what a guarantee looks like rather than
what a measurement looks like. The load-bearing empirical result is the *cost*:
identity CoNLL F1 and mention linking do not move across the revision rung
(intervals spanning zero), against the natural expectation that aggressive
revision discards correct facts.

**Method.** Three update types that append-only pipelines conflate:

| operation | fires when | effect |
| --- | --- | --- |
| `ELABORATE` | later text specifies what earlier text left open | refined in place; nothing retracted |
| `SUPERSEDE` | the **story world** changed | validity interval closed; kept as history |
| `REVISE` | the **reader** was wrong; it was never true | retracted, and so is anything derived from it |

The `SUPERSEDE`/`REVISE` split is the operational form of a narratological
distinction — an event in the story versus a disclosure in the telling — and is
decided by an inspectable lexicon (predicate mutability, evidential source,
revelation markers), never by a model. Seven structural invariants are checked
after every reading step, including prefix causality: no assertion may cite text
the reader has not reached.

**Scope, stated precisely.** This is a claim about *representations given
identical input*, established by holding extraction constant and varying only
the update policy. It is **not** a claim that the resulting state is a faithful
model of the text (see §5).

**Controls.** Every policy replays a byte-identical cached proposal stream, so
differences are attributable to the representation rather than to extractor
variance. Each rung of the ladder adds exactly one mechanism, so each is priced
separately: identity merging, fact revision, deferred commitment, and the cost
of causality against a non-causal oracle.

## 3. C2 — A benchmark needing no annotation and no judge (positive)

**Problem.** Generated stories have no gold. Human judgement is expensive; an
LLM judge is the thing being avoided.

**Method.** Plant the gold. Each premise fixes canon facts drawn from closed
vocabularies whose contradictions are enumerable — eye colour, metal, kinship,
trade, birthplace. A violation becomes a deterministic string test near a
mention of its subject, with a nearby correction ("not gold but silver")
correctly not counted. Canon is stated in chapter 1 and withheld afterwards, so
the benchmark tests memory rather than prompt-following.

**The metric that makes it trustworthy.** `on_premise` — does the chapter write
about the story it was asked for. Violations are only counted near a mention of
their subject, so a model that stops writing about the premise scores a perfect
violation rate while producing nothing usable. This is not hypothetical: a
fine-tuned writer here scored **0.054** against **0.367** for the base model, an
apparent 7× win, purely by degenerating into pastiche of its training corpus.
`on_premise` exposed it (1.000 against 0.765–0.812, with canon restatement
0.84 against 0.09) and the arm was excluded.

**Why this is the durable contribution.** It is gold by construction, so unlike
§5 it does not depend on extraction quality at all.

## 4. C3 — State conditioning does not help generation (negative, well powered)

Four designs, three runs, 12,600 chapters, 30 stories, paired within story.

| memory | violations ↓ | tokens | verdict |
| --- | --- | --- | --- |
| none | 0.487 | 112 | floor |
| previous chapter | 0.312 | 1,017 | strong, cheap |
| full transcript | 0.254 | 5,362 | best, 5× the context |
| serialised state digest | 0.588 | 357 | **worse than nothing** |
| digest + previous chapter | 0.379 | 1,194 | loses to previous alone |
| beat-conditioned retrieval | 0.571 | 212 | **worse than nothing** |
| retrieval + previous chapter | 0.317 | 1,098 | ties previous alone |
| + graph guard | 0.317 | 1,095 | no effect (ns) |

Serialised state is worse than no memory (+0.083, CI [+0.025, +0.142]).
Beat-conditioned retrieval — the design the prior literature predicts should
win — is also worse than no memory (+0.083, CI [+0.038, +0.125]) and
indistinguishable from the serialisation it was meant to fix. Retrieval plus
recent text ties recent text alone (+0.004, ns). The guard never moves a number.

**This is not an extraction ceiling.** The state held **0.492** of the planted
canon at the moment of writing and conditioning on it still added nothing.

**It replicates a published negative.** The Narrative World Model paper's own
ablation reports serialised current state at 0.358 against query-conditioned
retrieval at 0.898. Our `state-digest` is structurally their `State Memory`. We
reproduce their failure with a 3B open model — and, unlike them, find that
their success condition does not transfer.

## 5. C4 — The contradiction count does not measure a text (negative, and it
constrains C1)

We tested our own instrument's external validity and it failed. That test is
reported because it bounds what C1 may claim.

**Design.** 215 published novels and 30 generated stories, first 20 chapters
each, read with the *same* extraction prompt, reconciler and policy. If the
contradiction rate measures textual consistency, professionally edited novels
should score far below machine-generated ones.

**Result.** They do not separate: AUC **0.394**, 95% CI [0.259, 0.534] — the
interval spans chance. Published novels score **0.238**, which is not credible
as a rate of genuine self-contradiction in edited prose.

**Diagnosis.** The flagged "contradictions" in real novels are extraction
artefacts. In *Peter Pan*: `Mrs. Darling.occupation: [wife, mother]` — both
true; `Nana.material: [Newfoundland dog, dog]` — the same thing at different
precision; `Wendy.resides_in: [14, nursery]` — a house number and a room.

**Correction applied.** `occupation` is genuinely multi-valued and was
reclassified; refinement detection was broadened so a more precise restatement
is not counted as a rival value. This lowered both populations (human
0.267→0.238) and still did not separate them. **Residual extraction noise
dominates the absolute level.**

**What this licenses and forbids.** C1 stands: it is a controlled comparison of
policies on identical input, and that comparison is unaffected by a shared noise
floor. What is *forbidden* is reporting the contradiction rate as a measure of
how consistent a text is. We do not.

## 6. Design principles worth stating

1. **Hold extraction constant.** Every policy comparison replays a
   byte-identical cached stream, so representation and extractor never confound.
2. **Gold by construction beats gold by judgement.** The planted-canon benchmark
   needs no annotation and no model, which is why it survived when the
   extraction-dependent metric did not.
3. **Every headline metric needs a denominator that can expose its artefact.**
   `on_premise` for violation rate; the human-vs-machine test for the
   contradiction rate. One caught a 7× false positive; the other invalidated a
   claim we wanted to make.
4. **Pre-register the falsifiers.** Four were stated before the generation runs;
   three fired, and are reported as such.
5. **Price causality.** A non-causal oracle given the same extraction bounds
   what reading forward costs, separately from what the method buys.

## 7. Honest positioning

This is a paper about the limits of a popular idea, with one solid positive
result and a reusable benchmark. It is not a system paper claiming better story
generation, because the evidence does not support one.

The field is actively building state-conditioned writers. Evidence with
intervals that four such designs fail — plus a metric that catches the
degenerate-model artefact which makes them look like they work, plus a
demonstration that the obvious consistency metric measures its own extractor —
is a contribution of the kind that saves other people months.

## 8. What would extend it

1. **Mention-level identity gold** (BookCoref) to replace the alias-set proxy.
2. **Validate the `SUPERSEDE`/`REVISE` decision itself** against a few hundred
   hand-annotated conflict pairs. It is the conceptual centre and the least
   evaluated part.
3. **A second model family**, to show the negatives are not Qwen-specific.
4. **Human validation of the planted-canon metric** — do readers agree a flagged
   violation reads as a continuity error?
