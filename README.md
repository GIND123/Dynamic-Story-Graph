# Dynamic Story Graph

**What a reader can know at page 200, and what it costs to only know that.**

Computational narrative understanding reads finished books. Extraction over a
novel is run *retrospectively*: a system annotating chapter 2 may already have
seen chapter 40. That is the right design for describing a book and the wrong
design for anything that must read as a reader does — a drafting assistant that
must not know the ending, a model of suspense or surprise, or any measurement of
what a text makes available to a reader *when*.

This repository asks what changes under a **prefix-causal** constraint: at
character `s`, a system may condition on `text[0:s]` and on state derived from
`text[0:s]`, and on nothing else. It is enforced in code, on every item, and
raises rather than warns.

231 tests pass, 20 skipped. No LLM judge appears in any primary metric. No paid
API is used anywhere in the pipeline. Every number below is generated from a
committed artifact by the command printed beside it.

---

## 1. The measurement that motivates the project

Quote attribution on the Project Dialogism Novel Corpus is a solved-looking task:
Llama-3 8B zero-shot reaches **90.6** overall and **89.1** on non-explicit quotes
(Michel, Epure, Hennequin & Cerisara 2024, arXiv:2406.11380, Table 1). Every
published number is obtained non-causally — the system reads a window *centred*
on the quote.

We asked where the evidence actually sits. For the 10,743 explicit quotes whose
gold speaker has resolvable aliases, we located the speaker's name within ±250
characters:

| position of the gold speaker's name | count | share |
|---|---|---|
| neither side (tag further away, or a different surface) | 3,471 | 32.3% |
| **after the quote only — destroyed by a causal cut** | **3,148** | **29.3%** |
| both sides | 2,600 | 24.2% |
| before only | 1,524 | 14.2% |

> **A strict prefix-causal cut removes the decisive lexical cue for 29.3% of
> explicit quotes — about 8.8% of the corpus.**

The benchmark is substantially easier than it looks because, for a large slice of
it, the answer arrives after the question. This is a property of the corpus, not
of any system, and it holds for every result ever published on it.

```
python -m dsg.eval.causal_audit --report tag-position
```

Three annotation defects surfaced only by running the guards against real PDNC.
None appears in the corpus documentation, and each would silently inflate a
causal number:

| defect | scale | consequence if unguarded |
|---|---|---|
| quotes annotated as several spans | 10,734 / 37,131 (**28.9%**) | the interjected `," said Elizabeth, "` is handed to the model as part of "the quote" |
| speech tags shorter than a shingle | every explicit tag (~20 chars vs 40) | a shingle-based leak check sees nothing and passes |
| spans not in document order | 5 / 37,131 | the quote's range inverts and the run crashes |

Effective N after dropping quotes whose gold speaker has no resolvable alias:
**35,083 of 37,131**, across 28 novels.

---

## 2. The first prefix-causal number on PDNC

Qwen2.5-7B-Instruct, 1,200-character window, greedy decoding, gold candidate
lists (matching the literature's protocol). Paired bootstrap over 28 novels,
10,000 resamples.

| condition | evidence available | overall | explicit | non-explicit |
|---|---|---|---|---|
| `prior` | candidate list only | 0.139 | 0.144 | 0.138 |
| `recency` | last attributed speaker before `s` | 0.165 | 0.165 | 0.171 |
| **`text-causal`** | `text[s−W:s]` | **0.523** | **0.497** | **0.537** |
| `state-causal`† | `text[s−W:s]` + serialised state at `s` | 0.450 | 0.424 | 0.464 |
| `oracle-noncausal`‡ | `text[s−W:s+W]`, target marked | 0.696 | 0.850 | 0.644 |

| comparison | Δ | 95% CI | excludes 0 |
|---|---|---|---|
| `text-causal` − `recency` | +0.3579 | [+0.3253, +0.3887] | yes |
| `recency` − `prior` | +0.0257 | [+0.0098, +0.0419] | yes |

**`text-causal` = 0.523 is the headline: the first prefix-causal attribution
number on PDNC**, on 35,083 quotes with leakage enforced per item.

† **`state-causal` supports no claim in this repository.** It scores −0.073
below `text-causal` (CI [−0.0919, −0.0555]), but adding the state block also
adds ~1,400 characters between the recent text and the question, so the
contrast confounds content with prompt length. The `state-shuffled` placebo that
would separate them is specified in
[`docs/METHODOLOGY_V2.md`](docs/METHODOLOGY_V2.md) §2.4 and **was not run — the
compute budget for this project is exhausted.** The number is published here for
completeness and is excluded from every conclusion.

‡ **`oracle-noncausal` is superseded.** This run's oracle was asked about "the
next line" while its context bracketed the quote. The fix is implemented; the
rerun will not happen for the same reason. **No "cost of causality" figure is
claimed from this row.**

### The evidence that the causal cut is what bites

Without a valid internal oracle, the cross-paper gap (0.523 against a published
0.906) confounds model with setting and cannot be read as a causality price. One
piece of evidence survives that confound, because it compares **orderings within
a condition** rather than numbers across papers.

§1 predicts an asymmetry: a causal cut destroys the decisive cue for 29.3% of
*explicit* quotes, while anaphoric and implicit quotes never had a nearby name
cue to lose. So the explicit slice — the easy one in every published setting —
should become the hard one under a causal constraint.

| setting | overall | non-explicit | which slice is easier |
|---|---|---|---|
| BookNLP+, non-causal `[published]` | 78.5 | 68.9 | explicit (overall > non-explicit) |
| Llama-3 8B zero-shot, non-causal `[published]` | 90.6 | 89.1 | explicit (overall > non-explicit) |
| **`text-causal`, this work** | **52.3** | **53.7** | **non-explicit — the ordering inverts** |

Both published systems score *below* their overall accuracy on non-explicit
quotes; ours scores *above* it. Each row is an internal comparison within one
system and one prompt, so the inversion is robust to the model differences that
make the absolute numbers incomparable. **The slice the corpus audit says should
break is the slice that breaks.**

### Leakage is enforced, not asserted

The causal claim lives or dies here, so the guards are code:

- **G1 — offset assertion.** Every prompt is built by one function that can only
  slice `text[:s]`. A 40-character shingle check plus a literal check for short
  regions runs on *every* quote and raises. The literal check exists because
  `," said Elizabeth, "` is ~20 characters and slips under a shingle.
  ([`dsg/eval/attribution.py`](dsg/eval/attribution.py))
- **G2 — no gold at inference.** Gold-candidate and system-candidate regimes are
  reported separately and never mixed in one table.
- **G3 — memorization covariate.** Name cloze run with our own model over PDNC,
  correlated per novel with attribution accuracy.
- **G6 — provenance, not substrings.** The state digest is written by the
  extractor, so no substring check can vouch for it. Each snapshot carries the
  window index and offset it was taken at and is rejected if its coverage reaches
  past the quote. The window *containing* the quote is always excluded.
- **G4 — book-level splits only.** **G5 — no LLM judge**; exact match against
  gold after alias normalisation.

### The memorization check, and what it actually shows

Qwen has read these public-domain novels. Per novel, ρ = **+0.411** between name
cloze and attribution accuracy (permutation p = 0.030) — which looks like
contamination. But mean cloze is **0.054** (the model can barely recall these
books), and `text-causal` correlates *equally* (ρ = 0.403). Equal correlation
with and without state points at **book tractability** — famous, small-cast
novels are both easier to cloze and easier to attribute — rather than at
memorization. Reported as a complication that wants a difficulty control, not
explained away.

---

## 3. Persistent narrative state is self-poisoning

The reading study, on 28 PDNC novels and 100 LitBank excerpts, at three extractor
scales. Extraction is **held constant**: every policy replays a byte-identical
cached proposal stream, so differences are attributable to the representation and
not to extractor variance.

Carrying state forward buys a great deal and breaks the state in the same motion.
At 7B, `append-only` against `window-only`:

| metric | Δ | 95% CI | win split |
|---|---|---|---|
| identity CoNLL F1 | **+0.209** | [+0.182, +0.240] | 28–0 |
| mention linking | **+0.152** | [+0.124, +0.183] | 28–0 |
| self-contradictory slots | **+0.261** *(worse)* | [+0.233, +0.291] | 28–0 |

The accuracy gain is large at every scale but its shape moves, so the range is
reported rather than a single cell: CoNLL +0.209 (7B) / +0.088 (3B) / +0.063
(1.5B) / +0.081 (LitBank); mention linking +0.152 (7B) / +0.310 (3B) / +0.314
(1.5B).

The share of single-valued slots holding two or more live conflicting values:

| corpus / extractor | `window-only` | `append-only` |
|---|---|---|
| PDNC, Qwen2.5-7B | 0.110 | **0.371** |
| PDNC, Qwen2.5-3B | 0.101 | **0.423** |
| PDNC, Qwen2.5-1.5B‡ | 0.004 | **0.107** |
| LitBank, Qwen2.5-7B | 0.091 | **0.181** |

‡ Partly a yield comparison: the 1.5B extractor fails to produce parseable output
on 20.6% of windows (3,226/4,063), so its lower rates reflect a thinner state.

**An append-only store, given a real novel, turns 11–42% of its own slots
self-contradictory.** It cannot say "this was true then", "this was never true",
or "these two people are one", so every contradiction it creates is permanent.
That is the empirical result, and it is a measurement of what novels do to a
naive incremental pipeline.

### Reading the consistency number honestly

Policies equipped with the revision calculus report **0.000** self-contradictory
slots — on all 184 book-runs, at every scale, on both corpora. **This is a
conformance check, not an empirical finding, and we state it as one.**

Invariant I1 ([`dsg/invariants.py`](dsg/invariants.py)) is defined as "two live
conflicting values in one single-valued slot." `_close_and_link`
([`dsg/store.py`](dsg/store.py)) closes *every* conflicting assertion before
adding a new one. So I1 = 0 is a postcondition the calculus enforces by
construction; the metric and the mechanism are the same object, and 0/184 is what
a guarantee looks like rather than what a measurement looks like.

What that leaves is the question worth asking, and it is empirical:

> **What does enforcing consistency cost?**

The answer is nothing, and it was not obvious in advance — the natural worry is
that aggressive revision throws away correct facts.

| rung (7B) | identity CoNLL F1 | mention linking | self-contradictory slots |
|---|---|---|---|
| + identity merging | −0.002 (ns) | −0.005 (ns) | +0.007 *(worse)* |
| + fact revision | 0.000 (exact) | 0.000 (exact) | **−0.377** (0–28) |
| + deferred commitment | −0.003 (ns) | **+0.007** | 0.000 |
| cost of causality | −0.002 (ns) | **+0.028** | 0.000 |

Bold entries have a 95% interval excluding zero. The revision rung moves the
consistency metric by exactly the amount it is defined to move it and moves the
accuracy metrics by *exactly zero* — identity and fact revision are disjoint, and
the ladder shows it rather than assuming it.

Two findings on that ladder are genuinely empirical:

- **Identity merging without fact revision makes the state worse.** Merging two
  characters unifies their assertion slots; without revision the merged slot
  simply holds both rival values. Mention linking −0.016 at 3B and −0.023 at
  1.5B, both intervals excluding zero (not significant at 7B). The two mechanisms
  are complementary, not additive — merging is only safe once the state can
  revise.
- **Causality has a price, and it is small.** A non-causal oracle given the
  identical extraction is better on mention linking by +0.028 (7B, 25–2), +0.031
  (3B), +0.050 (1.5B, 28–0). Reading forward genuinely costs something. It is a
  few points, not a collapse, and it does not touch consistency.

---

## 4. Three negatives that bound the positive

These are reported at the same volume as the positive result, because they are
what make the positive claim narrow enough to be true.

### 4.1 State conditioning does not help generation

Five designs, four runs, **18,600 generated chapters**, 30 stories, conditions
paired within story. Canon is planted in chapter 1 and withheld afterwards;
violations are deterministic string tests near a mention of the subject. No
language model appears in any metric.

| memory at generation time | violations ↓ | prompt tokens |
|---|---|---|
| none | 0.487 | 112 |
| previous chapter | 0.312 | 1,017 |
| full transcript | 0.254 | 5,362 |
| serialised state digest | **0.588** | 357 |
| digest + previous chapter | 0.379 | 1,194 |
| beat-conditioned retrieval | **0.571** | 212 |
| retrieval + previous chapter | 0.317 | 1,098 |
| + graph guard | 0.317 | 1,095 |

Serialised state is worse than no memory (+0.083, CI [+0.025, +0.142]).
Beat-conditioned retrieval — the design the prior literature predicts should win
— is also worse than no memory (+0.083, CI [+0.038, +0.125]).

**This is not an extraction ceiling.** The state held **0.492** of the planted
canon at the moment of writing, and conditioning on it still added nothing.

**It replicates a published negative.** The Narrative World Model paper's own
ablation reports serialised current state at 0.358 against query-conditioned
retrieval at 0.898. Our `state-digest` is structurally their `State Memory`. We
reproduce their failure with a 3B open model and, unlike them, find their success
condition does not transfer.

### 4.2 The contradiction count does not measure a text

We tested our own instrument's external validity and it failed. 215 published
novels and 30 generated stories, first 20 chapters each, same extraction prompt,
reconciler and policy. If the contradiction rate measured textual consistency,
edited novels should score far below machine-generated ones.

**They do not separate: AUC 0.394, 95% CI [0.259, 0.534]** — the interval spans
chance. Published novels score 0.238, which is not credible as a rate of genuine
self-contradiction in edited prose. The flagged "contradictions" are extraction
artefacts: in *Peter Pan*, `Mrs. Darling.occupation: [wife, mother]` (both true),
`Nana.material: [Newfoundland dog, dog]` (the same thing at two precisions),
`Wendy.resides_in: [14, nursery]` (a house number and a room).

This bounds §3: the ladder is a controlled comparison of policies on identical
input, unaffected by a shared noise floor. Reporting the contradiction rate as a
measure of how consistent a *text* is would not be, and we do not.

### 4.3 The revision trace is a weaker instrument than it first appeared

Revision events trend steeply with discourse position — from 4.6% of operations
in the first twentieth of a book to 29.8% in the last (ρ = +0.82, p = 9×10⁻⁶).
Read alone that is a tidy narratological story: exposition first, reversal later.

It is mostly an artefact. As a book proceeds the state holds more assertions, so
*any* conflict-driven operation becomes mechanically more likely. Permuting the
reading order destroys narrative order while preserving state growth exactly:

| ordering (3B) | elaboration ρ | revision ρ |
|---|---|---|
| narrative | **−0.59** (p = 0.007) | **+0.82** (p = 9×10⁻⁶) |
| shuffled | +0.08 (p = 0.75) | **+0.69** (p = 8×10⁻⁴) |

The revision trend largely survives shuffling, so most of it is state growth. The
elaboration trend disappears, so that one is real. **The control reversed our
first reading of this result**, and the surviving signal is in the opposite
operation from the one we expected to carry it.

Scope: this holds at 3B and **reverses at 7B** (narrative elaboration ρ = −0.084,
p = 0.72). The claim is restricted accordingly rather than generalised.

---

## 5. Method

A small open-weight model **proposes**; a deterministic reconciler **disposes**.
The model reads one window plus a bounded digest and emits typed proposals. It
has no write access. Whether a conflict is a world change or a reader error is
decided by an inspectable predicate table and marker list
([`dsg/lexicon.py`](dsg/lexicon.py)), identical across every condition.

Three update types that append-only pipelines conflate:

| operation | fires when | effect |
|---|---|---|
| `ELABORATE` | later text specifies what earlier text left open | refined in place; nothing retracted |
| `SUPERSEDE` | the **story world** changed | validity interval closed; kept as history |
| `REVISE` | the **reader** was wrong; it was never true | retracted, and so is anything derived from it |

The `SUPERSEDE`/`REVISE` split is the operational form of the narratological
distinction between an event in the story and a disclosure in the telling — a
distinction static belief revision (AGM, TMS) does not make, because a knowledge
base has no notion of the world legitimately moving on.

Assertions carry two independent clocks: discourse time (while the reader held
the belief) and story time (while the fact held in the world). Seven structural
invariants are checked after every reading step, including prefix causality.
Memory is bounded by cast size, not book length.

**Deferred commitment** is the central lever. A definite description (`the
stranger`) creates a *provisional* node, promoted only once it has a proper name
and two attestations. If "the stranger" commits on first sight, the later reveal
that they are Magwitch requires a non-monotone rollback; held provisional, the
identical reveal is monotone refinement.

Full specification: [`docs/METHOD.md`](docs/METHOD.md).

### Identity constraints adopted after measurement

Merging is irreversible in practice: a fused node matches both name sets and
attracts further merges, so one bad link cascades. Three defects found by reading
the output — not by a test — materially changed the numbers:

1. **Titles were stripped before comparing surfaces**, so `Mrs. Bennet` and
   `Miss Bennet` scored 0.9 against each other. In this corpus a title is the
   primary distinguisher between people sharing a surname.
2. **Nothing prevented merging two separately named characters.** On *Pride and
   Prejudice* this collapsed 74 gold characters into 3 nodes.
3. **The link *bind* path bypassed the merge guard**, so even `append-only` was
   receiving identity resolution it should not have had.

Fixing these moved conflation on *Pride and Prejudice* from 1.33 to 0.05.

The obvious fix for (2) was to require corroboration. **Measured before
adopting, and it does not work:** across 205 links on one novel, `Mr. Darcy →
Elizabeth` was proposed nine times, exactly as often as the correct `Mr. Bingley
→ Bingley`. Repetition does not separate good links from bad, so the constraint
had to be structural rather than statistical.

---

## 6. Controls, and what they cost us

Every headline in this repository is paired with the control that could have
killed it. Three of them did:

| control | what it killed |
|---|---|
| `on_premise` (does the chapter write about the story it was asked for) | A fine-tuned writer scoring **0.054** against 0.367 — an apparent 7× win. It had degenerated into pastiche of its training corpus, reproducing even Project Gutenberg's hard-wrap indentation. A model that stops writing about the premise cannot violate its canon. `on_premise` 1.000 → 0.765; canon restated 0.84 → 0.09. The arm was excluded from every conclusion. |
| blind best-of-4 selection arm | A **0.054** gain from graph-based candidate ranking. Ranking four candidates and picking blind scored the same: graph − blind = **+0.004**, CI [−0.058, +0.063]. The gain belonged to extra sampling, not to the graph. |
| cross-run spread | A **baseline ordering**. The full-transcript condition moves 0.146 across runs — larger than almost every effect in this project — so its ranking against "previous chapter" is not established, and was retracted. |
| shuffled reading order | Our first reading of the revision profile (§4.3). |
| human-vs-machine discrimination | The claim that the contradiction rate measures a text (§4.2). |

**The portable principle**, replicated twice in this repository two months apart
on different tasks: *a headline metric that a degenerate policy can saturate
requires a paired denominator, reported together, always.* A contradiction rate
of 0.000 achieved by flagging everything, and a violation rate of 0.054 achieved
by writing off-premise, are the same failure.

### Pre-registration is auditable

Falsifiers were committed at **2026-09-03 17:19** (`2934dce`,
[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) §5). The earliest result
artifact reached the Hub at **2026-09-04 04:18:59**; the generation runs ran
2026-09-04 20:29 → 2026-09-06 04:15. Two independent timestamped systems, plan
before results in both.

Of the four pre-registered falsifiers, **three fired.** The project's original
framing — "premature commitment, not incrementality" — predicted an
identity-accuracy gain the experiments did not find. The claim was rewritten to
follow the evidence.

---

## 7. What is not claimed

- **Not competitive with supervised coreference.** MEIC-DT reports 81.27 CoNLL
  on LitBank. Our identity plane is an unsupervised alias-string partition over a
  different item set; the comparison would not mean anything.
- **The revision calculus does not improve identity accuracy** at any scale
  tested. That was a pre-registered falsifier and it fired.
- **The 0.000 consistency figure is conformance, not evidence** (§3).
- **The contradiction rate does not measure textual consistency** (AUC 0.394).
- **No cost-of-causality figure is claimed *for attribution*.** The E1 oracle
  that would price it is superseded and will not be rerun (§2); the cross-paper
  gap confounds model with setting, so only the explicit/non-explicit *ordering
  inversion* is claimed there. This does not affect §3, where the `retrospective`
  oracle replays the identical cached proposal stream with full lookahead and the
  causality cost is measured internally.
- **`state-causal` − `text-causal` = −0.073 supports nothing** — the length
  placebo was not run and cannot be (§2).
- **No "first" claim** stands until every related-work entry is checked against
  its published version; [`paper/RELATED_WORK.md`](paper/RELATED_WORK.md) tracks
  which were read directly.
- **One model family, one language.** 28 English novels plus 100 LitBank
  excerpts, Qwen2.5 throughout.
- **`SUPERSEDE` vs `REVISE` is decided by a lexicon** whose accuracy has not
  itself been evaluated against annotation. It is the conceptual centre and the
  least evaluated part.
- The `manuscript-memory-engine` graph-vs-baseline comparison is **invalid** (the
  graph reads gold speaker labels while baselines read raw prose) and is cited
  nowhere.

---

## 8. Reproducing

```bash
python -m dsg doctor                      # environment check
python -m pytest tests/ -q                # 231 passed, 20 skipped
python -m dsg.eval.causal_audit --report tag-position   # §1
python -m dsg report                      # regenerates every table from results.json
```

Results are read from `artifacts/results/*/results.json` and rendered to
`artifacts/report/*/results.md`. Comparisons are paired bootstraps over books
(10,000 resamples, 95% intervals) with win/loss splits reported. Identity is
scored against PDNC's human alias annotation and LitBank's gold coreference;
mention linking against PDNC's annotated referring expressions.

Corpora, weights, and the Modal training path:
[`docs/SETUP.md`](docs/SETUP.md).

---

## 9. Layout

| path | contents |
|---|---|
| [`dsg/`](dsg/) | the study: store, calculus, invariants, policies, evaluation, generation |
| [`dsg/eval/attribution.py`](dsg/eval/attribution.py) | E1 prompts and the leakage guards |
| [`dsg/store.py`](dsg/store.py) · [`dsg/invariants.py`](dsg/invariants.py) | the revision calculus and the seven invariants |
| [`docs/METHOD.md`](docs/METHOD.md) | formal specification, independent of implementation |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) · [`docs/GENERATION_FINDINGS.md`](docs/GENERATION_FINDINGS.md) | full results with every interval |
| [`docs/METHODOLOGY_V2.md`](docs/METHODOLOGY_V2.md) | current design, E1/E2 protocol, known reporting defects |
| [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) | the pre-registered plan and its falsifiers |
| [`paper/`](paper/) | draft, related-work survey, architecture |
| [`artifacts/results/`](artifacts/results/) | committed `results.json` behind every table |
| [`gnsm/`](gnsm/) | LoRA training infrastructure (Modal) |
| [`manuscript-memory-engine/`](manuscript-memory-engine/) | earlier model failure-profiling harness |

---

## 10. Status and open work

**The compute budget for this project is exhausted.** Everything below reflects
a closed experimental record: no further runs are planned or possible, and the
scope is bounded accordingly rather than deferred to future work.

**Complete and standing.** The causal audit of PDNC and its three annotation
defects (§1, pure corpus analysis, no model); the first prefix-causal attribution
number on 35,083 quotes with per-item leakage enforcement (§2); the reading study
at 1.5B/3B/7B on PDNC and 7B on LitBank, 184 book-runs (§3); four generation
runs, 18,600 chapters (§4.1); the external-validity test on 215 novels vs 30
stories (§4.2); the shuffled-order control (§4.3).

**Permanently unresolved, and excluded from all claims.** The utility of a
serialised state at attribution time — `state-causal` cannot be separated from
its length confound without the `state-shuffled` placebo, and no internal cost of
causality can be quantified without a corrected oracle. Neither run will happen.
These are reported as limits of the study, not as pending work.

**Fixable without compute, and still open.**

1. **Six reporting defects** catalogued in
   [`docs/METHODOLOGY_V2.md`](docs/METHODOLOGY_V2.md) §5, including a figure
   provenance error (`fig6-revision-profile.png` is byte-identical to the LitBank
   figure while the prose beside it reports 3B numbers). All are editorial.
2. **Citation verification** for the 2026 preprints the positioning rests on.

**What a continuation would need.** Mention-level identity gold (BookCoref) to
replace the alias-set proxy; validation of the `SUPERSEDE`/`REVISE` decision
against a few hundred hand-annotated conflict pairs; a second model family to
show the negatives are not Qwen-specific; human validation that a flagged canon
violation reads as a continuity error. The corpus-level results in §1 require no
GPU and are the cheapest of these to extend.

---

## References

Michel, Epure, Hennequin & Cerisara (2024). *Evaluating LLMs for Quotation
Attribution in Literary Texts.* arXiv:2406.11380 ·
Vishnubhotla, Hammond & Hirst (2022). *The Project Dialogism Novel Corpus.* LREC ·
Bamman, Popat & Shen (2019). *An Annotated Dataset of Literary Entities.* NAACL ·
Bamman, Lewke & Mansoor (2020). *An Annotated Dataset of Coreference in English
Literature.* LREC ·
Chang, Cramer, Soni & Bamman (2023). *Speak, Memory: An Archaeology of Books
Known to ChatGPT/GPT-4.* EMNLP ·
Bamman, Chang, Lucy & Zhou (2024). *On Classification with Large Language Models
in Cultural Analytics.* arXiv:2410.12029 ·
Martinelli, Bonomo, Huguet Cabot & Navigli (2025). *BookCoref: Coreference
Resolution at Book Scale.* ACL ·
Alchourrón, Gärdenfors & Makinson (1985). *On the Logic of Theory Change.* JSL ·
Doyle (1979). *A Truth Maintenance System.* ·
Genette (1980). *Narrative Discourse.* ·
Sternberg (1992). *Telling in Time (II).*

Full annotated survey, with per-entry verification status:
[`paper/RELATED_WORK.md`](paper/RELATED_WORK.md).
