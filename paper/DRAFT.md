# Memory Without Revision: Persistent Narrative State Is Self-Poisoning

*Working title. The original framing -- "premature commitment, not
incrementality" -- predicted an identity-accuracy gain the experiments did
not find; that was a pre-registered falsifier and it fired. The claim below
follows the result instead.*

*Draft. Results sections are filled from `artifacts/report/*/results.md`; every
number in this file is generated, not typed.*

---

## Abstract

*(to be written against final numbers)*

## 1. Introduction

A reader's model of a novel is built forward, under a hard constraint: at any
point they know only what they have been told. Fiction is written for that
constraint and exploits it. Identity is withheld and later disclosed; beliefs
are planted and later corrected; the world changes while what was true earlier
remains true of its moment.

Computational narrative understanding mostly ignores the constraint. Entity and
event extraction over novels is run *retrospectively*, over the finished text,
with unrestricted access to the ending while annotating the beginning. That is
the right design for producing a static description of a book. It is the wrong
design for any system that must read as a reader does — a generation assistant
that must not know chapter 40 while drafting chapter 5, a model of suspense or
surprise, or any measurement of what a text makes available to a reader *when*.

When such pipelines are made incremental the structure degrades, which is
usually attributed to incrementality itself: less context, worse extraction.
Our measurements say something more specific and less comfortable. **Carrying
state forward is both necessary and self-poisoning.** Across 28 novels, a
system that remembers earlier windows answers 31 points more mention-linking
questions correctly than one that reads each window in isolation — and at the
same time turns a third of its own state self-contradictory (+0.32 in the share
of single-valued slots holding rival live values). Memory buys accuracy and
manufactures contradiction in the same motion.

An append-only store cannot say "this was true then", "this was never true", or
"these two people are one", so every such contradiction it creates is
permanent. Giving it those operations removes the contradiction entirely — on
every one of 28 books — **at no measurable cost in accuracy**. What revision
buys is not a better answer; it is a state that still means something after
500,000 characters.

We report the negative half of this plainly: the revision calculus does **not**
improve identity accuracy at any model size we tested. That was a
pre-registered falsifier, and it fired.

We make three contributions.

1. **A revision calculus for narrative state** (§3) that separates three update
   types append-only pipelines conflate: monotone `ELABORATE` of deliberately
   underspecified structure, history-preserving `SUPERSEDE` of a changed world,
   and non-monotone `REVISE` of a mistaken reader belief. The
   `SUPERSEDE`/`REVISE` split is the operational form of the narratological
   distinction between an event in the story and a disclosure in the telling.
   The mechanisms are separable and we price each one: identity merging
   *without* fact revision makes the state strictly worse, so the two are
   complementary rather than additive.
2. **An implemented, prefix-causal system** (§4) in which a small open-weight
   model *proposes* typed updates over a bounded window and a deterministic
   reconciler *disposes* — classifying, resolving identity, and enforcing seven
   structural invariants after every reading step. Memory is bounded by cast
   size, not book length.
3. **A gold-scored evaluation protocol for time-indexed state** (§5), with no
   LLM in any primary metric: identity against human alias annotation over 28
   novels, mention linking probed at the reading position where each mention
   occurs, speaker attribution over 37,131 human-annotated quotations, and
   process-integrity measures. Reading forward is itself priced against a
   non-causal oracle given the same extraction: causality costs 3.1 points of
   mention accuracy at 3B and 5.0 at 1.5B.

Every trend against discourse position is reported with a shuffled-window
control, because state growth alone manufactures such trends — a control that
overturned our first reading of the revision profile (§7).

## 2. Related work

*Full survey with per-work positioning: [`RELATED_WORK.md`](RELATED_WORK.md).
Architecture and claim-to-evidence map: [`ARCHITECTURE.md`](ARCHITECTURE.md).*

**Incremental interpretation.** Chen (2026) draws the distinction between
revision-driven update and delayed elaboration, and argues that representing
early structure as underspecified converts rollbacks into monotone refinement.
That paper is explicitly conceptual: one worked seven-panel example, no
implementation, no corpus, no metrics. Its stated future work asks for exactly
four things — algorithmic criteria for classifying updates, corpus-scale
measurement of revision versus elaboration frequency, a hybrid architecture in
which a neural component proposes and a symbolic layer manages commitment, and
process-oriented metrics such as rollback frequency and the proportion of
monotone refinements. This paper supplies all four.

**Narrative memory for generation.** Narrative World Model (Saifullah et al.,
2026) pairs a narratology-typed temporal graph with query-conditioned retrieval
and beats generic temporal-KG agent memory (Graphiti/Zep) on multi-hop writer
QA. Its evidence is downstream: a frontier model reads each system's evidence
and is scored by another frontier model, on a partly proprietary corpus; its own
ablation finds the gain comes from retrieval conditioning rather than from the
typed representation. Our object of study is different — the *correctness of the
state itself*, measured against human annotation — and our constraint is
different: a 1.5B–14B open-weight model, no API anywhere in the pipeline.

**A versioned narrative world graph already exists.** Shadow-Loom (Wilmot,
2026) builds the closest architectural cousin: a typed, *versioned* world graph
with fabula and syuzhet indices on every node, beliefs carrying provenance, and
the language model confined to extraction, rendering and audit while a symbolic
core does the reasoning. It is, by its author's description, "offered as a
research artefact rather than as a benchmarked NLP model" — twenty hand-authored
fixtures, no quantitative evaluation — and its ingestion is a multi-pass
extraction over the *whole* text. Its versioning also serves a different
purpose: counterfactual forks ("what if Macbeth refuses?"), not correction of a
reader's belief as the reader acquires it. What is left open is precisely what
we do: build under a prefix-causal constraint, and measure.

**Coreference at book scale.** LitBank (Bamman et al., 2019) and BookCoref
(Martinelli et al., 2025) established gold coreference over literary text, the
latter at full-book length. Both are evaluated non-causally: a system may see
the whole document. We reuse this style of annotation under a prefix-causal
protocol, which changes the task rather than the data.

A separate line makes coreference *incremental* for efficiency — a fixed-size
cache with LRU or dual-cache eviction (Xia et al., 2020; Toshniwal et al., 2020;
Guo et al., 2023), a lightweight Transformer over historical clusters
(Martinelli et al., 2024), and memory-budgeted variants reaching 81.3 CoNLL F1
on LitBank (Luo et al., 2025). We take pains not to invite a comparison we
would lose and that would not mean anything: those are supervised coreference
systems scored on mention clusters, while our identity plane is unsupervised
and scored on a surface-string partition over gold alias sets. The word
"incremental" also differs. There it means *streaming under a memory budget*,
and the operative decision is eviction. Here it means *epistemically
constrained* — the system must not know what it has not read — and the
operative decision is commitment.

**Entity state tracking.** Small language models track entities in short
naturalistic narratives well (2026). Our question begins where that one ends:
what happens across 500,000 characters, when tracking must persist in an
external store that later text can contradict.

**Belief revision.** AGM revision and truth-maintenance systems formalise
non-monotone update under contradiction. Narrative adds a distinction those
frameworks do not make, because it does not arise in a static knowledge base:
a conflict may mean the world moved rather than that a belief was wrong.

## 3. The revision calculus

*(see `docs/METHOD.md` for the full specification; §3 of the paper reproduces
it with the discourse/story-time diagram)*

## 4. System

*(architecture, the propose/dispose split, deferred commitment, invariants)*

## 5. Experimental design

*(the policy ladder, shared-extraction control, three evaluation planes,
length control, model-scale ablation, paired bootstrap over books)*

## 6. Results

*(generated)*

## 7. The revision trace as a measurement instrument

*(where in a book the reader's model is refined versus corrected)*

## 8. Limitations

*(stated in advance in `docs/RESEARCH_PLAN.md` §5, reproduced with outcomes)*

## 9. Conclusion

## References

Core positioning:

- Chen (2026). Distinguishing Revision and Delayed Elaboration in Incremental
  Narrative Interpretation. CMN. arXiv:2608.21364.
- Wilmot (2026). Shadow-Loom: Causal Reasoning over Graphical World Model of
  Narratives. arXiv:2605.02475.
- Saifullah et al. (2026). Narrative World Model: Narratology-Grounded Writer
  Memory for Long-Form Fiction. arXiv:2607.05577.

Data and annotation:

- Bamman, Popat & Shen (2019). An Annotated Dataset of Literary Entities. NAACL.
- Bamman, Lewke & Mansoor (2020). An Annotated Dataset of Coreference in English
  Literature. LREC.
- Vishnubhotla, Hammond & Hirst (2022). The Project Dialogism Novel Corpus.
  LREC.
- Martinelli, Bonomo, Huguet Cabot & Navigli (2025). BookCoref: Coreference
  Resolution at Book Scale. ACL. arXiv:2507.12075.

Incremental coreference:

- Xia et al. (2020); Toshniwal et al. (2020). Incremental clustering for
  coreference.
- Guo et al. (2023). Dual cache for long-document neural coreference. ACL.
- Martinelli et al. (2024). Maverick: efficient coreference with a lightweight
  Transformer.
- Luo et al. (2025). MEIC-DT: Memory-Efficient Incremental Clustering.
  arXiv:2512.24711.

Foundations:

- Alchourrón, Gärdenfors & Makinson (1985). On the Logic of Theory Change. JSL.
- Doyle (1979). A Truth Maintenance System. Artificial Intelligence.
- Kamp (1981); Kamp & Reyle (1993). Discourse Representation Theory.
- Zwaan & Radvansky (1998). Situation Models in Language Comprehension and
  Memory. Psychological Bulletin.
- Genette (1980). Narrative Discourse: An Essay in Method.
- Sternberg (1992). Telling in Time (II): Chronology, Teleology, Narrativity.
- Liu et al. (2024). Lost in the Middle. TACL.
- Bamman, Chang, Lucy & Zhou (2024). On Classification with Large Language
  Models in Cultural Analytics. arXiv:2410.12029.

Metrics:

- Vilain et al. (1995) MUC; Bagga & Baldwin (1998) B³; Luo (2005) CEAF.

*The full annotated survey, including which entries were read directly and
which still need checking against the published version, is in
[`RELATED_WORK.md`](RELATED_WORK.md).*
