# Premature Commitment, Not Incrementality: Revision-Aware Narrative State for Book-Length Fiction

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

When such pipelines are made incremental, the structure degrades, and the
degradation is usually attributed to incrementality itself: less context, worse
extraction. **We argue the mechanism is different. The damage is done by
premature commitment** — writing a decision into the graph as settled when the
text has not settled it. An append-only store cannot express "this was true
then", "this was never true", or "these two people are one", so the errors it
makes are permanent and they compound.

We make three contributions.

1. **A revision calculus for narrative state** (§3) that separates three update
   types append-only pipelines conflate: monotone `ELABORATE` of deliberately
   underspecified structure, history-preserving `SUPERSEDE` of a changed world,
   and non-monotone `REVISE` of a mistaken reader belief. The
   `SUPERSEDE`/`REVISE` split is the operational form of the narratological
   distinction between an event in the story and a disclosure in the telling.
2. **An implemented, prefix-causal system** (§4) in which a small open-weight
   model *proposes* typed updates over a bounded window and a deterministic
   reconciler *disposes* — classifying, resolving identity, and enforcing seven
   structural invariants after every reading step. Memory is bounded by cast
   size, not book length.
3. **A gold-scored evaluation protocol for time-indexed state** (§5), with no
   LLM in any primary metric: identity against human alias annotation over 28
   novels, speaker attribution over 37,131 human-annotated quotations, and
   process-integrity measures over discourse position.

## 2. Related work

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

**Coreference at book scale.** LitBank (Bamman et al., 2019) and BookCoref
(Martinelli et al., 2025) established gold coreference over literary text, the
latter at full-book length. Both are evaluated non-causally: a system may see
the whole document. We reuse this style of annotation under a prefix-causal
protocol, which changes the task rather than the data.

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

- Bamman, Popat & Shen (2019). An Annotated Dataset of Literary Entities. NAACL.
- Chen (2026). Distinguishing Revision and Delayed Elaboration in Incremental
  Narrative Interpretation. CMN. arXiv:2608.21364.
- Martinelli, Bonomo, Huguet Cabot & Navigli (2025). BookCoref: Coreference
  Resolution at Book Scale. ACL.
- Saifullah et al. (2026). Narrative World Model: Narratology-Grounded Writer
  Memory for Long-Form Fiction. arXiv:2607.05577.
- Vishnubhotla, Hammond & Hirst (2022). The Project Dialogism Novel Corpus.
  LREC.
