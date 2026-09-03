# Related work for Dynamic Story Graph

What each line of work establishes, and what it leaves open that DSG occupies.
Entries marked **[verified]** were read directly; the rest are cited from the
standard literature and should be checked against the published version before
submission.

DSG's one-sentence position: *no prior system builds narrative state under a
strict prefix-causal constraint with an implemented revision calculus, and none
scores such a state against human annotation as a function of reading position.*

---

## 1. Incremental interpretation and update dynamics

The closest conceptual antecedent, and the one the paper must engage first.

| Work | Establishes | Leaves open |
| --- | --- | --- |
| **Chen (2026), *Distinguishing Revision and Delayed Elaboration in Incremental Narrative Interpretation*, CMN'26, arXiv:2608.21364** **[verified]** | The revision / delayed-elaboration distinction; that representing early structure as *underspecified* converts non-monotone rollbacks into monotone refinement. | Explicitly conceptual: one worked seven-panel manga example, **no implementation, no corpus, no metrics**. Its own future work asks for (a) algorithmic criteria for classifying updates, (b) corpus-scale measurement of revision vs elaboration frequency, (c) a hybrid neural-propose / symbolic-commit architecture, (d) process metrics such as rollback frequency and proportion of monotone refinements. **DSG supplies all four.** |
| Kamp (1981); Kamp & Reyle (1993) — Discourse Representation Theory | Meaning as incremental context update; the DRS as a state that grows with discourse. | No account of *retraction* when later discourse corrects earlier commitment. |
| Asher & Lascarides (2003) — SDRT | Rhetorical structure constrains how new content attaches to the existing representation. | Attachment, not revision; no reader-model correction. |
| Copestake et al. (2005) — Minimal Recursion Semantics; underspecified DRT | Underspecification as a principled representational strategy: hold structure open, add constraints monotonically. | Sentence/discourse scope, not book scale; not evaluated as a *state* over a long text. |
| DeVault, Sagae & Traum (2011) — incremental dialogue interpretation | Partial hypotheses maintained and updated under streaming input. | Dialogue turns, not 500,000-character narratives. |

**DSG's move.** Take the underspecification idea from semantics, the revision
idea from belief revision, and make the *choice between them* an implemented,
inspectable decision — then measure how often each fires on real novels.

## 2. Belief revision and non-monotonic reasoning

| Work | Establishes | Leaves open |
| --- | --- | --- |
| Alchourrón, Gärdenfors & Makinson (1985) — AGM | Rationality postulates for revising a belief set under contradiction; expansion / revision / contraction. | AGM has no notion of the *world changing*. Every conflict is an error to be repaired. |
| Doyle (1979) — Truth Maintenance Systems; de Kleer (1986) — ATMS | Dependency-directed backtracking: retracting a belief retracts what was derived from it. | Same gap; also no time-indexing of belief. |
| Doutre et al. (2024), *What killed the cat? Towards a logical formalization of curiosity (and suspense, and surprise) in narratives*, arXiv:2410.08597 | Narrative tension formalised over **non-monotonic** reasoning — the reader's inferences are defeasible by later text. | Logical framework; not built, not measured on corpora. |

**The gap DSG names.** Static knowledge bases have one kind of conflict.
Narrative has two, and they demand opposite treatment: the world moved on
(`SUPERSEDE` — the earlier belief was true *of its moment* and must be kept as
history) versus the reader was misled (`REVISE` — the earlier belief was never
true and must be retracted with its dependents). No belief-revision framework
makes this distinction, because outside narrative it rarely arises.

## 3. Situation models: why prefix-causal state is the right object

| Work | Establishes |
| --- | --- |
| **Zwaan & Radvansky (1998), *Situation Models in Language Comprehension and Memory*, Psychological Bulletin** **[verified]** | The event-indexing model: readers track at least five dimensions — time, space, entity/protagonist, motivation, causation — and integrate a new event more easily when it shares indices with prior events. |
| Gernsbacher (1990) — Structure Building Framework | Comprehension as laying a foundation, mapping coherent incoming information onto it, and **shifting** to a new substructure when incoming information does not cohere. |
| Graesser, Singer & Trabasso (1994) — constructionist inference | Readers build explanatory structure online, not retrospectively. |

This literature is the warrant for the whole design: the object a reader
maintains is time-indexed, incremental, and revisable. `dsg/schemas.py` keeps
two clocks (discourse time and story time) for exactly the reason the
event-indexing model separates presentation from situation.

## 4. Coreference at document and book scale

**This section must be written carefully, because it is where DSG is weakest
and a reviewer will look first.**

| Work | Establishes |
| --- | --- |
| Bamman, Popat & Shen (2019) — LitBank entities; Bamman, Lewke & Mansoor (2020) — LitBank coreference **[verified via MEIC-DT]** | Gold entity and coreference annotation over 100 literary excerpts (~2,100 words, ~291 mentions each). |
| **Martinelli, Bonomo, Huguet Cabot & Navigli (2025), *BookCoref: Coreference Resolution at Book Scale*, ACL, arXiv:2507.12075** **[verified]** | Gold-annotated coreference over whole books (30K–67K tokens); manually annotated test set, automatically generated train set. |
| Xia et al. (2020); Toshniwal et al. (2020) — incremental coreference **[verified via MEIC-DT]** | Fixed-size cache + linear classifier; LRU eviction. The originating incremental-clustering line. |
| Guo et al. (2023) — dual cache **[verified via MEIC-DT]** | LRU+LFU dual-cache eviction for long documents. |
| Martinelli et al. (2024) — Maverick **[verified via MEIC-DT]** | Lightweight Transformer over all mentions of historical clusters; SOTA on long text. |
| **Luo et al. (2025), *MEIC-DT*, arXiv:2512.24711** **[verified]** | Memory-efficient incremental clustering under an explicit memory budget; **81.27 CoNLL Avg-F1 on LitBank**. |
| Le Bourgeois et al. (2025), *The Elephant in the Coreference Room*, arXiv:2510.15594 | Full-length French fiction coreference; long annotated documents remain scarce. |

**The honest statement DSG must make.** Supervised incremental coreference is a
mature line reaching ~81 CoNLL F1 on LitBank. DSG's identity plane is *not* a
coreference system and is not competitive with one: it is unsupervised, prompt-
driven, and scores a **surface-string partition over gold alias sets**, not
mention-level clusters. The numbers are not comparable and the paper must say
so rather than inviting the comparison. What DSG adds to this line is
orthogonal: identity is one plane of a state that also carries time-indexed
facts and an explicit revision history, and it is built under a prefix-causal
constraint that the coreference literature does not impose.

Note the shared vocabulary and the different meaning: incremental coreference
"incremental" means *streaming for efficiency*, with a cache evicted under LRU.
DSG's "incremental" means *epistemically constrained* — the system must not
know what it has not read. Eviction is a memory-budget decision; deferral is a
commitment decision. Worth one paragraph, because the words collide.

## 5. Entity and world-state tracking

| Work | Establishes | Leaves open |
| --- | --- | --- |
| Henaff et al. (2017) — Recurrent Entity Networks | Latent per-entity memory slots updated as text streams. | Latent and uninspectable; no revision semantics; short texts. |
| Dalvi et al. (2018) — ProPara; Tandon et al. (2018) | Explicit state change over procedural text, with annotation. | Process descriptions, not narrative; no reader/world distinction. |
| Ghosh et al. (2023), *PASTA*, arXiv:2208.00329 | Participant states in narratives, often inferred rather than stated. | Short narratives; no persistent store. |
| *Entity tracking emerges in sub-billion parameter LMs*, arXiv:2608.18083 | Small LMs do track entities in naturalistic narrative. | Short texts; state stays inside the model, so it cannot be audited or corrected. |
| Chain and Causal Attention, arXiv:2410.05565; Structured-Sparse Attention, arXiv:2605.22476 | Architectural limits and remedies for entity tracking in transformers. | In-weights tracking; no external, revisable state. |

**DSG's move.** Put the state *outside* the model, where it can be inspected,
constrained, and corrected — and then show that the constraint layer, not the
model, is what keeps it consistent.

## 6. Character-centric literary NLP

| Work | Establishes |
| --- | --- |
| **Vishnubhotla, Hammond & Hirst (2022), *The Project Dialogism Novel Corpus*, LREC** **[verified — used as our primary corpus]** | 28 novels with gold character alias sets, quotation spans, speakers, addressees, and annotated referring expressions with character offsets. |
| Bamman, Underwood & Smith (2014); Bamman, O'Connor & Smith (2013) | Latent character-type models over literary text. |
| Elson, Dames & McKeown (2010) | Social networks extracted from dialogue in novels. |
| BookNLP | Pipeline tooling for literary NLP: entities, quotation attribution, events. |

PDNC is what makes DSG's evaluation possible at all: it is the only corpus that
supplies full novel text, gold alias sets, *and* offset-anchored referring
expressions, which is what a prefix-causal probe requires.

## 7. Narrative memory for LLMs

The most crowded neighbourhood, and where DSG is most likely to be confused
with something else.

| Work | Establishes | Leaves open |
| --- | --- | --- |
| **Saifullah et al. (2026), *Narrative World Model*, arXiv:2607.05577** **[verified]** | Narratology-typed temporal state graph + query-conditioned retrieval beats Graphiti/Zep on multi-hop writer QA (0.898 vs 0.574). | Frontier-API extraction; an LLM reader as the metric; partly proprietary corpus. Its own ablation finds the win comes from **retrieval conditioning**, not the typed representation — untyping never lowers accuracy. |
| ComoRAG, arXiv:2508.10419 | Cognitively-inspired memory-organised RAG for stateful long-narrative reasoning. | Retrieval for QA; state is a means, not the measured object. |
| Rasmussen et al. (2025) — Graphiti/Zep | Bi-temporal knowledge graph agent memory: when a fact was true, when it was ingested. | Generic entity-relation facts; append-dominant; no reader-model correction. |
| Edge et al. (2024) — GraphRAG; Guo et al. (2024) — LightRAG; Gutiérrez et al. (2024) — HippoRAG | Graph-structured retrieval over source-derived entity graphs. | Retrieval indices, not maintained world state. |
| Packer et al. (2023) — MemGPT; Chhikara et al. (2025) — Mem0 | Long-term agent memory with consolidation. | Conversational facts; no narrative time. |
| EvoWiki, arXiv:2608.23265 | Incremental state overwriting with traceable QA as knowledge evolves. | Meetings, not narrative; overwriting rather than a typed revision calculus. |
| Hindsight is 20/20, arXiv:2512.12818 | Agent memory that retains, recalls, reflects. | Agent episodes, not reading. |

**The distinction to draw sharply.** Every system above is evaluated by *what a
downstream reader can answer from the memory*, usually with an LLM judge. DSG
evaluates *whether the state itself is correct and internally consistent*,
against human annotation, with no LLM in any primary metric. Different object,
different evidence standard.

## 8. Long-form generation with explicit world state

| Work | Establishes | Leaves open |
| --- | --- | --- |
| Yang et al. (2022) — Re3; Yang et al. (2023) — DOC, arXiv:2212.10077 | Recursive reprompting, revision, and detailed outline control for multi-thousand-word stories. | Control *intended* structure; do not maintain an auditable accepted state. |
| **Wilmot (2026), *Shadow-Loom: Causal Reasoning over Graphical World Model of Narratives*, arXiv:2605.02475** **[verified]** | The closest architectural cousin: a **versioned** typed world graph, fabula and syuzhet indices on every node, belief with provenance, LLM confined to extraction / rendering / audit, symbolic core in typed code. | Explicitly "offered as a research artefact rather than as a benchmarked NLP model": **no quantitative evaluation**, 20 hand-authored fixtures, and ingestion is a five-pass extraction over the **whole text** — not prefix-causal. Its versioning serves *counterfactual forks* ("what if Macbeth refuses?"), not correction of a reader's belief while reading. |
| ConWriter, arXiv:2608.05169 | Transition-constrained stateful generation with lightweight neuro-symbolic consistency control. | Generation-side; consistency as a constraint on output, not a measured property of state. |
| World-State Transformations for Neuro-symbolic Interactive Storytelling, arXiv:2605.24719 | Typed world-state operations for interactive fiction. | Authored worlds, not extracted from existing novels. |
| Long Story Generation via Knowledge Graph and Literary Theory, arXiv:2508.03137; StoryWriter, arXiv:2506.16445 | KG- and multi-agent-scaffolded long story generation. | Generation quality, not state correctness. |
| **Lost in Stories: Consistency Bugs in Long Story Generation, arXiv:2603.05890** | Taxonomises consistency failures in long generated narratives. | Diagnoses the symptom in *generated* text; DSG measures the mechanism in *read* text. |

Shadow-Loom is the paper to position against most carefully, because a reviewer
who knows it will ask what is left. The answer is the two things it explicitly
does not do: **prefix-causality** and **measurement**.

## 9. Narratology

| Work | Used for |
| --- | --- |
| Shklovsky (1917/1965); Tomashevsky (1965); Genette (1980); Bal (2009) | *Fabula* vs *syuzhet* — story order vs telling order. This is the theoretical basis for DSG's two clocks, and for the `SUPERSEDE`/`REVISE` split: a change in fabula versus a disclosure in syuzhet. |
| Sternberg (1992) | Curiosity / suspense / surprise as functions of what the reader knows when. |
| Propp (1968); Greimas (1983) | Structural function inventories; background for typed narrative roles. |
| Ryan (1991); Doležel (1998); Pavel (1986) | Possible-worlds narratology — relevant if DSG's provisional nodes are framed modally. |
| Aristotle, *Poetics* — *anagnorisis* | The recognition scene: the canonical case where a reader's identity belief is corrected. Worth one sentence; it is the phenomenon `MERGE_COMMITTED` exists for. |

## 10. Long-context limits

| Work | Establishes |
| --- | --- |
| Liu et al. (2024) — *Lost in the Middle* | Models use long contexts unevenly, degrading on material in the middle. |
| This project's own prior study (`manuscript-memory-engine/FAILURE_PROFILE_REPORT.md`) | Advertised context windows are 3–18× larger than the usable ones on quote attribution; mid-size models track a story to only ~1.5K–7K tokens. |

The motivation for bounded-window construction with an external state, rather
than feeding a growing prefix into the model.

## 11. Benchmarks and evaluation

| Work | Relevance |
| --- | --- |
| Kočiský et al. (2018) — NarrativeQA | Established QA over stories as a test of narrative understanding. |
| FABLES; ConStory-Bench | Book-length faithfulness and consistency benchmarks (already fetched under `data/`, unused in the current study). |
| LitVISTA, arXiv:2601.06445; NarraBench, arXiv:2510.09869 | Recent narrative benchmark frameworks; useful for placing DSG's metric suite. |
| *The Challenge and Reward of Fair Play in Narrative*, arXiv:2507.13841 | Surprise and sense-making measured jointly; a natural companion to the revision-trace instrument. |
| Vilain et al. (1995) MUC; Bagga & Baldwin (1998) B³; Luo (2005) CEAF | The coreference metrics DSG reuses for its identity plane. |
| Bamman et al. (2024), *On classification with large language models in cultural analytics* | The methodological argument for not trusting an LLM as a measurement instrument — the reason no primary DSG metric uses an LLM judge. |

---

## What to claim, given all of the above

1. **The revision/elaboration distinction is not ours** — Chen (2026) states it.
   Ours is the first implementation, the first corpus-scale measurement, and the
   first process metrics for it, all of which that paper asks for by name.
2. **The versioned typed narrative graph is not unprecedented** — Shadow-Loom
   builds one. Ours is the first built under a prefix-causal constraint and the
   first evaluated against human annotation.
3. **We are not competitive with supervised coreference and should not imply
   otherwise.** Identity is one plane of a larger state, scored on a different
   item set.
4. **What is genuinely ours**: the `SUPERSEDE`/`REVISE` split as an implemented,
   lexicon-arbitrated decision; the prefix-causal evaluation protocol with
   position-indexed probes; the finding that persistent state is self-poisoning
   and that revision is the antidote at zero accuracy cost; and the negative
   result that the calculus does *not* improve accuracy.
