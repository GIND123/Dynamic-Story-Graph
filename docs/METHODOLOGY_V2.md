# Methodology v2 — three layers, and the experiment that is missing

**Status.** Design document. Numbers marked `[measured]` were computed in this
repo and are reproducible from the command given. Numbers marked `[published]`
are from cited work and were verified against the source PDF. Nothing here is
estimated, and no result cell is filled in until its run has produced it.

---

## 0. Why v2 exists

The v1 study (`docs/METHODOLOGY.md`) established a thesis with one positive and
three negatives. It is honest and the numbers reconcile. It has one structural
weakness and four reporting defects.

**The structural weakness.** Layer 2 (does the state *do* anything?) was tested
only against **generation**, which is the hardest possible utility test, and the
answer was a flat negative across five designs. A paper whose entire utility
layer is negative is a weaker paper than one that shows *where the boundary
lies*. We never tested whether the state helps an **interpretation-time** task,
which is (a) what the state is actually shaped for, (b) where established
benchmarks and published SOTA exist, and (c) cheap, because the corpus is
already annotated.

That is the gap v2 fills.

---

## 1. The three layers

| Layer | Question | v1 status | v2 action |
|---|---|---|---|
| **L1 Representation** | Can a maintained state stay coherent while reading? | Positive, well evidenced | Fix 4 reporting defects (§5). No new runs. |
| **L2 Utility** | Does the state help a downstream model? | Negative, generation only | **Add E1** (interpretation) and **E2** (long-horizon generation) |
| **L3 Measurement** | Does the contradiction count measure a text? | Negative, plus a reusable benchmark | Elevate `on_premise` to a *replicated* finding (§6) |

Revised thesis:

> A revision calculus keeps a maintained narrative state internally coherent
> (L1). That coherence transfers to interpretation-time tasks but **not** to
> generation (L2). And the state's own contradiction count is not a valid
> measure of textual consistency (L3). Representation, utility, and measurement
> are three separate problems, and solving the first does not solve the others.

---

## 2. E1 — Prefix-causal quote attribution on PDNC

The core new experiment. It is the only place in this project where a
**legitimate comparison to published SOTA** exists.

### 2.1 The task

For each gold quotation `q` at character span `[s, e)` in a PDNC novel, name the
speaker, conditioned **only** on `text[0:s]` and on state built causally from
`text[0:s]`. The system may never see `text[s:]`.

### 2.2 Why this is novel

Every published PDNC number is **non-causal**: the system reads a window
*centred* on the quote, which includes the following text where the attribution
cue usually lives.

| System | PDNC overall | non-explicit | Setting |
|---|---|---|---|
| BookNLP+ | 78.5 | 68.9 | non-causal `[published]` |
| Llama-3 8B zero-shot | **90.6** | **89.1** | non-causal `[published]` |

Source: Michel, Epure, Hennequin & Cerisara (2024), *Evaluating LLMs for
Quotation Attribution in Literary Texts*, arXiv:2406.11380, Table 1.

**No published number exists for the prefix-causal setting.** E1 produces the
first one, on an established benchmark, against a known non-causal ceiling.

### 2.3 What causality actually removes `[measured]`

This is the number that motivates the experiment. PDNC quote types:

| type | count | share |
|---|---|---|
| Implicit | 16,716 | 45.0% |
| Explicit | 11,185 | 30.1% |
| Anaphoric | 9,202 | 24.8% |
| (unlabelled) | 28 | 0.1% |
| **total** | **37,131** | |

For the 10,743 Explicit quotes whose gold speaker has resolvable aliases, where
does the speaker's name occur within ±250 characters of the quote?

| position of the gold speaker name | count | share |
|---|---|---|
| neither side (tag further away / different surface) | 3,472 | 32.3% |
| **after the quote only — destroyed by a causal cut** | **3,147** | **29.3%** |
| both sides | 2,600 | 24.2% |
| before only | 1,524 | 14.2% |

**So a strict prefix-causal cut removes the decisive lexical cue for 29.3% of
explicit quotes (~8.8% of the whole corpus).** The constraint is real and its
cost is now quantified rather than assumed. This is itself a publishable
measurement — it says how much of PDNC's attribution signal is *post-hoc*.

Reproduce: `python -m dsg.eval.causal_audit --report tag-position`

### 2.4 Conditions (a ladder, one mechanism per rung)

Every condition sees an identical candidate set and identical scoring. Only the
evidence changes.

| # | condition | evidence available | isolates |
|---|---|---|---|
| 0 | `prior` | candidate list only, pick corpus-majority speaker | floor |
| 1 | `recency` | last attributed speaker before `s` | trivial dialogue alternation |
| 2 | `text-causal` | `text[s-W:s]` only | what raw prefix text gives |
| 3 | `state-causal` | `text[s-W:s]` + serialised DSG state at `s` | **the state's contribution** |
| 4 | `state-retrieval` | `text[s-W:s]` + state entries for in-scope candidates | conditioning vs dumping |
| 5 | `oracle-noncausal` | `text[s-W:s+W]` | reproduces the published protocol; **prices causality** |

Condition 5 is the control that makes the whole thing interpretable: it runs the
literature's own setting with our model and prompt, so the gap between 5 and 3
is the cost of causality *measured on our own system*, not inferred by comparing
across papers.

### 2.5 What each outcome licenses

The experiment is designed to be informative under every result:

- **3 ≫ 2** → the state carries attribution signal that raw prefix text does not.
  This is a **positive utility result** and repairs the L2 hole.
- **3 ≈ 2** → the state adds nothing here either; the L2 negative generalises
  beyond generation, which is a stronger and more surprising negative than v1's.
- **3 ≈ 5** → causality is nearly free on this task once state is present. Strong
  result.
- **3 ≪ 5** → we have priced causality on a benchmark with known SOTA. Still first.

No outcome is unpublishable, and none requires a win. **We do not promise to
beat 90.6 and must not imply we will.**

### 2.6 Leakage controls — enforced in code, not by convention

This is where the experiment lives or dies. v1's prefix-causality is enforced at
**window** granularity (`prefix_end = proposal.end` in
`dsg/policies/runner.py:99`), i.e. a proposal may cite anything inside its own
3,200-char window. That is correct and honestly described for the reading study,
where the window *is* the reading step. **It is not sufficient for E1**, where a
quote sits inside a window and the rest of that window is future text.

Mandatory guards:

- **G1 — offset assertion.** Every prompt is built by a single function that
  takes `(text, s)` and can only slice `text[:s]`. A test asserts that no
  assembled prompt shares any 40-character substring with `text[s:]`. This runs
  on every quote, not a sample, and fails the run rather than warning.
- **G2 — no gold at inference.** Report two candidate regimes: *gold candidates*
  (matches the literature protocol, comparable to 90.6) and *system candidates*
  (derived from causal state, harder, honest). Never mix them in one table.
- **G3 — memorization covariate.** Qwen and Llama have read these public-domain
  novels. Per novel, compute name-cloze accuracy from `gpt4_books` and correlate
  it with attribution accuracy. Michel et al. ran this exact check and found
  memorization did not explain their gain; if our correlation is significant,
  the result is contaminated and must be reported as such.
- **G4 — book-level splits only.** If anything is ever tuned, split by novel,
  never by quote. Quote-level splits leak a book's cast across the boundary.
- **G5 — no LLM judge.** Scoring is exact match against PDNC gold speaker after
  alias normalisation. Deterministic.

### 2.7 Analysis

Paired bootstrap over the 28 novels, 10,000 resamples, 95% intervals, win/loss
split reported — identical machinery to the reading study
(`dsg/eval/bootstrap.py`), so the statistics need no new validation. Report
overall, explicit, and non-explicit accuracy separately, because the causal
constraint bites almost entirely on the explicit slice (§2.3).

---

## 3. E2 — the long-horizon generation test

v1's generation negative is real but was measured in a regime that favoured the
baseline it was competing against.

`manuscript-memory-engine/FAILURE_PROFILE_REPORT.md` Part 2 `[measured, prior]`:
advertised context windows are 3–18× larger than usable ones — Qwen2.5-7B holds
to only ~7,292 tokens of its advertised 131,072.

v1's `full-context` condition peaked at **5,308–5,398 prompt tokens** at 20
chapters — *below* that breaking point. So every graph condition was competing
against a full-transcript baseline that had **not yet degraded**.

**E2.** Extend to 60 chapters, where `full-context` crosses the usable window.
Pre-registered falsifier, stated here before the run:

> If graph conditioning still fails to beat `full-context` at 60 chapters, the
> L2 negative is established across the regime where the graph was *most*
> favoured, and the conditioning line is closed for good.

Caveat to state in the paper: the ladder covered six models
(`gemma2:9b`, `llama3.1:8b`, `mistral:7b`, `qwen2.5:72b`, `qwen2.5:7b`, `yi:9b`)
and **not** Qwen2.5-3B, the generator. The ~7.3K figure is a nearby proxy, not a
measurement of our generator. E2 should ladder the generator itself first.

---

## 4. Compute plan

| stage | work | hardware | note |
|---|---|---|---|
| E1 build | assemble 37,131 causal prompts | CPU, local | no model needed |
| E1 pilot | 3 novels × 6 conditions | M4 Pro / MLX 4-bit | validates guards before spend |
| E1 full | 28 novels × 6 conditions | A100 via vLLM | batched; conditions share prompts |
| E1 memo | name-cloze per novel | same | G3 covariate |
| E2 ladder | generator context ladder | A10G | 1 model, cheap |
| E2 full | 30 stories × 6 cond × 60 ch | A100 | ~2× v1 run 4 |

Local pilot first, always. The M4 Pro runs Qwen2.5-7B in 4-bit through
`dsg/llm/mlx_backend.py`, so guards G1–G5 are validated at zero cloud cost
before any GPU is billed.

---

## 5. Reporting defects to fix in v1 (no new runs needed)

1. **Scale-mixed headline.** v1 pairs "+0.310 mention linking" and "+0.088 CoNLL",
   both 3B — simultaneously the weakest CoNLL and strongest mention cell. The
   ordering inverts at 7B (+0.209 CoNLL, +0.152 mention). Report the range across
   scales, or lead with 7B (the run the headline ladder figure comes from).
2. **LitBank cannot support the mention-linking leg.** It has no quotations, so
   `mention_acc` is 0.000/undefined there — not a measured null.
3. **Shuffled-control claim is 3B-only and reverses at 7B** (narrative elaboration
   ρ=−0.084 p=0.72; shuffled ρ=+0.684 p=0.00088). Restrict it or demote it.
4. **Figure provenance.** `docs/figures/fig6-revision-profile.png` is byte-identical
   to the **LitBank** run's figure, which has no shuffled-control section at all,
   while the prose beside it reports 3B numbers.
5. **1.5B is partly a yield comparison** — parse success 3,226/4,063 (20.6% failure)
   vs 99.9% at 7B. Say so wherever 1.5B is cited as a replication.
6. Test count: 76, not 72.

---

## 6. `on_premise` is a replicated finding, not an anecdote

Two independent instances in this repo, ~2 months apart, different tasks:

| study | date | headline metric | degenerate strategy | denominator that caught it |
|---|---|---|---|---|
| Failure profile | 2026-07-11 | contradiction rate | flag *everything* → 0.000 | `over_flag` (=1.00 for llama3.1:8b, llama3.2:3b, yi:9b) |
| Generation run 1 | 2026-09-04 | violation rate | write off-premise → 0.054 | `on_premise` (1.000 → 0.765) |

State it as a general principle with two supporting instances:
**a headline metric that a degenerate policy can saturate requires a paired
denominator, reported together, always.** This is the most portable contribution
in the project.

---

## 7. Pre-registration is auditable — say so

The v1 falsifiers were committed at **2026-09-03 17:19** (`2934dce`,
`docs/RESEARCH_PLAN.md` §5). The earliest result artifact was uploaded to the
Hub at **2026-09-04 04:18:59**, and the four generation runs ran
**2026-09-04 20:29 → 2026-09-06 04:15**. Two independent timestamped systems,
plan before results in both.

E1 and E2 falsifiers are stated in §2.5 and §3 **of this document, before either
run**, and this document is committed before any E1/E2 artifact exists.

---

## 8. Claims that remain forbidden

Carried forward from v1 and still binding:

- Not competitive with supervised coreference (81.27 CoNLL on LitBank,
  MEIC-DT `[published, verified]`). Our identity plane is an unsupervised
  alias-string partition, a different item set.
- The contradiction rate does not measure textual consistency (AUC 0.394).
- No "first" claim until every related-work entry is checked against its
  published version.
- The 0.492 canon-capture figure does **not** rule out an extraction ceiling. It
  shows half the canon was present and unused. Say that instead.
- The `manuscript-memory-engine` "Graph 0.816 vs Long-Context 0.653" comparison
  is **invalid** — the graph reads gold PDNC speaker labels for neighbouring
  quotes while the baselines read raw prose. Do not cite it.
