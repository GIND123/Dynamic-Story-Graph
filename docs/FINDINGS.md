# Findings

Every reading-study number below is read from
`audit/repro/study_current_code.json`, which `python -m dsg.eval.study_repro`
builds from reruns of `python -m dsg study` (`results-current/` on the Hub,
pinned in `audit/repro/hub_manifest.json`). Numbers are recomputed under current
code (after commit `0be8e0d`, which made `occupation` multi-valued and widened
`refines()`); `audit/repro/study_current_code.md` lists published vs current
values. No conclusion weakened; one strengthened. Comparisons are paired
bootstraps over books (10,000 resamples, 95% intervals). No LLM judge appears in
any metric; identity is scored against PDNC's human alias annotation and
LitBank's gold coreference, and mention linking against PDNC's annotated
referring expressions.

**Status.** PDNC at 1.5B, 3B and 7B and LitBank at 7B are complete. This
document reports the 3B and 1.5B runs; the 7B run is in
`artifacts/report/pdnc-qwen7b/`.

Every trend against discourse position is reported with a shuffled-window
control, because state growth alone produces such trends (§2, F4).

---

## 1. The headline

Carrying narrative state forward is **necessary and self-poisoning**. It buys a
large amount of accuracy and, at the same time, fills the state with
contradictions that a stateless reader cannot have. Fact-level revision is the
antidote; whether it costs accuracy is untested (see below).

On 28 PDNC novels with a 3B extractor, moving one rung at a time:

| rung | identity CoNLL F1 | mention linking | self-contradictory slots | rollbacks |
| --- | --- | --- | --- | --- |
| persistent state (`append-only` − `window-only`) | **+0.088** | **+0.310** | **+0.320** *(worse)* | 0 |
| + identity merging | −0.003 (ns) | **−0.016** *(worse)* | **+0.017** *(worse)* | **+2.36** |
| + fact revision | 0.000 (ns) | 0.000 (ns) | **−0.439** | +33.1 |
| + deferred commitment | −0.002 (ns) | **+0.014** | 0.000 | **−2.57** |
| cost of causality (`retrospective` − `dsg-full`) | +0.005 (ns) | **+0.031** | 0.000 | ns |

Bold entries have a 95% interval excluding zero. Read down the fourth column:
persistent state introduces contradiction into a third of the state's slots, and
fact revision removes it — **on every one of the 28 books** (0–28 win split).
Both accuracy metrics read 0.000 on that rung, but identity and mention scoring never read fact assertions, so the 0.000 is guaranteed by construction, like I1. The first
real test of the cost is correct-value scoring in audit phase 4d.

In absolute terms, the share of single-valued slots holding two or more live
conflicting values:

| corpus / model | `window-only` | `append-only` | `dsg-full` |
| --- | --- | --- | --- |
| PDNC, Qwen2.5-3B | 0.102 | 0.422 | **0.000** |
| PDNC, Qwen2.5-1.5B | 0.004 | 0.090 | **0.000** |
| LitBank, Qwen2.5-7B | 0.092 | 0.185 | **0.000** |

## 2. The pre-registered falsifiers

Stated in `docs/RESEARCH_PLAN.md` §5 before any run.

**F1 — "`dsg-full` ≈ `append-only` on gold identity ⇒ premature commitment is
not the bottleneck." This falsifier FIRED.**
On identity CoNLL F1 the difference is indistinguishable from zero at every
model size (3B: −0.005, CI [−0.011, +0.000]; 1.5B: −0.003; LitBank: −0.003, CI
spans zero). Mention linking is flat at 3B (−0.002, ns) and lower at 1.5B
(−0.017, CI [−0.034, −0.004]). **The revision calculus does
not make the state more accurate.** What it does is make the state
*consistent* — a different claim, and the one the evidence supports. The paper's
framing follows the evidence rather than the other way round.

**F2 — "`dsg-full` ≪ `retrospective` with no closing of the gap ⇒ causality, not
commitment policy, is the binding constraint." Partially fired.**
Reading forward genuinely costs accuracy, and we can now price it: the
non-causal oracle is **+0.031** better on mention linking at 3B (CI [+0.018,
+0.045], 23–4) and **+0.050** at 1.5B (CI [+0.038, +0.064], 28–0). Both
intervals exclude zero. Causality is a real constraint, not an artefact — but
it is a few points, not a collapse, and it does not touch consistency, where
the causal system already matches the oracle exactly (both 0.000).

**F3 — "invariant violations near zero for `append-only` ⇒ the breakage premise
is wrong." Did not fire; the premise holds decisively.**
42.2% of `append-only`'s slots are self-contradictory at 3B on full novels.

**F4 — "revision events distribute uniformly over discourse position ⇒ the
instrument has no signal." Fired for revision; did not fire for elaboration.**

In narrative order both operations trend strongly with position: elaboration
falls (Spearman ρ = −0.850 at 3B, −0.741 at 7B) and revision rises steeply, from
4.4% of operations in the first twentieth of a book to 29.8% in the last
(3B, ρ = +0.82, p = 9×10⁻⁶). Read alone that is a tidy narratological story —
exposition first, reversal later.

It is mostly an artefact, and only a control shows it. As a book proceeds the
state holds more assertions, so *any* conflict-driven operation becomes
mechanically more likely. Permuting the reading order destroys narrative order
while preserving state growth exactly:

| ordering | elaboration ρ, 3B | elaboration ρ, 7B | revision ρ, 3B |
| --- | --- | --- | --- |
| narrative | **−0.850** (p = 2×10⁻⁶) | **−0.741** (p = 2×10⁻⁴) | **+0.82** (p = 9×10⁻⁶) |
| shuffled | −0.141 (p = 0.55) | +0.292 (p = 0.21) | **+0.68** (p = 1×10⁻³) |

The revision trend largely **survives** shuffling, so most of it is state
growth rather than a property of the text; the honest residual is the gap
between ρ = 0.82 and ρ = 0.68, which this design cannot cleanly separate. The
elaboration trend **disappears** under shuffling at both scales. That one is
real: elaboration is front-loaded in narrative order, and not explained by how
much state has accumulated.

This elaboration measure excludes the low-confidence overwrite rule
(`elaborate:provisional`), which replaces a provisional value rather than
refining it, so it differs from the one published earlier
(`audit/calculus/elaboration_trend.json`). Counting that rule as elaboration
gives ρ = −0.59 (p = 0.007) at 3B and ρ = −0.155 (p = 0.51) at 7B; the earlier
reading — real at 3B, reverses at 7B — came from that rule. *(Draft, pending audit phase 4c: in the measure that counts the rule, at 3B, 278 of the 972 operations counted in the elaboration rate (28.6%) are low-confidence `ELABORATE` overwrites; 1,263 of 2,708 (46.6%) at 7B (`audit/calculus/elaborate_census.json`). Whether the remaining `ELABORATE` operations are refinements depends on the 4c labels.)*

So the instrument has signal, but less than the uncontrolled numbers suggest,
and in the opposite operation from the one we expected to carry it.

## 3. What each mechanism actually buys

- **Persistent state** is where nearly all the accuracy comes from: +0.310
  mention linking over reading each window in isolation. It is also where all
  the inconsistency comes from.
- **Identity merging alone makes things worse.** On its own it costs mention
  accuracy (−0.016, CI excludes zero), raises self-contradictory slots
  (+0.017, CI [+0.003, +0.031]) and buys 2.4 rollbacks per book. Merging
  two characters unifies their assertion slots, and without revision the merged
  slot simply holds both rival values. The two mechanisms are complementary:
  merging is only safe once the state can revise.
- **Fact revision** is the load-bearing contribution: −0.439 self-contradictory
  slots, 0–28; its accuracy cost is untested (the 0.000 on the accuracy metrics
  is guaranteed by construction).
- **Deferred commitment** converts rollbacks into monotone updates. The size of
  the effect varies with how fast nodes become committed: −5.50 rollbacks per
  book at 1.5B (0–28, a 99% reduction), −0.60 on LitBank (92%), but only −2.57
  of 35.5 at 3B (7%). It also slightly *helps* mention linking (+0.014 at 3B,
  +0.006 at 1.5B, both intervals excluding zero), which is the opposite of the
  cost we expected to pay for holding identity open.

## 4. Three defects found by looking at the output

Recorded because they materially changed the numbers, and because the third was
found only by checking an assumption that seemed safe.

1. **Titles were stripped before comparing surfaces**, so `Mrs. Bennet` and
   `Miss Bennet` scored 0.9 against each other. In this corpus a title is the
   primary distinguisher between people sharing a surname.
2. **Nothing prevented merging two separately named characters.** A single bad
   link fuses two people; the fused node then matches both name sets and
   attracts more. On *Pride and Prejudice* this collapsed 74 gold characters
   into 3 nodes.
3. **The link *bind* path bypassed the merge guard and the policy check**, so
   even `append-only` was receiving identity resolution it should not have had.

Fixing these moved conflation on *Pride and Prejudice* from 1.33 to 0.05.

The obvious fix for (2) was to require corroboration — accept a name-to-name
link only if it is proposed more than once. **Measured before adopting, and it
does not work:** across 205 links on one novel, `Mr. Darcy -> Elizabeth` was
proposed nine times, exactly as often as the correct `Mr. Bingley -> Bingley`.
Repetition does not separate good links from bad, so the constraint had to be
structural rather than statistical.

*(Both are one-off diagnostics that predate the current-code rerun and were not
recomputed.)*

## 5. Limitations

- **No identity-accuracy gain.** Stated plainly: on the metrics that measure
  accuracy rather than consistency, the calculus is neutral.
- **One corpus of 28 English novels**, plus 100 LitBank excerpts. One model
  family.
- **Alias-level identity gold.** PDNC's alias sets are surface strings, not
  full mention-level coreference; the mention-linking plane partly compensates
  but covers only referring expressions inside quotations.
- **Extraction is the binding constraint on several metrics.** A 1.5B model
  fails to produce parseable output on 21% of windows, so the scale comparison
  is partly a comparison of yield; the parse rate is reported alongside it.
- **Nicknames are unrecoverable** under the merge constraint: `Lizzy` and
  `Elizabeth` share no surface material, so the guard that blocks
  `Jane -> Elizabeth` also blocks the correct `Lizzy -> Elizabeth`.
- **`SUPERSEDE` versus `REVISE` is decided by a lexicon**, and its accuracy has
  not itself been evaluated against annotation.
