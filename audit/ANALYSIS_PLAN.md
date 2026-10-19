# Audit analysis plan (phases 1c and 4c)

Fixed before any label is read. Code: `dsg/eval/agreement.py`,
`dsg/eval/labels.py`, `dsg/eval/multiplicity.py`.

## Splits

Dev and test are split by group, never by row, so nothing a rule could
memorise crosses the boundary:

| sheet | group | why |
|---|---|---|
| detector | story premise, by content fingerprint (`premise_fingerprint`) | the same planted canon recurs across runs and conditions; the 7B prose runs reuse story ids for different premises, so ids alone would merge distinct premises |
| calculus | novel (`book`) | a novel's cast, predicates and phrasing recur across its rows |

`grouped_split(groups, dev_fraction=0.5, seed=20261018)`. v2 rules are written
against dev only; v1 and v2 are reported on test.

## Agreement

On the 50 double-annotated rows of each sheet, overall and per stratum
(`agreement_by_stratum`):

- observed agreement
- Cohen's kappa
- positive and negative agreement (specific agreement on "yes" and "no"); for
  the calculus label, specific agreement for each category
- Gwet's AC1

Kappa is never reported alone: the sheets are skewed (most unflagged mentions
are "no"), and kappa falls under skew even when raw agreement is near perfect.
Per-stratum figures rest on few rows and are reported with their n.

## Disagreements on overlap rows

Fixed before any label is read.

1. **Agreement first.** Agreement statistics are computed on the two
   annotators' independent labels before anything else happens, and are then
   frozen. Nothing below changes them.
2. **Joint adjudication.** For every overlap row where the two labels differ
   (including `unsure` against a definite label), the two annotators then
   decide one label together. They see only their own sheets' columns, never
   the run, condition, rule or overlap columns. The result goes in a new
   `<label>_adjudicated` column, with a note saying why.
3. **Both originals are kept.** `labels_merged.csv` keeps the first annotator's
   label, the second annotator's label (`_b`) and the adjudicated label side by
   side. Nothing is overwritten.
4. **Which label enters the estimates.** On an overlap row, the adjudicated
   label, or the shared label where the two agreed. On every other row, the
   single annotator's label.
5. **Unresolved rows.** If adjudication cannot reach a label, the final label
   is `unsure`.
6. **`unsure` in estimates.** Rows whose final label is `unsure` are left out of
   precision and recall denominators and reported as a count per stratum. Each
   estimate is also reported with the `unsure` rows counted both ways (all as
   correct, all as incorrect) as a sensitivity bound.

## Estimates

Precision and recall are weighted by stratum: each sampled row stands for
`stratum_population / stratum_sampled` rows. Rows drawn by the bound-target
design carry that design's weights (`bound_stratum_*`) and are used only for
the bound; rows in both designs are labelled once.

## Generation comparisons: the multiple-comparison family

Fixed as of this commit and not changed after labels are read.

**Unit.** Paired bootstrap over the 30 stories, 100,000 resamples, seed 0,
percentile intervals; two-sided bootstrap p with the +1 correction
(`dsg/eval/multiplicity.py`).

**Family.** One family per metric. For each of the nine metrics in
`dsg.generate.score.METRICS`, the family is every pair in
`dsg.generate.score.COMPARISONS` whose two arms both appear in one of these four
runs: `gen-qwen3b-s30-c20`, `gen-v3-hybrid`, `gen-v4-retrieval`,
`gen-v5-selection`. That is 27 comparisons per metric, listed below. Any metric
added in phase 1d, such as violations per subject mention, is its own family of
the same 27 comparisons.

**Arms excluded from conclusions still count.** The five comparisons involving
run 1's tuned arm are in the family although that arm is excluded from every
conclusion. Keeping them makes the correction more conservative; dropping them
after seeing results would be a choice made with the data in view.

**Runs not in the family.** The 7B prose and smoke runs
(`gen-qwen7b-prose-*`) are not cited by any finding and are excluded, although
some `COMPARISONS` pairs could be computed on them.

**Detector versions.** In phase 1d each detector version (v1, v2) and the
other-subject bound is a separate analysis over the same family. Holm is applied
within each version and metric; versions are never pooled into one family.

**Correction.** Holm step-down at α = 0.05 within the family. Each comparison is
also reported with its interval at its Holm step level, α / (m − rank + 1).

**Wording.** A gap that fails the correction is described as "not significant
after correction", never as no difference.

| # | run | comparison (a − b) | note |
|---|---|---|---|
| 1 | `gen-qwen3b-s30-c20` | `base:rolling-summary` − `base:none` | |
| 2 | `gen-qwen3b-s30-c20` | `base:full-context` − `base:rolling-summary` | |
| 3 | `gen-qwen3b-s30-c20` | `base:dsg-state` − `base:none` | |
| 4 | `gen-qwen3b-s30-c20` | `base:dsg-state` − `base:rolling-summary` | |
| 5 | `gen-qwen3b-s30-c20` | `base:dsg-state` − `base:full-context` | |
| 6 | `gen-qwen3b-s30-c20` | `base:dsg-state` − `base:append-only-state` | |
| 7 | `gen-qwen3b-s30-c20` | `tuned:dsg-state` − `base:dsg-state` | excluded arm |
| 8 | `gen-qwen3b-s30-c20` | `tuned:full-context` − `base:full-context` | excluded arm |
| 9 | `gen-qwen3b-s30-c20` | `tuned:dsg-state` − `tuned:full-context` | excluded arm |
| 10 | `gen-qwen3b-s30-c20` | `tuned:dsg-repair` − `tuned:dsg-state` | excluded arm |
| 11 | `gen-qwen3b-s30-c20` | `tuned:dsg-repair` − `base:full-context` | excluded arm |
| 12 | `gen-v3-hybrid` | `base:dsg-state` − `base:none` | |
| 13 | `gen-v3-hybrid` | `base:dsg-state` − `base:full-context` | |
| 14 | `gen-v3-hybrid` | `base:dsg-hybrid` − `base:dsg-state` | |
| 15 | `gen-v3-hybrid` | `base:dsg-hybrid` − `base:last-chapter` | |
| 16 | `gen-v3-hybrid` | `base:dsg-hybrid` − `base:full-context` | |
| 17 | `gen-v3-hybrid` | `base:dsg-hybrid` − `base:none` | |
| 18 | `gen-v3-hybrid` | `base:dsg-hybrid-repair` − `base:dsg-hybrid` | |
| 19 | `gen-v4-retrieval` | `base:beat-retrieval` − `base:none` | |
| 20 | `gen-v4-retrieval` | `base:beat-hybrid` − `base:last-chapter` | |
| 21 | `gen-v4-retrieval` | `base:beat-hybrid` − `base:full-context` | |
| 22 | `gen-v4-retrieval` | `base:beat-hybrid` − `base:beat-retrieval` | |
| 23 | `gen-v4-retrieval` | `base:beat-hybrid-repair` − `base:beat-hybrid` | |
| 24 | `gen-v5-selection` | `base:last-chapter@bo4-graph` − `base:last-chapter` | |
| 25 | `gen-v5-selection` | `base:last-chapter@bo4-random` − `base:last-chapter` | |
| 26 | `gen-v5-selection` | `base:last-chapter@bo4-graph` − `base:last-chapter@bo4-random` | |
| 27 | `gen-v5-selection` | `base:last-chapter@bo4-graph` − `base:full-context` | |
