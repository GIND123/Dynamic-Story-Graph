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

## Estimates

Precision and recall are weighted by stratum: each sampled row stands for
`stratum_population / stratum_sampled` rows. Rows drawn by the bound-target
design carry that design's weights (`bound_stratum_*`) and are used only for
the bound; rows in both designs are labelled once.

## Generation comparisons

None of the generation gaps was named before its run. They are reported with
raw and Holm-corrected intervals, Holm applied within each metric across every
comparison computed on the four runs (`audit/generation/holm.json`).
