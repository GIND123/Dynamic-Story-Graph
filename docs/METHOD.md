# The revision calculus — specification

A precise statement of what the system does, independent of any implementation
detail, so the behaviour can be checked and criticised directly.

## 1. State

The reader's model after discourse step `t` is `S_t = (E_t, A_t, L_t)`.

**Entities** `E_t`. Each node carries an identifier, a canonical name, the
multiset of surface forms observed for it, a commitment level
`κ ∈ {provisional, committed}`, a status, and provenance spans.

**Assertions** `A_t`. Each assertion is

```
a = ⟨subject, predicate, object, polarity, ε, κ, σ, c, [d_open, d_close], [v_from, v_to], prov⟩
```

with two independent clocks:

| Clock | Meaning |
| --- | --- |
| `[d_open, d_close]` | **discourse time** — while the reader held this belief |
| `[v_from, v_to]` | **story time** — while the fact held in the world |

`ε` is the evidential source (`narrated`, `reported`, `implied`), `σ` the status
(`believed`, `superseded`, `retracted`, `merged_away`), `c` a confidence, and
`prov` a character span in the text.

Separating the two clocks is what lets the system record that a fact stopped
being *true* at one point and stopped being *believed* at another — the
difference between a plot event and a reveal.

**Log** `L_t`. Every update, typed, in order. The log is the object the process
metrics and the humanities-facing profile are computed from.

## 2. Prefix causality

At step `t` a system may read window `w_t` and any state derived from
`w_1 … w_{t-1}`, and nothing else. Formally, every `prov` span in `S_t` must end
at or before the last character of `w_t`. This is checked (invariant I6), not
assumed, so a bug that leaks future text shows up as a violation rather than as
a better score.

## 3. The operations

Given a candidate assertion `a'` at step `t` and the live assertions on the slot
`(subject, predicate)`:

| Condition | Operation | Effect |
| --- | --- | --- |
| slot empty | `ASSERT` | add `a'` |
| identical belief already live | `NOOP` | raise confidence, extend provenance |
| existing object is underspecified, or `a'` refines it | `ELABORATE` | update in place; nothing retracted |
| existing is provisional and weakly held | `ELABORATE` | as above |
| evidence carries a revelation marker | `REVISE` | retract, cascade to dependents |
| existing is `reported`, `a'` is `narrated` | `REVISE` | narration overrules a character's claim |
| predicate is immutable | `REVISE` | kinship and identity cannot change |
| otherwise (predicate mutable) | `SUPERSEDE` | close `v_to`, keep as history |

Identity has the parallel pair: folding a **provisional** node into another is
`MERGE_PROVISIONAL` (monotone); folding two **committed** nodes is
`MERGE_COMMITTED` (a rollback).

`{ASSERT, ELABORATE, MERGE_PROVISIONAL, NOOP}` are monotone: `S_t` remains a
substructure of `S_{t+1}`. `SUPERSEDE` is history-preserving but not monotone in
the believed set. `{REVISE, MERGE_COMMITTED, SPLIT}` are rollbacks.

**A policy that lacks an operation does not skip the update.** It asserts the
candidate anyway and leaves both beliefs live. That is the modelled failure of
append-only pipelines, and it is what invariant I1 then detects.

## 4. Deferred commitment

A reference that is a definite or indefinite description (`the stranger`,
`an old man`) or an ambiguous name creates a **provisional** node. A node is
promoted to **committed** once it has acquired a proper name *and* been
re-attested at least `k` times (`k = 2`).

This is the paper's central lever. If "the stranger" is committed on first
sight, the later reveal that they are Magwitch requires a non-monotone
`MERGE_COMMITTED`. If it is held provisional, the identical reveal is a monotone
`MERGE_PROVISIONAL` — the graph gains information instead of being torn up. The
`dsg-eager` ablation differs from `dsg-full` in this one rule.

## 5. Structural invariants

Checked after every step; violations are counted, never silently repaired.

| | Invariant |
| --- | --- |
| I1 | no two live, conflicting values in a single-valued slot with overlapping validity |
| I2 | no live assertion referencing a merged-away or retracted entity |
| I3 | no two committed live entities sharing a proper name |
| I4 | merge chains terminate at a live node and do not cycle |
| I5 | intervals are well-formed; a live assertion has no close time |
| I6 | no provenance beyond the prefix already read |
| I7 | one-way predicates do not flip back (the dead stay dead) |

## 6. What the model is and is not allowed to do

The language model reads one window plus a bounded digest of who has been met so
far, and emits typed proposals: entity mentions, identity links, facts with an
evidential label and a quoted evidence phrase, and speaker attributions. It has
**no write access**. Whether a conflict is a world change or a reader error is
decided by the predicate table and the marker list in `dsg/lexicon.py`, both of
which are fixed, inspectable, and identical across every condition.

Two consequences matter for the experiment. First, extraction can be cached and
replayed, so every policy sees a byte-identical proposal stream and differences
are attributable to the representation alone. Second, the part of the system
that does the reasoning under test is not a black box.

## 7. Complexity

Per step the system touches only the entities mentioned in the current window
and the assertions on their slots. Memory is bounded by the cast size, not by
book length, and the model's context is one window plus a fixed-size digest —
so cost is linear in book length with a constant context, against the quadratic
attention cost of re-reading a growing prefix.
