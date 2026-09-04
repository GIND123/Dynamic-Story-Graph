# Generation findings

Two runs, 9,000 generated chapters. 30 stories × 20 chapters, Qwen2.5-3B.
Canon is planted in chapter 1 and withheld afterwards; violations are
deterministic string tests near a mention of the subject; paired bootstrap over
stories. No LLM judge anywhere.

**Bottom line: a maintained graph state does not help long-form generation. The
single most effective memory is the previous chapter.** Reported here because
the result is clean, not because it is welcome.

## 1. Results after both fixes

Run 2 applies the two fixes the first run's diagnosis pointed at — ordering the
digest by durability so canon is not truncated in favour of churn, and dropping
the artificial digest-*or*-text choice. All conditions write about the premise
in ~100% of chapters, so these are like-for-like.

| memory | violation rate ↓ | canon capture ↑ | prompt tokens |
| --- | --- | --- | --- |
| none | 0.504 | — | 112 |
| **previous chapter** | **0.308** | — | 1,009 |
| full transcript | 0.329 | — | 5,398 |
| graph digest | 0.588 | 0.342 | 357 |
| digest + previous chapter | 0.379 | 0.487 | 1,194 |
| digest + previous chapter + guard | 0.367 | 0.512 | 1,196 |

| comparison | Δ violation | 95% CI | wins |
| --- | --- | --- | --- |
| digest − none | **+0.083** | [+0.025, +0.142] | 6–16 |
| digest − transcript | **+0.258** | [+0.192, +0.325] | 0–24 |
| hybrid − digest alone | **−0.208** | [−0.263, −0.154] | 25–1 |
| hybrid − previous chapter | **+0.071** | [+0.021, +0.121] | 5–18 |
| hybrid − transcript | +0.050 | [+0.000, +0.100] | ns |
| guard − no guard | −0.013 | [−0.046, +0.013] | ns |

## 2. What this says

**The digest alone is worse than no memory at all** (+0.083, interval excludes
zero). Handing a writer a partial, confident-looking summary is worse than
handing it nothing: it crowds the prompt while carrying a third of what it is
supposed to protect.

**The fixes worked, and were still not enough.** Durability ordering plus the
hybrid raised canon capture from 0.342 to 0.512 and cut violations by 0.208 —
a large, reliable improvement (25 of 30 stories). But the hybrid still loses to
simply pasting the previous chapter (+0.071, interval excludes zero). The
digest adds cost and subtracts accuracy.

**The graph guard does nothing measurable** (−0.013, ns). It fired on 76
chapters and repaired 50, but at that rate it cannot move a corpus-level number.

**More context is not better.** The previous chapter alone (0.308, 1,009
tokens) beats the whole transcript (0.329, 5,398 tokens). Recency, not volume,
is what protects continuity — consistent with what is already known about
models using long contexts unevenly, and worth reporting in its own right.

The one thing in the method's favour is narrow: the hybrid is statistically
indistinguishable from the full transcript (CI touches zero) at **4,204 fewer
prompt tokens**. That is a real efficiency property. It is not a reason to
prefer the digest, because the previous chapter is cheaper *and* better.

## 3. Why the state fails here

Extraction, as in the reading study. Capture by canon type at chapter 20:

| type | captured |
| --- | --- |
| eye colour | 0.50 |
| trade | 0.38 |
| hair | 0.33 |
| birthplace | 0.23 |
| **object material** | **0.12** |

The extractor is character-centric and largely ignores objects, so half the
planted canon never enters the state. A representation cannot protect what it
never captured, and no amount of work on the calculus changes that.

## 4. The artefact caught in run 1

Run 1 also included a LoRA fine-tuned on 6,876 state-conditioned examples. Its
violation rate was **0.054** against 0.367 for the base model — apparently a 7×
win. It was not real. The model had degenerated into pastiche of its training
corpus, reproducing even Project Gutenberg's hard-wrap indentation, and a model
that stops writing about the premise cannot violate its canon.

Two measurements expose it: base conditions write about the premise in 100% of
chapters against 77–81% for the tuned ones, and restate canon at 0.84 against
0.09. `on_premise` is now a first-class metric so this cannot be reported as a
win again. The tuned arm is excluded from every conclusion above.

## 5. What stands

The **reading** result is unaffected and remains well evidenced: the revision
calculus holds a read state at 0.000 self-contradictory slots against
0.370–0.423 for an append-only store, replicated at 1.5B / 3B / 7B, scored
against human annotation on 28 novels.

The **generation** result is negative and now has two independent runs and a
diagnosed cause behind it. On this evidence a maintained narrative graph is a
good instrument for *measuring* a text and a poor conditioning signal for
*writing* one.
