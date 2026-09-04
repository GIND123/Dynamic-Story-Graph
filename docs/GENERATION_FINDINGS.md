# Generation findings — and a false positive caught

30 stories × 20 chapters × 9 conditions, 5,400 chapters, Qwen2.5-3B base and a
LoRA trained on 6,876 state-conditioned examples from 215 novels. Canon is
planted in chapter 1 and withheld afterwards; violations are deterministic
string tests near a mention of the subject. Paired bootstrap over stories.

**Headline: the graph-state approach does not work for generation at this
scale, and the fine-tuned writer's apparent win is an artefact.**

## 1. The artefact, first

The raw table looked like a large win for the fine-tuned writer: violation rate
**0.054** against **0.367** for the base model with full context — nearly 7×
better.

It is not real. Violations are only counted near a mention of their subject, so
a model that stops writing about the premise cannot violate anything. The
fine-tune degenerated into pastiche of its training corpus:

> *"I have been in a great deal of trouble. I have had a great deal of trouble.
> I have had a great deal of trouble, and I have had a great deal of trouble."*
> — `tuned:full-context`, chapter 6

> *"After that, I had to pay attention to my mother…"* — `tuned:dsg-state`,
> chapter 6, in a story with no mother in it

It even reproduced Project Gutenberg's hard-wrap indentation, which is the
clearest sign of what it actually learned: the *surface form* of the corpus,
not the conditioning.

Two measurements make this quantitative, and the first was added only after the
prose was read:

| | base conditions | tuned conditions |
| --- | --- | --- |
| chapters that mention any premise character | **1.000** | 0.765 – 0.812 |
| canon facts actively restated | 0.838 – 0.887 | **0.092 – 0.163** |

So the tuned arm is not comparable to the base arm and is excluded from every
conclusion below. `on_premise` is now a first-class metric precisely so this
cannot be reported as a win again.

## 2. What the valid comparisons say

All base conditions write about the premise in 100% of chapters, so these are
like-for-like.

| comparison | Δ violation rate | 95% CI | wins |
| --- | --- | --- | --- |
| full-context − rolling-summary | **−0.146** | [−0.196, −0.096] | 21–2 |
| rolling-summary − none | −0.008 | [−0.075, +0.058] | ns |
| **dsg-state − none** | **+0.075** | [+0.008, +0.142] | 8–15 |
| **dsg-state − full-context** | **+0.229** | [+0.171, +0.288] | 0–25 |
| dsg-state − append-only-state | −0.017 | [−0.054, +0.021] | ns |

Three things follow, and two of them go against the method.

1. **Long context wins.** Pasting the transcript beats every alternative by a
   wide margin (0.367 vs 0.521 for no memory).
2. **The graph digest is worse than no memory at all** (+0.075, interval
   excludes zero). Not merely worse than the transcript — worse than giving the
   writer nothing.
3. **Revision makes no difference here.** `dsg-state` and `append-only-state`
   are indistinguishable, so the calculus that was decisive for *reading* buys
   nothing for *writing* in this setup.

## 3. Why the digest hurts

The mechanism is measurable: `canon_capture` — the share of planted facts the
state actually holds — is **0.321** for `dsg-state`. The digest carries a third
of the canon, so the writer is handed a confident-looking but impoverished
summary in place of the text. A partial canon is worse than none, because it
crowds the prompt while omitting most of what it is supposed to protect.

(`append-only-state` shows a higher capture, 0.550, but that is capture by
hoarding: it keeps contradictory values live, so the checker finds the correct
one among them. It is not evidence of a better representation.)

Extraction is therefore the binding constraint, exactly as it was in the
reading study. The state cannot protect what it never captured.

## 4. The one thing that did hold

Cost. `dsg-state` uses **325** prompt tokens at the final chapter against
**5,308** for `full-context` — a 4,983-token gap that widens with every
chapter, because a digest does not grow and a transcript does.

That is a real property and it is why the approach is worth repairing rather
than abandoning: at 100 chapters the transcript does not fit at all. But on
these results the digest is 16× cheaper *and decisively worse*, so the scaling
argument does not rescue it. It would only matter once capture is high enough
for the digest to be a faithful substitute.

## 5. What would have to change

In the order I would try them:

1. **Fix extraction, not the representation.** 0.321 capture is the whole
   story. Extracting canon-bearing facts reliably is a smaller, better-posed
   problem than anything downstream.
2. **Give the writer the digest *and* recent text.** The two are not rivals;
   the experiment forced a choice between them that no real system would make.
3. **Retrain properly.** The LoRA learned corpus style over conditioning —
   lower rank or fewer epochs, targets stripped of Gutenberg formatting, and a
   held-out check on premise adherence *during* training, not after.
4. Only then re-run this comparison.

## 6. Honest status

The reading result stands: the calculus holds a read state at 0.000
self-contradictory slots against 0.370–0.423 for append-only, replicated at
three model sizes. That is about maintaining a state, and it is well evidenced.

The generation result does not stand. On this evidence a maintained graph state
is not a useful conditioning signal for long-form generation, and the trained
writer failed outright. Both are reported here rather than buried.
