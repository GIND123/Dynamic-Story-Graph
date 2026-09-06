# Generation findings

Four runs, 18,600 generated chapters, 30 stories, conditions paired within
story. Canon is planted in chapter 1 and withheld afterwards; violations are
deterministic string tests near a mention of the subject. No language model
appears in any metric.

**Summary. Five ways of using a narrative graph at generation time were tested.
None of them helps. The fifth, ranking candidates, is the one the project's own
negative results predicted should work, and it does not either.**

## 1. What was tried

| use of the graph | design | outcome |
|---|---|---|
| carried as context | serialised state digest in the prompt | worse than no memory |
| carried plus recent text | digest and the previous chapter | loses to the previous chapter alone |
| queried by the next beat | retrieve only what the beat implicates | worse than no memory |
| queried plus recent text | retrieval and the previous chapter | ties the previous chapter alone |
| used to rank candidates | draw four, pick the most consistent | indistinguishable from picking blind |

## 2. The ranking result, and why the control mattered

Ranking was motivated by the finding in `METHODOLOGY.md` section 5 that absolute
contradiction counts are dominated by extraction noise while relative
comparisons on identical input are sound. Ranking four candidates for the same
chapter position is the relative case, so the noise should cancel in the
ordering.

| condition | violations | prompt tokens |
|---|---|---|
| previous chapter, one sample | 0.354 | 1,005 |
| previous chapter, four samples chosen blind | 0.296 | 1,012 |
| previous chapter, four samples ranked by the graph | 0.300 | 1,017 |
| full transcript, one sample | 0.400 | 5,360 |

| comparison | difference | interval | excludes zero |
|---|---|---|---|
| graph ranked minus one sample | -0.054 | [-0.129, +0.025] | no |
| blind choice minus one sample | -0.058 | [-0.142, +0.025] | no |
| **graph ranked minus blind choice** | **+0.004** | [-0.058, +0.063] | **no** |

The graph adds nothing over choosing blind from the same four candidates. Any
apparent gain belongs to the extra sampling, and even that does not clear the
interval.

Without the blind arm this would have been reported as a reduction of 0.054
from graph ranking. The control is what separates a mechanism from a budget.

## 3. A correction, and a limit of the benchmark

Earlier notes in this project stated that the previous chapter beats the full
transcript, and separately that the full transcript is best. Both were read from
single runs and neither is safe.

| condition | run 2 | run 3 | run 4 | spread |
|---|---|---|---|---|
| no memory | 0.504 | 0.487 | not run | 0.017 |
| previous chapter | 0.308 | 0.312 | 0.354 | 0.046 |
| **full transcript** | 0.329 | 0.254 | 0.400 | **0.146** |

The full transcript condition moves by 0.146 across runs, which is larger than
almost every effect discussed in this project. Its ordering against the previous
chapter is therefore **not established**, and the honest statement is that both
beat no memory while their ranking against each other is unresolved at this
sample size.

What survives the variance is the part that is large and repeated: no memory is
clearly worst, and every graph conditioning design lands at or below the memory
baselines rather than above them.

**Implication for the benchmark.** Conditions whose prompt grows without bound
are unstable across seeds. Effects below roughly 0.15 on such conditions need
several seeds before they are reported. Within run paired comparisons remain
valid, because premises and sampling seeds are shared; cross run comparisons of
a single condition are not.

## 4. The false positive caught in run 1

Run 1 included a LoRA fine tuned on 6,876 state conditioned examples. It scored
0.054 against 0.367 for the base model, an apparent sevenfold win. It was not
real: the model had degenerated into pastiche of its training corpus,
reproducing even the hard wrap indentation of Project Gutenberg, and a model
that stops writing about the premise cannot violate its canon.

| | base conditions | tuned conditions |
|---|---|---|
| chapters mentioning any premise character | 1.000 | 0.765 to 0.812 |
| canon facts actively restated | 0.838 to 0.887 | 0.092 to 0.163 |

`on_premise` is now a first class metric. The tuned arm is excluded from every
conclusion.

## 5. Why the conditioning line is closed

Five designs, four runs, 18,600 chapters, each falsifier stated in advance.
Three separate controls each removed an apparent effect: `on_premise` removed a
sevenfold win, the blind selection arm removed a 0.054 gain, and the cross run
spread removed a baseline ordering.

The state held 0.492 of the planted canon at the moment of writing and
conditioning on it still added nothing, so this is not an extraction ceiling.
The serialised digest result also replicates a published negative, the Narrative
World Model paper's State Memory condition, with a 3B open model.

The reading result in `METHODOLOGY.md` section 2 is unaffected. It compares
policies on identical input, where a shared noise floor cancels, and it holds at
three extractor sizes.
