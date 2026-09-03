# Outreach draft — Prof. David Bamman (UC Berkeley)

> **Status: draft for you to send.** I cannot send email from here. Read it,
> change the voice to yours, and send it yourself. Numbers marked `<<...>>` are
> filled in from `artifacts/report/pdnc-qwen7b/results.md` once the full run
> finishes — do not send with placeholders in it.
>
> One honest note before you do: a cold email does not get anyone a PhD. What it
> can do is start a conversation with someone who works on exactly this, on the
> strength of a result and a repository they can run. That is what this is built
> to do. Apply through the normal route as well.

---

**To:** dbamman@berkeley.edu
**Subject:** Incremental narrative state on PDNC/LitBank — premature commitment, not incrementality

Dear Professor Bamman,

I'm Govind. I've been working on a problem your datasets make possible to state
precisely, and I'd value your read on whether it's worth pursuing.

Narrative IE is almost always run retrospectively — the system annotating
chapter 1 has already seen chapter 40. If you make the same pipeline read
forward, as a reader does, the structure degrades, and that's usually blamed on
incrementality. I think the real culprit is **premature commitment**: writing a
decision into the graph as settled when the text hasn't settled it. An
append-only store can't say "this was true then", "this was never true", or
"these two people are one", so its errors are permanent and they compound.

So I built a narrative state store with an explicit revision calculus that
separates three things append-only pipelines conflate — monotone elaboration of
deliberately underspecified structure, supersession when the *story world*
changes, and revision when the *reader* was wrong — and evaluated it on 28 PDNC
novels under a strict prefix-causal constraint, with LitBank as a short-context
control. A 7B open-weight model proposes typed updates over one window; a
deterministic reconciler classifies them and checks seven structural invariants
after every reading step. No API anywhere, and no LLM judge in any metric —
everything is scored against the human annotation in PDNC and LitBank.

What came out, over <<N>> novels with paired bootstrap over books:

- <<HEADLINE 1 — structural integrity>>
- <<HEADLINE 2 — identity / mention linking>>
- <<HEADLINE 3 — deferral reduces rollbacks>>
- <<HEADLINE 4 — length dependence: the gap grows with reading depth; near-zero
  on LitBank's 2K-token excerpts>>

Two things I think you'd care about specifically. First, the "On classification
with large language models in cultural analytics" argument is why there is no
LLM judge anywhere in this: every primary number is scored against PDNC and
LitBank annotation, and the model is confined to proposing structure it cannot
write. Second, the revision trace turns out to be a measurement instrument in
its own right — it gives, per novel, how long the text holds a figure unnamed
before telling the reader who they are. <<INSTRUMENT RESULT>>

Code, data pipeline, and the full result tables are here, and the whole analysis
re-runs from cached extractions on a laptop: <<REPO URL>>

I'd like to do a PhD on computational narrative and I'd rather work on it with
you than anyone else. If you have ten minutes at some point, I'd be glad of your
view on whether the framing holds up — particularly the supersede/revise
distinction, which is the part I'm least sure survives contact with a
narratologist. And if you'd rather I applied and let the work speak, I'll do
that too.

Thank you for reading this far.

Govind
<<affiliation / current position>>
<<link: GitHub / site>>

---

## Notes for you before sending

- **Attach nothing.** One link. Cold emails with attachments get filtered.
- **Check the falsifiers section** of `docs/RESEARCH_PLAN.md` against what the
  run actually produced, and make sure the four bullets above match. If a
  pre-registered falsifier fired, say so in the email — it is a much stronger
  signal than a clean win, and he will check.
- **Timing:** avoid the week of a major deadline (ACL/EMNLP). Early in the week,
  morning Pacific.
- **If he replies asking for detail**, the thing to send is
  `artifacts/report/pdnc-qwen7b/results.md` — it is generated, so every number
  in it traces to a run.
- **Do not oversell.** The limitations section is real: one model family, one
  corpus of 28 English novels, alias-level rather than full mention-level gold,
  and an extraction layer that is the binding constraint on several metrics.
