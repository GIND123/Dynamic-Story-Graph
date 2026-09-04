# DSG — Dynamic Story Graph

Revision-aware, prefix-causal narrative state: build the reader's model of a
novel **as the novel is read**, and let it be corrected without breaking.

```bash
python -m dsg doctor                                   # environment + corpus check
python -m pytest tests/dsg -q                          # 33 tests, no weights needed

# extraction (pick one)
modal run dsg/infra/modal_extract.py --model qwen7b    # GPU, whole corpus
python -m dsg extract --backend qwen3b --books 3       # local Apple-Silicon MLX

python -m dsg study  --proposals artifacts/proposals/pdnc-qwen7b-w3200 \
                     --out artifacts/results/pdnc-qwen7b
python -m dsg report --results artifacts/results/pdnc-qwen7b --out artifacts/report
```

## The idea in one paragraph

Incremental narrative extraction is usually blamed for being lossy. It is not
incrementality that hurts — it is **premature commitment**. A reader who writes
down "the stranger" as a *committed* entity has to tear the graph up when the
stranger turns out to be Magwitch; a reader who writes it down as *deliberately
underspecified* only has to fill a slot. DSG makes that distinction executable,
and separates three update types that append-only pipelines conflate:

| Operation | When | Monotone? | What happens to the old belief |
| --- | --- | --- | --- |
| `ELABORATE` | later text *specifies* what earlier text left open | yes | refined in place; earlier state stays a valid substructure |
| `SUPERSEDE` | the **story world** changed | history-preserving | validity interval closed, kept as history |
| `REVISE` | the **reader** was wrong; it was never true | no (rollback) | retracted, and so is anything derived from it |

`SUPERSEDE` vs `REVISE` is the operational form of the narratological split
between a change in the story and a revelation in the telling. It is decided by
an inspectable table (`dsg/lexicon.py`) — predicate mutability, evidential
source, and surface revelation markers — never by the language model.

## Architecture

```
text ──► windows ──► [LLM proposes]  ──► [calculus disposes] ──► state
        prefix-causal   typed, cached      classify · resolve · check
                        never writes       ELABORATE / SUPERSEDE / REVISE
                                           + 7 structural invariants
```

The model sees one window and a bounded digest of names met so far, and returns
line-oriented proposals. It has no write access to the store. Everything after
extraction is deterministic, so a change to the calculus is re-scored for free.

| Module | Role |
| --- | --- |
| `schemas.py` | assertions with two clocks — discourse time (when the reader learned it) and story time (when it held) |
| `lexicon.py` | predicate mutability, revelation markers, underspecification |
| `matching.py` | conservative surface matching; a shared surname never merges anyone |
| `store.py` | the calculus, the policy ladder, the revision log |
| `invariants.py` | seven structural checks run after every reading step |
| `proposals.py` | prompt, tolerant line parser, policy-agnostic digest |
| `policies/runner.py` | replays one cached proposal stream under any policy |
| `eval/` | identity (gold aliases), speaker attribution, process integrity, paired bootstrap |

## The policy ladder

Every policy replays a **byte-identical** proposal stream, so differences are
differences of representation, not of extractor quality.

| Policy | State | Merge identity | Elaborate / supersede / revise facts | Commitment | Causal |
| --- | --- | --- | --- | --- | --- |
| `window-only` | — | — | — | — | yes |
| `append-only` | yes | — | — | eager | yes |
| `dsg-merge` | yes | **yes** | — | eager | yes |
| `dsg-eager` | yes | yes | **yes** | eager | yes |
| `dsg-full` | yes | yes | yes | **deferred** | yes |
| `retrospective` | yes | yes | yes | deferred | **no** |

Note that `ELABORATE` sits on the revision side of the line. It is monotone
in information, but it still rewrites a stored value, and a store that only
appends cannot do that — it has to add a second assertion and hold both. The
baselines are therefore not quietly granted it.

Each rung adds exactly one mechanism, so each is separately priced:

- `append-only` → `dsg-merge` isolates **identity merging**;
- `dsg-merge` → `dsg-eager` isolates **fact-level revision**;
- `dsg-eager` → `dsg-full` isolates **deferred commitment**, holding the
  resolution decisions themselves fixed — the two policies bind surfaces
  identically and differ only in whether the resulting update is a monotone
  elaboration or a rollback;
- `dsg-full` → `retrospective` prices **causality**, by handing the same
  calculus full lookahead.

## Evaluation

No LLM judge appears in any primary metric.

- **Identity** — the entity registry induces a partition over the gold alias
  strings a human annotator assigned to each character (PDNC). Scored with
  MUC / B³ / CEAF-e and their CoNLL average. The item set is fixed per book
  *before* any system runs, so extracting less cannot raise the score.
- **Speaker attribution** — every gold quotation, answered from state built
  only from earlier text.
- **Process integrity** — structural violations, monotone fraction, rollback
  count, and the revision profile over discourse position.
- **Length control** — the identical protocol on LitBank's ~2K-token excerpts,
  where premature commitment has little time to hurt.

## Generation: consistency at length

The reading state is only interesting if it constrains writing. `dsg/generate/`
asks whether a story can get **longer without breaking its own graph**.

The measurement problem is that a generated story has no gold. So the gold is
planted: each premise fixes canon facts drawn from closed vocabularies whose
contradictions are enumerable — eye colour, metal, kinship, trade, birthplace —
which makes a violation a deterministic string test near a mention of the
subject. A nearby correction ("not gold but silver") does not count. No model
and no human judgement enters the measurement.

**Canon is supplied in chapter 1 only.** From chapter 2 each condition keeps
whatever its own memory carries, so the experiment tests memory rather than
prompt-following.

| Condition | What the writer is given for chapter t |
| --- | --- |
| `none` | premise + outline beat |
| `last-chapter` | + the previous chapter verbatim |
| `rolling-summary` | + a running summary the model maintains |
| `full-context` | + every previous chapter, truncated head-and-tail so the opening (where canon was established) survives |
| `append-only-state` | + a state digest that can never be revised |
| `dsg-state` | + a state digest under the full revision calculus |

The last pair differs only in whether what the writer is shown may be revised.
Chapters are read back into state with `apply_window`, the same function the
novel study uses — a second implementation would make the two settings
incomparable.

Reported alongside accuracy: **context tokens per chapter**. A digest does not
grow with the story; a transcript does, which is what makes the comparison to
`full-context` about scaling rather than only about accuracy.

```bash
modal run --detach dsg/infra/modal_generate.py::main --stories 30 --chapters 20
python -m dsg gen-report --generation artifacts/generation/<run>/generation.json                          --out artifacts/report/generation
```

## Corpora

Fetched to `data/`, never redistributed here.

| Corpus | What it supplies | Source |
| --- | --- | --- |
| PDNC (28 novels) | full text, gold alias sets, 37,131 gold quotations with speakers | [Priya22/project-dialogism-novel-corpus](https://github.com/Priya22/project-dialogism-novel-corpus) |
| LitBank (100 excerpts) | gold coreference, short-context control | [dbamman/litbank](https://github.com/dbamman/litbank) |

## Reproducing

The GPU stage is the only paid step and runs once per (corpus, model); the
proposal stream is cached to `artifacts/proposals/` and everything downstream is
CPU-deterministic. `--backend mock` runs the entire pipeline with no weights and
no network, which is what CI exercises.
