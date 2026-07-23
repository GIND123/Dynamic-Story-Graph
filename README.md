# Dynamic Narration Graph

Research workspace for long-context consistent story generation.

## Projects

- [`gnsm/`](gnsm/) — Grounded Narrative State Model: symbolic extraction,
  learned narrative state, supervised transitions, conditioned generation, and
  deterministic consistency verification.
- [`manuscript-memory-engine/`](manuscript-memory-engine/) — the existing graph
  memory baseline and evaluation harness.
- [`Dataset Generation/`](Dataset%20Generation/) and
  [`Long Model Comparison/`](Long%20Model%20Comparison/) — research notebooks.

Start with `python -m gnsm demo` for a dependency-light end-to-end GNSM run.
See [`gnsm/README.md`](gnsm/README.md) for setup, architecture, and extension
points.

## Run on Colab (clone → paste → run)

Open [`gnsm/colab/GNSM_Colab.ipynb`](gnsm/colab/GNSM_Colab.ipynb) in Colab, or
paste this into a fresh **GPU** Colab cell (Runtime → Change runtime type → GPU):

```python
!git clone https://github.com/GIND123/Dynamic-Narration-Graph.git
%cd Dynamic-Narration-Graph
from gnsm.colab.bootstrap import ensure_hf_token; ensure_hf_token()  # activate ambient HF token, no prompt
!python -m gnsm.colab.bootstrap   # keeps Colab's CUDA torch, installs the rest, prints a report
!python -m gnsm demo              # deterministic end-to-end loop (no downloads)
!python -m gnsm smoke --json      # trains the neural state stack on the GPU
```

CUDA target is **cu121** (compatible with Colab and CUDA 12.1–12.4 hosts).
Hugging Face auth is never prompted: gated models (Llama-3.1, Gemma, ...) are
reached through a token that is already active in the session — an `HF_TOKEN`
env var, a prior `huggingface-cli login`, or a Colab secret named `HF_TOKEN`
(or `hf`). Public models need nothing. Full notes:
[`gnsm/colab/README.md`](gnsm/colab/README.md).

---

## Datasets

Eight corpora back the project. None are redistributed here — every one is
fetched from its original source into `data/`, which is git-ignored.

| Dataset | Role in GNSM | Source |
| --- | --- | --- |
| **LitBank** | NER + coref + event supervision for the extraction plane | [dbamman/litbank](https://github.com/dbamman/litbank) |
| **BookCoref** | Full-book coref → breaks LitBank's ~2K-token ceiling | [sapienzanlp/bookcoref](https://huggingface.co/datasets/sapienzanlp/bookcoref) |
| **PDNC** | Quote/speaker edge training + quote-attribution eval | [Priya22/project-dialogism-novel-corpus](https://github.com/Priya22/project-dialogism-novel-corpus) |
| **EvolvTrip / LitCharToM** | Gold temporal belief/desire/emotion/intention → supervises the soft internal-state heads | [yangbh217/EvolvTrip](https://huggingface.co/datasets/yangbh217/EvolvTrip) |
| **ConStory-Bench** | Primary consistency evaluation (CED / GRR, 5 categories / 19 subtypes) | [jayden8888/ConStory-Bench](https://huggingface.co/datasets/jayden8888/ConStory-Bench) |
| **FABLES** | Book-length faithfulness eval | [mungg/FABLES](https://github.com/mungg/FABLES) |
| **GPT4-Books** | Name-cloze eval and memorization guardrail | [bamman-group/gpt4-books](https://github.com/bamman-group/gpt4-books) |

Roles and split policies are registered in
[`gnsm/data/manifests/datasets.yaml`](gnsm/data/manifests/datasets.yaml).

### Train / eval boundary

`constory_bench`, `fables`, and `gpt4_books` are marked `trainable: false` in the
manifest. FABLES additionally ships a **canary string** at the top level of its
JSON declaring that the benchmark must never appear in training corpora — keep
that key intact in any derived artifact rather than stripping it, so the
contamination check still works downstream.

### Loader status

Only three datasets have working loaders in this repo today:

| Dataset | Loader |
| --- | --- |
| PDNC | `manuscript-memory-engine/ingestion/pdnc.py` |
| LitBank | `manuscript-memory-engine/ingestion/litbank.py` |
| GPT4-Books | `manuscript-memory-engine/evals/metrics/gpt4_books_cloze.py` |

BookCoref, EvolvTrip/LitCharToM, ConStory-Bench, and FABLES are downloaded and
verified but still need adapters written against
`manuscript-memory-engine/ingestion/base.py`. All four are plain JSON / JSONL /
Parquet on disk after the steps below.

---

## Loading the datasets in Colab

Run these after the bootstrap cell above. Each cell uses explicit paths rather
than environment variables, so they are safe to run in any order and after a
runtime restart.

### Step 1 — clone the six git-hosted corpora

```python
%cd /content/Dynamic-Narration-Graph
!mkdir -p data
%cd data

# LitBank
!git clone --depth 1 https://github.com/dbamman/litbank.git litbank

# PDNC
!git clone --depth 1 https://github.com/Priya22/project-dialogism-novel-corpus.git pdnc

# EvolvTrip / LitCharToM  (code + schema docs only; data comes in Step 2)
!git clone --depth 1 https://github.com/Bernard-Yang/EvolvTrip.git evolvtrip

# ConStory-Bench  (evaluation code only; prompts come in Step 2)
!git clone --depth 1 https://github.com/Picrew/ConStory-Bench.git constory_bench

# FABLES
!git clone --depth 1 https://github.com/mungg/FABLES.git fables

# GPT4-Books
!git clone --depth 1 https://github.com/bamman-group/gpt4-books.git gpt4_books

%cd /content/Dynamic-Narration-Graph
!du -sh data/*
```

Expected: `litbank` 116M, `pdnc` 40M, `gpt4_books` 41M, `constory_bench` 5.6M,
`fables` 4.6M, `evolvtrip` 424K.

`fatal: destination path ... already exists` on a re-run is harmless — git is
refusing to overwrite an existing clone.

### Step 2 — pull the two Hugging Face corpora

Cloned over plain git rather than the `datasets` library, so this is immune to
`huggingface_hub` / `datasets` version drift in the Colab image.

```python
%cd /content/Dynamic-Narration-Graph/data
!git lfs install

# ConStory-Bench — 2,000 prompts + generated stories + per-model evaluations
!git clone https://huggingface.co/datasets/jayden8888/ConStory-Bench constory_bench/hf_data

# EvolvTrip / LitCharToM — the GitHub repo above is docs-only; data lives here
!git clone https://huggingface.co/datasets/yangbh217/EvolvTrip evolvtrip_data

%cd /content/Dynamic-Narration-Graph
!du -sh data/constory_bench/hf_data data/evolvtrip_data
```

Expected: `constory_bench/hf_data` **5.5G**, `evolvtrip_data` 11M.

The ConStory-Bench clone is large because LFS pulls generated stories for ~38
models. Only `prompts.parquet` is needed to run the benchmark on your own model;
`stories.parquet` and `evaluations/*.csv` are other models' outputs for
leaderboard comparison. To skip them:

```python
%cd /content/Dynamic-Narration-Graph/data
!GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/jayden8888/ConStory-Bench constory_bench/hf_data
!cd constory_bench/hf_data && git lfs pull --include="prompts.parquet"
%cd /content/Dynamic-Narration-Graph
```

### Step 3 — unzip FABLES

The annotations ship compressed in the GitHub clone.

```python
import zipfile, json

root = "/content/Dynamic-Narration-Graph/data/fables"
with zipfile.ZipFile(f"{root}/data/FABLES.json.zip") as z:
    z.extractall(f"{root}/data")

data = json.load(open(f"{root}/data/FABLES.json"))
print("top-level keys:", list(data))          # ['canary', 'FABLES']
print("books:", len(data["FABLES"]))          # 26
```

Structure is `FABLES[book][summarizer_model]` → `general_comment`, `summary`,
`claims`. Five summarizers per book: GPT-3.5-TURBO, GPT-4, GPT-4-TURBO, MIXTRAL,
CLAUDE-3-OPUS. Note the sibling `canary` key described above.

FABLES contains claim-level annotations only — the source books are recent
commercial titles and are not redistributed.

### Step 4 — BookCoref (needs a temporary version pin)

BookCoref publishes annotations without book text; the `sentences` field is
reconstructed at load time by a dataset script that fetches Project Gutenberg
texts through the Wayback Machine. Script-based loading was removed in
`datasets` 4.0, so this step pins `datasets==3.6.0`, downloads, then restores.

**4a — pin, then restart the runtime**

```python
# pins from the BookCoref dataset card
!pip install -q "datasets==3.6.0" "deepdiff==8.5.0" "spacy==3.8.7" "nltk==3.9.1"

# hub version compatible with datasets 3.6.0 (<1.0); --force-reinstall avoids
# the mixed-file state that a plain downgrade can leave behind
!pip install -q --force-reinstall --no-deps "huggingface_hub==0.34.4"

print("Now: Runtime > Restart session, then run 4b.")
```

> ### ⚠️ RESTART THE RUNTIME NOW — RUNTIME → RESTART SESSION
>
> THE PIN DOES NOT TAKE EFFECT UNTIL YOU RESTART. `datasets` IS ALREADY
> IMPORTED IN THE KERNEL, SO PIP ONLY SWAPS FILES ON DISK. SKIPPING THIS
> RESTART MAKES STEP 4b FAIL WITH
> `RuntimeError: Dataset scripts are no longer supported`.

Dependency-conflict warnings about `gradio` are unrelated and safe to ignore.

After restarting, confirm the pin took before spending 35 minutes on the download:

```python
import datasets, huggingface_hub
print("datasets:", datasets.__version__)   # must be 3.6.0
print("hub:", huggingface_hub.__version__) # must be 0.34.4
```

**4b — download (~35 minutes)**

```python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset

bookcoref = load_dataset("sapienzanlp/bookcoref", trust_remote_code=True)
print(bookcoref)

out = "/content/Dynamic-Narration-Graph/data/bookcoref"
os.makedirs(out, exist_ok=True)
for split, ds in bookcoref.items():
    ds.to_json(f"{out}/{split}.jsonl")
    print(split, ds.num_rows, "docs")
```

`trust_remote_code=True` is required and is supported again under 3.6.0. Most of
the runtime is tokenizing 52 Gutenberg books; it happens once. Expected splits:
45 train / 5 validation / 3 test, ~103M on disk.

If it fails on a missing spaCy model, run
`!python -m spacy download en_core_web_sm` and re-run. Interrupted Wayback
fetches are cached, so re-running resumes.

**4c — restore versions, then restart again**

```python
!pip install -q -U "datasets>=4"
!pip install -q --force-reinstall --no-deps "huggingface_hub==0.36.2"
print("Now: Runtime > Restart session, then run Step 5.")
```

> ### ⚠️ RESTART THE RUNTIME NOW — RUNTIME → RESTART SESSION
>
> THE KERNEL IS STILL HOLDING `datasets` 3.6.0 IN MEMORY. WITHOUT THIS
> RESTART THE REST OF THE PIPELINE RUNS AGAINST THE OLD PINNED VERSIONS.

`0.36.2` matches what `gnsm.colab.bootstrap` installs. Nothing downstream reads
BookCoref through `datasets` — the adapter reads the JSONL directly.

### Step 5 — set dataset roots

> ### ⚠️ RE-RUN THIS CELL AFTER EVERY RUNTIME RESTART
>
> ENVIRONMENT VARIABLES DO NOT SURVIVE A RESTART. IF STEP 6 RAISES
> `KeyError: 'PDNC_ROOT'`, THIS CELL HAS NOT BEEN RUN IN THE CURRENT SESSION.

```python
import os, glob
from pathlib import Path

REPO = Path("/content/Dynamic-Narration-Graph")
DATA = REPO / "data"

# GPT4-Books nests model_output/chatgpt_results under a version-dependent
# subfolder, so locate it rather than hardcoding the path.
hits = glob.glob(str(DATA / "gpt4_books" / "**" / "model_output" / "chatgpt_results"), recursive=True)
assert hits, "chatgpt_results not found — check the gpt4_books clone"
gpt4_root = str(Path(hits[0]).parent.parent)

env = {
    # --- read by manuscript-memory-engine/app/config.py ---
    "PDNC_ROOT": str(DATA / "pdnc"),
    "LITBANK_ROOT": str(DATA / "litbank"),
    "GPT4BOOKS_ROOT": gpt4_root,

    # --- no loader yet; names follow gnsm/data/manifests/datasets.yaml ---
    "BOOKCOREF_ROOT": str(DATA / "bookcoref"),
    "EVOLVTRIP_ROOT": str(DATA / "evolvtrip_data"),       # HF data, not the docs repo
    "LITCHARTOM_ROOT": str(DATA / "evolvtrip_data"),      # same repo hosts both
    "EVOLVTRIP_DOCS": str(DATA / "evolvtrip"),            # README = schema reference
    "CONSTORY_BENCH_ROOT": str(DATA / "constory_bench"),
    "FABLES_ROOT": str(DATA / "fables"),

    # loader smoke tests need no real API key
    "LLM_MODE": "fake",
}
os.environ.update(env)

# subprocesses (python -m ingestion.cli, ...) read this via pydantic-settings
(REPO / "manuscript-memory-engine" / ".env").write_text(
    "\n".join(f"{k}={v}" for k, v in env.items()) + "\n"
)

for k, v in env.items():
    status = ("OK" if Path(v).exists() else "MISSING") if k.endswith(("_ROOT", "_DOCS")) else "set"
    print(f"{k:22} {status:7} {v}")
```

All `_ROOT` / `_DOCS` lines should read `OK`. `LLM_MODE` reads `set` — it is a
mode string, not a path.

### Step 6 — verify every dataset

```python
!pip install -q pydantic pydantic-settings

import sys, os, json
import pandas as pd
from pathlib import Path
sys.path.insert(0, "/content/Dynamic-Narration-Graph/manuscript-memory-engine")

from ingestion.pdnc import PDNCLoader
from ingestion.litbank import LitBankLoader
from evals.metrics.gpt4_books_cloze import book_path, read_book

# --- PDNC: speaker/addressee ---
novels = sorted(p.name for p in (Path(os.environ["PDNC_ROOT"]) / "data").iterdir() if p.is_dir())
ir = PDNCLoader(Path(os.environ["PDNC_ROOT"]) / "data" / novels[0]).load()
print(f"PDNC        {len(novels)} novels | {novels[0]}: {len(ir.entities)} chars, "
      f"{len(ir.quotations)} quotes, {len(ir.segments)} segments")

# --- LitBank: entities/events ---
docs = sorted(p.stem for p in (Path(os.environ["LITBANK_ROOT"]) / "entities" / "tsv").glob("*.tsv"))
ir2 = LitBankLoader(os.environ["LITBANK_ROOT"], docs[0]).load()
print(f"LITBANK     {len(docs)} docs | {docs[0]}: {len(ir2.entities)} entities, {len(ir2.events)} events")

# --- GPT4-Books: name cloze ---
cloze_dir = Path(os.environ["GPT4BOOKS_ROOT"], "model_output", "chatgpt_results")
books = sorted(p.stem for p in cloze_dir.glob("*.txt"))
rows = read_book(book_path(books[0], os.environ["GPT4BOOKS_ROOT"]))
print(f"GPT4BOOKS   {len(books)} books | {books[0]}: {len(rows)} cloze rows | gold[0]={rows[0].gold!r}")

# --- BookCoref: full-book coref (sentences must be populated) ---
for split in ["train", "validation", "test"]:
    with open(f"{os.environ['BOOKCOREF_ROOT']}/{split}.jsonl") as f:
        row = json.loads(f.readline())
    n_tok = sum(len(s) for s in row["sentences"])
    print(f"BOOKCOREF   {split:10} {row['doc_key']:28} tokens={n_tok:7} "
          f"clusters={len(row['clusters'])} chars={len(row['characters'])}")

# --- EvolvTrip / LitCharToM: temporal ToM ---
for fname in ["all_books_current.json", "all_books_with_prev.json", "ood_test_book.json"]:
    d = json.load(open(f"{os.environ['EVOLVTRIP_ROOT']}/{fname}"))
    print(f"EVOLVTRIP   {fname:26} {len(d)} records | keys={list(d[0])[:4]}...")

# --- ConStory-Bench: consistency prompts ---
prompts = pd.read_parquet(f"{os.environ['CONSTORY_BENCH_ROOT']}/hf_data/prompts.parquet")
print(f"CONSTORY    {prompts.shape[0]} prompts | task types: "
      f"{sorted(prompts['task_type'].unique())}")

# --- FABLES: faithfulness annotations ---
fab = json.load(open(f"{os.environ['FABLES_ROOT']}/data/FABLES.json"))
print(f"FABLES      {len(fab['FABLES'])} books x "
      f"{len(next(iter(fab['FABLES'].values())))} summarizers | canary present: {'canary' in fab}")
```

Reference output from a clean run:

```text
PDNC        28 novels | AHandfulOfDust: 104 chars, 2337 quotes, 375 segments
LITBANK     100 docs | 1023_bleak_house_brat: 154 entities, 61 events
GPT4BOOKS   571 books | 1023_bleak_house: 100 cloze rows | gold[0]='Charley'
BOOKCOREF   train      o_pioneers!_24               tokens=  67463 clusters=12 chars=12
BOOKCOREF   validation the_boxcar_children_42796    tokens=  30819 clusters=10 chars=10
BOOKCOREF   test       siddhartha_2500              tokens=  47785 clusters=9 chars=9
EVOLVTRIP   all_books_current.json     638 records | keys=['book_name', 'character', 'plot_index', 'plot_summary']...
EVOLVTRIP   all_books_with_prev.json   638 records | keys=['book_name', 'character', 'plot_index', 'plot_summary']...
EVOLVTRIP   ood_test_book.json         779 records | keys=['messages', 'corresponding_triples', 'triples', 'plot_id']...
CONSTORY    2000 prompts | task types: ['completion', 'continuation', 'expansion', 'generation']
FABLES      26 books x 5 summarizers | canary present: True
```

---

## Data layout

```text
data/
├── litbank/                    LitBank clone (entities/, events/, coref/)
├── pdnc/                       PDNC clone (data/<Novel>/)
├── gpt4_books/                 GPT4-Books clone; root is data/
├── bookcoref/                  train|validation|test .jsonl with sentences
├── evolvtrip/                  docs + schema reference only
├── evolvtrip_data/             EvolvTrip + LitCharToM JSON
├── constory_bench/             evaluation code
│   └── hf_data/                prompts.parquet, stories.parquet, evaluations/
└── fables/data/FABLES.json     faithfulness annotations
```

Total ≈ 5.9 GB, dominated by ConStory-Bench's generated stories.

`data/` is git-ignored; corpora are never committed. See
[`gnsm/data/README.md`](gnsm/data/README.md) for the intended normalized layout
(`raw/`, `processed/`, `cache/`, `manifests/`).

## Notes

- The dataset steps run fine on CPU. Switch to **Runtime → Change runtime type →
  GPU** before training.

> ### ⚠️ SWITCHING TO GPU RESETS THE RUNTIME
>
> `/content` SURVIVES, SO THE DOWNLOADED DATA IS SAFE AND NOTHING NEEDS
> RE-DOWNLOADING. ONLY STEP 5 MUST BE RE-RUN TO RESTORE THE ENVIRONMENT
> VARIABLES.

- `python -m ingestion.cli pdnc <Novel>` writes into Neo4j and Redis from
  `manuscript-memory-engine/docker-compose.yml`, and will not work in stock
  Colab. Step 6 exercises the same loaders without the graph write. For the graph
  itself in Colab, point `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` at a
  hosted instance.
