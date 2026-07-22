# GNSM on Colab / cloud GPU

Everything needed to clone the repo into a fresh GPU runtime and run it.

## Fastest path — the notebook

Open [`GNSM_Colab.ipynb`](GNSM_Colab.ipynb) in Colab and run the cells top to
bottom. Set **Runtime → Change runtime type → GPU** first (a free T4 is enough
for the demo, the neural smoke, and small generators).

## Fastest path — one drop-in cell

Paste this into a fresh Colab cell (GPU runtime) and run it:

```python
# 1. clone
!git clone https://github.com/GIND123/Dynamic-Narration-Graph.git
%cd Dynamic-Narration-Graph

# 2. bootstrap the CUDA stack (keeps Colab's torch, installs the rest, prints a report)
!python -m gnsm.colab.bootstrap

# 3. run
!python -m gnsm demo            # deterministic end-to-end reference loop (no downloads)
!python -m gnsm smoke --json    # trains the neural state stack on GPU (wiring check)
```

## What the bootstrap does

`python -m gnsm.colab.bootstrap` (stdlib only, safe before any install):

1. prints `nvidia-smi`;
2. installs a **CUDA 12.1 (cu121)** build of torch *only if torch is missing*
   — Colab already ships a CUDA-matched torch, so it is kept as-is;
3. installs the GNSM package (editable) and `requirements-colab.txt`;
4. loads an `HF_TOKEN` from Colab **Secrets** if present (for gated models);
5. prints the `gnsm doctor` runtime report.

Flags: `--no-torch`, `--no-requirements`, `--no-editable`, `--no-doctor`.

## CUDA / torch policy

| Runtime | torch | CUDA target |
|---|---|---|
| Google Colab GPU | preinstalled, **kept** | matches Colab driver (12.x) |
| Bare GPU box, no torch | installed by bootstrap | **cu121** wheels |

cu121 wheels run on CUDA 12.1–12.4 hosts. To move the target, edit
`TORCH_CUDA_INDEX` / `CUDA_TAG` in [`bootstrap.py`](bootstrap.py).

## Authentication (Hugging Face) — no prompt

GNSM never runs an interactive `login()` and never asks for a token. Gated
models are reached through a token that is **already active in the session**.
The token is discovered, in order, from:

1. an environment variable — `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`,
   `HUGGINGFACE_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`);
2. a cached `huggingface-cli login` (typical on an HPC login node);
3. a Colab **secret** named `HF_TOKEN` — lowercase `hf` is also accepted.

On Colab, add the secret under the 🔑 panel and toggle *notebook access*. Because
each `!python …` line is a subprocess, run `ensure_hf_token()` from a **kernel**
cell first (the notebook and the drop-in cell already do) so those subprocesses
inherit `HF_TOKEN`:

```python
from gnsm.colab.bootstrap import ensure_hf_token
ensure_hf_token()   # sets HF_TOKEN in this kernel if a token is found; no prompt
```

On an HPC node the token is usually already an env var or a cached login, so
nothing extra is needed. `python -m gnsm doctor` prints whether a token is active
(never the value). `from_pretrained` then uses it implicitly — no `token=`
argument required.

### Models used by GNSM

| Model | Gating | Notes |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | **gated** (access granted) | comparability generator |
| `google/gemma-*`, `meta-llama/Llama-3*` | **gated** (access granted) | optional generators |
| `Qwen/Qwen2.5-*-Instruct` | public (Apache-2.0) | headline generators |
| `answerdotai/ModernBERT-large` | public | state-encoder node init |
| `microsoft/deberta-v3-large` | public | encoder alternative |

Only the gated rows need the token; everything else loads with no auth.

## Commands you get after bootstrapping

```bash
python -m gnsm doctor            # python/torch/CUDA/GPU report (+ --json, --require-gpu)
python -m gnsm demo              # no-download reference generation loop
python -m gnsm smoke --json      # GPU training smoke over the real neural modules
python -m gnsm extract FILE.txt  # print an extracted scene graph
python -m gnsm.training.stage0_probe features.npy labels.npy   # P0 go/no-go
```

## Memory guide (single GPU)

| Generator | Precision | Fits on |
|---|---|---|
| ≤1B (e.g. Qwen2.5-0.5B) | fp16/bf16, no quant | T4 (16 GB) |
| 7–8B (Llama-3.1-8B) | 4-bit NF4 (`quantize=True`) | T4 (16 GB) |
| 14B (Qwen2.5-14B) | 4-bit NF4 | L4 / A100 |
| 72B (scale ceiling) | 4-bit NF4 | A100 80 GB |

The encoder + transition modules are <500 M params and train on a T4; scene
encodings are cached, so the graph-transformer passes happen once.
