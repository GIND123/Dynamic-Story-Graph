"""E1 on Modal: batched prefix-causal quote attribution over all of PDNC.

The leakage guard runs **locally**, before anything is shipped. Prompts are
built and whitelist-verified on the client, and only verified prompt strings
cross the wire -- the GPU never receives novel text, so there is no path by
which the remote side could widen a context beyond what was checked. That is a
stronger guarantee than verifying remotely, and it is why this app takes prompts
rather than books.

Batching is safe here in a way it is not for extraction: attribution items are
independent, because each prompt already carries its own causal prefix. There is
no lockstep constraint, so vLLM can batch as wide as it likes.

    # smoke: 2 novels, 40 quotes each, cheapest GPU
    modal run dsg/infra/modal_attribution.py::main --novels 2 --limit 40 --gpu l4

    # full run -- always --detach, see modal_extract.py for why
    modal run --detach dsg/infra/modal_attribution.py::main --model qwen7b --gpu a10g

    # recover a run whose client disconnected
    modal run dsg/infra/modal_attribution.py::fetch --run-id e1-qwen7b
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("dsg-attribution")

MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
}
GPU_FOR = {"l4": "L4", "a10g": "A10G", "a100": "A100-40GB"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.11.0",
        "transformers>=4.51,<5",
        "huggingface-hub>=0.26",
    )
    .env({"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("dsg")
)

weights = modal.Volume.from_name("dsg-weights", create_if_missing=True)
results = modal.Volume.from_name("dsg-results", create_if_missing=True)
_VOLUMES = {"/weights": weights, "/results": results}
_TIMEOUT = 60 * 60 * 6


def _generate(payload: dict, model_id: str, max_tokens: int) -> dict:
    """Run one novel's prompts through vLLM. Prompts arrive pre-verified."""
    import os

    os.environ.setdefault("HF_HOME", "/weights/hf")
    from vllm import LLM, SamplingParams

    prompts = payload["prompts"]
    llm = LLM(
        model=model_id,
        download_dir="/weights/hf",
        dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=False,
    )
    # Greedy: attribution is a single-label decision, and sampling noise would
    # show up as a condition difference that is not one.
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.time()
    outs = llm.generate(prompts, params)
    replies = [o.outputs[0].text.strip() for o in outs]
    return {
        "novel": payload["novel"],
        "keys": payload["keys"],
        "replies": replies,
        "seconds": round(time.time() - t0, 1),
    }


@app.function(image=image, gpu="L4", timeout=_TIMEOUT, volumes=_VOLUMES)
def attribute_l4(payload: dict, model_id: str, max_tokens: int = 16) -> dict:
    return _generate(payload, model_id, max_tokens)


@app.function(image=image, gpu="A10G", timeout=_TIMEOUT, volumes=_VOLUMES)
def attribute_a10g(payload: dict, model_id: str, max_tokens: int = 16) -> dict:
    return _generate(payload, model_id, max_tokens)


@app.function(image=image, gpu="A100-40GB", timeout=_TIMEOUT, volumes=_VOLUMES)
def attribute_a100(payload: dict, model_id: str, max_tokens: int = 16) -> dict:
    return _generate(payload, model_id, max_tokens)


_FN = {"l4": attribute_l4, "a10g": attribute_a10g, "a100": attribute_a100}


@app.function(image=image, volumes={"/results": results}, timeout=600)
def store(run_id: str, blob: dict) -> str:
    """Persist a finished run remotely, so a dropped client loses nothing."""
    out = Path("/results") / "attribution" / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob))
    results.commit()
    return str(out)


@app.function(image=image, volumes={"/results": results}, timeout=600)
def read_result(run_id: str) -> dict:
    path = Path("/results") / "attribution" / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no stored run at {path}")
    return json.loads(path.read_text())


@app.local_entrypoint()
def main(
    model: str = "qwen7b",
    gpu: str = "a10g",
    novels: int = 0,
    limit: int = 0,
    window: int = 1200,
    conditions: str = "",
    run_id: str = "",
    out: str = "artifacts/e1",
) -> None:
    """Build and verify prompts locally, then fan out over novels on the GPU."""
    from dsg.eval.attribution_run import (
        ALL_CONDITIONS,
        build_prompt,
        load_novels,
    )

    model_id = MODELS.get(model, model)
    conds = tuple(c.strip() for c in conditions.split(",") if c.strip()) or ALL_CONDITIONS
    run = run_id or f"e1-{model}-w{window}"

    novel_data = load_novels(limit=novels or None)
    if not novel_data:
        raise SystemExit("no PDNC novels found under data/pdnc/data")

    # --- local prompt construction; every causal prompt passes the whitelist ---
    import random

    rng = random.Random(0)
    payloads, meta = [], {}
    for nd in novel_data:
        quotes = list(nd.quotes)
        if limit and len(quotes) > limit:
            quotes = sorted(rng.sample(quotes, limit), key=lambda q: q.start)
        ordered = sorted(nd.quotes, key=lambda q: q.start)
        prev, last = {}, None
        for q in ordered:
            prev[q.start] = last
            last = q.speaker

        prompts, keys = [], []
        for cond in conds:
            for q in quotes:
                prompts.append(
                    build_prompt(nd, q, cond, window=window, recent=prev.get(q.start))
                )
                keys.append([cond, q.quote_id, q.speaker, q.quote_type])
        payloads.append({"novel": nd.name, "prompts": prompts, "keys": keys})
        meta[nd.name] = {"candidates": nd.candidates,
                         "aliases": {k: sorted(v) for k, v in nd.aliases.items()}}

    total = sum(len(p["prompts"]) for p in payloads)
    print(
        f"E1: {len(payloads)} novels, {len(conds)} conditions, {total:,} prompts "
        f"(all causal prompts whitelist-verified locally) -> {model_id} on {gpu}"
    )

    fn = _FN[gpu]
    t0 = time.time()
    replies = {}
    for res in fn.map(payloads, kwargs={"model_id": model_id}):
        replies[res["novel"]] = {"keys": res["keys"], "replies": res["replies"]}
        print(f"  done {res['novel'][:30]:30} {len(res['replies']):6,} in {res['seconds']}s")

    blob = {
        "run_id": run,
        "model": model_id,
        "gpu": gpu,
        "window_chars": window,
        "conditions": list(conds),
        "seconds": round(time.time() - t0, 1),
        "raw": replies,
        "meta": meta,
    }
    print("stored remotely at", store.remote(run, blob))

    dest = Path(out) / f"{run}-raw.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(blob))
    print(f"wrote {dest}")
    print(f"score with: python -m dsg.eval.attribution_score --raw {dest}")


@app.local_entrypoint()
def fetch(run_id: str, out: str = "artifacts/e1") -> None:
    """Recover a stored run after a client disconnect."""
    blob = read_result.remote(run_id)
    dest = Path(out) / f"{run_id}-raw.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(blob))
    print(f"wrote {dest}")
