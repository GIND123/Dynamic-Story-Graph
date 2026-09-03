"""GPU extraction on Modal: the one expensive stage of the pipeline.

Everything downstream -- the calculus, the five policies, every metric -- is
deterministic CPU work over the cached proposal stream, so the GPU is used once
per (corpus, model) and never again.

Batching without breaking causality. Windows within a book are strictly
sequential: window ``t``'s prompt carries a digest built only from windows
``< t``. So instead of batching within a book, this batches *across* books in
lockstep -- step ``t`` submits window ``t`` of every book at once. Each book
still sees only its own prefix, while vLLM gets a batch as wide as the corpus.

    modal run dsg/infra/modal_extract.py --books 2 --max-windows 8   # smoke
    modal run dsg/infra/modal_extract.py --model qwen7b              # full run
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("dsg-extract")

MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct-AWQ",
}

GPU_FOR = {"qwen1.5b": "L4", "qwen3b": "L4", "qwen7b": "A10G", "qwen14b": "A100-40GB"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.11.0",
        "transformers>=4.51,<5",  # vLLM 0.11 uses tokenizer APIs dropped in v5
        "huggingface-hub>=0.26",
    )
    .env({"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("dsg")
)

weights = modal.Volume.from_name("dsg-weights", create_if_missing=True)
results = modal.Volume.from_name("dsg-results", create_if_missing=True)


def _run_extraction(
    books: list[dict],
    model_id: str,
    window_chars: int,
    lead_chars: int,
    max_windows: int | None,
    max_tokens: int,
) -> dict:
    """Lockstep batched extraction. Runs inside the GPU container."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from dsg.proposals import (
        SurfaceLedger,
        build_prompt,
        build_resolve_prompt,
        needs_resolution,
        parse_proposal,
        parse_resolve,
    )
    from dsg.windows import iter_windows

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        disable_log_stats=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    resolve_params = SamplingParams(temperature=0.0, max_tokens=120)

    # Materialise each book's windows once.
    plans = []
    for b in books:
        wins = list(iter_windows(b["text"], window_chars, lead_chars))
        if max_windows is not None:
            wins = wins[:max_windows]
        plans.append(
            {"id": b["id"], "windows": wins, "ledger": SurfaceLedger(), "out": []}
        )
    depth = max((len(p["windows"]) for p in plans), default=0)
    print(f"[dsg] {len(plans)} books, deepest {depth} windows, model {model_id}", flush=True)

    t0 = time.time()
    for step in range(depth):
        active = [p for p in plans if step < len(p["windows"])]
        prompts = []
        for p in active:
            window, lead = p["windows"][step]
            text = build_prompt(window, lead, p["ledger"].digest())
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        outputs = llm.generate(prompts, params, use_tqdm=False)
        staged = []
        for p, out in zip(active, outputs):
            window, _ = p["windows"][step]
            staged.append((p, window, parse_proposal(out.outputs[0].text, window)))

        # Second pass: identity questions left open by the first, still
        # batched across books so the GPU stays saturated.
        asks, prompts2 = [], []
        for p, window, proposal in staged:
            unnamed, names = needs_resolution(proposal, p["ledger"])
            if not unnamed:
                continue
            asks.append((proposal, unnamed, names))
            prompts2.append(
                tokenizer.apply_chat_template(
                    [{"role": "user",
                      "content": build_resolve_prompt(window, unnamed, names)}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
        if prompts2:
            for (proposal, unnamed, names), out in zip(
                asks, llm.generate(prompts2, resolve_params, use_tqdm=False)
            ):
                proposal.links.extend(
                    parse_resolve(out.outputs[0].text, unnamed, names)
                )

        for p, _window, proposal in staged:
            p["ledger"].update(proposal)
            p["out"].append(proposal.to_json())
        if step % 10 == 0 or step == depth - 1:
            done = sum(len(p["out"]) for p in plans)
            rate = done / max(1e-9, time.time() - t0)
            print(
                f"[dsg] step {step + 1}/{depth} batch={len(active)} "
                f"windows={done} {rate:.1f} win/s "
                f"eta={(sum(len(p['windows']) for p in plans) - done) / max(rate, 1e-9) / 60:.1f}m",
                flush=True,
            )

    elapsed = time.time() - t0
    total = sum(len(p["out"]) for p in plans)
    ok = sum(1 for p in plans for w in p["out"] if w.get("parse_ok", True))
    print(f"[dsg] done: {total} windows in {elapsed / 60:.1f} min, parse_ok {ok}/{total}", flush=True)
    return {
        "meta": {
            "model": model_id, "window_chars": window_chars, "lead_chars": lead_chars,
            "max_tokens": max_tokens, "seconds": round(elapsed, 1),
            "windows": total, "parse_ok": ok,
        },
        "books": {p["id"]: p["out"] for p in plans},
    }


_VOLUMES = {"/root/.cache/huggingface": weights, "/results": results}
_TIMEOUT = 60 * 60 * 4


def _entry(
    books: list[dict],
    model_id: str,
    window_chars: int,
    lead_chars: int,
    max_windows: int | None,
    max_tokens: int,
    run_id: str,
) -> dict:
    payload = _run_extraction(
        books, model_id, window_chars, lead_chars, max_windows, max_tokens
    )
    out = Path("/results") / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    results.commit()
    return payload


# Modal requires each GPU class to be its own globally scoped function.
@app.function(image=image, gpu="L4", timeout=_TIMEOUT, volumes=_VOLUMES)
def extract_l4(
    books: list[dict], model_id: str, window_chars: int = 3200, lead_chars: int = 400,
    max_windows: int | None = None, max_tokens: int = 420, run_id: str = "default",
) -> dict:
    return _entry(books, model_id, window_chars, lead_chars, max_windows, max_tokens, run_id)


@app.function(image=image, gpu="A10G", timeout=_TIMEOUT, volumes=_VOLUMES)
def extract_a10g(
    books: list[dict], model_id: str, window_chars: int = 3200, lead_chars: int = 400,
    max_windows: int | None = None, max_tokens: int = 420, run_id: str = "default",
) -> dict:
    return _entry(books, model_id, window_chars, lead_chars, max_windows, max_tokens, run_id)


@app.function(image=image, gpu="A100-40GB", timeout=_TIMEOUT, volumes=_VOLUMES)
def extract_a100(
    books: list[dict], model_id: str, window_chars: int = 3200, lead_chars: int = 400,
    max_windows: int | None = None, max_tokens: int = 420, run_id: str = "default",
) -> dict:
    return _entry(books, model_id, window_chars, lead_chars, max_windows, max_tokens, run_id)


_FNS = {"L4": extract_l4, "A10G": extract_a10g, "A100-40GB": extract_a100}


@app.local_entrypoint()
def main(
    model: str = "qwen7b",
    books: int = 0,
    max_windows: int = 0,
    window_chars: int = 0,
    max_tokens: int = 420,
    corpus: str = "pdnc",
    out_dir: str = "artifacts/proposals",
    gpu: str = "",
    run_id: str = "",
):
    """Ship the corpus to a GPU, run extraction, write proposals locally."""
    from dsg.data.registry import WINDOW_CHARS, load

    model_id = MODELS.get(model, model)
    gpu_name = gpu or GPU_FOR.get(model, "A10G")
    run_id = run_id or f"{corpus}-{model}-w{window_chars}"

    window_chars = window_chars or WINDOW_CHARS.get(corpus, 3200)
    novels = load(corpus, limit=books or None)
    payload_books = [{"id": n.book_id, "text": n.text} for n in novels]
    total_chars = sum(len(b["text"]) for b in payload_books)
    print(
        f"[local] corpus={corpus} books={len(payload_books)} "
        f"chars={total_chars:,} model={model_id} gpu={gpu_name}"
    )

    fn = _FNS[gpu_name]
    result = fn.remote(
        payload_books, model_id,
        window_chars=window_chars, max_windows=max_windows or None,
        max_tokens=max_tokens, run_id=run_id,
    )

    out_root = Path(out_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    for book_id, windows in result["books"].items():
        path = out_root / f"{book_id}.jsonl"
        with path.open("w") as fh:
            fh.write(json.dumps({"meta": {**result["meta"], "book_id": book_id}}) + "\n")
            for w in windows:
                fh.write(json.dumps(w) + "\n")
    (out_root / "meta.json").write_text(json.dumps(result["meta"], indent=2))
    print(f"[local] wrote {len(result['books'])} books to {out_root}")
    print(f"[local] {json.dumps(result['meta'])}")
