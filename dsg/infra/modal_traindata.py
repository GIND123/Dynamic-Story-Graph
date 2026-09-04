"""Build the state-conditioned continuation dataset from public-domain novels.

One training example is: *given the graph state a reader would hold after
chapters 1..t-1, and a one-line brief for chapter t, write chapter t*. The
state is built causally with the same calculus the reading study measured, so
the model is trained on exactly the object it will be conditioned on at
inference.

    modal run --detach dsg/infra/modal_traindata.py::main --books 200 --chapters 32
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("dsg-traindata")

MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.11.0",
        "transformers>=4.51,<5",
        "huggingface-hub>=0.26",
        "pyarrow>=15",
    )
    .env({"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("dsg")
)

weights = modal.Volume.from_name("dsg-weights", create_if_missing=True)
results = modal.Volume.from_name("dsg-results", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

def _push_partial(rows: list[dict], run_id: str, done: int, total: int) -> None:
    """Mirror progress to the Hub as it is made, not only at the end."""
    try:
        from dsg.hub import DATASET_REPO, push_jsonl

        push_jsonl(rows, DATASET_REPO, "train.jsonl",
                   message=f"{run_id}: {len(rows)} rows, chapter {done}/{total}")
    except Exception as exc:  # never let a Hub hiccup kill a GPU run
        print(f"[data] hub push failed ({exc}); checkpoint is safe on the volume",
              flush=True)


BEAT_PROMPT = """Here is a chapter from a novel.

<<<CHAPTER>>>
{chapter}
<<<END>>>

In ONE sentence of at most 30 words, say what happens in it. Write it as an
instruction to a novelist, e.g. "Hesper confronts her brother about the debt and
he refuses to answer." Output only that sentence.

BRIEF:"""


@app.function(
    image=image, gpu="A10G", timeout=60 * 60 * 5,
    volumes={"/root/.cache/huggingface": weights, "/results": results},
    secrets=[hf_secret],
)
def build(
    n_books: int = 200,
    shards: int = 8,
    max_chapters: int = 32,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    run_id: str = "dsg-traindata",
    push: bool = True,
    checkpoint_every: int = 2,
) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from dsg.data.gutenberg import load_books
    from dsg.generate.run import (
        WRITE_EXTRACT_PROMPT,
        parse_write_extraction,
        replay_extraction,
    )
    from dsg.policies.runner import apply_window
    from dsg.schemas import Span, Window
    from dsg.store import POLICIES, NarrativeState

    print(f"[data] loading up to {n_books} fiction novels from {shards} shards", flush=True)
    books = load_books(
        limit=n_books, shards=shards,
        min_chars=120_000, max_chars=1_200_000, min_chapters=12,
    )
    # Chapters are re-cut to roughly the length the writer is asked to produce,
    # so training and inference share a unit.
    from dsg.data.gutenberg import segment_chapters

    seen_titles: set[str] = set()
    plans = []
    for book in books:
        key = book.title.strip().lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        chapters = segment_chapters(book.text, min_chars=1800, max_chars=4500)
        if len(chapters) < 8:
            continue
        plans.append({
            "book": book, "chapters": chapters[:max_chapters],
            "state": NarrativeState(POLICIES["dsg-full"]), "offset": 0,
        })
    depth = max((len(p["chapters"]) for p in plans), default=0)
    print(f"[data] {len(plans)} novels, deepest {depth} chapters, model {model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id, max_model_len=8192, gpu_memory_utilization=0.90,
              disable_log_stats=True)
    beat_params = SamplingParams(temperature=0.0, max_tokens=80)
    extract_params = SamplingParams(temperature=0.0, max_tokens=460)

    def chat(prompts: list[str]) -> list[str]:
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                tokenize=False,
            )
            for p in prompts
        ]

    # --- resume ---------------------------------------------------------
    # Every extraction is cached, so a run that dies mid-way is replayed on the
    # CPU rather than re-billed on the GPU. This matters more than it looks:
    # the failure mode being defended against is a dropped connection, which
    # kills the client, which kills the container.
    ckpt_path = Path("/results") / f"{run_id}-checkpoint.json"
    rows: list[dict] = []
    cache: dict[str, str] = {}
    start_step = 0
    if ckpt_path.exists():
        saved = json.loads(ckpt_path.read_text())
        rows = saved.get("rows", [])
        cache = saved.get("extractions", {})
        start_step = int(saved.get("completed_steps", 0))
        print(f"[data] resuming from checkpoint: {len(rows)} rows, "
              f"{start_step} chapters done", flush=True)
        for plan in plans:
            plan["offset"] = replay_extraction(
                plan["state"], plan["chapters"], cache,
                plan["book"].text_id, start_step, plan["offset"],
            )

    def save_checkpoint(completed: int) -> None:
        ckpt_path.write_text(json.dumps({
            "rows": rows, "extractions": cache, "completed_steps": completed,
            "run_id": run_id, "model": model_id,
        }))
        results.commit()

    t0 = time.time()
    for step in range(start_step, depth):
        active = [p for p in plans if step < len(p["chapters"])]
        texts = [p["chapters"][step] for p in active]

        beats = llm.generate(
            chat([BEAT_PROMPT.format(chapter=t[:6000]) for t in texts]),
            beat_params, use_tqdm=False,
        )
        for plan, text, beat in zip(active, texts, beats, strict=False):
            rows.append({
                "book_id": plan["book"].text_id,
                "title": plan["book"].title,
                "author": plan["book"].author,
                "chapter": step + 1,
                "state": plan["state"].fact_digest(),
                "beat": beat.outputs[0].text.strip().split("\n")[0][:300],
                "target": text,
                "target_chars": len(text),
            })

        # Read the chapter into the state, so the next example is conditioned on
        # everything up to but not including its own target.
        exts = llm.generate(
            chat([WRITE_EXTRACT_PROMPT.format(chapter=t[:6000]) for t in texts]),
            extract_params, use_tqdm=False,
        )
        for plan, text, ext in zip(active, texts, exts, strict=False):
            cache[f"{plan['book'].text_id}:{step}"] = ext.outputs[0].text
            start = plan["offset"]
            window = Window(index=step, start=start, end=start + len(text), text=text)
            proposal = parse_write_extraction(ext.outputs[0].text, window)
            plan["state"].step(step, window.end)
            apply_window(plan["state"], proposal, Span(window.start, window.end))
            plan["state"].close_step()
            plan["offset"] = window.end + 2

        if (step + 1) % checkpoint_every == 0 or step == depth - 1:
            save_checkpoint(step + 1)
            if push:
                _push_partial(rows, run_id, step + 1, depth)
            rate = len(rows) / max(1e-9, time.time() - t0)
            print(f"[data] chapter {step + 1}/{depth} books={len(active)} "
                  f"rows={len(rows)} {rate:.1f} row/s (checkpointed)", flush=True)

    elapsed = time.time() - t0
    meta = {
        "model": model_id, "novels": len(plans), "rows": len(rows),
        "max_chapters": max_chapters, "seconds": round(elapsed, 1),
        "mean_target_chars": sum(r["target_chars"] for r in rows) // max(1, len(rows)),
        "books": [p["book"].to_json() for p in plans],
    }
    out = Path("/results") / f"{run_id}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows}))
    results.commit()
    print(f"[data] done: {len(rows)} rows from {len(plans)} novels in "
          f"{elapsed / 60:.1f} min", flush=True)

    if push:
        from dsg.hub import DATASET_REPO, push_jsonl

        url = push_jsonl(rows, DATASET_REPO, "train.jsonl", message=f"{run_id}: {len(rows)} rows")
        meta_url = push_jsonl([meta], DATASET_REPO, "meta.jsonl", message=run_id)
        print(f"[data] pushed -> {url}\n[data] meta -> {meta_url}", flush=True)
        meta["hub_url"] = url
    return meta


@app.local_entrypoint()
def main(
    books: int = 200, shards: int = 8, chapters: int = 32,
    model: str = "qwen7b", run_id: str = "dsg-traindata", push: bool = True,
    checkpoint_every: int = 2,
):
    meta = build.remote(
        n_books=books, shards=shards, max_chapters=chapters,
        model_id=MODELS.get(model, model), run_id=run_id, push=push,
        checkpoint_every=checkpoint_every,
    )
    meta.pop("books", None)
    print(f"[local] {json.dumps(meta, indent=2)}")
