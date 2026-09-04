"""Long-form generation under six memory conditions, batched on Modal.

Generation within a story is sequential, so batching is across (story,
condition) pairs in lockstep: at chapter t every run submits at once. Each
chapter is up to three batched rounds -- write, summarise, read-back.

    modal run dsg/infra/modal_generate.py::main --stories 3 --chapters 4   # smoke
    modal run --detach dsg/infra/modal_generate.py::main                   # full run

--detach is mandatory for a full run: without it Modal cancels the remote
function when the local client disconnects. Results are also written to the
dsg-results volume, so a disconnected run is recoverable with ::fetch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("dsg-generate")

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
_VOLUMES = {"/root/.cache/huggingface": weights, "/results": results}


def _run_generation(
    premise_dicts: list[dict],
    model_id: str,
    conditions: tuple[str, ...],
    chapters: int,
    words: int,
    max_model_len: int,
) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from dsg.generate.canon import Premise
    from dsg.generate.run import (
        apply_extraction,
        build_summary_prompt,
        chapter_prompts,
        clean_chapter,
        extraction_prompt,
        init_runs,
        record,
        state_targets,
        summary_targets,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id, max_model_len=max_model_len,
        gpu_memory_utilization=0.90, disable_log_stats=True,
    )
    chapter_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=900, seed=0)
    aux_params = SamplingParams(temperature=0.0, max_tokens=420)

    def chat(prompts: list[str]) -> list[str]:
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False,
            )
            for p in prompts
        ]

    premises = [Premise.from_json(d) for d in premise_dicts]
    by_id = {p.story_id: p for p in premises}
    runs = init_runs(premises, conditions)
    print(f"[gen] {len(premises)} stories x {len(conditions)} conditions "
          f"= {len(runs)} runs, {chapters} chapters, model {model_id}", flush=True)

    records: list[dict] = []
    t0 = time.time()
    for chapter in range(1, chapters + 1):
        prompts = chapter_prompts(runs, by_id, chapter, chapters, words)
        outs = llm.generate(chat(prompts), chapter_params, use_tqdm=False)
        for run, out in zip(runs, outs, strict=False):
            run.chapters.append(clean_chapter(out.outputs[0].text))

        # Rolling summaries, batched.
        targets = summary_targets(runs)
        if targets:
            sums = llm.generate(
                chat([build_summary_prompt(r, r.chapters[-1]) for r in targets]),
                aux_params, use_tqdm=False,
            )
            for run, out in zip(targets, sums, strict=False):
                run.summary = out.outputs[0].text.strip()

        # Read the new chapter into the state-carrying runs, batched.
        stateful = state_targets(runs)
        if stateful:
            exts = llm.generate(
                chat([extraction_prompt(r, r.chapters[-1]) for r in stateful]),
                aux_params, use_tqdm=False,
            )
            for run, out in zip(stateful, exts, strict=False):
                apply_extraction(run, out.outputs[0].text, run.chapters[-1])

        for run in runs:
            rec = record(run, by_id[run.story_id], run.chapters[-1], chapter)
            payload = rec.to_json()
            payload["text"] = rec.text
            records.append(payload)

        done = sum(len(r.chapters) for r in runs)
        rate = done / max(1e-9, time.time() - t0)
        print(f"[gen] chapter {chapter}/{chapters} runs={len(runs)} "
              f"{rate:.1f} chap/s eta={(len(runs) * chapters - done) / max(rate, 1e-9) / 60:.1f}m",
              flush=True)

    elapsed = time.time() - t0
    print(f"[gen] done: {len(records)} chapters in {elapsed / 60:.1f} min", flush=True)
    return {
        "meta": {
            "model": model_id, "stories": len(premises), "conditions": list(conditions),
            "chapters": chapters, "words": words, "seconds": round(elapsed, 1),
        },
        "premises": [p.to_json() for p in premises],
        "records": records,
    }


def _entry(premise_dicts, model_id, conditions, chapters, words, max_model_len, run_id):
    payload = _run_generation(
        premise_dicts, model_id, tuple(conditions), chapters, words, max_model_len
    )
    out = Path("/results") / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    results.commit()
    return payload


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 5, volumes=_VOLUMES)
def generate_a10g(premise_dicts: list[dict], model_id: str, conditions: list[str],
                  chapters: int, words: int, max_model_len: int, run_id: str) -> dict:
    return _entry(premise_dicts, model_id, conditions, chapters, words,
                  max_model_len, run_id)


@app.function(image=image, gpu="L4", timeout=60 * 60 * 5, volumes=_VOLUMES)
def generate_l4(premise_dicts: list[dict], model_id: str, conditions: list[str],
                chapters: int, words: int, max_model_len: int, run_id: str) -> dict:
    return _entry(premise_dicts, model_id, conditions, chapters, words,
                  max_model_len, run_id)


@app.function(image=image, gpu="A100-40GB", timeout=60 * 60 * 5, volumes=_VOLUMES)
def generate_a100(premise_dicts: list[dict], model_id: str, conditions: list[str],
                  chapters: int, words: int, max_model_len: int, run_id: str) -> dict:
    return _entry(premise_dicts, model_id, conditions, chapters, words,
                  max_model_len, run_id)


_FNS = {"A10G": generate_a10g, "L4": generate_l4, "A100-40GB": generate_a100}


@app.function(image=image, volumes={"/results": results}, timeout=600)
def read_result(run_id: str) -> dict:
    path = Path("/results") / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no generation result for '{run_id}'")
    return json.loads(path.read_text())


def _write_local(payload: dict, out_dir: str, run_id: str) -> Path:
    root = Path(out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "generation.json").write_text(json.dumps(payload, indent=2))
    return root


@app.local_entrypoint()
def fetch(run_id: str, out_dir: str = "artifacts/generation"):
    payload = read_result.remote(run_id)
    print(f"[local] wrote {_write_local(payload, out_dir, run_id)}")


@app.local_entrypoint()
def main(
    model: str = "qwen7b",
    stories: int = 30,
    chapters: int = 16,
    words: int = 500,
    n_canon: int = 8,
    max_model_len: int = 16384,
    seed: int = 0,
    gpu: str = "",
    run_id: str = "",
    out_dir: str = "artifacts/generation",
):
    from dsg.generate.canon import build_premises
    from dsg.generate.conditions import CONDITIONS

    model_id = MODELS.get(model, model)
    gpu_name = gpu or GPU_FOR.get(model, "A10G")
    run_id = run_id or f"gen-{model}-s{stories}-c{chapters}"
    premises = build_premises(n=stories, n_canon=n_canon, seed=seed)
    print(f"[local] {len(premises)} stories x {len(CONDITIONS)} conditions x "
          f"{chapters} chapters, model={model_id} gpu={gpu_name}")

    payload = _FNS[gpu_name].remote(
        [p.to_json() for p in premises], model_id, list(CONDITIONS),
        chapters, words, max_model_len, run_id,
    )
    print(f"[local] wrote {_write_local(payload, out_dir, run_id)}")
    print(f"[local] {json.dumps(payload['meta'])}")
