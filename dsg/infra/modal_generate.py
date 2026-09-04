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
hf_secret = modal.Secret.from_name("hf-token")
_VOLUMES = {"/root/.cache/huggingface": weights, "/results": results}


def _window_at(chapters: list[str], index: int):
    """The character span chapter ``index`` occupies in the story so far."""
    from dsg.schemas import Window

    start = sum(len(c) + 2 for c in chapters[:index])
    text = chapters[index]
    return Window(index=index, start=start, end=start + len(text), text=text)


def _generate_split(llm, prompts, params, use_lora, lora_path):
    """Route each prompt to the base model or the adapter, preserving order.

    With no adapter available the tuned arm is served by the base weights; the
    run metadata records ``lora_repo``, so a result produced that way cannot be
    mistaken for a base-versus-tuned comparison.
    """
    from vllm.lora.request import LoRARequest

    texts: list[str] = [""] * len(prompts)
    request = LoRARequest("dsg-writer", 1, lora_path) if lora_path else None
    groups = {
        False: [i for i, flag in enumerate(use_lora) if not flag],
        True: [i for i, flag in enumerate(use_lora) if flag],
    }
    for tuned, idx in groups.items():
        if not idx:
            continue
        outs = llm.generate(
            [prompts[i] for i in idx], params, use_tqdm=False,
            lora_request=request if tuned else None,
        )
        for i, out in zip(idx, outs, strict=False):
            texts[i] = out.outputs[0].text
    return texts


def _run_generation(
    premise_dicts: list[dict],
    model_id: str,
    conditions: tuple[str, ...],
    chapters: int,
    words: int,
    max_model_len: int,
    lora_repo: str = "",
    run_id: str = "gen",
    checkpoint_every: int = 1,
) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from dsg.generate.canon import Premise
    from dsg.generate.conditions import variant_of
    from dsg.generate.repair import repair_instruction
    from dsg.generate.run import (
        apply_extraction,
        build_summary_prompt,
        chapter_prompts,
        check_chapter_against_state,
        clean_chapter,
        extraction_prompt,
        guarded_targets,
        init_runs,
        record,
        restore_runs,
        state_targets,
        summary_targets,
        tuned_targets,
    )

    # The tuned conditions are served by the same base weights plus a LoRA, so
    # base and tuned answer identical prompts on one GPU and every comparison
    # stays within-story.
    lora_path = ""
    if lora_repo:
        import json as _json

        from huggingface_hub import hf_hub_download, snapshot_download

        lora_path = snapshot_download(lora_repo)
        # An adapter only loads onto the base it was trained on. Catch the
        # mismatch here rather than as an opaque shape error mid-run.
        try:
            cfg = _json.loads(
                Path(hf_hub_download(lora_repo, "adapter_config.json")).read_text()
            )
            trained_on = cfg.get("base_model_name_or_path", "")
            if trained_on and trained_on != model_id:
                raise RuntimeError(
                    f"adapter {lora_repo} was trained on {trained_on}, "
                    f"but generation is running {model_id}"
                )
        except FileNotFoundError:
            print("[gen] adapter has no config to check against", flush=True)
        print(f"[gen] serving adapter {lora_repo}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id, max_model_len=max_model_len,
        gpu_memory_utilization=0.90, disable_log_stats=True,
        enable_lora=bool(lora_path), max_lora_rank=64,
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
    extraction_log: dict[str, str] = {}
    repairs_attempted = repairs_accepted = 0

    # Generated chapters are expensive and unrecoverable, so they are written to
    # the volume after every chapter. Losing the connection then costs one
    # chapter, not the run.
    ckpt_path = Path("/results") / f"{run_id}-checkpoint.json"

    def save_checkpoint(done: int) -> None:
        # Everything a resume needs: the prose, the rolling summaries, and the
        # raw extractions, so state is rebuilt on CPU instead of re-billed.
        ckpt_path.write_text(json.dumps({
            "run_id": run_id, "chapters_done": done, "records": records,
            "premises": [p.to_json() for p in premises],
            "meta": {"model": model_id, "conditions": list(conditions),
                     "chapters": chapters, "words": words,
                     "lora_repo": lora_repo, "stories": len(premises)},
            "repairs_attempted": repairs_attempted,
            "repairs_accepted": repairs_accepted,
            "runs": [
                {"story_id": r.story_id, "condition": r.condition,
                 "chapters": r.chapters, "summary": r.summary}
                for r in runs
            ],
            "extractions": extraction_log,
        }))
        results.commit()

    start_chapter = 0
    if ckpt_path.exists():
        saved = json.loads(ckpt_path.read_text())
        if saved.get("meta", {}).get("conditions") == list(conditions):
            start_chapter = int(saved.get("chapters_done", 0))
            records.extend(saved.get("records", []))
            repairs_attempted = int(saved.get("repairs_attempted", 0))
            repairs_accepted = int(saved.get("repairs_accepted", 0))
            extraction_log.update(saved.get("extractions", {}))
            restored = restore_runs(runs, saved, start_chapter, extraction_log)
            print(f"[gen] resuming after chapter {start_chapter}/{chapters} "
                  f"({restored} runs restored)", flush=True)

    t0 = time.time()
    for chapter in range(start_chapter + 1, chapters + 1):
        prompts = chapter_prompts(runs, by_id, chapter, chapters, words)
        # What each condition costs to *ask*: the whole point of carrying a
        # digest rather than the transcript is that it does not grow.
        prompt_sizes = [
            (len(p), len(tokenizer(p).input_ids)) for p in prompts
        ]
        outs = _generate_split(
            llm, chat(prompts), chapter_params, tuned_targets(runs), lora_path
        )
        for run, text in zip(runs, outs, strict=False):
            run.chapters.append(clean_chapter(text))

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
        extractions: dict[int, str] = {}
        if stateful:
            exts = llm.generate(
                chat([extraction_prompt(r, r.chapters[-1]) for r in stateful]),
                aux_params, use_tqdm=False,
            )
            for run, out in zip(stateful, exts, strict=False):
                extractions[id(run)] = out.outputs[0].text
                extraction_log[
                    f"{run.story_id}|{run.condition}|{len(run.chapters) - 1}"
                ] = out.outputs[0].text

        # The guard: a chapter that would contradict an immutable established
        # fact is sent back with the contradiction named, once.
        guarded = [r for r in guarded_targets(runs) if id(r) in extractions]
        retry_runs, retry_prompts = [], []
        for run in guarded:
            conflicts = check_chapter_against_state(
                run, extractions[id(run)], run.chapters[-1]
            )
            if not conflicts:
                continue
            index = runs.index(run)
            retry_runs.append(run)
            retry_prompts.append(prompts[index] + repair_instruction(conflicts))
            repairs_attempted += 1
        if retry_runs:
            fixed = _generate_split(
                llm, chat(retry_prompts), chapter_params,
                [variant_of(r.condition) == "tuned" for r in retry_runs], lora_path,
            )
            for run, text in zip(retry_runs, fixed, strict=False):
                run.chapters[-1] = clean_chapter(text)
            refresh = llm.generate(
                chat([extraction_prompt(r, r.chapters[-1]) for r in retry_runs]),
                aux_params, use_tqdm=False,
            )
            for run, out in zip(retry_runs, refresh, strict=False):
                extractions[id(run)] = out.outputs[0].text
                extraction_log[
                    f"{run.story_id}|{run.condition}|{len(run.chapters) - 1}"
                ] = out.outputs[0].text
                if not check_chapter_against_state(
                    run, extractions[id(run)], run.chapters[-1]
                ):
                    repairs_accepted += 1

        for run in stateful:
            apply_extraction(run, extractions.get(id(run), ""), run.chapters[-1])

        for run, (p_chars, p_tokens) in zip(runs, prompt_sizes, strict=False):
            rec = record(
                run, by_id[run.story_id], run.chapters[-1], chapter,
                prompt_chars=p_chars, prompt_tokens=p_tokens,
            )
            payload = rec.to_json()
            payload["text"] = rec.text
            records.append(payload)

        if chapter % checkpoint_every == 0 or chapter == chapters:
            save_checkpoint(chapter)
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
            "repairs_attempted": repairs_attempted,
            "repairs_accepted": repairs_accepted,
            "lora_repo": lora_repo,
        },
        "premises": [p.to_json() for p in premises],
        "records": records,
    }


def _entry(premise_dicts, model_id, conditions, chapters, words, max_model_len,
           run_id, lora_repo=""):
    payload = _run_generation(
        premise_dicts, model_id, tuple(conditions), chapters, words,
        max_model_len, lora_repo, run_id,
    )
    out = Path("/results") / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    results.commit()
    try:
        from dsg.hub import ARTIFACT_REPO, push_file

        push_file(out, ARTIFACT_REPO, f"generation/{run_id}.json",
                  repo_type="model", message=f"{run_id} generation")
    except Exception as exc:
        print(f"[gen] hub push failed ({exc}); volume copy is safe", flush=True)
    return payload


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 5, volumes=_VOLUMES,
              secrets=[hf_secret])
def generate_a10g(premise_dicts: list[dict], model_id: str, conditions: list[str],
                  chapters: int, words: int, max_model_len: int, run_id: str,
                  lora_repo: str = "") -> dict:
    return _entry(premise_dicts, model_id, conditions, chapters, words,
                  max_model_len, run_id, lora_repo)


@app.function(image=image, gpu="L4", timeout=60 * 60 * 5, volumes=_VOLUMES,
              secrets=[hf_secret])
def generate_l4(premise_dicts: list[dict], model_id: str, conditions: list[str],
                chapters: int, words: int, max_model_len: int, run_id: str,
                lora_repo: str = "") -> dict:
    return _entry(premise_dicts, model_id, conditions, chapters, words,
                  max_model_len, run_id, lora_repo)


@app.function(image=image, gpu="A100-40GB", timeout=60 * 60 * 5, volumes=_VOLUMES,
              secrets=[hf_secret])
def generate_a100(premise_dicts: list[dict], model_id: str, conditions: list[str],
                  chapters: int, words: int, max_model_len: int, run_id: str,
                  lora_repo: str = "") -> dict:
    return _entry(premise_dicts, model_id, conditions, chapters, words,
                  max_model_len, run_id, lora_repo)


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
    lora_repo: str = "",
    ladder: str = "base",
    out_dir: str = "artifacts/generation",
):
    from dsg.generate.canon import build_premises
    from dsg.generate.conditions import BASE_LADDER, CONDITIONS, RETRIEVAL_LADDER

    model_id = MODELS.get(model, model)
    gpu_name = gpu or GPU_FOR.get(model, "A10G")
    conditions = {
        "base": BASE_LADDER,
        "retrieval": RETRIEVAL_LADDER,
        "all": CONDITIONS,
    }.get(ladder, BASE_LADDER)
    run_id = run_id or f"gen-{model}-s{stories}-c{chapters}"
    premises = build_premises(n=stories, n_canon=n_canon, seed=seed)
    print(f"[local] {len(premises)} stories x {len(conditions)} conditions x "
          f"{chapters} chapters, model={model_id} gpu={gpu_name}")

    payload = _FNS[gpu_name].remote(
        [p.to_json() for p in premises], model_id, list(conditions),
        chapters, words, max_model_len, run_id, lora_repo,
    )
    print(f"[local] wrote {_write_local(payload, out_dir, run_id)}")
    print(f"[local] {json.dumps(payload['meta'])}")
