"""Train the writer: a LoRA that conditions chapter generation on graph state.

The model is taught to *use* the state, not merely to have it in context. One
example is (state after chapters 1..t-1, brief for chapter t) -> chapter t, from
public-domain novels, with the state built by exactly the calculus used at
inference. Loss is on the chapter only; the prompt is masked.

    modal run --detach dsg/infra/modal_train.py::main --model qwen3b --epochs 2
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("dsg-train")

MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.51,<5",
        "peft>=0.13",
        "accelerate>=1.1",
        "datasets>=3.0",
        "huggingface-hub>=0.26",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2",
    )
    .add_local_python_source("dsg")
)

weights = modal.Volume.from_name("dsg-weights", create_if_missing=True)
results = modal.Volume.from_name("dsg-results", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")


_NON_FACTS = ("not specified", "not mentioned", "not stated", "unknown",
              "not given", "not described", "unclear", "n/a")


def clean_state(digest: str) -> str:
    """Drop non-answers the extractor slipped past the exact-match filter.

    A line saying a property is "not specified in this chapter" carries nothing,
    and training on it teaches the writer to produce the same non-answer. The
    parse-time filter matches exact values, so phrasal variants survive; this
    catches them before they reach the model.
    """
    kept = []
    for line in (digest or "").splitlines():
        head, _, tail = line.partition(":")
        if not tail:
            kept.append(line)
            continue
        facts = [
            f.strip() for f in tail.split(";")
            if f.strip() and not any(n in f.lower() for n in _NON_FACTS)
        ]
        if facts:
            kept.append(f"{head}: " + "; ".join(facts))
        elif head.strip().startswith("-"):
            kept.append(head.rstrip())
    return "\n".join(kept)


def _build_examples(rows: list[dict], tokenizer, max_len: int) -> list[dict]:
    """Tokenise, mask the prompt, keep only what fits."""
    from dsg.generate.prompt import state_memory, writer_prompt

    out = []
    for row in rows:
        prompt = writer_prompt(
            title=row.get("title", ""),
            memory=state_memory(clean_state(row.get("state", ""))),
            beat=row.get("beat", ""),
            chapter=int(row.get("chapter", 1)),
            words=max(200, min(900, len(row.get("target", "")) // 6)),
        )
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer(chat, add_special_tokens=False).input_ids
        target_ids = tokenizer(
            row["target"] + tokenizer.eos_token, add_special_tokens=False
        ).input_ids
        if len(prompt_ids) >= max_len - 64:
            continue
        target_ids = target_ids[: max_len - len(prompt_ids)]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        out.append({"input_ids": input_ids, "labels": labels})
    return out


@app.function(
    image=image, gpu="A100-40GB", timeout=60 * 60 * 5,
    volumes={"/root/.cache/huggingface": weights, "/results": results},
    secrets=[hf_secret],
)
def train(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    dataset_repo: str = "",
    epochs: float = 2.0,
    lr: float = 1e-4,
    rank: int = 16,
    max_len: int = 2048,
    batch_size: int = 2,
    grad_accum: int = 8,
    val_books: int = 12,
    run_id: str = "dsg-writer-qwen3b",
    push: bool = True,
    save_every: int = 50,
) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    from dsg.hub import DATASET_REPO, MODEL_REPO

    dataset_repo = dataset_repo or DATASET_REPO
    hub_repo = f"{MODEL_REPO}-{run_id.split('-')[-1]}"
    path = hf_hub_download(dataset_repo, "train.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print(f"[train] {len(rows)} rows from {dataset_repo}", flush=True)

    # Split by book so no novel appears on both sides.
    books = sorted({r["book_id"] for r in rows})
    held_out = set(books[:val_books])
    train_rows = [r for r in rows if r["book_id"] not in held_out]
    val_rows = [r for r in rows if r["book_id"] in held_out]
    print(f"[train] {len(train_rows)} train / {len(val_rows)} val "
          f"({len(books) - len(held_out)}/{len(held_out)} novels)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = _build_examples(train_rows, tokenizer, max_len)
    val_ds = _build_examples(val_rows, tokenizer, max_len)
    print(f"[train] tokenised {len(train_ds)} / {len(val_ds)} examples", flush=True)

    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        input_ids, labels, mask = [], [], []
        for b in batch:
            gap = longest - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad] * gap)
            labels.append(b["labels"] + [-100] * gap)
            mask.append([1] * len(b["input_ids"]) + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    peft_model.print_trainable_parameters()

    out_dir = Path("/results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    class Checkpoint(TrainerCallback):
        """Persist the adapter as training proceeds, not only at the end.

        The connection this runs over is unreliable, and losing the client kills
        the container. Writing the adapter to the volume every ``save_every``
        steps -- and mirroring it to the Hub -- means an interrupted run costs
        the steps since the last save rather than the whole run.
        """

        def on_save(self, args, state, control, **kw):  # noqa: D102
            peft_model.save_pretrained(str(out_dir))
            (out_dir / "progress.json").write_text(json.dumps({
                "global_step": state.global_step,
                "epoch": state.epoch,
                "log_history": state.log_history[-20:],
            }, indent=2))
            results.commit()
            print(f"[train] checkpoint at step {state.global_step}", flush=True)
            if push:
                try:
                    from dsg.hub import push_folder

                    push_folder(out_dir, hub_repo, repo_type="model",
                                message=f"{run_id} step {state.global_step}")
                except Exception as exc:
                    print(f"[train] hub push failed ({exc}); volume copy is safe",
                          flush=True)

    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, eval_strategy="steps", eval_steps=save_every,
        save_strategy="steps", save_steps=save_every, save_total_limit=1,
        bf16=True, report_to=[],
        gradient_checkpointing=True, remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model, args=args, train_dataset=train_ds,
        eval_dataset=val_ds, data_collator=collate, callbacks=[Checkpoint()],
    )
    t0 = time.time()
    baseline = trainer.evaluate()
    print(f"[train] val loss before training: {baseline['eval_loss']:.4f}", flush=True)
    trainer.train()
    final = trainer.evaluate()
    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    meta = {
        "model": model_id, "run_id": run_id, "rows": len(rows),
        "train_examples": len(train_ds), "val_examples": len(val_ds),
        "epochs": epochs, "lr": lr, "rank": rank, "max_len": max_len,
        "val_loss_before": round(float(baseline["eval_loss"]), 4),
        "val_loss_after": round(float(final["eval_loss"]), 4),
        "seconds": round(elapsed, 1),
        "held_out_books": sorted(held_out),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    results.commit()
    print(f"[train] val loss {meta['val_loss_before']} -> {meta['val_loss_after']} "
          f"in {elapsed / 60:.1f} min", flush=True)

    if push:
        from dsg.hub import push_folder

        url = push_folder(out_dir, hub_repo, repo_type="model",
                          message=f"{run_id}: final, val {meta['val_loss_after']}")
        print(f"[train] pushed -> {url}", flush=True)
        meta["hub_url"] = url
    return meta


@app.local_entrypoint()
def main(
    model: str = "qwen3b", epochs: float = 2.0, lr: float = 1e-4, rank: int = 16,
    max_len: int = 2048, batch_size: int = 2, grad_accum: int = 8,
    val_books: int = 12, run_id: str = "", push: bool = True,
    save_every: int = 50,
):
    run_id = run_id or f"dsg-writer-{model}"
    meta = train.remote(
        model_id=MODELS.get(model, model), epochs=epochs, lr=lr, rank=rank,
        max_len=max_len, batch_size=batch_size, grad_accum=grad_accum,
        val_books=val_books, run_id=run_id, push=push, save_every=save_every,
    )
    meta.pop("held_out_books", None)
    print(f"[local] {json.dumps(meta, indent=2)}")
