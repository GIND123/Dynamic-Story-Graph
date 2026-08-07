"""Graph-state-conditioned generation via prefix-tuning (Li & Liang, 2021,
"Prefix-Tuning: Optimizing Continuous Prompts for Generation"; Lester et al.,
2021, "The Power of Scale for Parameter-Efficient Prompt Tuning").

`StatePrefixAdapter` (gnsm/generation/adapter.py) projects a narrative state
vector into `prefix_tokens` continuous embeddings; those are prepended to a
frozen LM's own token embeddings via `inputs_embeds`, so the frozen model's
weights are never touched -- only the adapter is trained (see
gnsm/training/train_adapter.py).

This is deliberately a separate, dedicated path from
`gnsm.generation.huggingface.HuggingFaceFrozenGenerator.generate()`, which
conditions via a text prompt (symbolic constraints as bullet points) -- a
different, already-real mechanism that this doesn't replace.
"""

from __future__ import annotations

from typing import Any


def build_prefix_embeds(adapter: Any, global_state: Any) -> Any:
    """StatePrefixAdapter(global_state) -> (batch, prefix_tokens, model_dim)."""

    return adapter(global_state)


def generate_with_state_prefix(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    global_state: Any,
    prompt_text: str = "",
    max_new_tokens: int = 128,
) -> list[str]:
    """Generate continuations conditioned on `global_state` via a soft prefix.

    `global_state` is `(batch, state_dim)`. `prompt_text` (optional, shared
    across the batch) is tokenized and its embeddings appended after the
    prefix, so generation continues from there.
    """

    import torch

    device = next(model.parameters()).device
    global_state = global_state.to(device)
    batch_size = global_state.shape[0]

    prefix_embeds = build_prefix_embeds(adapter, global_state)  # (B, prefix_tokens, model_dim)
    # Match the frozen LM's dtype (it may be bf16/fp16 while the adapter is
    # float32) -- torch.cat and the model's own matmuls both require it.
    embedding_dtype = model.get_input_embeddings().weight.dtype
    inputs_embeds = prefix_embeds.to(embedding_dtype)
    attention_mask = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=device)

    if prompt_text:
        tokenized = tokenizer([prompt_text] * batch_size, return_tensors="pt").to(device)
        prompt_embeds = model.get_input_embeddings()(tokenized["input_ids"])
        inputs_embeds = torch.cat([inputs_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, tokenized["attention_mask"]], dim=1)

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # With inputs_embeds, generate() returns only the newly generated tokens
    # (there's no input_ids prefix in the output to slice off).
    return [str(tokenizer.decode(row, skip_special_tokens=True)) for row in output_ids]
