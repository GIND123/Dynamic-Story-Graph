"""Emit real generated continuations for qualitative inspection.

This produces artifacts, not scores. With no LLM judge and no gold-annotated
consistency labels available for this corpus (see
`gnsm.eval.extraction_coverage` for why the rule-based verifier cannot supply
one on EvolvTrip), the honest thing to show alongside the quantitative loss
result is the actual text the model produces, next to the gold continuation,
so a reader can judge it themselves.

Emits, per example: the gold next scene, and the continuation generated from
the real narrative-state prefix -- optionally alongside the shuffled-state
control, so the pair can be compared directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def generate_samples(
    examples: list[Any],
    encoder_state_dict: dict[str, Any],
    adapter_state_dict: dict[str, Any],
    n_samples: int = 8,
    max_new_tokens: int = 60,
    device: str = "auto",
    lm_name: str | None = None,
    state_mode: str = "real",
    seed: int = 0,
) -> list[dict[str, str]]:
    import torch

    from gnsm.generation.adapter import StatePrefixAdapter
    from gnsm.generation.state_prefix import generate_with_state_prefix
    from gnsm.training.adapter_data import collate_for_adapter
    from gnsm.training.batch_config import BatchConfig
    from gnsm.training.train_adapter import (
        DEFAULT_LM,
        TrainAdapterConfig,
        apply_state_mode,
        load_frozen_encoder,
        load_frozen_lm,
    )
    from gnsm.training.train_state import resolve_device

    torch.manual_seed(seed)
    config = TrainAdapterConfig(device=device, lm_name=lm_name or DEFAULT_LM)
    resolved = torch.device(resolve_device(device))

    encoder = load_frozen_encoder(config, encoder_state_dict, resolved)
    model, tokenizer = load_frozen_lm(config.lm_name, resolved, config.lm_dtype)
    adapter = StatePrefixAdapter(
        state_dim=config.hidden_dim,
        model_dim=int(model.config.hidden_size),
        prefix_tokens=config.prefix_tokens,
    ).to(resolved)
    adapter.load_state_dict(adapter_state_dict)
    adapter.eval()

    chosen = examples[:n_samples]
    batch_config = BatchConfig(
        nodes=config.nodes,
        edges_per_graph=config.edges_per_graph,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
    )
    batch = collate_for_adapter(chosen, batch_config, seed)
    with torch.no_grad():
        _nodes, global_state = encoder(batch.node_features.to(resolved))
        global_state = apply_state_mode(global_state, state_mode)

    generated = generate_with_state_prefix(
        model, tokenizer, adapter, global_state, max_new_tokens=max_new_tokens
    )
    return [
        {
            "book": example.book,
            "character": example.character,
            "step_from": str(example.step_from),
            "step_to": str(example.step_to),
            "state_mode": state_mode,
            "gold_next_scene": example.action_text[:600],
            "generated": text.strip(),
        }
        for example, text in zip(chosen, generated, strict=True)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm sample-generations",
        description="Emit real generated continuations for qualitative inspection.",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/evolvtrip_data/all_books_current.json")
    )
    parser.add_argument("--encoder-hf-repo", default="GOVINDFROM/DNG-GNSM")
    parser.add_argument("--encoder-run-id", default="evolvtrip-v2-earlystop")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--state-mode", default="real", choices=["real", "shuffled", "zero"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch

    from gnsm.training.evolvtrip_adapter import load_examples
    from gnsm.training.train_adapter import _load_encoder_state

    examples = load_examples(args.data)
    encoder_state_dict = _load_encoder_state(args)
    adapter_state = torch.load(args.adapter_checkpoint, map_location="cpu")
    adapter_state_dict = adapter_state.get("adapter", adapter_state)

    samples = generate_samples(
        examples,
        encoder_state_dict,
        adapter_state_dict,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        lm_name=args.lm,
        state_mode=args.state_mode,
        seed=args.seed,
    )
    text = json.dumps(samples, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
