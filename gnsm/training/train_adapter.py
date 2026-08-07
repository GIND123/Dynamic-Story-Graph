"""Stage C trainer: graph-state-conditioned generation via prefix tuning.

Trains ONLY :class:`gnsm.generation.adapter.StatePrefixAdapter`. Both the
narrative-state encoder (a `GraphStateEncoder` checkpoint produced by
`train_state.py`) and the language model are frozen, so the single learned
mapping is "narrative state -> soft prompt the LM can read". The technique is
standard prefix tuning (Li & Liang, 2021, "Prefix-Tuning: Optimizing
Continuous Prompts for Generation"; Lester et al., 2021, "The Power of Scale
for Parameter-Efficient Prompt Tuning"); what is specific to GNSM is the
*source* of the prefix -- a learned scene-graph state rather than a free
parameter.

Objective: cross-entropy language-model loss on the gold next-scene text
(EvolvTrip's next `scenario`/`plot_summary`, via
`gnsm.training.adapter_data`). Prefix positions are masked out of the labels
(-100) so the loss only scores real tokens.

Design note -- the encoder is loaded frozen from a *best* checkpoint
(`checkpointing.resume_best_from_hub`), not retrained here, so the encoder's
own reported validation loss stays the citable artifact for that stage and
Stage C's numbers are attributable to the adapter alone.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gnsm.exceptions import OptionalDependencyError
from gnsm.training.adapter_data import collate_for_adapter
from gnsm.training.batch_config import BatchConfig
from gnsm.training.splitting import shuffled_batches, train_val_split
from gnsm.training.train_state import resolve_device

DEFAULT_LM = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(slots=True)
class TrainAdapterConfig:
    """Encoder geometry fields must match the checkpoint being loaded --
    they are used to rebuild the module before `load_state_dict`."""

    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "auto"
    patience: int = 5
    prefix_tokens: int = 8
    max_target_tokens: int = 128
    lm_name: str = DEFAULT_LM
    lm_dtype: str = "auto"
    # "real"      -- condition on each example's own narrative state.
    # "shuffled"  -- condition on another example's state (permuted within the
    #                batch). The control condition: identical state
    #                distribution, but the state<->text correspondence is
    #                destroyed, so beating it isolates *example-specific*
    #                information in the state rather than "a prefix helps".
    # "zero"      -- condition on a zero vector (no state signal at all).
    state_mode: str = "real"
    # Encoder geometry (defaults match the EvolvTrip runs in this repo).
    hidden_dim: int = 128
    layers: int = 2
    heads: int = 4
    nodes: int = 8
    edges_per_graph: int = 8
    input_dim: int = 64
    edge_types: int = 13


def load_frozen_encoder(config: TrainAdapterConfig, state_dict: dict[str, Any], device: Any) -> Any:
    """Rebuild a GraphStateEncoder and load trained weights, frozen + eval."""

    from gnsm.state.neural import GraphStateEncoder

    encoder = GraphStateEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        heads=config.heads,
        edge_types=config.edge_types,
    ).to(device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def resolve_lm_dtype(requested: str, device: Any) -> Any:
    """Pick the LM's compute dtype.

    float32 on CPU is deliberate, not a fallback: bf16/fp16 matmuls on CPU are
    unaccelerated and run orders of magnitude slower, which makes the CPU smoke
    path unusable. On CUDA, prefer bf16 where supported (Ampere+) and fp16
    otherwise (e.g. T4/Turing), mirroring
    `gnsm.generation.huggingface.HuggingFaceFrozenGenerator._resolve_dtype`.
    """

    import torch

    named = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if requested in named:
        return named[requested]
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_frozen_lm(lm_name: str, device: Any, dtype: str = "auto") -> tuple[Any, Any]:
    """Load the base LM + tokenizer, frozen and in eval mode."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise OptionalDependencyError(
            "Adapter training requires transformers: pip install -e '.[training]'."
        ) from exc

    from gnsm.generation.huggingface import local_or_hub

    weights = local_or_hub(lm_name)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    resolved_dtype = resolve_lm_dtype(dtype, device)
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=resolved_dtype).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def apply_state_mode(global_state: Any, mode: str) -> Any:
    """Transform encoder states into the requested experimental condition.

    See TrainAdapterConfig.state_mode. "shuffled" permutes states across the
    batch, so each example is conditioned on some other example's state --
    with a batch of 1 there is nothing to permute against, which would
    silently make the control identical to the treatment, so that is rejected
    rather than quietly mis-measured.
    """

    import torch

    if mode == "real":
        return global_state
    if mode == "zero":
        return torch.zeros_like(global_state)
    if mode == "shuffled":
        batch_size = global_state.shape[0]
        if batch_size < 2:
            raise ValueError(
                "state_mode='shuffled' needs batch_size >= 2 to permute against; "
                f"got a batch of {batch_size}."
            )
        # A derangement-ish roll: guarantees no example keeps its own state.
        return torch.roll(global_state, shifts=1, dims=0)
    raise ValueError(f"Unknown state_mode {mode!r}; expected real, shuffled, or zero.")


def build_prefixed_inputs(
    model: Any,
    tokenizer: Any,
    prefix_embeds: Any,
    target_texts: tuple[str, ...],
    max_target_tokens: int,
    device: Any,
) -> tuple[Any, Any, Any]:
    """Concatenate [state prefix | gold-text embeddings] into one sequence.

    Returns (inputs_embeds, attention_mask, labels) where labels are -100 over
    the prefix positions and over padding, so the LM loss scores only real
    target tokens.
    """

    import torch

    tokenized = tokenizer(
        list(target_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_target_tokens,
    ).to(device)
    input_ids = tokenized["input_ids"]
    token_embeds = model.get_input_embeddings()(input_ids)

    # The adapter trains in float32 while the LM may be loaded in bf16/fp16;
    # torch.cat requires a single dtype. Cast toward the LM (the frozen side),
    # keeping the adapter's own parameters in float32 -- gradients flow back
    # through this cast normally.
    inputs_embeds = torch.cat([prefix_embeds.to(token_embeds.dtype), token_embeds], dim=1)

    prefix_len = prefix_embeds.shape[1]
    prefix_mask = torch.ones(
        (input_ids.shape[0], prefix_len), dtype=tokenized["attention_mask"].dtype, device=device
    )
    attention_mask = torch.cat([prefix_mask, tokenized["attention_mask"]], dim=1)

    labels = input_ids.masked_fill(tokenized["attention_mask"] == 0, -100)
    prefix_labels = torch.full(
        (input_ids.shape[0], prefix_len), -100, dtype=labels.dtype, device=device
    )
    labels = torch.cat([prefix_labels, labels], dim=1)

    return inputs_embeds, attention_mask, labels


def run(
    config: TrainAdapterConfig,
    examples: list[Any],
    encoder_state_dict: dict[str, Any],
    checkpoint_cb: Callable[[int, float, dict[str, Any], dict[str, Any]], None] | None = None,
    best_checkpoint_cb: Callable[[int, float, dict[str, Any], dict[str, Any]], None] | None = None,
    plot_cb: (
        Callable[[list[int], list[float], list[int], list[float], int | None, float | None], None]
        | None
    ) = None,
) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            "Adapter training requires the training extra: pip install -e '.[training]'."
        ) from exc

    from gnsm.generation.adapter import StatePrefixAdapter

    if len(examples) < 2:
        raise ValueError(f"Need at least 2 examples to train and validate, got {len(examples)}.")

    torch.manual_seed(config.seed)
    device = torch.device(resolve_device(config.device))
    batch_config = BatchConfig(
        nodes=config.nodes,
        edges_per_graph=config.edges_per_graph,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
    )

    encoder = load_frozen_encoder(config, encoder_state_dict, device)
    model, tokenizer = load_frozen_lm(config.lm_name, device, config.lm_dtype)
    model_dim = int(model.config.hidden_size)

    adapter = StatePrefixAdapter(
        state_dim=config.hidden_dim, model_dim=model_dim, prefix_tokens=config.prefix_tokens
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr)

    train_examples, val_examples = train_val_split(examples, config.val_fraction, config.seed)

    def forward(batch_examples: list[Any]) -> Any:
        batch = collate_for_adapter(batch_examples, batch_config, config.seed)
        node_features = batch.node_features.to(device)
        with torch.no_grad():
            _node_embeddings, global_state = encoder(node_features)
            global_state = apply_state_mode(global_state, config.state_mode)
        prefix_embeds = adapter(global_state)
        inputs_embeds, attention_mask, labels = build_prefixed_inputs(
            model, tokenizer, prefix_embeds, batch.target_texts, config.max_target_tokens, device
        )
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def evaluate() -> float:
        adapter.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for val_batch in shuffled_batches(val_examples, config.batch_size, seed=config.seed):
                total += float(forward(val_batch).detach().cpu()) * len(val_batch)
                n += len(val_batch)
        adapter.train()
        return total / n if n else float("nan")

    train_steps: list[int] = []
    train_losses: list[float] = []
    val_epochs: list[int] = []
    val_losses: list[float] = []

    step = 0
    initial_loss = float("nan")
    final_loss = float("nan")
    best_val_loss = float("inf")
    best_step: int | None = None
    epochs_without_improvement = 0
    early_stopped = False
    epochs_run = 0

    for epoch in range(config.epochs):
        epochs_run = epoch + 1
        for batch_examples in shuffled_batches(
            train_examples, config.batch_size, seed=config.seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = forward(batch_examples)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            if step == 0:
                initial_loss = value
            final_loss = value
            train_steps.append(step)
            train_losses.append(value)
            if checkpoint_cb is not None:
                checkpoint_cb(
                    step, value, {"adapter": adapter.state_dict()}, optimizer.state_dict()
                )
            step += 1

        val_loss = evaluate()
        val_epochs.append(step)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            epochs_without_improvement = 0
            if best_checkpoint_cb is not None:
                best_checkpoint_cb(
                    step, val_loss, {"adapter": adapter.state_dict()}, optimizer.state_dict()
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                early_stopped = True
                break

    if plot_cb is not None:
        plot_cb(train_steps, train_losses, val_epochs, val_losses, best_step, best_val_loss)

    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in encoder.parameters()
    )
    return {
        "device": str(device),
        "steps": step,
        "epochs_configured": config.epochs,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "patience": config.patience,
        "lm_name": config.lm_name,
        "state_mode": config.state_mode,
        "model_dim": model_dim,
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "final_val_loss": round(val_losses[-1], 4) if val_losses else None,
        "best_val_loss": round(best_val_loss, 4) if best_step is not None else None,
        "best_step": best_step,
        "loss_decreased": bool(final_loss < initial_loss),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
        "config": asdict(config),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm train-adapter",
        description="Train a StatePrefixAdapter to condition a frozen LM on GNSM narrative state.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/evolvtrip_data/all_books_current.json"),
        help="Path to EvolvTrip's all_books_current.json.",
    )
    parser.add_argument(
        "--encoder-hf-repo",
        default="GOVINDFROM/DNG-GNSM",
        help="HF repo holding the trained encoder checkpoint.",
    )
    parser.add_argument(
        "--encoder-run-id",
        default="evolvtrip-v2-earlystop",
        help="run_id whose BEST checkpoint supplies the frozen encoder weights.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=None,
        help="Local state.pt to use instead of downloading from the Hub (offline/testing).",
    )
    parser.add_argument("--lm", default=DEFAULT_LM, help="Frozen base LM to condition.")
    parser.add_argument(
        "--lm-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="LM compute dtype; auto = float32 on CPU, bf16/fp16 on CUDA.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--state-mode",
        default="real",
        choices=["real", "shuffled", "zero"],
        help="real = own state; shuffled = another example's state (control); zero = none.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N examples.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--hf-repo", default=None, help="Push checkpoints + the loss plot here.")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--run-id", default="adapter-primary")
    parser.add_argument("--plot-dir", type=Path, default=Path(".gnsm_checkpoints/plots"))
    args = parser.parse_args(argv)

    from gnsm.training.evolvtrip_adapter import load_examples

    examples = load_examples(args.data)
    if args.limit:
        examples = examples[: args.limit]

    config = TrainAdapterConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        prefix_tokens=args.prefix_tokens,
        max_target_tokens=args.max_target_tokens,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        lm_name=args.lm,
        lm_dtype=args.lm_dtype,
        state_mode=args.state_mode,
    )

    encoder_state_dict = _load_encoder_state(args)

    checkpoint_cb = None
    best_checkpoint_cb = None
    manager = None
    run_id = args.run_id
    if args.hf_repo:
        from gnsm.training.checkpointing import attach_to_run

        checkpoint_cb, best_checkpoint_cb, _resume, manager = attach_to_run(
            args.hf_repo, args.checkpoint_every, False, run_id=run_id
        )

    def plot_cb(
        train_steps: list[int],
        train_losses: list[float],
        val_epochs: list[int],
        val_losses: list[float],
        best_step: int | None,
        best_val_loss: float | None,
    ) -> None:
        from gnsm.training.plotting import plot_loss_curves

        png_path, pdf_path = plot_loss_curves(
            train_steps,
            train_losses,
            args.plot_dir,
            stem=run_id,
            title="GNSM StatePrefixAdapter — EvolvTrip gold-continuation LM loss",
            val_steps=val_epochs,
            val_losses=val_losses,
            best_step=best_step,
            best_val_loss=best_val_loss,
        )
        if manager is not None:
            manager.push_artifact(png_path, f"plots/{run_id}/loss.png")
            manager.push_artifact(pdf_path, f"plots/{run_id}/loss.pdf")

    result = run(
        config,
        examples,
        encoder_state_dict,
        checkpoint_cb=checkpoint_cb,
        best_checkpoint_cb=best_checkpoint_cb,
        plot_cb=plot_cb,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"device={result['device']}  lm={result['lm_name']}  "
            f"epochs {result['epochs_run']}/{result['epochs_configured']}"
            f"{' (early stopped)' if result['early_stopped'] else ''}  "
            f"loss {result['initial_loss']} -> {result['final_loss']}  "
            f"best_val_loss={result['best_val_loss']} @step={result['best_step']}  "
            f"trainable={result['trainable_parameters']:,} frozen={result['frozen_parameters']:,}"
        )
    return 0 if result["loss_decreased"] else 1


def _load_encoder_state(args: argparse.Namespace) -> dict[str, Any]:
    """Trained encoder weights, from a local file or the Hub's best.json."""

    import torch

    if args.encoder_checkpoint is not None:
        state = torch.load(args.encoder_checkpoint, map_location="cpu")
    else:
        from gnsm.training.checkpointing import resume_best_from_hub

        state = resume_best_from_hub(
            args.encoder_hf_repo,
            args.encoder_run_id,
            Path(".gnsm_checkpoints") / "encoder" / args.encoder_run_id,
        )
        if state is None:
            raise SystemExit(
                f"No best checkpoint found for run_id {args.encoder_run_id!r} in "
                f"{args.encoder_hf_repo!r}. Train the encoder first "
                "(python -m gnsm.training.train_state --hf-repo ... --run-id ...), "
                "or pass --encoder-checkpoint <path/to/state.pt>."
            )
    if "encoder" not in state:
        raise SystemExit(
            f"Checkpoint has no 'encoder' key (found: {sorted(state)}). "
            "Expected a train_state.py checkpoint."
        )
    return dict(state["encoder"])


if __name__ == "__main__":
    raise SystemExit(main())
