"""Stage B trainer: real supervised training of the encoder, grounded decode
heads, and transition model against EvolvTrip -- gnsm.training.smoke's
synthetic-batch counterpart, but on real (book, character, plot_index) data.

See gnsm/training/evolvtrip_adapter.py's module docstring for exactly how a
raw EvolvTrip record becomes node/edge/attribute/emotion/delta tensors (a
documented v1 design, not a final one), and gnsm/docs/benchmark_and_publication_plan.md
for how this fits the rest of the P2/P3 build-out.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gnsm.exceptions import OptionalDependencyError
from gnsm.training.evolvtrip_adapter import (
    ATTRIBUTE_CLASSES,
    DELTA_CLASSES,
    EDGE_TYPES,
    EMOTION_CLASSES,
    BatchConfig,
    EvolvTripExample,
    collate_batch,
    load_examples,
)


@dataclass(slots=True)
class TrainStateConfig:
    epochs: int = 100
    batch_size: int = 16
    hidden_dim: int = 128
    layers: int = 2
    heads: int = 4
    nodes: int = 8
    edges_per_graph: int = 8
    input_dim: int = 64
    lr: float = 1e-3
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "auto"
    patience: int = 10  # stop after this many epochs with no val-loss improvement


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split(
    examples: list[EvolvTripExample], val_fraction: float, seed: int
) -> tuple[list[EvolvTripExample], list[EvolvTripExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def _batches(
    examples: list[EvolvTripExample], batch_size: int, seed: int
) -> list[list[EvolvTripExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def run(
    config: TrainStateConfig,
    examples: list[EvolvTripExample],
    checkpoint_cb: Callable[[int, float, dict[str, Any], dict[str, Any]], None] | None = None,
    best_checkpoint_cb: Callable[[int, float, dict[str, Any], dict[str, Any]], None] | None = None,
    resume_state: dict[str, Any] | None = None,
    plot_cb: (
        Callable[[list[int], list[float], list[int], list[float], int | None, float | None], None]
        | None
    ) = None,
) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise OptionalDependencyError(
            "Real training requires the training extra: pip install -e '.[training]'."
        ) from exc

    from gnsm.state.losses import grounded_state_loss
    from gnsm.state.neural import GraphStateEncoder, GroundedStateHeads, NeuralTransitionModel

    if len(examples) < 2:
        raise ValueError(
            f"Need at least 2 EvolvTrip examples to train and validate, got {len(examples)}."
        )

    torch.manual_seed(config.seed)
    device = torch.device(resolve_device(config.device))
    batch_config = BatchConfig(
        nodes=config.nodes,
        edges_per_graph=config.edges_per_graph,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
    )

    train_examples, val_examples = _split(examples, config.val_fraction, config.seed)

    encoder = GraphStateEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        heads=config.heads,
        edge_types=EDGE_TYPES,
    ).to(device)
    heads = GroundedStateHeads(
        hidden_dim=config.hidden_dim,
        edge_types=EDGE_TYPES,
        attribute_classes=ATTRIBUTE_CLASSES,
        emotion_classes=EMOTION_CLASSES,
    ).to(device)
    transition = NeuralTransitionModel(
        hidden_dim=config.hidden_dim, delta_classes=DELTA_CLASSES
    ).to(device)
    parameters = (
        list(encoder.parameters()) + list(heads.parameters()) + list(transition.parameters())
    )
    optimizer = torch.optim.AdamW(parameters, lr=config.lr)

    start_step = 0
    if resume_state is not None:
        encoder.load_state_dict(resume_state["encoder"])
        heads.load_state_dict(resume_state["heads"])
        transition.load_state_dict(resume_state["transition"])
        optimizer.load_state_dict(resume_state["optimizer"])
        start_step = resume_state["step"]

    def forward(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        node_features = batch["node_features"].to(device)
        next_node_features = batch["next_node_features"].to(device)
        action_features = batch["action_features"].to(device)
        edge_pairs = batch["edge_pairs"].to(device)
        edge_labels = batch["edge_labels"].to(device)
        attribute_labels = batch["attribute_labels"].to(device)
        emotion_labels = batch["emotion_labels"].to(device)
        delta_labels = batch["delta_labels"].to(device)

        batch_size = node_features.shape[0]
        node_embeddings, global_state = encoder(node_features)
        with torch.no_grad():
            _, encoded_next = encoder(next_node_features)
        batch_index = (
            torch.arange(batch_size, device=device).view(-1, 1).expand(-1, edge_pairs.shape[1])
        )
        source = node_embeddings[batch_index, edge_pairs[..., 0]].reshape(-1, config.hidden_dim)
        target = node_embeddings[batch_index, edge_pairs[..., 1]].reshape(-1, config.hidden_dim)
        head_out = heads(source, target, global_state)
        predicted_state, delta_logits = transition(global_state, action_features)
        loss, _components = grounded_state_loss(
            edge_logits=head_out["edge_logits"],
            edge_labels=edge_labels,
            attribute_logits=head_out["attribute_logits"],
            attribute_labels=attribute_labels,
            delta_logits=delta_logits,
            delta_labels=delta_labels,
            predicted_state=predicted_state,
            encoded_next_state=encoded_next,
            emotion_logits=head_out["emotion_logits"],
            emotion_labels=emotion_labels,
        )
        return loss

    def evaluate() -> float:
        encoder.eval()
        heads.eval()
        transition.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for val_batch in _batches(val_examples, config.batch_size, seed=config.seed):
                batch = collate_batch(val_batch, batch_config, seed=config.seed)
                total += float(forward(batch).detach().cpu()) * len(val_batch)
                n += len(val_batch)
        encoder.train()
        heads.train()
        transition.train()
        return total / n if n else float("nan")

    def state_dicts() -> dict[str, Any]:
        return {
            "encoder": encoder.state_dict(),
            "heads": heads.state_dict(),
            "transition": transition.state_dict(),
        }

    train_steps: list[int] = []
    train_losses: list[float] = []
    val_epochs: list[int] = []
    val_losses: list[float] = []

    step = start_step
    initial_loss = float("nan")
    final_loss = float("nan")
    best_val_loss = float("inf")
    best_step: int | None = None
    epochs_without_improvement = 0
    early_stopped = False
    epochs_run = 0

    for epoch in range(config.epochs):
        epochs_run = epoch + 1
        for batch_examples in _batches(train_examples, config.batch_size, seed=config.seed + epoch):
            batch = collate_batch(batch_examples, batch_config, seed=config.seed)
            optimizer.zero_grad(set_to_none=True)
            loss = forward(batch)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            if step == start_step:
                initial_loss = value
            final_loss = value
            train_steps.append(step)
            train_losses.append(value)
            if checkpoint_cb is not None:
                checkpoint_cb(step, value, state_dicts(), optimizer.state_dict())
            step += 1

        val_loss = evaluate()
        val_epochs.append(step)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            epochs_without_improvement = 0
            if best_checkpoint_cb is not None:
                best_checkpoint_cb(step, val_loss, state_dicts(), optimizer.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                early_stopped = True
                break

    if plot_cb is not None:
        plot_cb(train_steps, train_losses, val_epochs, val_losses, best_step, best_val_loss)

    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    return {
        "device": str(device),
        "steps": step,
        "epochs_configured": config.epochs,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "patience": config.patience,
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "final_val_loss": round(val_losses[-1], 4) if val_losses else None,
        "best_val_loss": round(best_val_loss, 4) if best_step is not None else None,
        "best_step": best_step,
        "loss_decreased": bool(final_loss < initial_loss),
        "trainable_parameters": int(trainable),
        "config": asdict(config),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm train-state",
        description="Train the GNSM neural stack on real EvolvTrip data.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/evolvtrip_data/all_books_current.json"),
        help="Path to EvolvTrip's all_books_current.json.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Upper bound; early stopping usually ends the run sooner.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after this many epochs with no val-loss improvement.",
    )
    parser.add_argument(
        "--device", default="auto", help="auto (default), cuda, cpu, or an explicit device string"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit metrics as JSON")
    parser.add_argument(
        "--hf-repo", default=None, help="Push periodic checkpoints + the loss plot here."
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50, help="Push a checkpoint every N steps."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest checkpoint in --hf-repo."
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(".gnsm_checkpoints/plots"),
        help="Where to save the loss-curve PNG/PDF locally.",
    )
    args = parser.parse_args(argv)

    examples = load_examples(args.data)

    config = TrainStateConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
    )

    checkpoint_cb = None
    best_checkpoint_cb = None
    resume_state = None
    manager = None
    run_id = f"train-state-{int(time.time())}"
    if args.hf_repo:
        from gnsm.training.checkpointing import attach_to_run

        checkpoint_cb, resume_state, manager = attach_to_run(
            args.hf_repo, args.checkpoint_every, args.resume, run_id=run_id
        )

        def best_checkpoint_cb(
            step: int, val_loss: float, model_state: dict, optimizer_state: dict
        ) -> None:
            manager.push_best(step, val_loss, {**model_state, "optimizer": optimizer_state})

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
            title="GNSM state encoder — EvolvTrip train/val loss",
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
        checkpoint_cb=checkpoint_cb,
        best_checkpoint_cb=best_checkpoint_cb,
        resume_state=resume_state,
        plot_cb=plot_cb,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"device={result['device']}  "
            f"epochs {result['epochs_run']}/{result['epochs_configured']}"
            f"{' (early stopped)' if result['early_stopped'] else ''}  "
            f"loss {result['initial_loss']} -> {result['final_loss']}  "
            f"val_loss={result['final_val_loss']}  "
            f"best_val_loss={result['best_val_loss']} @step={result['best_step']}  "
            f"params={result['trainable_parameters']:,}  "
            f"train_n={result['n_train_examples']}  val_n={result['n_val_examples']}"
        )
    return 0 if result["loss_decreased"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
