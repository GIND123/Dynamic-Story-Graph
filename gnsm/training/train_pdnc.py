"""CLI entry point for real training on PDNC (quote/speaker/addressee),
funneling into the shared engine in gnsm.training.train_state.run. See
gnsm/training/pdnc_adapter.py's docstring for the tensor design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnsm.training.pdnc_adapter import (
    ATTRIBUTE_CLASSES,
    DELTA_CLASSES,
    EDGE_TYPES,
    EMOTION_CLASSES,
    collate_batch,
    load_examples,
)
from gnsm.training.train_state import TrainStateConfig
from gnsm.training.train_state import run as run_train_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm train-pdnc", description="Train the GNSM neural stack on real PDNC data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/pdnc/data"),
        help="Path to PDNC's data/ directory (one subfolder per novel).",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Upper bound; early stopping usually ends sooner."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--nodes", type=int, default=4, help="1 speaker + up to 3 addressees.")
    parser.add_argument("--edges-per-graph", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
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
        "--checkpoint-every", type=int, default=500, help="Push a checkpoint every N steps."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest checkpoint in --hf-repo."
    )
    parser.add_argument(
        "--run-id",
        default="pdnc-primary",
        help="Checkpoints are namespaced under this id; --resume must reuse the original run's id.",
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
        nodes=args.nodes,
        edges_per_graph=args.edges_per_graph,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        edge_types=EDGE_TYPES,
        attribute_classes=ATTRIBUTE_CLASSES,
        emotion_classes=EMOTION_CLASSES,
        delta_classes=DELTA_CLASSES,
    )

    checkpoint_cb = None
    best_checkpoint_cb = None
    resume_state = None
    manager = None
    run_id = args.run_id
    if args.hf_repo:
        from gnsm.training.checkpointing import attach_to_run

        checkpoint_cb, best_checkpoint_cb, resume_state, manager = attach_to_run(
            args.hf_repo, args.checkpoint_every, args.resume, run_id=run_id
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
            title="GNSM state encoder — PDNC train/val loss",
            val_steps=val_epochs,
            val_losses=val_losses,
            best_step=best_step,
            best_val_loss=best_val_loss,
        )
        if manager is not None:
            manager.push_artifact(png_path, f"plots/{run_id}/loss.png")
            manager.push_artifact(pdf_path, f"plots/{run_id}/loss.pdf")

    result = run_train_state(
        config,
        examples,
        collate_batch,
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
