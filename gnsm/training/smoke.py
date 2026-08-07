"""End-to-end GPU smoke test for the learned C4/C5 modules.

Purpose: after cloning into a fresh GPU runtime, prove the neural stack trains.
It fabricates a small, fixed synthetic dataset of scene graphs and runs a few
optimizer steps through the real modules in :mod:`gnsm.state.neural` and the real
objective in :mod:`gnsm.state.losses`. If the loss drops on the reported device,
the CUDA path works. This is not a research result - it is a wiring check.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from gnsm.exceptions import OptionalDependencyError


@dataclass(slots=True)
class SmokeConfig:
    batch_size: int = 8
    nodes: int = 6
    edges_per_graph: int = 10
    input_dim: int = 64
    hidden_dim: int = 128
    layers: int = 2
    heads: int = 4
    edge_types: int = 8
    attribute_classes: int = 5
    emotion_classes: int = 6
    delta_classes: int = 12
    steps: int = 60
    lr: float = 1e-3
    seed: int = 0
    device: str = "auto"


def resolve_device(requested: str) -> str:
    """Pick a torch device string, honouring an explicit request."""

    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def plan(config: SmokeConfig) -> dict[str, int]:
    """Pure description of the synthetic problem (testable without torch)."""

    return {
        "graphs": config.batch_size,
        "node_rows": config.batch_size * config.nodes,
        "edge_rows": config.batch_size * config.edges_per_graph,
        "delta_rows": config.batch_size,
        "optimizer_steps": config.steps,
    }


def run(
    config: SmokeConfig | None = None,
    checkpoint_cb: Callable[[int, float, dict[str, Any], dict[str, Any]], None] | None = None,
    resume_state: dict[str, Any] | None = None,
) -> dict[str, object]:
    config = config or SmokeConfig()
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise OptionalDependencyError(
            "The GPU smoke requires the training extra: "
            "pip install -e '.[training]' (or requirements-colab.txt)."
        ) from exc

    from gnsm.state.losses import grounded_state_loss
    from gnsm.state.neural import (
        GraphStateEncoder,
        GroundedStateHeads,
        NeuralTransitionModel,
    )

    torch.manual_seed(config.seed)
    device = torch.device(resolve_device(config.device))

    encoder = GraphStateEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        heads=config.heads,
        edge_types=config.edge_types,
    ).to(device)
    heads = GroundedStateHeads(
        hidden_dim=config.hidden_dim,
        edge_types=config.edge_types,
        attribute_classes=config.attribute_classes,
        emotion_classes=config.emotion_classes,
    ).to(device)
    transition = NeuralTransitionModel(
        hidden_dim=config.hidden_dim,
        delta_classes=config.delta_classes,
    ).to(device)

    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def randint(high: int, *shape: int) -> torch.Tensor:
        return torch.randint(0, high, shape, generator=generator).to(device)

    # A fixed synthetic batch the model should be able to overfit.
    node_features = torch.randn(
        config.batch_size, config.nodes, config.input_dim, generator=generator
    ).to(device)
    next_node_features = torch.randn(
        config.batch_size, config.nodes, config.input_dim, generator=generator
    ).to(device)
    action_features = torch.randn(config.batch_size, config.hidden_dim, generator=generator).to(
        device
    )
    edge_pairs = torch.randint(
        0, config.nodes, (config.batch_size, config.edges_per_graph, 2), generator=generator
    ).to(device)
    edge_rows = config.batch_size * config.edges_per_graph
    edge_labels = randint(config.edge_types, edge_rows)
    # GroundedStateHeads decodes attributes from the same `source` rows it decodes
    # edges from, so attribute labels are sized per edge-source row (B*E) here.
    attribute_labels = randint(config.attribute_classes, edge_rows)
    delta_labels = randint(config.delta_classes, config.batch_size)

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

    batch_index = (
        torch.arange(config.batch_size, device=device)
        .view(-1, 1)
        .expand(-1, config.edges_per_graph)
    )

    def forward() -> torch.Tensor:
        node_embeddings, global_state = encoder(node_features)
        with torch.no_grad():
            _, encoded_next = encoder(next_node_features)
        source = node_embeddings[batch_index, edge_pairs[..., 0]].reshape(-1, config.hidden_dim)
        target = node_embeddings[batch_index, edge_pairs[..., 1]].reshape(-1, config.hidden_dim)
        head_out = heads(source, target, global_state)
        attribute_logits = head_out["attribute_logits"].reshape(-1, config.attribute_classes)
        predicted_state, delta_logits = transition(global_state, action_features)
        loss, _ = grounded_state_loss(
            edge_logits=head_out["edge_logits"],
            edge_labels=edge_labels,
            attribute_logits=attribute_logits,
            attribute_labels=attribute_labels,
            delta_logits=delta_logits,
            delta_labels=delta_labels,
            predicted_state=predicted_state,
            encoded_next_state=encoded_next,
        )
        return loss

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    encoder.train()
    heads.train()
    transition.train()

    initial_loss = float("nan")
    final_loss = float("nan")
    for step in range(start_step, config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = forward()
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if step == start_step:
            initial_loss = value
        final_loss = value
        if checkpoint_cb is not None:
            checkpoint_cb(
                step,
                value,
                {
                    "encoder": encoder.state_dict(),
                    "heads": heads.state_dict(),
                    "transition": transition.state_dict(),
                },
                optimizer.state_dict(),
            )

    peak_memory_gb = None
    if device.type == "cuda":
        peak_memory_gb = round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)

    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    result: dict[str, object] = {
        "device": str(device),
        "steps": config.steps,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "loss_decreased": bool(final_loss < initial_loss),
        "trainable_parameters": int(trainable),
        "peak_memory_gb": peak_memory_gb,
        "nn_module_check": isinstance(encoder, nn.Module),
        "config": asdict(config),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm smoke", description="Train the GNSM neural stack on a synthetic batch."
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--device", default="auto", help="auto (default), cuda, cpu, or an explicit device string"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit metrics as JSON")
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="Push periodic checkpoints to this private HF repo (owner/name).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50, help="Push a checkpoint every N steps."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest checkpoint in --hf-repo."
    )
    parser.add_argument(
        "--run-id",
        default="smoke-primary",
        help="Checkpoints are namespaced under this id; --resume must reuse the original run's id.",
    )
    args = parser.parse_args(argv)

    config = SmokeConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
        seed=args.seed,
    )

    checkpoint_cb = None
    resume_state = None
    if args.hf_repo:
        from gnsm.training.checkpointing import attach_to_run

        checkpoint_cb, _best_checkpoint_cb, resume_state, _manager = attach_to_run(
            args.hf_repo,
            args.checkpoint_every,
            args.resume,
            run_id=args.run_id,
        )

    result = run(config, checkpoint_cb=checkpoint_cb, resume_state=resume_state)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"device={result['device']}  "
            f"loss {result['initial_loss']} -> {result['final_loss']}  "
            f"(decreased={result['loss_decreased']})  "
            f"params={result['trainable_parameters']:,}  "
            f"peak_mem_gb={result['peak_memory_gb']}"
        )
    return 0 if result["loss_decreased"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
