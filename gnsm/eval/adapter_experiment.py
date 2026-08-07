"""Run the Stage C experiment matrix: {seeds} x {state conditions}.

The claim under test is narrow and falsifiable: *does conditioning a frozen LM
on GNSM's learned narrative state predict the true next scene better than
conditioning on a state that carries no example-specific information?*

Conditions (see `gnsm.training.train_adapter.apply_state_mode`):
  real      -- each example's own encoder state (treatment)
  shuffled  -- another example's state; identical state distribution, broken
               correspondence (the control that isolates example-specific
               information rather than "having a prefix at all")
  zero      -- no state signal whatsoever (floor)

Every condition trains its own adapter from scratch on the same split with the
same seed, so the only difference is the conditioning signal. Results are
reported as mean +/- bootstrap CI across seeds, with an exact paired sign test
(`gnsm.training.stats`) on the per-seed pairs -- not by eyeballing two numbers.

This trains adapters and reports validation loss; it does not generate or
judge free-form text (see gnsm/docs/adapter_results.md for what that does and
does not license claiming).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gnsm.training.stats import bootstrap_ci, bootstrap_paired_difference, paired_sign_test

DEFAULT_CONDITIONS = ("real", "shuffled", "zero")


def run_matrix(
    examples: list[Any],
    encoder_state_dict: dict[str, Any],
    seeds: list[int],
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    epochs: int = 30,
    batch_size: int = 8,
    patience: int = 5,
    lr: float = 1e-4,
    device: str = "auto",
    lm_name: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train one adapter per (condition, seed) and collect best val losses."""

    from gnsm.training.train_adapter import DEFAULT_LM, TrainAdapterConfig
    from gnsm.training.train_adapter import run as run_adapter

    per_condition: dict[str, list[float]] = {condition: [] for condition in conditions}
    runs: list[dict[str, Any]] = []

    for condition in conditions:
        for seed in seeds:
            config = TrainAdapterConfig(
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                lr=lr,
                seed=seed,
                device=device,
                lm_name=lm_name or DEFAULT_LM,
                state_mode=condition,
            )
            result = run_adapter(config, examples, encoder_state_dict)
            best = result["best_val_loss"]
            per_condition[condition].append(float(best))
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "best_val_loss": best,
                    "epochs_run": result["epochs_run"],
                    "early_stopped": result["early_stopped"],
                }
            )
            if progress:
                print(
                    f"[{condition:>8} seed={seed}] best_val_loss={best} "
                    f"epochs_run={result['epochs_run']} early_stopped={result['early_stopped']}",
                    flush=True,
                )

    summary: dict[str, Any] = {"per_condition": {}, "runs": runs, "seeds": seeds}
    for condition, values in per_condition.items():
        ci = bootstrap_ci(values, confidence=0.95, seed=0)
        summary["per_condition"][condition] = {
            "values": values,
            "mean": round(ci.mean, 4),
            "ci_low": round(ci.ci_low, 4),
            "ci_high": round(ci.ci_high, 4),
            "n_seeds": len(values),
        }

    # Paired comparisons against the treatment. Lower loss is better, so
    # "real wins" means real < control on that seed.
    summary["comparisons"] = {}
    if "real" in per_condition:
        for control in conditions:
            if control == "real":
                continue
            test = paired_sign_test(per_condition["real"], per_condition[control])
            paired = bootstrap_paired_difference(
                per_condition["real"], per_condition[control], confidence=0.95, seed=0
            )
            summary["comparisons"][f"real_vs_{control}"] = {
                "mean_delta": round(paired.mean_difference, 4),
                "delta_ci_low": round(paired.ci_low, 4),
                "delta_ci_high": round(paired.ci_high, 4),
                "delta_ci_excludes_zero": paired.significant,
                "real_better_on_n_seeds": test.n_negative,
                "control_better_on_n_seeds": test.n_positive,
                "ties": test.n_ties,
                "sign_test_p_value": round(test.p_value, 4),
                "sign_test_p_floor": round(2 / (2 ** max(1, len(seeds))), 4),
                "note": (
                    "negative mean_delta favours 'real' (lower loss is better); "
                    "the sign test cannot go below its p_floor at this seed count, "
                    "so read the bootstrap CI on the paired difference alongside it"
                ),
            }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm adapter-experiment",
        description="Seeds x state-condition matrix for the Stage C adapter.",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/evolvtrip_data/all_books_current.json")
    )
    parser.add_argument("--encoder-hf-repo", default="GOVINDFROM/DNG-GNSM")
    parser.add_argument("--encoder-run-id", default="evolvtrip-v2-earlystop")
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds.")
    parser.add_argument(
        "--conditions", default=",".join(DEFAULT_CONDITIONS), help="Comma-separated state modes."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write the summary JSON here.")
    args = parser.parse_args(argv)

    from gnsm.training.evolvtrip_adapter import load_examples
    from gnsm.training.train_adapter import _load_encoder_state

    examples = load_examples(args.data)
    if args.limit:
        examples = examples[: args.limit]

    encoder_state_dict = _load_encoder_state(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())

    summary = run_matrix(
        examples,
        encoder_state_dict,
        seeds=seeds,
        conditions=conditions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        device=args.device,
        lm_name=args.lm,
    )
    summary["n_examples"] = len(examples)

    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
