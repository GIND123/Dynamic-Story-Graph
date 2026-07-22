"""Stage B trainer assembly for the encoder and grounded decode heads."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnsm.exceptions import OptionalDependencyError
from gnsm.training.common import TrainingRun


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("checkpoints/state"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run = TrainingRun("state-encoder-transition", args.output, args.seed)
    run.prepare()
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise OptionalDependencyError(
            "Stage B requires `python -m pip install -e '.[training]'`."
        ) from exc
    raise SystemExit(
        "Dataset-specific batching is intentionally adapter-owned. Implement a dataset adapter "
        "that emits node_features, graph labels, and next-scene deltas, then call the modules in "
        "gnsm.state.neural and gnsm.state.losses."
    )


if __name__ == "__main__":
    raise SystemExit(main())
