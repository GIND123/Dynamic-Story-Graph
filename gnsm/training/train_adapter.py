"""Stage C conditioning-adapter trainer entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnsm.exceptions import OptionalDependencyError
from gnsm.training.common import TrainingRun


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("checkpoints/adapter"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    TrainingRun("state-conditioning-adapter", args.output, args.seed).prepare()
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise OptionalDependencyError(
            "Stage C requires `python -m pip install -e '.[training]'`."
        ) from exc
    raise SystemExit(
        "Connect the selected frozen generator to StatePrefixAdapter or "
        "StateCrossAttentionAdapter, then optimize gold-continuation LM loss plus consistency loss."
    )


if __name__ == "__main__":
    raise SystemExit(main())
