"""P0 go/no-go: measure whether narrative state linearly decodes labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gnsm.state.probes import LinearProbe


def run(features_path: Path, labels_path: Path, threshold: float = 0.6) -> dict[str, object]:
    features = np.load(features_path)
    labels = np.load(labels_path).astype(str)
    split = max(1, int(len(features) * 0.8))
    probe = LinearProbe().fit(features[:split], labels[:split])
    accuracy = probe.score(features[split:], labels[split:]) if split < len(features) else 1.0
    return {
        "accuracy": accuracy,
        "threshold": threshold,
        "decision": "go" if accuracy >= threshold else "no-go",
        "train_samples": split,
        "test_samples": len(features) - split,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    result = run(args.features, args.labels, args.threshold)
    print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
