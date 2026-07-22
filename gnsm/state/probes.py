"""Stage-0 decodability probes using a closed-form linear classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class LinearProbe:
    regularization: float = 1e-3
    weights: NDArray[np.float64] | None = None
    classes: NDArray[np.str_] | None = None

    def fit(self, features: NDArray[np.floating], labels: NDArray[np.str_]) -> LinearProbe:
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError("expected features [n, d] and labels [n]")
        if len(features) != len(labels):
            raise ValueError("feature and label counts differ")
        classes, encoded = np.unique(labels, return_inverse=True)
        targets = np.eye(len(classes), dtype=np.float64)[encoded]
        x = np.column_stack([features, np.ones(len(features))]).astype(np.float64)
        identity = np.eye(x.shape[1], dtype=np.float64)
        self.weights = np.linalg.solve(x.T @ x + self.regularization * identity, x.T @ targets)
        self.classes = classes
        return self

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.str_]:
        if self.weights is None or self.classes is None:
            raise RuntimeError("probe must be fit before prediction")
        x = np.column_stack([features, np.ones(len(features))]).astype(np.float64)
        return self.classes[np.argmax(x @ self.weights, axis=1)]

    def score(self, features: NDArray[np.floating], labels: NDArray[np.str_]) -> float:
        return float(np.mean(self.predict(features) == labels))
