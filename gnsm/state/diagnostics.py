"""Representation-collapse tripwires tracked as first-class metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def collapse_diagnostics(embeddings: NDArray[np.floating]) -> dict[str, float]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [samples, dimensions]")
    if embeddings.shape[0] == 0:
        return {"effective_rank": 0.0, "mean_variance": 0.0, "min_variance": 0.0}
    variances = np.var(embeddings, axis=0)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = singular_values**2
    if float(energy.sum()) == 0.0:
        effective_rank = 0.0
    else:
        probabilities = energy / energy.sum()
        nonzero = probabilities[probabilities > 0]
        entropy = -float(np.sum(nonzero * np.log(nonzero)))
        effective_rank = float(np.exp(entropy))
    return {
        "effective_rank": effective_rank,
        "mean_variance": float(np.mean(variances)),
        "min_variance": float(np.min(variances)),
    }


def is_collapse_suspected(
    diagnostics: dict[str, float],
    *,
    minimum_rank: float = 2.0,
    minimum_mean_variance: float = 1e-5,
) -> bool:
    return (
        diagnostics.get("effective_rank", 0.0) < minimum_rank
        or diagnostics.get("mean_variance", 0.0) < minimum_mean_variance
    )
