"""Holm correction for the generation comparisons.

The generation report computes a paired bootstrap for every pair in
``dsg.generate.score.COMPARISONS`` and every metric, and reads "the interval
excludes zero" off each one separately. None of those pairs was named as a test
before the runs, so the family is every comparison actually computed: within
each metric, all comparisons across the four runs the findings rest on.

For each comparison this reports the raw percentile interval (the same draws as
``dsg.eval.bootstrap.paired_bootstrap``, asserted equal), a two-sided bootstrap
p-value, the Holm-adjusted p-value, and a Holm-level interval: the percentile
interval at the level Holm's step-down uses for that comparison's rank,
alpha / (m - rank + 1). Holm has no exact simultaneous interval; this one agrees
with the Holm decision except where the step-down has already stopped, which the
``holm_rejects`` flag records.

    python -m dsg.eval.multiplicity --generation artifacts/hub/generation \\
        --out audit/generation/holm.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dsg.eval.bootstrap import paired_bootstrap
from dsg.eval.provenance import record as provenance
from dsg.generate.score import COMPARISONS, METRICS, score_runs

RUNS = ("gen-qwen3b-s30-c20", "gen-v3-hybrid", "gen-v4-retrieval", "gen-v5-selection")
N_BOOT = 10_000
ALPHA = 0.05
SEED = 0          # paired_bootstrap's default, used for every reported interval


def _draws(a: list[float], b: list[float], n_boot: int = N_BOOT, seed: int = SEED):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    return delta, delta[idx].mean(axis=1)


def bootstrap_p(means: np.ndarray) -> float:
    """Two-sided bootstrap p-value, with the +1 correction so it is never 0."""
    n = len(means)
    low = (int((means <= 0).sum()) + 1) / (n + 1)
    high = (int((means >= 0).sum()) + 1) / (n + 1)
    return min(1.0, 2 * min(low, high))


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, monotone, in input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _interval(means: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def collect(generation_dir: Path) -> list[dict]:
    rows = []
    for run in RUNS:
        payload = json.loads((generation_dir / f"{run}.json").read_text())
        by = defaultdict(dict)
        for s in score_runs(payload):
            by[s.condition][s.story_id] = s.as_dict()
        for a, b in COMPARISONS:
            if a not in by or b not in by:
                continue
            stories = sorted(set(by[a]) & set(by[b]))
            if not stories:
                continue
            for metric in METRICS:
                xs = [float(by[a][s][metric]) for s in stories]
                ys = [float(by[b][s][metric]) for s in stories]
                delta, means = _draws(xs, ys)
                lo, hi = _interval(means, ALPHA)
                ref = paired_bootstrap(xs, ys, n_boot=N_BOOT, alpha=ALPHA, seed=SEED)
                assert abs(lo - ref.ci_low) < 1e-12 and abs(hi - ref.ci_high) < 1e-12
                rows.append({
                    "run": run, "comparison": f"{a}_vs_{b}", "metric": metric,
                    "n_stories": len(stories), "mean_delta": float(delta.mean()),
                    "ci_raw": [lo, hi], "excludes_zero_raw": bool(lo > 0 or hi < 0),
                    "p_raw": bootstrap_p(means), "_means": means,
                })
    return rows


def correct(rows: list[dict]) -> dict[str, dict]:
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_metric[r["metric"]].append(r)
    families = {}
    for metric, fam in sorted(by_metric.items()):
        m = len(fam)
        adjusted = holm([r["p_raw"] for r in fam])
        order = sorted(range(m), key=lambda i: fam[i]["p_raw"])
        stopped = False
        for rank, i in enumerate(order):
            r = fam[i]
            level = ALPHA / (m - rank)
            r["holm_rank"] = rank + 1
            r["holm_level"] = level
            r["p_holm"] = adjusted[i]
            r["ci_holm_level"] = list(_interval(r.pop("_means"), level))
            rejects = not stopped and r["p_raw"] <= level
            stopped = stopped or not rejects
            r["holm_rejects"] = rejects
        families[metric] = {"family_size": m, "comparisons": fam}
    return families


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.multiplicity", description=__doc__)
    p.add_argument("--generation", default="artifacts/hub/generation")
    p.add_argument("--out", default="audit/generation/holm.json")
    args = p.parse_args(argv)
    gen = Path(args.generation)
    families = correct(collect(gen))
    out = {
        "provenance": provenance("dsg.eval.multiplicity", argv, {"bootstrap": SEED},
                                 [gen / f"{r}.json" for r in RUNS]),
        "runs": list(RUNS), "n_boot": N_BOOT, "alpha": ALPHA,
        "family": "within each metric, every comparison computed across the four runs",
        "families": families,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    vr = families["violation_rate"]
    print(f"violation_rate family size {vr['family_size']}")
    for r in vr["comparisons"]:
        print(f"{r['run']:20} {r['comparison']:55} {r['mean_delta']:+.3f} "
              f"raw [{r['ci_raw'][0]:+.3f},{r['ci_raw'][1]:+.3f}] p={r['p_raw']:.4f} "
              f"holm p={r['p_holm']:.4f} rejects={r['holm_rejects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
