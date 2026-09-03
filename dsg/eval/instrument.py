"""The revision trace read as a measurement of the text.

Once identity resolution is deferred rather than guessed, the system records
something no retrospective pipeline can: *how long a novel keeps a figure
unnamed before the reader is told who they are*. That quantity -- identity
resolution latency -- is a property of the narration, not of the model, and it
is measurable only from a prefix-causal state.

The claim tested here is deliberately modest and pre-registered: first-person
narration bounds what the reader may know to what the narrator knows, so
first-person texts should hold identities open longer than third-person ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from dsg.data.pdnc import Novel
from dsg.policies.runner import RunResult
from dsg.schemas import CommitLevel, Op


@dataclass(slots=True)
class LatencyScore:
    book: str
    person: str
    genre: str
    n_resolved: int
    n_unresolved: int
    mean_latency: float          # in normalised discourse position (0-1)
    median_latency: float
    max_latency: float
    unresolved_fraction: float
    provisional_created: int

    def as_dict(self) -> dict[str, float | str]:
        return {
            "book": self.book, "person": self.person, "genre": self.genre,
            "n_resolved": self.n_resolved, "n_unresolved": self.n_unresolved,
            "mean_latency": self.mean_latency, "median_latency": self.median_latency,
            "max_latency": self.max_latency,
            "unresolved_fraction": self.unresolved_fraction,
            "provisional_created": self.provisional_created,
        }


def resolution_latency(result: RunResult, novel: Novel) -> LatencyScore:
    state = result.state
    windows = max(1, result.windows)
    latencies: list[float] = []

    for record in state.log:
        if record.op is not Op.MERGE_PROVISIONAL or not record.payload:
            continue
        absorbed = state.entities.get(record.payload)
        if absorbed is None:
            continue
        gap = (record.index - absorbed.first_seen) / windows
        if gap >= 0:
            latencies.append(gap)

    # A description the book never ties to a name: the question stays open.
    unresolved = [
        n for n in state.live_entities()
        if n.commit is CommitLevel.PROVISIONAL and not n.proper_names
    ]
    created = sum(
        1 for r in state.log
        if r.op is Op.ASSERT and "provisional entity" in r.detail and "description" in r.detail
    )
    total = len(latencies) + len(unresolved)
    return LatencyScore(
        book=novel.book_id,
        person=novel.meta.get("person", "unknown"),
        genre=novel.meta.get("genre", "unknown"),
        n_resolved=len(latencies),
        n_unresolved=len(unresolved),
        mean_latency=sum(latencies) / len(latencies) if latencies else 0.0,
        median_latency=median(latencies) if latencies else 0.0,
        max_latency=max(latencies) if latencies else 0.0,
        unresolved_fraction=len(unresolved) / total if total else 0.0,
        provisional_created=created,
    )


@dataclass(slots=True)
class GroupComparison:
    metric: str
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    excludes_zero: bool

    def as_dict(self) -> dict:
        return {
            "metric": self.metric, "group_a": self.group_a, "group_b": self.group_b,
            "n_a": self.n_a, "n_b": self.n_b, "mean_a": self.mean_a,
            "mean_b": self.mean_b, "delta": self.delta,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "excludes_zero": self.excludes_zero,
        }


def compare_groups(
    scores: list[LatencyScore], key: str = "person", metric: str = "mean_latency",
    a: str = "first", b: str = "third", n_boot: int = 10000, seed: int = 0,
) -> GroupComparison | None:
    """Unpaired bootstrap on the difference of group means.

    The groups are unequal and small (7 first-person novels against 21), so this
    resamples each group independently rather than pretending to a paired design
    the data does not support.
    """
    import numpy as np

    xs = [float(getattr(s, metric)) for s in scores if getattr(s, key) == a]
    ys = [float(getattr(s, metric)) for s in scores if getattr(s, key) == b]
    if len(xs) < 2 or len(ys) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr_x, arr_y = np.asarray(xs), np.asarray(ys)
    boots = (
        arr_x[rng.integers(0, len(xs), size=(n_boot, len(xs)))].mean(axis=1)
        - arr_y[rng.integers(0, len(ys), size=(n_boot, len(ys)))].mean(axis=1)
    )
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return GroupComparison(
        metric=metric, group_a=a, group_b=b, n_a=len(xs), n_b=len(ys),
        mean_a=float(arr_x.mean()), mean_b=float(arr_y.mean()),
        delta=float(arr_x.mean() - arr_y.mean()),
        ci_low=float(lo), ci_high=float(hi),
        excludes_zero=bool(lo > 0 or hi < 0),
    )


def position_trend(
    profiles: dict[str, dict], policy: str = "dsg-full"
) -> dict[str, object]:
    """Do elaboration and revision move with position in the book?

    Reported with a shuffled-window control. As a book proceeds the state holds
    more assertions, so *any* conflict-driven operation becomes mechanically
    more likely -- a rising revision rate could be pure state growth rather than
    a property of the narration. Shuffling the reading order destroys narrative
    order while preserving state growth exactly, so a trend that survives the
    shuffle is an artefact and one that disappears is a property of the text.
    """
    from scipy import stats

    bins: dict[int, dict[str, list[float]]] = {}
    for by_policy in profiles.values():
        for entry in by_policy.get(policy, []):
            slot = bins.setdefault(entry["bin"], {"elab": [], "rev": []})
            slot["elab"].append(float(entry["elaboration_rate"]))
            slot["rev"].append(float(entry["revision_rate"]))
    if len(bins) < 4:
        return {}
    xs = sorted(bins)
    out: dict[str, object] = {"n_bins": len(xs)}
    for key, name in (("elab", "elaboration"), ("rev", "revision")):
        ys = [sum(bins[x][key]) / len(bins[x][key]) for x in xs]
        rho, p = stats.spearmanr(xs, ys)
        out[name] = {
            "rho": float(rho), "p_value": float(p),
            "first_bin": ys[0], "last_bin": ys[-1],
            "series": [{"bin": x, "rate": y} for x, y in zip(xs, ys, strict=False)],
        }
    return out


def analyse(
    results: dict[str, RunResult], novels: dict[str, Novel], policy: str = "dsg-full"
) -> dict:
    scores = [
        resolution_latency(r, novels[b])
        for b, r in results.items()
        if r.policy == policy and b in novels
    ]
    comparisons = []
    for metric in ("mean_latency", "median_latency", "unresolved_fraction", "max_latency"):
        comparison = compare_groups(scores, "person", metric)
        if comparison is not None:
            comparisons.append(comparison.as_dict())
    return {
        "per_book": [s.as_dict() for s in scores],
        "by_narrative_person": comparisons,
    }
