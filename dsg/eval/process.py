"""Plane C -- process integrity, and the revision trace as an instrument.

These are the measurements the conceptual literature on incremental narrative
interpretation asks for but has never reported on a corpus: how often an update
is a monotone refinement rather than a rollback, how often the state breaks its
own structural constraints, and where in a book revisions actually land.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dsg import invariants
from dsg.policies.runner import RunResult
from dsg.schemas import CommitLevel, Op, Status


@dataclass(slots=True)
class ProcessScore:
    windows: int
    ops_total: int
    monotone: int
    rollback: int
    monotone_fraction: float
    violations: int
    violations_per_100_windows: float
    violations_by_code: dict[str, int] = field(default_factory=dict)
    ops_by_type: dict[str, int] = field(default_factory=dict)
    live_entities: int = 0
    committed_entities: int = 0
    provisional_entities: int = 0
    live_assertions: int = 0
    superseded_assertions: int = 0
    retracted_assertions: int = 0
    contradiction_density: float = 0.0
    parse_failures: int = 0
    links_proposed: int = 0
    links_bound: int = 0
    links_merged: int = 0
    links_unresolved: int = 0

    def as_dict(self) -> dict[str, float]:
        base = {
            "windows": float(self.windows), "ops_total": float(self.ops_total),
            "monotone": float(self.monotone), "rollback": float(self.rollback),
            "monotone_fraction": self.monotone_fraction,
            "violations": float(self.violations),
            "violations_per_100w": self.violations_per_100_windows,
            "live_entities": float(self.live_entities),
            "committed_entities": float(self.committed_entities),
            "provisional_entities": float(self.provisional_entities),
            "live_assertions": float(self.live_assertions),
            "superseded": float(self.superseded_assertions),
            "retracted": float(self.retracted_assertions),
            "contradiction_density": self.contradiction_density,
            "parse_failures": float(self.parse_failures),
            "links_proposed": float(self.links_proposed),
            "links_bound": float(self.links_bound),
            "links_merged": float(self.links_merged),
            "links_unresolved": float(self.links_unresolved),
        }
        base.update({f"viol_{k}": float(v) for k, v in self.violations_by_code.items()})
        base.update({f"op_{k}": float(v) for k, v in self.ops_by_type.items()})
        return base


def score_process(result: RunResult) -> ProcessScore:
    state = result.state
    counts = state.op_counts()
    ops_total = len(state.log)
    live_nodes = state.live_entities()
    live_assertions = sum(1 for a in state.assertions.values() if a.live)
    windows = max(1, result.windows)
    return ProcessScore(
        windows=result.windows,
        ops_total=ops_total,
        monotone=counts["monotone"],
        rollback=counts["rollback"],
        monotone_fraction=counts["monotone"] / ops_total if ops_total else 0.0,
        violations=len(state.violations),
        violations_per_100_windows=100 * len(state.violations) / windows,
        violations_by_code=invariants.summarize(state.violations),
        ops_by_type={op.value: counts[op.value] for op in Op},
        live_entities=len(live_nodes),
        committed_entities=sum(1 for n in live_nodes if n.commit is CommitLevel.COMMITTED),
        provisional_entities=sum(1 for n in live_nodes if n.commit is CommitLevel.PROVISIONAL),
        live_assertions=live_assertions,
        superseded_assertions=sum(
            1 for a in state.assertions.values() if a.status is Status.SUPERSEDED
        ),
        retracted_assertions=sum(
            1 for a in state.assertions.values() if a.status is Status.RETRACTED
        ),
        contradiction_density=len(state.violations) / max(1, live_assertions),
        parse_failures=result.parse_failures,
        links_proposed=result.links_proposed,
        links_bound=result.links_bound,
        links_merged=result.links_merged,
        links_unresolved=result.links_unresolved,
    )


def revision_profile(result: RunResult, n_bins: int = 20) -> list[dict[str, float]]:
    """Where in the book the reader's model gets refined versus corrected.

    This is the humanities-facing output: a per-book curve of elaboration and
    revision events over discourse position, computed from the same log that
    the integrity metrics read.
    """
    windows = max(1, result.windows)
    bins = [
        {"bin": i, "elaborate": 0, "supersede": 0, "revise": 0,
         "merge_provisional": 0, "merge_committed": 0, "assert": 0}
        for i in range(n_bins)
    ]
    for record in result.state.log:
        b = min(n_bins - 1, int(record.index / windows * n_bins))
        key = record.op.value
        if key in bins[b]:
            bins[b][key] += 1
    for b in bins:
        total = sum(b[k] for k in ("elaborate", "supersede", "revise",
                                   "merge_provisional", "merge_committed", "assert"))
        b["total"] = total
        b["revision_rate"] = (b["revise"] + b["merge_committed"]) / total if total else 0.0
        b["elaboration_rate"] = (b["elaborate"] + b["merge_provisional"]) / total if total else 0.0
    return bins


def growth_curve(result: RunResult) -> list[dict[str, int]]:
    return result.state.trace
