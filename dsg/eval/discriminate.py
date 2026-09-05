"""Does the instrument tell human-written fiction from machine-generated?

A consistency metric is only an instrument if it discriminates. Published
novels are professionally edited: their continuity is close to airtight. Stories
written by a 3B model chapter by chapter are not. If the reader model cannot
separate the two, it is measuring noise; if it can, it is measuring something
real, and the separation itself is the calibration.

The comparison is controlled where it matters. Both corpora are read with the
*same* extraction prompt, the *same* reconciler and the *same* policy, over the
same number of chapters of similar length, so the only difference is who wrote
the text.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from dsg import invariants
from dsg.generate.run import parse_write_extraction
from dsg.policies.runner import apply_window
from dsg.schemas import Span, Window
from dsg.store import POLICIES, NarrativeState


@dataclass(slots=True)
class DocScore:
    doc_id: str
    source: str                 # "human" | "machine"
    chapters: int
    inconsistent_slot_rate: float
    inconsistent_slots: int
    total_slots: int
    live_facts: int
    entities: int

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "source": self.source, "chapters": self.chapters,
            "inconsistent_slot_rate": self.inconsistent_slot_rate,
            "inconsistent_slots": self.inconsistent_slots,
            "total_slots": self.total_slots,
            "live_facts": self.live_facts, "entities": self.entities,
        }


def read_document(
    chapters: list[str], extractions: list[str], policy: str = "append-only"
) -> NarrativeState:
    """Replay cached extractions into a state. No model, no GPU."""
    state = NarrativeState(POLICIES[policy])
    offset = 0
    for index, (text, raw) in enumerate(zip(chapters, extractions, strict=False)):
        window = Window(index=index, start=offset, end=offset + len(text), text=text)
        proposal = parse_write_extraction(raw, window)
        state.step(index, window.end)
        apply_window(state, proposal, Span(window.start, window.end))
        state.close_step()
        offset = window.end + 2
    return state


def score_document(
    doc_id: str, source: str, chapters: list[str], extractions: list[str],
    policy: str = "append-only",
) -> DocScore:
    state = read_document(chapters, extractions, policy)
    rate, bad, total = invariants.inconsistent_slot_rate(state.assertions.values())
    return DocScore(
        doc_id=doc_id, source=source, chapters=len(chapters),
        inconsistent_slot_rate=rate, inconsistent_slots=bad, total_slots=total,
        live_facts=sum(1 for a in state.assertions.values() if a.live),
        entities=len(state.live_entities()),
    )


def separation(human: list[float], machine: list[float]) -> dict:
    """How cleanly the two populations come apart.

    AUC is the probability that a randomly drawn machine document scores higher
    than a randomly drawn human one -- the natural read for a discriminator, and
    it needs no threshold. Cliff's delta reports the same ordering as an effect
    size.
    """
    if not human or not machine:
        return {}
    wins = ties = 0
    for m in machine:
        for h in human:
            if m > h:
                wins += 1
            elif m == h:
                ties += 1
    n = len(human) * len(machine)
    auc = (wins + 0.5 * ties) / n
    return {
        "auc": auc,
        "cliffs_delta": 2 * auc - 1,
        "human_mean": mean(human), "human_median": median(human), "n_human": len(human),
        "machine_mean": mean(machine), "machine_median": median(machine),
        "n_machine": len(machine),
    }


def bootstrap_auc(
    human: list[float], machine: list[float], n_boot: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval on the AUC, resampling documents in both arms."""
    import numpy as np

    rng = np.random.default_rng(seed)
    h, m = np.asarray(human), np.asarray(machine)
    out = []
    for _ in range(n_boot):
        hs = h[rng.integers(0, len(h), len(h))]
        ms = m[rng.integers(0, len(m), len(m))]
        out.append((ms[:, None] > hs[None, :]).mean()
                   + 0.5 * (ms[:, None] == hs[None, :]).mean())
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)
