"""Structural invariants checked after every discourse step.

These are the operational content of 'the graph must not break'. A policy that
cannot revise will accumulate violations here as the book goes on; that
accumulation, measured against discourse position, is one of the paper's
headline results rather than an implementation detail.
"""

from __future__ import annotations

from collections.abc import Iterable

from dsg.lexicon import ONE_WAY_FALSE, compatible, is_multi_valued
from dsg.matching import MatchKind, match_kind
from dsg.schemas import Assertion, EntityNode, Status, Violation

# I1 two live, conflicting values in a single-valued slot with overlapping validity
# I2 a live assertion pointing at a dead (merged/retracted) entity
# I3 two committed live entities sharing a proper name
# I4 a merge chain that cycles or dangles
# I5 a malformed interval (closed before opened, live but closed)
# I6 provenance outside the prefix the reader has seen
# I7 a one-way predicate flipped back (the dead walking again)
CODES = ("I1", "I2", "I3", "I4", "I5", "I6", "I7")


def _intervals_overlap(a: Assertion, b: Assertion) -> bool:
    a_lo = a.valid_from if a.valid_from is not None else a.d_open
    b_lo = b.valid_from if b.valid_from is not None else b.d_open
    a_hi = a.valid_to if a.valid_to is not None else float("inf")
    b_hi = b.valid_to if b.valid_to is not None else float("inf")
    return a_lo <= b_hi and b_lo <= a_hi


def check(
    entities: dict[str, EntityNode],
    assertions: Iterable[Assertion],
    index: int,
    prefix_end: int,
) -> list[Violation]:
    out: list[Violation] = []
    live_assertions = [a for a in assertions if a.status is Status.BELIEVED]

    # I1 -- conflicting live values in one single-valued slot
    slots: dict[tuple[str, str], list[Assertion]] = {}
    for a in live_assertions:
        if is_multi_valued(a.predicate):
            continue
        slots.setdefault(a.slot, []).append(a)
    for (subject, predicate), group in slots.items():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                same = a.object.strip().lower() == b.object.strip().lower()
                if same and a.polarity == b.polarity:
                    continue
                # A more precise restatement is not a rival value.
                if a.polarity == b.polarity and compatible(a.object, b.object):
                    continue
                if _intervals_overlap(a, b):
                    out.append(
                        Violation(
                            index, "I1",
                            f"{subject}.{predicate}: '{a.object}'({a.polarity}) vs "
                            f"'{b.object}'({b.polarity}) both live and overlapping",
                        )
                    )

    # I2 -- live assertion referencing a dead entity
    for a in live_assertions:
        for role, ref in (("subject", a.subject), ("object", a.object)):
            node = entities.get(ref)
            if node is not None and not node.live:
                out.append(
                    Violation(index, "I2",
                              f"assertion {a.id} {role} -> {ref} ({node.status})")
                )

    # I3 -- two committed live entities claiming the same proper name
    live_nodes = [n for n in entities.values() if n.live and n.commit == "committed"]
    for i, n in enumerate(live_nodes):
        for m in live_nodes[i + 1 :]:
            shared = {
                s for s in n.proper_names
                for t in m.proper_names
                if match_kind(s, t) in (MatchKind.EXACT, MatchKind.TITLED)
            }
            if shared:
                out.append(
                    Violation(index, "I3",
                              f"{n.id} and {m.id} share proper name(s) {sorted(shared)}")
                )

    # I4 -- merge chains must terminate at a live node without cycling
    for node in entities.values():
        seen: set[str] = set()
        cur = node
        while cur.merged_into is not None:
            if cur.id in seen:
                out.append(Violation(index, "I4", f"merge cycle at {node.id}"))
                break
            seen.add(cur.id)
            nxt = entities.get(cur.merged_into)
            if nxt is None:
                out.append(
                    Violation(index, "I4",
                              f"{cur.id} merged into missing {cur.merged_into}")
                )
                break
            cur = nxt

    # I5 -- interval well-formedness
    for a in assertions:
        if a.d_close is not None and a.d_close < a.d_open:
            out.append(Violation(index, "I5", f"{a.id} closed {a.d_close} < open {a.d_open}"))
        if a.status is Status.BELIEVED and a.d_close is not None:
            out.append(Violation(index, "I5", f"{a.id} live but has d_close"))
        if a.valid_to is not None and a.valid_from is not None and a.valid_to < a.valid_from:
            out.append(Violation(index, "I5", f"{a.id} valid_to < valid_from"))

    # I6 -- prefix causality: nothing may cite text the reader has not reached
    for a in assertions:
        if a.provenance is not None and a.provenance.end > prefix_end:
            out.append(
                Violation(index, "I6",
                          f"{a.id} cites char {a.provenance.end} > prefix {prefix_end}")
            )

    # I7 -- one-way predicates cannot flip back
    for (subject, predicate), group in slots.items():
        if predicate not in ONE_WAY_FALSE:
            continue
        ordered = sorted(group, key=lambda a: a.d_open)
        seen_false = False
        for a in ordered:
            if not a.polarity or a.object.strip().lower() in {"false", "no", "dead"}:
                seen_false = True
            elif seen_false:
                out.append(
                    Violation(index, "I7",
                              f"{subject}.{predicate} returned to true after false")
                )
    return out


def inconsistent_slot_rate(
    assertions: Iterable[Assertion],
) -> tuple[float, int, int]:
    """Share of single-valued slots holding two or more live conflicting values.

    The raw I1 count is pairwise and therefore quadratic in how badly a slot has
    broken: one slot with twenty rival values contributes 190 violations. That
    is a real signal but an unreadable headline, so this reports the bounded
    quantity a reader actually wants -- what fraction of the state's slots are
    self-contradictory -- alongside it.
    """
    slots: dict[tuple[str, str], list[Assertion]] = {}
    for a in assertions:
        if a.status is not Status.BELIEVED or is_multi_valued(a.predicate):
            continue
        slots.setdefault(a.slot, []).append(a)
    if not slots:
        return 0.0, 0, 0

    def conflicted(group: list[Assertion]) -> bool:
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.polarity != b.polarity:
                    return True
                if a.object.strip().lower() == b.object.strip().lower():
                    continue
                if compatible(a.object, b.object):
                    continue
                return True
        return False

    bad = sum(1 for group in slots.values() if conflicted(group))
    return bad / len(slots), bad, len(slots)


def summarize(violations: Iterable[Violation]) -> dict[str, int]:
    counts = dict.fromkeys(CODES, 0)
    for v in violations:
        counts[v.code] = counts.get(v.code, 0) + 1
    counts["total"] = sum(counts[c] for c in CODES)
    return counts
