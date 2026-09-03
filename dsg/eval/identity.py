"""Plane A -- identity, scored against human alias annotation.

The question is the one the whole paper turns on: does the reader's model keep
one person as one node? PDNC gives, per novel, the set of surface forms a human
annotator judged to denote each character. Those alias sets induce a gold
partition; the entity registry induces a system partition over the same items;
standard coreference measures compare them.

Two design choices matter for fairness:

* The item set is fixed **per book, before any system runs** -- the gold aliases
  that actually occur in the novel text. A system cannot improve its score by
  extracting less, because everything it misses stays in the item set as a
  singleton.
* Aliases the annotation itself leaves ambiguous (a surname shared by siblings)
  are excluded upstream in ``Novel.alias_to_character``. Scoring them would
  measure annotation noise, not system behaviour.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from dsg.data.pdnc import Novel
from dsg.store import NarrativeState


@dataclass(slots=True)
class IdentityScore:
    b3_precision: float
    b3_recall: float
    b3_f1: float
    ceaf_f1: float
    muc_f1: float
    conll_f1: float
    b3_f1_multi: float
    coverage: float
    fragmentation: float
    conflation: float
    n_items: int
    n_gold_clusters: int
    n_system_clusters: int
    observed_items: int

    def as_dict(self) -> dict[str, float]:
        return {
            "b3_p": self.b3_precision, "b3_r": self.b3_recall, "b3_f1": self.b3_f1,
            "ceaf_f1": self.ceaf_f1, "muc_f1": self.muc_f1, "conll_f1": self.conll_f1,
            "b3_f1_multi": self.b3_f1_multi, "coverage": self.coverage,
            "fragmentation": self.fragmentation, "conflation": self.conflation,
            "n_items": float(self.n_items), "n_gold_clusters": float(self.n_gold_clusters),
            "n_system_clusters": float(self.n_system_clusters),
            "observed_items": float(self.observed_items),
        }


def occurring_aliases(novel: Novel, min_len: int = 3) -> dict[str, str]:
    """Gold aliases that actually appear in the text, mapped to their character."""
    text = novel.text.lower()
    out: dict[str, str] = {}
    for alias, char_id in novel.alias_to_character().items():
        alias = alias.strip()
        if len(alias) < min_len:
            continue
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
            out[alias] = char_id
    return out


def system_partition(state: NarrativeState, items: dict[str, str]) -> dict[str, str]:
    """Assign each gold alias to the live node that most often observed it."""
    best: dict[str, tuple[int, str]] = {}
    for node in state.live_entities():
        node_id = state.deref(node.id)
        for surface, count in node.surfaces.items():
            key = surface.strip().lower()
            if key not in items:
                continue
            prior = best.get(key)
            if prior is None or count > prior[0]:
                best[key] = (count, node_id)
    # Anything the system never saw stays alone -- an honest penalty, applied
    # identically to every policy because the item set is fixed in advance.
    return {
        alias: (best[alias][1] if alias in best else f"__unseen__{alias}")
        for alias in items
    }


def _clusters(assignment: dict[str, str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for item, cluster in assignment.items():
        out.setdefault(cluster, set()).add(item)
    return out


def _b3(gold: dict[str, str], system: dict[str, str]) -> tuple[float, float, float]:
    g, s = _clusters(gold), _clusters(system)
    precision = recall = 0.0
    for item in gold:
        gi, si = g[gold[item]], s[system[item]]
        inter = len(gi & si)
        precision += inter / len(si)
        recall += inter / len(gi)
    n = len(gold) or 1
    p, r = precision / n, recall / n
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _ceaf_e(gold: dict[str, str], system: dict[str, str]) -> float:
    """CEAF-e with a greedy alignment (exact Hungarian is unnecessary here:
    clusters are small and the greedy bound is within noise of the optimum)."""
    g, s = _clusters(gold), _clusters(system)
    pairs = sorted(
        (
            (2 * len(gi & si) / (len(gi) + len(si)), gk, sk)
            for gk, gi in g.items()
            for sk, si in s.items()
            if gi & si
        ),
        key=lambda t: -t[0],
    )
    used_g, used_s, total = set(), set(), 0.0
    for score, gk, sk in pairs:
        if gk in used_g or sk in used_s:
            continue
        used_g.add(gk)
        used_s.add(sk)
        total += score
    p = total / (len(s) or 1)
    r = total / (len(g) or 1)
    return 2 * p * r / (p + r) if p + r else 0.0


def _muc(gold: dict[str, str], system: dict[str, str]) -> float:
    g, s = _clusters(gold), _clusters(system)

    def links(a: dict[str, set[str]], b: dict[str, set[str]], mapping: dict[str, str]) -> float:
        num = den = 0.0
        for cluster in a.values():
            partitions = Counter(mapping[item] for item in cluster)
            num += len(cluster) - len(partitions)
            den += len(cluster) - 1
        return num / den if den else 0.0

    r = links(g, s, system)
    p = links(s, g, gold)
    return 2 * p * r / (p + r) if p + r else 0.0


def score_identity(state: NarrativeState, novel: Novel, items: dict[str, str] | None = None) -> IdentityScore:
    gold = items if items is not None else occurring_aliases(novel)
    if not gold:
        return IdentityScore(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    system = system_partition(state, gold)
    observed = sum(1 for a in gold if not system[a].startswith("__unseen__"))

    p, r, f1 = _b3(gold, system)
    ceaf = _ceaf_e(gold, system)
    muc = _muc(gold, system)
    conll = (muc + f1 + ceaf) / 3

    # B3 restricted to characters the annotation gives more than one surface
    # form. Singleton gold clusters are scored correct by any system that
    # simply never groups anything, so the unrestricted figure has a high
    # floor; this subset is where grouping is actually under test.
    sizes: dict[str, int] = {}
    for char_id in gold.values():
        sizes[char_id] = sizes.get(char_id, 0) + 1
    multi = {a: c for a, c in gold.items() if sizes[c] > 1}
    b3_multi = _b3(multi, {a: system[a] for a in multi})[2] if multi else 0.0

    # Fragmentation: a gold character split across several system nodes.
    by_char: dict[str, set[str]] = {}
    for alias, char_id in gold.items():
        node = system[alias]
        if node.startswith("__unseen__"):
            continue
        by_char.setdefault(char_id, set()).add(node)
    fragmentation = (
        sum(len(v) - 1 for v in by_char.values()) / len(by_char) if by_char else 0.0
    )

    # Conflation: one system node holding several gold characters.
    by_node: dict[str, set[str]] = {}
    for alias, char_id in gold.items():
        node = system[alias]
        if node.startswith("__unseen__"):
            continue
        by_node.setdefault(node, set()).add(char_id)
    conflation = (
        sum(len(v) - 1 for v in by_node.values()) / len(by_node) if by_node else 0.0
    )

    return IdentityScore(
        b3_precision=p, b3_recall=r, b3_f1=f1, ceaf_f1=ceaf, muc_f1=muc,
        conll_f1=conll, b3_f1_multi=b3_multi,
        coverage=observed / len(gold),
        fragmentation=fragmentation, conflation=conflation,
        n_items=len(gold), n_gold_clusters=len(set(gold.values())),
        n_system_clusters=len({v for v in system.values() if not v.startswith("__unseen__")}),
        observed_items=observed,
    )


def node_to_character(state: NarrativeState, novel: Novel) -> dict[str, str]:
    """Majority-vote map from system node id to gold character id."""
    alias_map = novel.alias_to_character()
    votes: dict[str, Counter] = {}
    for node in state.live_entities():
        node_id = state.deref(node.id)
        counter = votes.setdefault(node_id, Counter())
        for surface, count in node.surfaces.items():
            char_id = alias_map.get(surface.strip().lower())
            if char_id:
                counter[char_id] += count
    return {n: c.most_common(1)[0][0] for n, c in votes.items() if c}
