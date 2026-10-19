"""Inter-annotator agreement and grouped dev/test splits for the audit labels.

Agreement is reported four ways, per stratum, because the audit sheets are
heavily skewed (most unflagged mentions are "no"): Cohen's kappa collapses under
skew even when raw agreement is near perfect, so it is shown beside positive and
negative (specific) agreement and Gwet's AC1, which is stable under prevalence.

Splits are grouped so nothing a rule could memorise crosses the boundary: by
story premise for the detector (the same planted canon recurs across runs and
conditions) and by novel for the calculus. Premises are identified by content,
because the 7B prose runs reuse story ids for different premises.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict

EXCLUDED = {"n/a", ""}


def _pairs(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    return [(x, y) for x, y in zip(a, b, strict=True)
            if x not in EXCLUDED and y not in EXCLUDED]


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def gwet_ac1(pairs: list[tuple[str, str]]) -> float | None:
    """Gwet's AC1 for two raters over any number of categories."""
    n = len(pairs)
    cats = sorted({c for p in pairs for c in p})
    if not n or len(cats) < 2:
        return None if not n else 1.0
    po = sum(x == y for x, y in pairs) / n
    pi = {k: (sum(x == k for x, _ in pairs) + sum(y == k for _, y in pairs)) / (2 * n)
          for k in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (len(cats) - 1)
    return (po - pe) / (1 - pe)


def specific_agreement(pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    """Per-category specific agreement, 2*n_kk / (n_k. + n_.k).

    For a yes/no label, the "yes" entry is positive agreement and the "no"
    entry is negative agreement.
    """
    out: dict[str, float | None] = {}
    for k in sorted({c for p in pairs for c in p}):
        both = sum(x == k and y == k for x, y in pairs)
        either = sum(x == k for x, _ in pairs) + sum(y == k for _, y in pairs)
        out[k] = 2 * both / either if either else None
    return out


def agreement(a: list[str], b: list[str]) -> dict:
    pairs = _pairs(a, b)
    n = len(pairs)
    spec = specific_agreement(pairs)
    return {
        "n": n,
        "observed": sum(x == y for x, y in pairs) / n if n else None,
        "cohen_kappa": cohen_kappa(pairs),
        "gwet_ac1": gwet_ac1(pairs),
        "positive_agreement": spec.get("yes"),
        "negative_agreement": spec.get("no"),
        "specific_agreement": spec,
    }


def agreement_by_stratum(rows: list[dict], label: str, stratum: str) -> dict[str, dict]:
    """Agreement on ``label`` vs ``label_b`` in merged rows, overall and per stratum."""
    doubled = [r for r in rows if r.get("annotator_b")]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in doubled:
        groups[r.get(stratum) or "(none)"].append(r)
    out = {"all": agreement([r[label] for r in doubled], [r[f"{label}_b"] for r in doubled])}
    for g in sorted(groups):
        rs = groups[g]
        out[g] = agreement([r[label] for r in rs], [r[f"{label}_b"] for r in rs])
    return out


def premise_fingerprint(premise: dict) -> str:
    """A content id for a story premise: same canon, same fingerprint."""
    body = json.dumps({"canon": premise["canon"], "characters": premise["characters"],
                       "objects": premise["objects"]}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def grouped_split(groups: dict[str, str], dev_fraction: float, seed: int
                  ) -> dict[str, str]:
    """Assign every item to 'dev' or 'test' so that no group spans both.

    ``groups`` maps item id to group id. Groups are shuffled with ``seed`` and
    taken into dev until it holds ``dev_fraction`` of the items.
    """
    sizes = Counter(groups.values())
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    target = dev_fraction * len(groups)
    dev: set[str] = set()
    held = 0
    for g in order:
        if held >= target:
            break
        dev.add(g)
        held += sizes[g]
    return {item: ("dev" if g in dev else "test") for item, g in groups.items()}
