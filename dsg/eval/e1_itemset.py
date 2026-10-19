"""Rescore saved E1 predictions on the published PDNC item set.

Published PDNC attribution numbers (Michel et al. 2024, Table 1) are on the
22-novel first release and on characters who utter at least ten quotes in a
novel. E1 scored all 28 novels and every annotated speaker, so its
explicit/non-explicit ordering was compared across different item sets. This
rescores the stored replies -- no model is run -- on both item sets, with the
same answer parser and scorer the run used, and checks that the all-items
figures reproduce the reported ones.

    python -m dsg.eval.e1_itemset --out audit/e1/published_itemset_v3.json

``--raw`` and ``--scored`` take a path or a name from ``audit/e1/hub_manifest.json``
(default: the v3 run), resolved by ``dsg.eval.e1_artifacts``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from dsg.eval.attribution import score_attribution
from dsg.eval.attribution_run import parse_answer
from dsg.eval.e1_artifacts import resolve
from dsg.eval.multiplicity import ALPHA, N_BOOT, _draws, _interval, bootstrap_p
from dsg.eval.provenance import record as provenance

# Added in PDNC's June 2023 release (Michel et al. 2024, Table 5, middle block).
SECOND_RELEASE = frozenset({"APassageToIndia", "MansfieldPark", "WinnieThePooh",
                            "WhereAngelsFearToTread", "OliverTwist", "HardTimes"})
MIN_QUOTES = 10
METRICS = ("accuracy", "accuracy_explicit", "accuracy_non_explicit")


def speaker_counts(folder: Path) -> Counter:
    with (folder / "quotation_info.csv").open(encoding="utf-8") as fh:
        return Counter((row.get("speaker") or "").strip() for row in csv.DictReader(fh))


def rescore(raw: dict, pdnc_root: Path, condition: str) -> dict:
    per = {"all": {}, "published": {}}
    micro = {"all": Counter(), "published": Counter()}
    for novel, payload in sorted(raw["raw"].items()):
        meta = raw["meta"][novel]
        aliases = {k: set(v) for k, v in meta["aliases"].items()}
        counts = speaker_counts(pdnc_root / novel)
        rows = [(parse_answer(reply, meta["candidates"]), gold, qtype)
                for (cond, _qid, gold, qtype), reply in zip(payload["keys"], payload["replies"],
                                                            strict=True)
                if cond == condition]
        sets = {"all": rows}
        if novel not in SECOND_RELEASE:
            sets["published"] = [r for r in rows if counts[r[1]] >= MIN_QUOTES]
        for name, rs in sets.items():
            if not rs:
                continue
            s = score_attribution([p for p, _, _ in rs], [g for _, g, _ in rs],
                                  [t for _, _, t in rs], alias_sets=aliases).as_dict()
            per[name][novel] = {"n": len(rs), **{m: s[m] for m in METRICS}}
            for p, g, t in rs:
                ok = score_attribution([p], [g], [t], alias_sets=aliases).as_dict()["accuracy"]
                explicit = t == "Explicit"
                micro[name]["n"] += 1
                micro[name]["hit"] += ok
                micro[name]["n_explicit" if explicit else "n_non_explicit"] += 1
                micro[name]["hit_explicit" if explicit else "hit_non_explicit"] += ok
    out = {}
    for name in per:
        novels = per[name]
        # Non-explicit minus explicit, paired over novels: the inversion test.
        keep = sorted(n for n, v in novels.items()
                      if v["accuracy_explicit"] is not None
                      and v["accuracy_non_explicit"] is not None)
        delta, means = _draws([novels[n]["accuracy_non_explicit"] for n in keep],
                              [novels[n]["accuracy_explicit"] for n in keep])
        out_delta = {"n_novels": len(keep), "mean_delta": float(delta.mean()),
                     "ci": list(_interval(means, ALPHA)), "p": bootstrap_p(means),
                     "novels_non_explicit_higher": int((delta > 0).sum()),
                     "novels_explicit_higher": int((delta < 0).sum()), "n_boot": N_BOOT}
        out[name] = {
            "non_explicit_minus_explicit": out_delta,
            "novels": len(novels), "quotes": sum(v["n"] for v in novels.values()),
            "macro": {m: sum(v[m] for v in novels.values()) / len(novels) for m in METRICS},
            "micro": {
                "accuracy": micro[name]["hit"] / micro[name]["n"],
                "accuracy_explicit": micro[name]["hit_explicit"] / micro[name]["n_explicit"],
                "accuracy_non_explicit": (micro[name]["hit_non_explicit"]
                                          / micro[name]["n_non_explicit"]),
            },
            "per_novel": novels,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.e1_itemset", description=__doc__)
    p.add_argument("--raw", default="e1-qwen7b-v3-raw.json",
                   help="a path, or a name from audit/e1/hub_manifest.json")
    p.add_argument("--scored", default="e1-qwen7b-v3.json", help="the run's own scored file")
    p.add_argument("--pdnc", type=Path, default=Path("data/pdnc/data"))
    p.add_argument("--condition", default="text-causal")
    p.add_argument("--out", type=Path, default=Path("audit/e1/published_itemset.json"))
    args = p.parse_args(argv)
    args.raw, args.scored = resolve(args.raw), resolve(args.scored)
    raw = json.loads(args.raw.read_text())
    result = rescore(raw, args.pdnc, args.condition)
    scored = json.loads(args.scored.read_text())["per_novel"]
    reported = sum(v[args.condition]["accuracy"] for v in scored.values()) / len(scored)
    assert abs(reported - result["all"]["macro"]["accuracy"]) < 1e-9, "does not reproduce"
    out = {"provenance": provenance("dsg.eval.e1_itemset", argv, {}, [args.raw, args.scored]),
           "condition": args.condition, "min_quotes": MIN_QUOTES,
           "second_release_excluded": sorted(SECOND_RELEASE),
           "reproduces_reported_macro": reported, **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for name in ("all", "published"):
        r = result[name]
        print(name, r["novels"], "novels", r["quotes"], "quotes | macro",
              {k: round(v, 4) for k, v in r["macro"].items()}, "| micro",
              {k: round(v, 4) for k, v in r["micro"].items()})
        d = r["non_explicit_minus_explicit"]
        print("   non-explicit - explicit (per-novel, paired):", round(d["mean_delta"], 4),
              [round(x, 4) for x in d["ci"]], "p", round(d["p"], 4),
              f"{d['novels_non_explicit_higher']}-{d['novels_explicit_higher']} novels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
