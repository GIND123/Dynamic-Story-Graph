"""What the logged ELABORATE operations actually are.

The process metrics count every ELABORATE as a monotone refinement, and the
revision profile's elaboration rate counts ELABORATE plus provisional merges.
But one ELABORATE path -- an existing assertion that is provisional with
confidence below 0.5 -- overwrites the old value in place, which is not
monotone in information. This replays cached proposals under ``dsg-full`` and
splits every ELABORATE by what produced it:

* ``entity:promotion`` -- a provisional node committed (``_maybe_promote``)
* ``entity:named`` -- an underspecified node received its first proper name
* ``entity:bind`` -- an identity link bound a description to a node (runner)
* ``assertion:<rule>`` -- an assertion conflict the classifier called
  ELABORATE, by rule (``elaborate:provisional`` is the overwrite)

    python -m dsg.eval.elaborate_census --out audit/calculus/elaborate_census.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dsg.eval.provenance import record as provenance
from dsg.experiments.run_study import load_book_proposals
from dsg.policies.runner import run_policy
from dsg.schemas import Op

RUNS = {
    "pdnc-qwen1.5b-w3200": "artifacts/hub/proposals/pdnc-qwen1.5b-w3200",
    "pdnc-qwen3b-w3200": "artifacts/hub/proposals/pdnc-qwen3b-w3200",
    "pdnc-qwen7b-w3200": "artifacts/hub/proposals/pdnc-qwen7b-w3200",
    "litbank-qwen7b-w1200": "artifacts/hub/proposals/litbank-qwen7b-w1200",
}
OVERWRITE = "assertion:elaborate:provisional"


def kind(record) -> str:
    if record.rule:
        return f"assertion:{record.rule}"
    if record.detail == "provisional -> committed":
        return "entity:promotion"
    if record.detail.startswith("underspecified node named"):
        return "entity:named"
    if record.detail.startswith("bound "):
        return "entity:bind"
    return "entity:other"


def census(run_dir: Path) -> dict:
    kinds: Counter = Counter()
    ops: Counter = Counter()
    for _, proposals in sorted(load_book_proposals(run_dir).items()):
        state = run_policy(proposals, "dsg-full", causal=True).state
        for r in state.log:
            ops[r.op.value] += 1
            if r.op is Op.ELABORATE:
                kinds[kind(r)] += 1
    elaborate = sum(kinds.values())
    assertion = sum(v for k, v in kinds.items() if k.startswith("assertion:"))
    monotone = sum(ops[o.value] for o in (Op.ASSERT, Op.ELABORATE, Op.MERGE_PROVISIONAL, Op.NOOP))
    profile_numerator = ops[Op.ELABORATE.value] + ops[Op.MERGE_PROVISIONAL.value]
    over = kinds[OVERWRITE]
    return {
        "elaborate_by_kind": dict(sorted(kinds.items())),
        "elaborate_total": elaborate,
        "entity_elaborate": elaborate - assertion,
        "assertion_elaborate": assertion,
        "overwrite": over,
        "overwrite_share_of_all_elaborate": over / elaborate if elaborate else 0.0,
        "overwrite_share_of_assertion_elaborate": over / assertion if assertion else 0.0,
        "overwrite_share_of_monotone_ops": over / monotone if monotone else 0.0,
        "overwrite_share_of_profile_elaboration": (
            over / profile_numerator if profile_numerator else 0.0),
        "ops": dict(sorted(ops.items())),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.elaborate_census", description=__doc__)
    p.add_argument("--out", default="audit/calculus/elaborate_census.json")
    args = p.parse_args(argv)
    inputs = [f for d in RUNS.values() for f in sorted(Path(d).glob("*.jsonl"))]
    out = {"provenance": provenance("dsg.eval.elaborate_census", argv, {}, inputs),
           "policy": "dsg-full", "runs": {n: census(Path(d)) for n, d in RUNS.items()}}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    for name, c in out["runs"].items():
        print(name, json.dumps({k: c[k] for k in c if k != "ops"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
