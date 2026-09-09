"""Score a raw Modal E1 run into the per-novel shape the reporter expects.

Kept separate from generation so scoring is pure CPU work over a stored blob:
a run can be rescored after a change to normalisation or alias handling without
touching the GPU, which is the same separation the reading study uses.

    python -m dsg.eval.attribution_score --raw artifacts/e1/e1-qwen7b-w1200-raw.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dsg.eval.attribution import score_attribution
from dsg.eval.attribution_run import parse_answer


def score_raw(blob: dict) -> dict:
    """Group replies by (novel, condition) and score each group."""
    per_novel: dict[str, dict[str, dict]] = {}
    for novel, payload in blob["raw"].items():
        meta = blob["meta"][novel]
        candidates = meta["candidates"]
        aliases = {k: set(v) for k, v in meta["aliases"].items()}

        grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for (cond, _qid, gold, qtype), reply in zip(
            payload["keys"], payload["replies"], strict=True
        ):
            grouped[cond].append((reply, gold, qtype))

        per_novel[novel] = {}
        for cond, rows in grouped.items():
            preds = [parse_answer(r, candidates) for r, _g, _t in rows]
            golds = [g for _r, g, _t in rows]
            types = [t for _r, _g, t in rows]
            per_novel[novel][cond] = score_attribution(
                preds, golds, types, alias_sets=aliases
            ).as_dict()

    return {
        "backend": blob.get("model", "?"),
        "window_chars": blob.get("window_chars"),
        "conditions": blob.get("conditions", []),
        "seconds": blob.get("seconds"),
        "repeated_text_diagnostics": blob.get("repeated_text_diagnostics", 0),
        "per_novel": per_novel,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    blob = json.loads(args.raw.read_text())
    scored = score_raw(blob)
    dest = args.out or args.raw.with_name(args.raw.name.replace("-raw", ""))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(scored, indent=2))

    for novel in sorted(scored["per_novel"]):
        for cond, s in sorted(scored["per_novel"][novel].items()):
            print(
                f"  {novel[:26]:26} {cond:18} n={int(s['n']):5} "
                f"acc={s['accuracy']:.3f} non-exp={s['accuracy_non_explicit']:.3f}"
            )
    print(f"\nwrote {dest}")
    print(f"report with: python -m dsg.eval.attribution_report --result {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
