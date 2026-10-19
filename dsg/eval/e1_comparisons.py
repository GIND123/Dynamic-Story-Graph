"""Paired comparisons for one saved E1 run, with Holm correction.

The family is the six contrasts ``dsg.eval.attribution_report.comparisons``
defines, plus ``state-causal`` vs ``state-shuffled``: the placebo contrast
METHODOLOGY_V2 §2.4 names as the one that isolates the state's content, which
the report never computed. Only pairs whose arms both exist in the run are in
the family. Per-novel accuracies are rescored from the stored replies with the
current parser and asserted equal to the run's own scored file.

    python -m dsg.eval.e1_comparisons \\
        --raw ../Dynamic-Story-Graph/artifacts/e1/e1-qwen7b-v3-raw.json \\
        --scored ../Dynamic-Story-Graph/artifacts/e1/e1-qwen7b-v3.json \\
        --out audit/e1/v3_comparisons.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsg.eval.attribution_score import score_raw
from dsg.eval.multiplicity import ALPHA, N_BOOT, SEED, _draws, _interval, bootstrap_p, holm
from dsg.eval.provenance import record as provenance

FAMILY = (
    ("state-causal", "text-causal"),
    ("state-retrieval", "text-causal"),
    ("state-retrieval", "state-causal"),
    ("text-causal", "recency"),
    ("recency", "prior"),
    ("oracle-noncausal", "state-causal"),
    ("state-causal", "state-shuffled"),
)


def compare(per_novel: dict, metric: str = "accuracy") -> list[dict]:
    novels = sorted(per_novel)
    rows = []
    for a, b in FAMILY:
        if not all(a in per_novel[n] and b in per_novel[n] for n in novels):
            continue
        xs = [per_novel[n][a][metric] for n in novels]
        ys = [per_novel[n][b][metric] for n in novels]
        delta, means = _draws(xs, ys)
        rows.append({
            "comparison": f"{a} - {b}", "n_novels": len(novels),
            "mean_a": sum(xs) / len(xs), "mean_b": sum(ys) / len(ys),
            "mean_delta": float(delta.mean()), "ci_raw": list(_interval(means, ALPHA)),
            "p_raw": bootstrap_p(means), "wins_a": int((delta > 0).sum()),
            "wins_b": int((delta < 0).sum()), "_means": means,
        })
    adjusted = holm([r["p_raw"] for r in rows])
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    stopped = False
    for rank, i in enumerate(order):
        r, level = rows[i], ALPHA / (len(rows) - rank)
        r["p_holm"] = adjusted[i]
        r["ci_holm_level"] = list(_interval(r.pop("_means"), level))
        rejects = not stopped and r["p_raw"] <= level
        stopped = stopped or not rejects
        r["significant_after_correction"] = rejects
    return rows


def reply_identity(blob: dict, base: str = "text-causal") -> dict[str, float]:
    """Share of quotes where each arm's raw reply equals ``base``'s, verbatim.

    Under greedy decoding an arm whose prompt equals ``base``'s gives the same
    reply except for run-to-run nondeterminism, so a share near 1 means the arm
    did not get a different prompt.
    """
    same: dict[str, int] = {}
    total = 0
    for payload in blob["raw"].values():
        by: dict[str, dict[str, str]] = {}
        for (cond, qid, *_), reply in zip(payload["keys"], payload["replies"], strict=True):
            by.setdefault(qid, {})[cond] = reply
        for replies in by.values():
            total += 1
            for cond, reply in replies.items():
                if cond != base:
                    same[cond] = same.get(cond, 0) + (reply == replies[base])
    return {cond: n / total for cond, n in sorted(same.items())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.eval.e1_comparisons", description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--scored", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    blob = json.loads(args.raw.read_text())
    per_novel = score_raw(blob)["per_novel"]
    stored = json.loads(args.scored.read_text())["per_novel"]
    for n, conds in per_novel.items():
        for c, s in conds.items():
            assert abs(s["accuracy"] - stored[n][c]["accuracy"]) < 1e-12, (n, c)
    arms = {c: sum(per_novel[n][c]["accuracy"] for n in per_novel) / len(per_novel)
            for c in blob["conditions"]}
    out = {
        "provenance": provenance("dsg.eval.e1_comparisons", argv,
                                 {"bootstrap": SEED}, [args.raw, args.scored]),
        "run_id": blob.get("run_id"), "model": blob.get("model"),
        "window_chars": blob.get("window_chars"), "n_boot": N_BOOT,
        "arms_macro_accuracy": arms,
        "reply_identity_with_text_causal": reply_identity(blob),
        "comparisons": compare(per_novel),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for c, v in arms.items():
        print(f"{c:18} {v:.4f}")
    print("reply identity with text-causal:",
          {k: round(v, 4) for k, v in out["reply_identity_with_text_causal"].items()})
    for r in out["comparisons"]:
        print(f"{r['comparison']:36} {r['mean_delta']:+.4f} raw [{r['ci_raw'][0]:+.4f}, "
              f"{r['ci_raw'][1]:+.4f}] holm p {r['p_holm']:.4f} "
              f"{'SIG' if r['significant_after_correction'] else 'ns'} "
              f"wins {r['wins_a']}-{r['wins_b']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
