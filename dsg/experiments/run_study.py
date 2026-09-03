"""Replay cached proposals through every policy and score all three planes.

Everything here is deterministic CPU work: the GPU pass has already happened
and its output is on disk, so the whole study can be re-scored after a change
to the calculus without spending another cent of compute.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dsg.data.pdnc import Novel
from dsg.data.registry import load as load_corpus
from dsg.eval import examples as ex
from dsg.eval import instrument
from dsg.eval.bootstrap import paired_bootstrap
from dsg.eval.identity import occurring_aliases, score_identity
from dsg.eval.mention import build_probes, score_mentions
from dsg.eval.process import growth_curve, revision_profile, score_process
from dsg.eval.speaker import score_speakers
from dsg.policies.runner import run_policy
from dsg.proposals import WindowProposal, load_proposals

DEFAULT_POLICIES = [
    "window-only", "append-only", "dsg-merge", "dsg-eager", "dsg-full", "retrospective"
]
PREFIX_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load_book_proposals(run_dir: Path) -> dict[str, list[WindowProposal]]:
    out: dict[str, list[WindowProposal]] = {}
    for path in sorted(Path(run_dir).glob("*.jsonl")):
        _, proposals = load_proposals(path)
        if proposals:
            out[path.stem] = proposals
    return out


def _prefix_curve(
    proposals: list[WindowProposal], novel: Novel, policy: str, causal: bool
) -> list[dict[str, float]]:
    """Identity quality as a function of how much of the book has been read.

    The gold item set is recomputed on each prefix, so a checkpoint is never
    penalised for characters the reader has not met yet.
    """
    curve = []
    n = len(proposals)
    for fraction in PREFIX_FRACTIONS:
        k = max(1, int(round(n * fraction)))
        head = proposals[:k]
        prefix_end = head[-1].end
        prefix_novel = Novel(
            book_id=novel.book_id,
            text=novel.text[:prefix_end],
            characters=novel.characters,
            quotes=[q for q in novel.quotes if q.end <= prefix_end],
        )
        items = occurring_aliases(prefix_novel)
        if not items:
            continue
        result = run_policy(head, policy, causal=causal, text_length=prefix_end)
        score = score_identity(result.state, prefix_novel, items=items)
        curve.append(
            {
                "fraction": fraction, "windows": k, "chars": prefix_end,
                "conll_f1": score.conll_f1, "b3_f1": score.b3_f1,
                "b3_f1_multi": score.b3_f1_multi, "ceaf_f1": score.ceaf_f1,
                "muc_f1": score.muc_f1,
                "coverage": score.coverage, "fragmentation": score.fragmentation,
                "conflation": score.conflation, "n_items": score.n_items,
                "violations": len(result.state.violations),
            }
        )
    return curve


def run_study(
    proposals_dir: Path,
    out_dir: Path,
    policies: list[str] | None = None,
    with_curves: bool = True,
    limit: int | None = None,
    corpus: str = "pdnc",
) -> dict:
    policies = policies or DEFAULT_POLICIES
    proposals_dir, out_dir = Path(proposals_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_book = load_book_proposals(proposals_dir)
    if limit:
        by_book = dict(list(by_book.items())[:limit])
    novels = {n.book_id: n for n in load_corpus(corpus) if n.book_id in by_book}
    missing = sorted(set(by_book) - set(novels))
    if missing:
        print(f"[warn] no gold annotation for: {missing}")

    started = time.time()
    records: list[dict] = []
    example_rows: list[dict] = []
    latency_results: dict[str, object] = {}
    curves: dict[str, dict[str, list]] = {}
    profiles: dict[str, dict[str, list]] = {}
    growth: dict[str, dict[str, list]] = {}

    for book_id, proposals in by_book.items():
        novel = novels.get(book_id)
        if novel is None:
            continue
        items = occurring_aliases(novel)
        probes = build_probes(novel)
        for policy in policies:
            causal = policy != "retrospective"
            result = run_policy(
                proposals, policy, causal=causal,
                text_length=len(novel.text), probes=probes,
            )
            identity = score_identity(result.state, novel, items=items)
            mentions = score_mentions(result, novel)
            speaker = score_speakers(result, novel)
            process = score_process(result)
            row = {
                "book": book_id, "policy": policy,
                "chars": len(novel.text), "windows": len(proposals),
                "gold_quotes": len(novel.quotes), "gold_characters": len(novel.characters),
                **identity.as_dict(), **mentions.as_dict(),
                **speaker.as_dict(), **process.as_dict(),
            }
            records.append(row)
            if policy == "dsg-full":
                example_rows.extend(
                    e.as_dict() for e in ex.collect(result, novel, limit=25)
                )
                latency_results[book_id] = result
            profiles.setdefault(book_id, {})[policy] = revision_profile(result)
            growth.setdefault(book_id, {})[policy] = growth_curve(result)
            if with_curves:
                curves.setdefault(book_id, {})[policy] = _prefix_curve(
                    proposals, novel, policy, causal
                )
            row["speaker_by_decile"] = speaker.by_decile
            row["mention_by_decile"] = mentions.by_decile
        print(f"[study] {book_id}: {len(policies)} policies done", flush=True)

    payload = {
        "meta": {
            "proposals_dir": str(proposals_dir),
            "corpus": corpus,
            "books": len(by_book),
            "policies": policies,
            "seconds": round(time.time() - started, 1),
        },
        "records": records,
        "curves": curves,
        "profiles": profiles,
        "growth": growth,
        "examples": example_rows,
        "instrument": instrument.analyse(latency_results, novels),
        "comparisons": compare(records, policies),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return payload


COMPARISON_METRICS = (
    "conll_f1", "mention_acc", "mention_acc_answered", "mention_coverage",
    "b3_f1_multi", "b3_f1", "ceaf_f1", "muc_f1", "fragmentation", "conflation", "coverage",
    "speaker_acc", "speaker_acc_matched", "violations_per_100w",
    "monotone_fraction", "rollback", "live_entities",
)

# Pairs that answer a specific question, rather than every combination.
COMPARISON_PAIRS = (
    ("dsg-full", "append-only"),      # does the calculus help at all?
    ("dsg-full", "window-only"),      # does persistent state help at all?
    ("dsg-full", "dsg-merge"),        # does fact-level revision add anything?
    ("dsg-full", "dsg-eager"),        # does *deferred commitment* add anything?
    ("dsg-full", "retrospective"),    # what does causality cost?
    ("dsg-eager", "append-only"),
    ("append-only", "window-only"),
)


def compare(records: list[dict], policies: list[str]) -> dict:
    by_policy: dict[str, dict[str, dict[str, float]]] = {}
    for row in records:
        by_policy.setdefault(row["policy"], {})[row["book"]] = row

    out: dict[str, dict] = {}
    for a, b in COMPARISON_PAIRS:
        if a not in by_policy or b not in by_policy:
            continue
        books = sorted(set(by_policy[a]) & set(by_policy[b]))
        if not books:
            continue
        entry: dict[str, dict] = {"n_books": len(books)}
        for metric in COMPARISON_METRICS:
            xs = [by_policy[a][bk].get(metric, 0.0) for bk in books]
            ys = [by_policy[b][bk].get(metric, 0.0) for bk in books]
            entry[metric] = paired_bootstrap(xs, ys).as_dict()
        out[f"{a}_vs_{b}"] = entry
    return out


def summarize(payload: dict) -> str:
    """A compact per-policy table, means over books."""
    records = payload["records"]
    policies = payload["meta"]["policies"]
    cols = ["conll_f1", "mention_acc", "mention_acc_answered", "b3_f1_multi", "coverage", "fragmentation", "conflation",
            "speaker_acc", "speaker_acc_matched", "violations_per_100w",
            "monotone_fraction", "rollback", "live_entities"]
    lines = ["| policy | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for policy in policies:
        rows = [r for r in records if r["policy"] == policy]
        if not rows:
            continue
        cells = []
        for c in cols:
            vals = [r.get(c, 0.0) for r in rows]
            mean = sum(vals) / len(vals)
            cells.append(f"{mean:.3f}" if abs(mean) < 100 else f"{mean:.1f}")
        lines.append(f"| {policy} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
