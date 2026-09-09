"""E1 runner -- build causal attribution prompts, score them, never leak.

Conditions differ only in what evidence reaches the prompt; the candidate set,
the question, and the scoring are identical everywhere, so a difference between
conditions is attributable to the evidence and nothing else.

    python -m dsg.eval.attribution_run --novels 2 --limit 100 --backend mock
    python -m dsg.eval.attribution_run --novels 3 --limit 300 \
        --backend mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit

``oracle-noncausal`` is the only condition permitted to read past the quote; it
reproduces the published protocol so causality can be priced on our own system
rather than inferred across papers.

Every other condition is guarded by ``CausalContext.verify_evidence``, a
whitelist: each evidence block must be provably a slice of ``text[:start]`` or
of the spoken words. That is the sound check. The blacklist
(``CausalContext.verify``) is kept only as a diagnostic, because novels repeat
themselves -- a line of dialogue spoken twice appears both before and after the
quote, and a model reading the *earlier* occurrence has not seen the future. Such
hits are counted and reported, not treated as failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dsg.eval.attribution import (
    CausalContext,
    LeakageError,
    normalise_name,
    quoted_only,
    score_attribution,
)
from dsg.eval.attribution_state import (
    DEFAULT_PROPOSALS,
    EMPTY,
    Snapshot,
    build_snapshots,
    placebo_for,
    snapshot_for,
)
from dsg.eval.causal_audit import DEFAULT_ROOT, alias_sets, quote_spans

CAUSAL_CONDITIONS = (
    "prior",
    "recency",
    "text-causal",
    "state-causal",
    "state-shuffled",
    "state-retrieval",
)
ALL_CONDITIONS = (*CAUSAL_CONDITIONS, "oracle-noncausal")


@dataclass(slots=True)
class Quote:
    novel: str
    quote_id: str
    start: int
    end: int
    text: str
    speaker: str
    quote_type: str
    spans: tuple[tuple[int, int], ...] = ()


@dataclass(slots=True)
class NovelData:
    name: str
    text: str
    quotes: list[Quote]
    aliases: dict[str, set[str]]
    # Candidate list = the novel's annotated cast. This matches the published
    # PDNC protocol (gold candidates), and is reported as such: it is regime G2a.
    candidates: list[str] = field(default_factory=list)


def load_novels(root: Path = DEFAULT_ROOT, limit: int | None = None) -> list[NovelData]:
    out: list[NovelData] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        tp, qp = folder / "novel_text.txt", folder / "quotation_info.csv"
        if not (tp.exists() and qp.exists()):
            continue
        text = tp.read_text(encoding="utf-8")
        aliases = alias_sets(folder)
        quotes: list[Quote] = []
        with qp.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                spans = quote_spans(row)
                speaker = (row.get("speaker") or "").strip()
                if not spans or speaker not in aliases:
                    continue
                start, end = spans[0][0], spans[-1][1]
                quotes.append(
                    Quote(
                        novel=folder.name,
                        quote_id=(row.get("quoteID") or f"{start}").strip(),
                        start=start,
                        end=end,
                        text=quoted_only(text, spans),
                        speaker=speaker,
                        quote_type=(row.get("quoteType") or "unlabelled").strip(),
                        spans=spans,
                    )
                )
        if quotes:
            out.append(
                NovelData(folder.name, text, quotes, aliases, sorted(aliases))
            )
        if limit and len(out) >= limit:
            break
    return out


def _candidate_block(candidates: list[str]) -> str:
    return "\n".join(f"- {c}" for c in candidates)


def build_prompt(
    novel: NovelData,
    quote: Quote,
    condition: str,
    *,
    window: int,
    recent: str | None,
    snapshot: Snapshot = EMPTY,
    placebo: Snapshot = EMPTY,
) -> str:
    """Assemble one prompt. Causal conditions are verified before returning."""
    ctx = CausalContext(novel.text, quote.start, quote.end, quote.spans)
    cands = _candidate_block(novel.candidates)
    head = (
        "You are reading a novel and must identify who speaks a line of dialogue.\n"
        f"Characters in this novel:\n{cands}\n\n"
    )
    def ask_for(where: str) -> str:
        return (
            f'\nThe line of dialogue {where}:\n"{quote.text.strip()}"\n\n'
            "Reply with the speaker's name exactly as it appears in the character "
            "list above, and nothing else.\nSpeaker:"
        )

    # Causal conditions end exactly at the quote, so "comes next" is literally
    # true. The oracle's context brackets the quote, so the same wording would
    # be false there and the model would have to guess which line is meant --
    # a prompt artefact that would depress the oracle and could manufacture a
    # "causality is free" result. Each condition gets the wording that matches
    # what it was actually shown.
    ask = ask_for("that comes next")

    if condition == "prior":
        return head + ask_for("below")
    if condition == "recency":
        hint = f"\nThe most recent speaker before this line was: {recent or 'unknown'}.\n"
        return head + hint + ask
    label = f"{novel.name}:{quote.quote_id}"
    if condition in (
        "text-causal", "state-causal", "state-shuffled", "state-retrieval"
    ):
        evidence = ctx.tail(window)
        # Sound guard: prove the evidence came from a sanctioned slice.
        ctx.verify_evidence(evidence, label=label)
        body = f"Text leading up to the line:\n...{evidence}"
        if condition == "state-retrieval":
            # Query-conditioned: surface only the state lines the recent text
            # implicates, instead of dumping the store. Same provenance rule.
            snapshot.assert_causal(quote.start, label=label)
            rendered = snapshot.render_retrieved(evidence)
            if rendered:
                body += f"\n\nThe reader's notes, filtered to this scene:\n{rendered}\n"
        if condition == "state-shuffled":
            # Length- and format-matched placebo: a real state block, but one
            # built at a different point in the same book. If state-causal ties
            # this, the effect of "adding state" is prompt length and shape, not
            # what the state says. Same control logic as the blind best-of-4 arm
            # in the generation study, and the shuffled arm in Stage C.
            placebo.assert_causal(quote.start, label=label)
            rendered = placebo.render()
            if rendered:
                body += f"\n\nThe reader's notes at this point:\n{rendered}\n"
        if condition == "state-causal":
            # The state block is DERIVED text, not a slice of the novel, so the
            # substring whitelist cannot vouch for it. Its causality rests on
            # provenance instead: the snapshot must not reach past the quote.
            snapshot.assert_causal(quote.start, label=label)
            rendered = snapshot.render()
            if rendered:
                body += f"\n\nThe reader's notes at this point:\n{rendered}\n"
        return head + body + ask
    if condition == "oracle-noncausal":
        # The published protocol: a window centred on the quote. Deliberately NOT
        # verified -- reading forward is exactly what this condition measures.
        # The target is marked inline, because the context brackets it and the
        # model would otherwise have to infer which line is being asked about.
        lo = max(0, quote.start - window)
        hi = min(len(novel.text), quote.end + window)
        ctx = novel.text[lo:hi]
        first = quote.spans[0] if quote.spans else (quote.start, quote.end)
        seg = novel.text[first[0]:first[1]]
        if seg and seg in ctx:
            ctx = ctx.replace(seg, f">>>{seg}<<<", 1)
        return head + f"Surrounding text (target marked >>>...<<<):\n...{ctx}..." + ask_for(
            "marked >>>...<<< above"
        )
    raise ValueError(f"unknown condition: {condition}")


def parse_answer(raw: str, candidates: list[str]) -> str | None:
    """Map a model's reply onto the candidate list; refuse to guess wildly."""
    reply = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
    norm = normalise_name(reply)
    if not norm:
        return None
    by_norm = {normalise_name(c): c for c in candidates}
    if norm in by_norm:
        return by_norm[norm]
    # Longest candidate contained in the reply, so "Speaker: Mr. Darcy" resolves
    # without letting a short name match inside a longer one.
    hits = [c for c in candidates if normalise_name(c) in norm]
    return max(hits, key=len) if hits else reply


class MockBackend:
    """Deterministic stand-in so the pipeline is testable with no weights."""

    name = "mock"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def generate(self, prompt: str, max_tokens: int = 16, temperature: float = 0.0) -> str:
        block = [ln[2:] for ln in prompt.splitlines() if ln.startswith("- ")]
        return self._rng.choice(block) if block else ""


def get_backend(kind: str, model: str):
    if kind == "mock":
        return MockBackend()
    if kind == "mlx":
        from dsg.llm.mlx_backend import MLXBackend

        return MLXBackend(model)
    raise ValueError(f"unknown backend: {kind}")


def run(
    novels: list[NovelData],
    conditions: tuple[str, ...],
    backend,
    *,
    window: int,
    limit_per_novel: int | None,
    seed: int,
    proposals: Path | None = None,
) -> dict:
    rng = random.Random(seed)
    per_novel: dict[str, dict[str, dict]] = {}
    # Blacklist hits are repeated-text diagnostics, not failures (see module docstring).
    repeats = 0
    t0 = time.time()

    for nd in novels:
        quotes = list(nd.quotes)
        if limit_per_novel and len(quotes) > limit_per_novel:
            quotes = sorted(rng.sample(quotes, limit_per_novel), key=lambda q: q.start)
        # "Most recent speaker" is derived only from quotes strictly earlier in
        # the book, so it is causal by construction.
        ordered = sorted(nd.quotes, key=lambda q: q.start)
        prev_by_start = {}
        last = None
        for q in ordered:
            prev_by_start[q.start] = last
            last = q.speaker

        snaps: list[Snapshot] = []
        if {"state-causal", "state-shuffled", "state-retrieval"} & set(conditions):
            path = (proposals or DEFAULT_PROPOSALS) / f"{nd.name}.jsonl"
            if path.exists():
                snaps = build_snapshots(path)
            else:
                print(f"  !! no cached proposals for {nd.name}; state-causal = empty")

        per_novel[nd.name] = {}
        for cond in conditions:
            preds, golds, types = [], [], []
            for q in quotes:
                prompt = build_prompt(
                    nd, q, cond,
                    window=window,
                    recent=prev_by_start.get(q.start),
                    snapshot=snapshot_for(snaps, q.start) if snaps else EMPTY,
                    placebo=placebo_for(snaps, q.start) if snaps else EMPTY,
                )
                if cond in (
                    "text-causal", "state-causal", "state-shuffled", "state-retrieval"
                ):
                    try:
                        CausalContext(
                            nd.text, q.start, q.end, q.spans
                        ).verify(prompt, label=f"{nd.name}:{q.quote_id}")
                    except LeakageError:
                        repeats += 1
                reply = backend.generate(prompt, max_tokens=16, temperature=0.0)
                preds.append(parse_answer(reply, nd.candidates))
                golds.append(q.speaker)
                types.append(q.quote_type)
            score = score_attribution(preds, golds, types, alias_sets=nd.aliases)
            per_novel[nd.name][cond] = score.as_dict()
            print(
                f"  {nd.name[:26]:26} {cond:18} "
                f"n={score.n:4} acc={score.accuracy:.3f} "
                f"non-exp={score.accuracy_non_explicit:.3f}",
                flush=True,
            )

    return {
        "backend": getattr(backend, "name", "?"),
        "window_chars": window,
        "conditions": list(conditions),
        "seed": seed,
        "repeated_text_diagnostics": repeats,
        "seconds": round(time.time() - t0, 1),
        "per_novel": per_novel,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--novels", type=int, default=2, help="how many novels")
    ap.add_argument("--limit", type=int, default=100, help="quotes per novel")
    ap.add_argument("--window", type=int, default=1200, help="context chars")
    ap.add_argument("--backend", choices=["mock", "mlx"], default="mock")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS,
                    help="cached proposal stream that state-causal replays")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    conds = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    novels = load_novels(args.root, limit=args.novels)
    if not novels:
        raise SystemExit(f"no PDNC novels under {args.root}")
    print(
        f"E1 pilot: {len(novels)} novel(s), <={args.limit} quotes each, "
        f"{len(conds)} conditions, backend={args.backend}"
    )
    result = run(
        novels, conds, get_backend(args.backend, args.model),
        window=args.window, limit_per_novel=args.limit, seed=args.seed,
        proposals=args.proposals,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    else:
        json.dump(result, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
