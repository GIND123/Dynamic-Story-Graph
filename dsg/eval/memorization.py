"""G3 -- a memorization covariate for E1.

Qwen has read these public-domain novels. If attribution accuracy tracks how
well the model has memorised a book, then E1 is partly measuring recall of the
training corpus rather than inference from the causal prefix. Michel et al.
(2024) ran exactly this check before claiming Llama-3 was the new state of the
art on PDNC, and found memorization did not explain their gain; E1 is not
entitled to skip it.

Name cloze, after Chang et al. (2023) and the GPT4-Books protocol: take a
passage containing exactly one occurrence of one character's name, mask it, and
ask the model to restore it. A model that has memorised the book can do this; a
model reasoning from context alone mostly cannot, because the masked name is
rarely recoverable from local evidence.

Built over PDNC directly rather than borrowing GPT4-Books' released numbers.
Those are ChatGPT's, and the covariate has to describe *our* model on *our*
novels, which also avoids depending on a fuzzy title match between corpora --
only 16 of 28 PDNC novels align, and some of those alignments are wrong
("Emma" fuzzy-matches a different book entirely).

    python -m dsg.eval.memorization --novels 3 --per-novel 40 --backend mlx
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dsg.eval.attribution import normalise_name
from dsg.eval.causal_audit import DEFAULT_ROOT

MASK = "[MASK]"
_PAD = 450  # characters of context on each side


@dataclass(slots=True)
class ClozeItem:
    novel: str
    name: str
    passage: str  # already masked


def _proper(name: str) -> bool:
    """Single- or multi-word capitalised name, no honorific-only forms."""
    if len(name) < 4 or not name[0].isupper():
        return False
    return bool(re.fullmatch(r"[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)*", name))


def build_items(
    text: str, novel: str, names: list[str], *, n: int, rng: random.Random
) -> list[ClozeItem]:
    """Sample passages where exactly one name occurs, so the answer is unambiguous."""
    usable = [n_ for n_ in names if _proper(n_)]
    items: list[ClozeItem] = []
    seen: set[int] = set()
    # Candidate sites: every occurrence of every usable name.
    sites: list[tuple[int, str]] = []
    for name in usable:
        for m in re.finditer(re.escape(name), text):
            sites.append((m.start(), name))
    rng.shuffle(sites)

    for pos, name in sites:
        if len(items) >= n:
            break
        if any(abs(pos - s) < _PAD for s in seen):
            continue  # keep passages from overlapping
        lo, hi = max(0, pos - _PAD), min(len(text), pos + len(name) + _PAD)
        window = text[lo:hi]
        # Exactly one occurrence of the target inside the window, and no other
        # character's name that could make the answer ambiguous.
        if window.count(name) != 1:
            continue
        masked = window.replace(name, MASK)
        items.append(ClozeItem(novel=novel, name=name, passage=masked))
        seen.add(pos)
    return items


def build_prompt(item: ClozeItem) -> str:
    return (
        "The following passage is from a novel. One character's name has been "
        f"replaced with {MASK}.\n\n"
        f"{item.passage}\n\n"
        f"Reply with the single name that belongs in {MASK}, and nothing else.\n"
        "Name:"
    )


def score(replies: list[str], items: list[ClozeItem]) -> dict:
    per_novel: dict[str, list[int]] = {}
    for reply, item in zip(replies, items, strict=True):
        first = (reply or "").strip().splitlines()[0] if (reply or "").strip() else ""
        hit = int(normalise_name(first) == normalise_name(item.name))
        per_novel.setdefault(item.novel, []).append(hit)
    return {
        novel: {"n": len(v), "cloze_accuracy": sum(v) / len(v) if v else 0.0}
        for novel, v in per_novel.items()
    }


def load_texts(root: Path = DEFAULT_ROOT, limit: int | None = None) -> dict[str, tuple[str, list[str]]]:
    """novel -> (text, character names), reusing PDNC's own annotation."""
    from dsg.eval.causal_audit import alias_sets

    out: dict[str, tuple[str, list[str]]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        tp = folder / "novel_text.txt"
        if not tp.exists():
            continue
        out[folder.name] = (tp.read_text(encoding="utf-8"), sorted(alias_sets(folder)))
        if limit and len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--novels", type=int, default=0)
    ap.add_argument("--per-novel", type=int, default=50)
    ap.add_argument("--backend", choices=["mock", "mlx"], default="mock")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from dsg.eval.attribution_run import get_backend

    rng = random.Random(args.seed)
    texts = load_texts(args.root, limit=args.novels or None)
    items: list[ClozeItem] = []
    for novel, (text, names) in texts.items():
        items += build_items(text, novel, names, n=args.per_novel, rng=rng)
    print(f"name cloze: {len(items):,} items over {len(texts)} novels, backend={args.backend}")

    backend = get_backend(args.backend, args.model)
    replies = [backend.generate(build_prompt(i), max_tokens=12, temperature=0.0) for i in items]
    result = score(replies, items)

    for novel in sorted(result):
        r = result[novel]
        print(f"  {novel[:30]:30} n={r['n']:4} cloze={r['cloze_accuracy']:.3f}")

    blob = {"model": args.model if args.backend != "mock" else "mock", "per_novel": result}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(blob, indent=2))
        print(f"\nwrote {args.out}")
    else:
        json.dump(blob, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
