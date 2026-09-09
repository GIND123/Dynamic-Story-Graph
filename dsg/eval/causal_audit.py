"""What a prefix-causal cut actually costs on PDNC, measured rather than assumed.

Published quote-attribution numbers are non-causal: the system reads a window
centred on the quote, so it sees the trailing speech tag. This module measures
how much of PDNC's attribution signal lives *after* each quote, which is exactly
what a prefix-causal protocol gives up.

Reads PDNC's raw CSVs directly rather than going through ``dsg.data.pdnc``,
because the DSG loader drops ``quoteType`` and the whole point here is the split
by type.

    python -m dsg.eval.causal_audit --report tag-position
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("data/pdnc/data")
DEFAULT_WINDOW = 250
_MIN_ALIAS_LEN = 3


def _literal(raw: str | None, fallback):
    """PDNC stores lists as Python literals in CSV cells."""
    if not raw:
        return fallback
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return fallback


def alias_sets(folder: Path) -> dict[str, set[str]]:
    """Main Name -> surface forms, keeping only forms long enough to match safely.

    Very short aliases ("I", "he") would match everywhere and turn the audit into
    noise, so they are dropped rather than counted as evidence.
    """
    out: dict[str, set[str]] = {}
    path = folder / "character_info.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Main Name") or "").strip()
            if not name:
                continue
            forms = {name} | {
                str(a).strip() for a in _literal(row.get("Aliases"), []) if str(a).strip()
            }
            keep = {f for f in forms if len(f) >= _MIN_ALIAS_LEN}
            if keep:
                out[name] = keep
    return out


def quote_spans(row: dict[str, str]) -> tuple[tuple[int, int], ...]:
    """Every quoted segment, in order.

    28.9% of PDNC quotes are split across two or more segments, and the
    narration between them is where the speech tag lives. Callers that collapse
    this to (first_start, last_end) hand that tag to the model.
    """
    raw = _literal(row.get("quoteByteSpans") or row.get("quoteSpans"), [])
    out: list[tuple[int, int]] = []
    for span in raw:
        try:
            s, e = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            return ()
        if e > s:
            out.append((s, e))
    # PDNC does not guarantee document order: 5 of 37,131 quotes list a later
    # segment first (e.g. HardTimes Q224 = [[83859, 83964], [83789, 83837]]).
    # Callers take spans[0][0] as the quote's start and spans[-1][1] as its end,
    # which inverts the range for those, so sort before returning.
    return tuple(sorted(out))


def quote_span(row: dict[str, str]) -> tuple[int, int] | None:
    """Outer bounds of a quote. For span-accurate work use ``quote_spans``."""
    spans = quote_spans(row)
    return (spans[0][0], spans[-1][1]) if spans else None


@dataclass(slots=True)
class TagPosition:
    """Where the gold speaker's name sits relative to each explicit quote."""

    window: int
    analysed: int = 0
    skipped: int = 0
    counts: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = Counter()

    def add(self, before: bool, after: bool) -> None:
        self.analysed += 1
        self.counts["both" if (before and after) else
                    "before" if before else
                    "after" if after else "neither"] += 1

    @property
    def lost_to_causal_cut(self) -> int:
        """Quotes whose only in-window name cue follows the quote."""
        return self.counts["after"]

    def as_dict(self) -> dict[str, object]:
        n = self.analysed or 1
        return {
            "window_chars": self.window,
            "explicit_quotes_analysed": self.analysed,
            "skipped_no_alias_or_span": self.skipped,
            "counts": dict(self.counts),
            "shares": {k: v / n for k, v in self.counts.items()},
            "lost_to_causal_cut": self.lost_to_causal_cut,
            "lost_to_causal_cut_share": self.lost_to_causal_cut / n,
        }


def audit_tag_position(root: Path = DEFAULT_ROOT, window: int = DEFAULT_WINDOW) -> TagPosition:
    result = TagPosition(window=window)
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit(f"no PDNC novels under {root} -- clone the corpus first")
    for folder in folders:
        text_path = folder / "novel_text.txt"
        quotes_path = folder / "quotation_info.csv"
        if not (text_path.exists() and quotes_path.exists()):
            continue
        text = text_path.read_text(encoding="utf-8")
        aliases = alias_sets(folder)
        with quotes_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("quoteType") or "").strip() != "Explicit":
                    continue
                forms = aliases.get((row.get("speaker") or "").strip())
                span = quote_span(row)
                if not forms or span is None:
                    result.skipped += 1
                    continue
                start, end = span
                before = text[max(0, start - window) : start]
                after = text[end : end + window]
                result.add(
                    any(f in before for f in forms),
                    any(f in after for f in forms),
                )
    return result


def quote_type_counts(root: Path = DEFAULT_ROOT) -> Counter:
    counts: Counter = Counter()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        path = folder / "quotation_info.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                raw = (row.get("quoteType") or "").strip()
                counts[raw if raw and raw.lower() != "nan" else "unlabelled"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", choices=["tag-position", "quote-types", "all"], default="all")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args(argv)

    payload: dict[str, object] = {}

    if args.report in ("quote-types", "all"):
        counts = quote_type_counts(args.root)
        total = sum(counts.values())
        payload["quote_types"] = {"total": total, "counts": dict(counts)}
        if not args.json:
            print(f"PDNC quote types  (total {total:,})")
            for k, v in counts.most_common():
                print(f"  {k:14} {v:7,}  {v / total:6.1%}")
            print()

    if args.report in ("tag-position", "all"):
        tp = audit_tag_position(args.root, args.window)
        payload["tag_position"] = tp.as_dict()
        if not args.json:
            label = {
                "neither": "neither side",
                "after": "AFTER only  (lost to causal cut)",
                "both": "both sides",
                "before": "BEFORE only",
            }
            print(
                f"Explicit quotes analysed: {tp.analysed:,} "
                f"(window +/-{tp.window} chars, skipped {tp.skipped:,})"
            )
            for k, v in tp.counts.most_common():
                print(f"  {label.get(k, k):34} {v:7,}  {v / tp.analysed:6.1%}")
            print(
                f"\n-> a strict prefix-causal cut removes the in-window name cue for "
                f"{tp.lost_to_causal_cut:,} "
                f"({tp.lost_to_causal_cut / tp.analysed:.1%}) of explicit quotes"
            )

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
