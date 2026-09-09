"""Project Dialogism Novel Corpus loader.

PDNC is the backbone of the study because it is self-contained and gives three
kinds of human annotation over the *same* full novel text: character alias
sets, quotation spans with gold speakers, and the referring expression that
introduces each quote. That lets identity and speaker attribution both be
scored against people, at book scale, with character offsets that make
prefix-causality checkable.

Cite: Vishnubhotla, Hammond & Hirst (2022), *The Project Dialogism Novel
Corpus: A Dataset for Character-Focused Narrative Analysis*, LREC.
"""

from __future__ import annotations

import ast
import csv
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_ROOT = Path("data/pdnc/data")
INDEX_PATH = Path("data/pdnc/PDNC-Novel-Index.csv")


@dataclass(slots=True)
class GoldCharacter:
    char_id: str
    name: str
    aliases: set[str]
    gender: str = ""
    category: str = ""


@dataclass(slots=True)
class GoldQuote:
    quote_id: str
    start: int
    end: int
    text: str
    speaker: str
    addressees: tuple[str, ...] = ()
    referring_expression: str = ""


@dataclass(slots=True)
class GoldMention:
    """A referring expression inside a quotation, linked to a character.

    PDNC annotates these with exact character offsets, which makes them usable
    as time-indexed probes: at the point in the book where this mention occurs,
    does the reader's state know who it refers to?
    """

    start: int
    end: int
    text: str
    entity: str


@dataclass(slots=True)
class Novel:
    book_id: str
    text: str
    characters: list[GoldCharacter] = field(default_factory=list)
    quotes: list[GoldQuote] = field(default_factory=list)
    mentions: list[GoldMention] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)

    def alias_to_character(self) -> dict[str, str]:
        """Aliases that identify exactly one character.

        Ambiguous strings are dropped rather than arbitrarily assigned: in this
        corpus 'Miss Bennet' belongs to more than one Bennet, and scoring a
        system on a label the annotation itself does not disambiguate would
        measure noise.
        """
        owners: dict[str, set[str]] = {}
        for c in self.characters:
            for a in c.aliases:
                key = a.strip().lower()
                if key:
                    owners.setdefault(key, set()).add(c.char_id)
        return {a: next(iter(o)) for a, o in owners.items() if len(o) == 1}

    def character_by_name(self) -> dict[str, GoldCharacter]:
        return {c.name: c for c in self.characters}


def _parse_aliases(raw: str) -> set[str]:
    raw = (raw or "").strip()
    if not raw:
        return set()
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {raw}
    if isinstance(value, set | list | tuple):
        return {str(v).strip() for v in value if str(v).strip()}
    return {str(value).strip()}


def _parse_spans(raw: str) -> list[tuple[int, int]]:
    try:
        value = ast.literal_eval((raw or "").strip() or "[]")
    except (ValueError, SyntaxError):
        return []
    out = []
    for pair in value if isinstance(value, list) else []:
        if isinstance(pair, list | tuple) and len(pair) == 2:
            out.append((int(pair[0]), int(pair[1])))
    return out


@lru_cache(maxsize=1)
def _index(path: str = str(INDEX_PATH)) -> dict[str, dict[str, str]]:
    """Per-novel bibliographic metadata: genre, narrative person, year.

    Narrative person is the interesting one. A first-person narrator bounds what
    the reader may know, so it is a natural predictor of how long a text keeps a
    figure unnamed -- which is exactly what this system can now measure.
    """
    if not Path(path).exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            folder = (row.get("Folder Name") or "").strip()
            if not folder:
                continue
            person = (row.get("Narrative Person") or "").strip()
            out[folder] = {
                "title": (row.get("Novel Title") or "").strip(),
                "genre": (row.get("Genre") or "").strip(),
                "person": {"1.0": "first", "3.0": "third"}.get(person, person),
                "year": (row.get("Year of First Publication") or "").strip(),
            }
    return out


def _parse_mentions(row: dict[str, str]) -> list[GoldMention]:
    """Flatten the per-sub-quote mention lists into (span, entity) pairs."""
    try:
        texts = ast.literal_eval((row.get("mentionTextsList") or "[]").strip() or "[]")
        spans = ast.literal_eval((row.get("mentionSpansList") or "[]").strip() or "[]")
        entities = ast.literal_eval((row.get("mentionEntitiesList") or "[]").strip() or "[]")
    except (ValueError, SyntaxError):
        return []
    out: list[GoldMention] = []
    for sub_t, sub_s, sub_e in zip(texts, spans, entities, strict=False):
        if not (isinstance(sub_t, list) and isinstance(sub_s, list) and isinstance(sub_e, list)):
            continue
        for text, span, entity in zip(sub_t, sub_s, sub_e, strict=False):
            if not (isinstance(span, list | tuple) and len(span) == 2):
                continue
            # An expression annotated with several referents is genuinely
            # ambiguous; scoring it would measure the annotation, not the state.
            names = entity if isinstance(entity, list) else [entity]
            names = [str(n).strip() for n in names if str(n).strip()]
            if len(names) != 1:
                continue
            out.append(
                GoldMention(
                    start=int(span[0]), end=int(span[1]),
                    text=str(text).strip(), entity=names[0],
                )
            )
    return out


def load_novel(folder: Path) -> Novel:
    folder = Path(folder)
    text = (folder / "novel_text.txt").read_text(encoding="utf-8")

    characters: list[GoldCharacter] = []
    char_path = folder / "character_info.csv"
    if char_path.exists():
        with char_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("Main Name") or "").strip()
                if not name:
                    continue
                aliases = _parse_aliases(row.get("Aliases", ""))
                aliases.add(name)
                characters.append(
                    GoldCharacter(
                        char_id=str(row.get("Character ID", name)).strip(),
                        name=name,
                        aliases=aliases,
                        gender=(row.get("Gender") or "").strip(),
                        category=(row.get("Category") or "").strip(),
                    )
                )

    quotes: list[GoldQuote] = []
    mentions: list[GoldMention] = []
    quote_path = folder / "quotation_info.csv"
    if quote_path.exists():
        with quote_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                spans = _parse_spans(row.get("quoteByteSpans", ""))
                if not spans:
                    continue
                start = min(s for s, _ in spans)
                end = max(e for _, e in spans)
                addressees = _parse_aliases(row.get("addressees", ""))
                mentions.extend(_parse_mentions(row))
                quotes.append(
                    GoldQuote(
                        quote_id=(row.get("quoteID") or "").strip(),
                        start=start,
                        end=end,
                        text=(row.get("quoteText") or "").strip(),
                        speaker=(row.get("speaker") or "").strip(),
                        addressees=tuple(sorted(addressees)),
                        referring_expression=(row.get("referringExpression") or "").strip(),
                    )
                )
    quotes.sort(key=lambda q: q.start)
    mentions.sort(key=lambda m: m.start)
    return Novel(
        book_id=folder.name, text=text, characters=characters, quotes=quotes,
        mentions=mentions, meta=dict(_index().get(folder.name, {})),
    )


@lru_cache(maxsize=1)
def _folders(root: str) -> tuple[Path, ...]:
    base = Path(root)
    return tuple(sorted(p for p in base.iterdir() if (p / "novel_text.txt").exists()))


def list_books(root: Path | str = DEFAULT_ROOT) -> list[str]:
    return [p.name for p in _folders(str(root))]


def load_corpus(
    limit: int | None = None,
    root: Path | str = DEFAULT_ROOT,
    only: list[str] | None = None,
) -> list[Novel]:
    """Load novels shortest-first, so a truncated run is still a valid sample."""
    folders = list(_folders(str(root)))
    if only:
        wanted = {o.lower() for o in only}
        folders = [f for f in folders if f.name.lower() in wanted]
    folders.sort(key=lambda p: (p / "novel_text.txt").stat().st_size)
    if limit:
        folders = folders[:limit]
    return [load_novel(f) for f in folders]
