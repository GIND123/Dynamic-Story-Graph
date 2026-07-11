"""PDNC loader — Project Dialogism Novel Corpus (full novels).

Repo: github.com/Priya22/project-dialogism-novel-corpus
Per-novel folder under the corpus `data/` directory contains:
  - character_info.csv : canonical "Main Name" + "Aliases" + "Category"
  - quotation_info.csv  (newer)  OR  quotations.csv  (older) : one row per quote
    with quoteID, quoteText, quoteType {Explicit|Implicit|Anaphoric}, speaker,
    addressees.
  - novel_text.txt : full novel text.

NETWORK NOTE: the corpus could not be cloned in this environment (no network),
so column/file detection is done defensively and case-insensitively at runtime;
the loader adapts to whichever names are present and records the resolved
filename. See DECISIONS.md.

ENTITY RESOLUTION: character_info aliases are the canonical resolver. Every
speaker/addressee surface form is collapsed to its Main Name BEFORE the IR is
built, so one character == one entity == one node.
"""

from __future__ import annotations

import ast
import csv
import logging
from pathlib import Path

from ingestion.base import (
    DatasetIR,
    DatasetLoader,
    Entity,
    Quotation,
    chunk_text,
)

logger = logging.getLogger(__name__)

# Candidate filenames (version-dependent); first existing wins.
_QUOTE_FILE_CANDIDATES = ("quotation_info.csv", "quotations.csv")
_CHARACTER_FILE_CANDIDATES = ("character_info.csv", "characters.csv")
_TEXT_FILE_CANDIDATES = ("novel_text.txt", "text.txt", "novel.txt")

_VALID_QUOTE_TYPES = {"explicit", "implicit", "anaphoric"}


def _find_existing(folder: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def _find_column(fieldnames: list[str], *candidates: str) -> str | None:
    """Case-insensitive header lookup; returns the actual header or None."""
    lowered = {fn.lower().strip(): fn for fn in fieldnames}
    for cand in candidates:
        hit = lowered.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _parse_list_cell(cell: str | None) -> list[str]:
    """Parse a collection-ish CSV cell into clean string items.

    The real PDNC data is mixed: the character-info Aliases column uses Python
    *set* literals ("{'Mrs. Collins', 'Miss Lucas'}") for some rows and *list*
    literals ("['Colonel Forster']") for others, while quotation addressees use
    list literals. Handles set/list/tuple literals, plus semicolon-/comma-
    separated values and plain single values. Set items are sorted so output is
    deterministic (Python set ordering is not). Empty -> [].
    """
    if not cell:
        return []
    text = cell.strip()
    if not text:
        return []
    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    ):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                items = [str(x).strip() for x in parsed if str(x).strip()]
                return sorted(items) if isinstance(parsed, set) else items
        except (ValueError, SyntaxError):
            pass
    separator = ";" if ";" in text else ","
    return [part.strip().strip("'\"") for part in text.split(separator) if part.strip()]


def _normalize_quote_type(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in _VALID_QUOTE_TYPES:
        return value
    logger.warning("Unknown PDNC quoteType %r; defaulting to 'explicit'", raw)
    return "explicit"


class PDNCLoader(DatasetLoader):
    """Load one PDNC novel folder into the normalized IR."""

    def __init__(self, novel_dir: str | Path) -> None:
        self.novel_dir = Path(novel_dir)
        if not self.novel_dir.is_dir():
            raise FileNotFoundError(f"PDNC novel folder not found: {self.novel_dir}")
        self.folder_name = self.novel_dir.name

    def namespace(self) -> str:
        return f"pdnc:{self.folder_name}"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _load_characters(self) -> tuple[list[Entity], dict[str, str]]:
        """Return (character entities, alias->canonical map).

        The alias map includes each canonical name mapped to itself so lookups
        are total. Matching is case-insensitive on the alias key.
        """
        path = _find_existing(self.novel_dir, _CHARACTER_FILE_CANDIDATES)
        if path is None:
            raise FileNotFoundError(
                f"No character info file in {self.novel_dir} "
                f"(looked for {_CHARACTER_FILE_CANDIDATES})"
            )
        entities: list[Entity] = []
        alias_map: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            name_col = _find_column(fields, "Main Name", "name", "character")
            alias_col = _find_column(fields, "Aliases", "alias", "aliases")
            for row in reader:
                main = (row.get(name_col) or "").strip() if name_col else ""
                if not main:
                    continue
                # Real PDNC alias sets include the Main Name itself; drop it so we
                # don't create an Alias node identical to the canonical character.
                aliases = [
                    a
                    for a in _parse_list_cell(row.get(alias_col) if alias_col else None)
                    if a != main
                ]
                entities.append(Entity(name=main, type="CHARACTER", aliases=aliases))
                alias_map[main.lower()] = main
                for alias in aliases:
                    alias_map[alias.lower()] = main
        logger.info("PDNC: loaded %d characters from %s", len(entities), path.name)
        return entities, alias_map

    def _resolve(self, surface: str, alias_map: dict[str, str]) -> str:
        """Collapse a surface form to its canonical Main Name (else itself)."""
        return alias_map.get(surface.strip().lower(), surface.strip())

    def _load_quotations(self, alias_map: dict[str, str]) -> list[Quotation]:
        path = _find_existing(self.novel_dir, _QUOTE_FILE_CANDIDATES)
        if path is None:
            raise FileNotFoundError(
                f"No quotation file in {self.novel_dir} "
                f"(looked for {_QUOTE_FILE_CANDIDATES})"
            )
        logger.info("PDNC: using quotation file %s", path.name)
        quotations: list[Quotation] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            id_col = _find_column(fields, "quoteID", "quote_id", "id")
            text_col = _find_column(fields, "quoteText", "quote_text", "text")
            type_col = _find_column(fields, "quoteType", "quote_type", "type")
            speaker_col = _find_column(fields, "speaker")
            addr_col = _find_column(fields, "addressees", "addressee")
            for i, row in enumerate(reader):
                speaker_raw = (
                    (row.get(speaker_col) or "").strip() if speaker_col else ""
                )
                if not speaker_raw:
                    continue
                addressees = [
                    self._resolve(a, alias_map)
                    for a in _parse_list_cell(row.get(addr_col) if addr_col else None)
                ]
                ref = (row.get(id_col) or f"q{i}").strip() if id_col else f"q{i}"
                quotations.append(
                    Quotation(
                        speaker_name=self._resolve(speaker_raw, alias_map),
                        addressee_names=addressees,
                        quote_type=_normalize_quote_type(
                            row.get(type_col) if type_col else None
                        ),
                        text=(row.get(text_col) or "").strip() if text_col else "",
                        ref=ref,
                    )
                )
        logger.info("PDNC: loaded %d quotations", len(quotations))
        return quotations

    def _load_segments(self) -> list:
        path = _find_existing(self.novel_dir, _TEXT_FILE_CANDIDATES)
        if path is None:
            logger.warning("PDNC: no novel text file in %s; 0 segments", self.novel_dir)
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        return chunk_text(text)

    # ------------------------------------------------------------------

    def load(self) -> DatasetIR:
        entities, alias_map = self._load_characters()
        quotations = self._load_quotations(alias_map)
        segments = self._load_segments()
        # Events are modeled from quotations as dialogue acts in project.py;
        # PDNC has no separate event annotation layer in scope.
        return DatasetIR(
            entities=entities,
            events=[],
            quotations=quotations,
            segments=segments,
        )
