"""Read PDNC character metadata (aliases + Category) from character_info.csv.

The graph does NOT store PDNC Category or aliases on Character nodes (verified:
Character nodes carry only uid/name/manuscript_id/source). Category is the
ground-truth importance label and aliases are the identity resolver, so both are
read from the source dataset file. Reuses the verified ingestion.pdnc parsers so
this stays consistent with how the data was ingested (and is not re-implemented).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.pdnc import (
    _CHARACTER_FILE_CANDIDATES,
    _find_column,
    _find_existing,
    _parse_list_cell,
)


@dataclass
class PdncChars:
    """Resolved PDNC character metadata for a single novel."""

    alias_map: dict[str, str] = field(
        default_factory=dict
    )  # surface_lower -> canonical
    category: dict[str, str] = field(default_factory=dict)  # canonical -> lowercased
    canonical_names: set[str] = field(default_factory=set)

    def normalize(self, name: str | None) -> str | None:
        """Collapse any surface form to its canonical Main Name (else itself)."""
        if not name:
            return name
        return self.alias_map.get(name.strip().lower(), name.strip())

    def filtered_names(self, categories: tuple[str, ...]) -> set[str]:
        return {c for c in self.canonical_names if self.category.get(c) in categories}


def read_pdnc_characters(novel_dir: str | Path) -> PdncChars:
    """Parse character_info.csv into a PdncChars (aliases + Category)."""
    path = _find_existing(Path(novel_dir), _CHARACTER_FILE_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            f"No character info file under {novel_dir} "
            f"(looked for {_CHARACTER_FILE_CANDIDATES})"
        )
    chars = PdncChars()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        name_col = _find_column(fields, "Main Name", "name", "character")
        alias_col = _find_column(fields, "Aliases", "alias", "aliases")
        cat_col = _find_column(fields, "Category", "category")
        for row in reader:
            main = (row.get(name_col) or "").strip() if name_col else ""
            if not main:
                continue
            chars.canonical_names.add(main)
            chars.alias_map[main.lower()] = main
            for alias in _parse_list_cell(row.get(alias_col) if alias_col else None):
                if alias != main:
                    chars.alias_map[alias.lower()] = main
            category = (row.get(cat_col) or "").strip().lower() if cat_col else ""
            if category:
                chars.category[main] = category
    return chars
