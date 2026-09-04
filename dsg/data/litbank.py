"""LitBank as a short-context control.

LitBank documents are ~2,100-token excerpts, so they are the opposite regime
from a full novel: an incremental reader barely has time to drift. Running the
identical protocol here answers the obvious objection to the main result --
that the policies differ for some reason other than reading depth. If premature
commitment is the mechanism, the gap between policies should be small on 2K
tokens and large on 500K, and that length dependence is itself the evidence.

Gold clusters are converted to the same surface-string partition used for PDNC,
so one metric implementation serves both corpora.

Cite: Bamman, Popat & Shen (2019), *An Annotated Dataset of Literary Entities*,
NAACL.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from dsg.data.pdnc import GoldCharacter, Novel

DEFAULT_ROOT = Path("data/litbank/coref/conll")

_NO_SPACE_BEFORE = set(".,;:!?)]}'\"’”%")
_NO_SPACE_AFTER = set("([{“‘$")
_PRONOUNS = {
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "himself", "herself", "itself", "themselves",
    "myself", "ourselves", "yourself", "who", "whom", "whose", "that", "which",
}


def _detokenize(tokens: list[str]) -> str:
    out: list[str] = []
    for token in tokens:
        if out and (token[:1] in _NO_SPACE_BEFORE or out[-1][-1:] in _NO_SPACE_AFTER):
            out.append(token)
        elif out:
            out.append(" " + token)
        else:
            out.append(token)
    return "".join(out)


def _parse_conll(path: Path) -> tuple[list[str], dict[int, list[tuple[int, int]]]]:
    """Return the token stream and cluster id -> [(start, end)] mention spans."""
    tokens: list[str] = []
    clusters: dict[int, list[tuple[int, int]]] = defaultdict(list)
    open_spans: dict[int, list[int]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.rstrip("\n").split("\t")
        if len(cells) < 4:
            continue
        index = len(tokens)
        tokens.append(cells[3])
        label = cells[-1].strip()
        if not label or label == "_":
            continue
        for part in label.split("|"):
            part = part.strip()
            if not part:
                continue
            cid = int(re.sub(r"[^0-9]", "", part) or -1)
            if cid < 0:
                continue
            if part.startswith("(") and part.endswith(")"):
                clusters[cid].append((index, index))
            elif part.startswith("("):
                open_spans[cid].append(index)
            elif part.endswith(")") and open_spans[cid]:
                clusters[cid].append((open_spans[cid].pop(), index))
    return tokens, dict(clusters)


def load_document(path: Path) -> Novel:
    tokens, clusters = _parse_conll(Path(path))
    text = _detokenize(tokens)
    characters: list[GoldCharacter] = []
    for cid, spans in clusters.items():
        names = set()
        for start, end in spans:
            surface = " ".join(tokens[start : end + 1]).strip()
            if not surface or surface.lower() in _PRONOUNS:
                continue
            # Keep proper-name-like mentions: the same item type PDNC scores.
            if surface[:1].isupper() and len(surface) > 2:
                names.add(surface)
        if names:
            characters.append(
                GoldCharacter(char_id=str(cid), name=sorted(names)[0], aliases=names)
            )
    return Novel(book_id=Path(path).stem, text=text, characters=characters, quotes=[])


@lru_cache(maxsize=1)
def _paths(root: str) -> tuple[Path, ...]:
    return tuple(sorted(Path(root).glob("*.conll")))


def load_corpus(
    limit: int | None = None, root: Path | str = DEFAULT_ROOT, only: list[str] | None = None
) -> list[Novel]:
    paths = list(_paths(str(root)))
    if only:
        wanted = {o.lower() for o in only}
        paths = [p for p in paths if p.stem.lower() in wanted]
    if limit:
        paths = paths[:limit]
    docs = [load_document(p) for p in paths]
    return [d for d in docs if d.characters]
