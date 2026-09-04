"""One place that knows which corpora exist, so runners stay corpus-agnostic."""

from __future__ import annotations

from collections.abc import Callable

from dsg.data.pdnc import Novel

_LOADERS: dict[str, Callable[..., list[Novel]]] = {}


def loader(name: str) -> Callable[..., list[Novel]]:
    if not _LOADERS:
        from dsg.data import litbank, pdnc

        _LOADERS["pdnc"] = pdnc.load_corpus
        _LOADERS["litbank"] = litbank.load_corpus
    if name not in _LOADERS:
        raise KeyError(f"unknown corpus '{name}' (have: {sorted(_LOADERS)})")
    return _LOADERS[name]


def load(name: str, limit: int | None = None, only: list[str] | None = None) -> list[Novel]:
    return loader(name)(limit=limit, only=only)


# Corpus-appropriate window sizes: a 2K-token LitBank excerpt needs a finer
# window than a 500K-character novel to yield a comparable number of reading
# steps.
WINDOW_CHARS = {"pdnc": 3200, "litbank": 1200}
