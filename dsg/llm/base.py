"""Backend contract plus an on-disk cache.

Every generation is cached by (model, prompt, decoding params). Re-running an
experiment after a policy change therefore costs nothing: the expensive
proposal pass happens once per (book, model) and every policy reads the same
cached stream, which is also what makes the extractor-held-constant comparison
exact rather than approximate.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str: ...


class GenerationCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS gen ("
            " key TEXT PRIMARY KEY, model TEXT, prompt TEXT, output TEXT, meta TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def key(model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        blob = json.dumps([model, prompt, max_tokens, temperature], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT output FROM gen WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, model: str, prompt: str, output: str, meta: dict | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO gen VALUES (?,?,?,?,?)",
                (key, model, prompt, output, json.dumps(meta or {})),
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM gen").fetchone()[0]
        return {"entries": int(n)}


class CachedBackend:
    """Wraps any backend so repeated prompts never re-run the model."""

    def __init__(self, inner: Backend, cache: GenerationCache) -> None:
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.hits = 0
        self.misses = 0

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        key = GenerationCache.key(self.name, prompt, max_tokens, temperature)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        out = self.inner.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        self.cache.put(key, self.name, prompt, out)
        return out
