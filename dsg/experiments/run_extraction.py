"""Local (MLX) extraction, for development and for running without Modal."""

from __future__ import annotations

import time
from pathlib import Path

from dsg.data.registry import WINDOW_CHARS, load
from dsg.llm import build_backend
from dsg.proposals import extract_book, save_proposals


def run(
    out_dir: Path,
    backend_name: str = "qwen3b",
    books: int | None = None,
    max_windows: int | None = None,
    window_chars: int = 0,
    corpus: str = "pdnc",
    cache: Path | None = Path("artifacts/cache/generations.sqlite"),
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(backend_name, cache_path=cache)
    window_chars = window_chars or WINDOW_CHARS.get(corpus, 3200)
    for novel in load(corpus, limit=books):
        started = time.time()
        proposals = extract_book(
            novel.text, backend, window_chars=window_chars, max_windows=max_windows
        )
        save_proposals(
            out_dir / f"{novel.book_id}.jsonl",
            proposals,
            {
                "model": backend_name, "book_id": novel.book_id,
                "window_chars": window_chars, "windows": len(proposals),
                "seconds": round(time.time() - started, 1),
            },
        )
        ok = sum(p.parse_ok for p in proposals)
        print(
            f"[extract] {novel.book_id}: {len(proposals)} windows "
            f"parse_ok={ok} in {(time.time() - started) / 60:.1f} min",
            flush=True,
        )
    return out_dir
