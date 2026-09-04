"""Project Gutenberg fiction as the training corpus.

Public-domain novels are the only large source of *real* long-form fiction that
can be redistributed and that no licence forbids training on. We take the
English dump, keep what the catalogue marks as fiction at novel length, and cut
it into chapters, because the unit the writer model is trained on is a chapter
given everything before it.

Cite: Project Gutenberg; HF mirror ``sedthh/gutenberg_english``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

REPO = "sedthh/gutenberg_english"
N_SHARDS = 37

# Gutenberg boilerplate wrappers.
_START_RE = re.compile(r"\*\*\*\s*START OF (?:THIS |THE )?PROJECT GUTENBERG[^*]*\*\*\*", re.I)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THIS |THE )?PROJECT GUTENBERG[^*]*\*\*\*", re.I)

# Chapter headings as they actually appear in 19th/20th-century novels.
_CHAPTER_RE = re.compile(
    r"^[ \t]*(?:"
    r"CHAPTER\s+[IVXLCDM\d]+|"
    r"CHAPTER\s+[A-Z][A-Z\-]+|"
    r"[IVXLCDM]{1,7}\.?|"
    r"\d{1,3}\.?"
    r")[ \t]*$",
    re.MULTILINE,
)

_FICTION_HINTS = ("fiction", "novel", "romance", "stories", "tales")
_EXCLUDE_HINTS = ("juvenile", "poetry", "drama", "periodical", "dictionary",
                  "bibliography", "encyclopedia")


@dataclass(slots=True)
class GutenbergBook:
    text_id: str
    title: str
    author: str
    subjects: str
    text: str
    chapters: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "text_id": self.text_id, "title": self.title, "author": self.author,
            "subjects": self.subjects, "n_chapters": len(self.chapters),
            "chars": len(self.text),
        }


def strip_boilerplate(text: str) -> str:
    start = _START_RE.search(text)
    if start:
        text = text[start.end() :]
    end = _END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def segment_chapters(
    text: str, min_chars: int = 1500, max_chars: int = 20_000
) -> list[str]:
    """Split on chapter headings, falling back to paragraph blocks.

    Chapters that come out too short are folded into the previous one and
    over-long ones are cut at a paragraph boundary, so the training unit has a
    predictable size without ever splitting mid-paragraph.
    """
    marks = [m.start() for m in _CHAPTER_RE.finditer(text)]
    if len(marks) >= 4:
        bounds = marks + [len(text)]
        raw = [text[bounds[i] : bounds[i + 1]].strip() for i in range(len(bounds) - 1)]
    else:
        raw = [b.strip() for b in re.split(r"\n\s*\n\s*\n+", text) if b.strip()]

    merged: list[str] = []
    for block in raw:
        if merged and len(block) < min_chars:
            merged[-1] = merged[-1] + "\n\n" + block
        else:
            merged.append(block)

    out: list[str] = []
    for chapter in merged:
        while len(chapter) > max_chars:
            cut = chapter.rfind("\n\n", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            out.append(chapter[:cut].strip())
            chapter = chapter[cut:].strip()
        if len(chapter) >= min_chars:
            out.append(chapter)
    return out


def is_fiction(subjects: str, title: str) -> bool:
    blob = f"{subjects} {title}".lower()
    if any(bad in blob for bad in _EXCLUDE_HINTS):
        return False
    return any(hint in blob for hint in _FICTION_HINTS)


def iter_books(
    shards: int = 4,
    min_chars: int = 120_000,
    max_chars: int = 1_200_000,
    min_chapters: int = 10,
    token: str | None = None,
) -> Iterator[GutenbergBook]:
    """Stream fiction novels of usable length, already cut into chapters."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    names = sorted(
        s.rfilename
        for s in api.dataset_info(REPO).siblings
        if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet")
    )[:shards]

    for name in names:
        path = hf_hub_download(REPO, name, repo_type="dataset", token=token)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64):
            cols = batch.schema.names
            for i in range(batch.num_rows):
                row = {k: batch.column(k)[i].as_py() for k in cols}
                try:
                    meta = json.loads(row.get("METADATA") or "{}")
                except json.JSONDecodeError:
                    meta = {}
                if (meta.get("language") or "").lower() not in ("en", "english", ""):
                    continue
                subjects = str(meta.get("subjects") or "")
                title = str(meta.get("title") or "").replace("\r", " ").strip()
                if not is_fiction(subjects, title):
                    continue
                text = strip_boilerplate(row.get("TEXT") or "")
                if not (min_chars <= len(text) <= max_chars):
                    continue
                chapters = segment_chapters(text)
                if len(chapters) < min_chapters:
                    continue
                yield GutenbergBook(
                    text_id=str(meta.get("text_id", "?")),
                    title=title,
                    author=str(meta.get("authors") or "").replace("\r", " ").strip(),
                    subjects=subjects,
                    text=text,
                    chapters=chapters,
                )


def load_books(limit: int = 200, **kw) -> list[GutenbergBook]:
    out: list[GutenbergBook] = []
    for book in iter_books(**kw):
        out.append(book)
        if len(out) >= limit:
            break
    return out
