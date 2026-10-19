"""Find a saved E1 file: the Hub copy first, a local folder as fallback.

Every E1 number is computed from files listed in ``audit/e1/hub_manifest.json``,
which pins the Hub revision and each file's SHA-256. ``resolve`` accepts either
an existing path (used as is) or a manifest name such as
``e1-qwen7b-v3-raw.json``. A name is downloaded from the pinned Hub revision;
if the Hub is unreachable, the local folders are tried in order: ``$DSG_E1_LOCAL``,
``artifacts/e1``, then ``../Dynamic-Story-Graph/artifacts/e1`` (where the files
were written). Whichever copy is used must match the manifest hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MANIFEST = Path("audit/e1/hub_manifest.json")
LOCAL_DIRS = ("artifacts/e1", "../Dynamic-Story-Graph/artifacts/e1")


class ArtifactError(RuntimeError):
    """A named E1 file could not be found, or did not match its hash."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _from_hub(manifest: dict, entry: dict) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(manifest["repo"], entry["hub_path"],
                                repo_type=manifest["repo_type"],
                                revision=manifest["revision"]))


def resolve(name_or_path: str | Path, manifest_path: Path = MANIFEST,
            fetch=_from_hub) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    manifest = json.loads(Path(manifest_path).read_text())
    entry = next((e for e in manifest["files"] if e["name"] == str(name_or_path)), None)
    if entry is None:
        raise ArtifactError(f"{name_or_path} is neither a file nor in {manifest_path}")
    candidates = []
    try:
        candidates.append(fetch(manifest, entry))
    except Exception as exc:  # no network, no huggingface_hub, missing revision
        print(f"[e1] Hub copy unavailable ({type(exc).__name__}); trying local folders")
    local = [os.environ["DSG_E1_LOCAL"]] if os.environ.get("DSG_E1_LOCAL") else []
    candidates += [Path(d) / entry["name"] for d in (*local, *LOCAL_DIRS)]
    for candidate in candidates:
        if candidate.exists():
            if _sha256(candidate) != entry["sha256"]:
                raise ArtifactError(f"{candidate} does not match the manifest hash")
            return candidate
    raise ArtifactError(f"no copy of {entry['name']} on the Hub or in {list(LOCAL_DIRS)}")
