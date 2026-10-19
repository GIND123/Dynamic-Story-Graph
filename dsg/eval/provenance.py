"""What an audit sheet was built from, so it can be rebuilt byte for byte."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    dirty = run("status", "--porcelain", "--untracked-files=no", "--", "dsg")
    return {"commit": run("rev-parse", "HEAD"), "dsg_tree_dirty": bool(dirty)}


def record(module: str, argv: list[str] | None, seeds: dict[str, int],
           inputs: list[Path]) -> dict[str, object]:
    args = list(sys.argv[1:] if argv is None else argv)
    return {
        "command": " ".join(["python", "-m", module, *args]),
        "seeds": seeds,
        "git": git_state(),
        "python": sys.version.split()[0],
        "inputs": {str(p): sha256(p) for p in sorted(inputs)},
    }
