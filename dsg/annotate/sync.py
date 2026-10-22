"""Back up one annotator's files to GitHub: commit them, rebase, push.

Only the annotator's own CSVs are ever staged, so other work in the checkout is
left alone, and two annotators never edit the same file, so the rebase onto the
other person's pushes cannot conflict. Commits carry ``[skip ci]``: labels are
data, and pushing every few minutes should not start CI runs. (The workflows'
path filters already leave ``audit/`` out; the marker keeps it that way if they
change. Never quote the marker in a code commit's message: GitHub skips CI for
any push whose message contains it.)
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path


class SyncError(RuntimeError):
    """A git step failed; the labels are still saved on disk."""


class GitSync:
    def __init__(self, root: Path, paths: list[Path], me: str, branch: str = "main",
                 remote: str = "origin", enabled: bool = True):
        self.root, self.me, self.branch, self.remote = Path(root), me, branch, remote
        self.paths = [str(Path(p).relative_to(self.root)) for p in paths]
        self.enabled = enabled
        self.lock = threading.RLock()     # held by saves too, so git never sees half a file
        self.dirty = False
        self.status: dict = {"enabled": enabled, "last_push": None, "error": None,
                             "pending": False}

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        done = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True)
        if check and done.returncode != 0:
            raise SyncError(f"git {' '.join(args)}: {(done.stderr or done.stdout).strip()}")
        return done

    def _on_branch(self) -> None:
        head = self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if head != self.branch:
            raise SyncError(f"checkout is on '{head}', not '{self.branch}'; not syncing")

    def _pull(self) -> None:
        done = self._git("pull", "--rebase", "--autostash", self.remote, self.branch,
                         check=False)
        if done.returncode != 0:
            self._git("rebase", "--abort", check=False)
            raise SyncError(f"git pull: {(done.stderr or done.stdout).strip()}")

    def _commit(self, message: str) -> bool:
        if not self._git("status", "--porcelain", "--", *self.paths).stdout.strip():
            return False
        self._git("add", "--", *self.paths)
        self._git("commit", "-m", message, "--", *self.paths)
        return True

    def _only_ours(self) -> None:
        """Refuse to push anyone's other unpushed work along with the labels."""
        changed = self._git("diff", "--name-only", f"{self.remote}/{self.branch}..HEAD").stdout
        others = sorted(set(changed.splitlines()) - set(self.paths))
        if others:
            raise SyncError("other unpushed commits touch " + ", ".join(others[:3])
                            + "; push or move them yourself, then sync again")

    def mark_dirty(self) -> None:
        with self.lock:
            self.dirty = True
            self.status["pending"] = True

    def pull(self) -> None:
        """At start-up: bring in the other annotator's pushes."""
        if not self.enabled:
            return
        with self.lock:
            try:
                self._on_branch()
                self._pull()
                self.status["error"] = None
            except SyncError as exc:
                self.status["error"] = str(exc)

    def sync(self, summary: str = "") -> dict:
        """Commit our files if they changed, rebase onto the remote, push."""
        if not self.enabled:
            return self.status
        with self.lock:
            try:
                self._on_branch()
                message = f"Labels: {self.me}{' ' + summary if summary else ''} [skip ci]"
                self._commit(message)
                for attempt in range(2):
                    self._pull()
                    self._only_ours()
                    push = self._git("push", self.remote, f"HEAD:{self.branch}", check=False)
                    if push.returncode == 0:
                        break
                    if attempt == 1:
                        raise SyncError(f"git push: {(push.stderr or push.stdout).strip()}")
                self.dirty = False
                self.status.update(last_push=time.strftime("%H:%M:%S"), error=None,
                                   pending=False)
            except SyncError as exc:
                self.status["error"] = str(exc)
        return self.status

    def start(self, interval: float, summary=lambda: "") -> threading.Thread:
        """Push every ``interval`` seconds while there are unsynced saves."""
        def loop() -> None:
            while True:
                time.sleep(interval)
                if self.dirty:
                    self.sync(summary())

        thread = threading.Thread(target=loop, name="label-sync", daemon=True)
        if self.enabled:
            thread.start()
        return thread
