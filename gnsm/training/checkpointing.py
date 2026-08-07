"""Periodic checkpoint push/resume against a private Hugging Face Hub repo.

The Hub, not local disk, is the source of truth for "the latest checkpoint" so
that a restart -- by the same person or anyone else with repo access -- can
resume training from wherever it left off. ``latest.json`` is only written
after its checkpoint folder has fully uploaded, so an interrupted push leaves
it pointing at the previous good checkpoint rather than a partial one.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gnsm.exceptions import OptionalDependencyError

# This repo's local .env uses the bare `HF` name; `HF_TOKEN` is the convention
# gnsm/colab/bootstrap.py and the Hub libraries themselves read.
_HF_ENV_VARS = ("HF_TOKEN", "HF")


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def resolve_token(explicit: str | None = None) -> str:
    """Resolve an HF token: explicit arg, else HF_TOKEN, else the bare HF var."""

    if explicit:
        return explicit
    _load_dotenv_if_present()
    for name in _HF_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        "No Hugging Face token found. Set HF_TOKEN or HF in the environment "
        "(this repo's .env uses the bare HF var), or pass token= explicitly."
    )


def _hf_api(token: str) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise OptionalDependencyError(
            "Checkpoint push/resume requires huggingface_hub: "
            "pip install -e '.[training]' (or add huggingface_hub)."
        ) from exc
    return HfApi(token=token)


def _not_found_errors() -> tuple[type[Exception], ...]:
    try:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import (  # type: ignore[import-not-found,no-redef]
            EntryNotFoundError,
            RepositoryNotFoundError,
        )
    return (EntryNotFoundError, RepositoryNotFoundError)


@dataclass(slots=True)
class CheckpointConfig:
    hf_repo_id: str
    local_dir: Path
    checkpoint_every_steps: int = 50
    checkpoint_every_seconds: float = 300.0
    token: str | None = None


class CheckpointManager:
    """Owns push cadence, repo creation, and heartbeats for one training run."""

    def __init__(self, config: CheckpointConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.token = resolve_token(config.token)
        self._last_push_time = time.monotonic()
        self._repo_ready = False

    def _ensure_repo(self) -> None:
        if self._repo_ready:
            return
        _hf_api(self.token).create_repo(self.config.hf_repo_id, private=True, exist_ok=True)
        self._repo_ready = True

    def should_checkpoint(self, step: int) -> bool:
        if step > 0 and step % self.config.checkpoint_every_steps == 0:
            return True
        return (time.monotonic() - self._last_push_time) >= self.config.checkpoint_every_seconds

    def push(self, step: int, loss: float, state: dict[str, Any]) -> None:
        try:
            import torch
        except ImportError as exc:
            raise OptionalDependencyError(
                "Checkpoint push requires torch: pip install -e '.[training]'."
            ) from exc

        self._ensure_repo()
        api = _hf_api(self.token)

        ckpt_dir = self.config.local_dir / f"checkpoint-step-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_dir / "state.pt")

        path_in_repo = f"checkpoints/checkpoint-step-{step}"
        api.upload_folder(
            folder_path=str(ckpt_dir), repo_id=self.config.hf_repo_id, path_in_repo=path_in_repo
        )
        # Promotion happens only after the folder upload above fully succeeds.
        latest = json.dumps({"step": step, "path": path_in_repo, "run_id": self.run_id})
        api.upload_file(
            path_or_fileobj=latest.encode("utf-8"),
            path_in_repo="latest.json",
            repo_id=self.config.hf_repo_id,
        )
        self._last_push_time = time.monotonic()
        self.write_heartbeat(step, loss)

    def push_best(self, step: int, val_loss: float, state: dict[str, Any]) -> None:
        """Push the best-so-far checkpoint (by validation loss) to a stable
        ``checkpoints/best`` path, overwriting the previous best. Call this
        only when the caller has confirmed ``val_loss`` improved -- this
        method doesn't track "best" itself, so the run's early-stopping loop
        stays the single source of truth for what counts as an improvement.
        """

        try:
            import torch
        except ImportError as exc:
            raise OptionalDependencyError(
                "Checkpoint push requires torch: pip install -e '.[training]'."
            ) from exc

        self._ensure_repo()
        api = _hf_api(self.token)

        best_dir = self.config.local_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, best_dir / "state.pt")

        path_in_repo = "checkpoints/best"
        api.upload_folder(
            folder_path=str(best_dir), repo_id=self.config.hf_repo_id, path_in_repo=path_in_repo
        )
        # Promoted only after the folder upload above fully succeeds, same as latest.json.
        best = json.dumps(
            {"step": step, "val_loss": val_loss, "path": path_in_repo, "run_id": self.run_id}
        )
        api.upload_file(
            path_or_fileobj=best.encode("utf-8"),
            path_in_repo="best.json",
            repo_id=self.config.hf_repo_id,
        )

    def write_heartbeat(self, step: int, loss: float) -> None:
        self._ensure_repo()
        payload = json.dumps(
            {"run_id": self.run_id, "step": step, "loss": loss, "timestamp": time.time()}
        )
        _hf_api(self.token).upload_file(
            path_or_fileobj=payload.encode("utf-8"),
            path_in_repo=f"heartbeats/{self.run_id}.json",
            repo_id=self.config.hf_repo_id,
        )

    def push_artifact(self, local_path: Path, path_in_repo: str) -> None:
        """Push an arbitrary file (a plot, a report) to the repo, e.g. under
        ``plots/{run_id}/``."""

        self._ensure_repo()
        _hf_api(self.token).upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=self.config.hf_repo_id,
        )


def resume_from_hub(
    hf_repo_id: str, local_dir: Path, token: str | None = None
) -> dict[str, Any] | None:
    """Download and load the latest checkpoint, or None for a fresh start."""

    try:
        import torch
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise OptionalDependencyError(
            "Resuming requires torch and huggingface_hub: pip install -e '.[training]'."
        ) from exc

    resolved = resolve_token(token)
    try:
        latest_path = hf_hub_download(repo_id=hf_repo_id, filename="latest.json", token=resolved)
    except _not_found_errors():
        return None

    latest = json.loads(Path(latest_path).read_text())
    snapshot_dir = snapshot_download(
        repo_id=hf_repo_id,
        allow_patterns=[f"{latest['path']}/*"],
        local_dir=str(local_dir),
        token=resolved,
    )
    state = torch.load(Path(snapshot_dir) / latest["path"] / "state.pt", map_location="cpu")
    state["step"] = latest["step"]
    return state


def read_heartbeat(hf_repo_id: str, run_id: str, token: str | None = None) -> dict[str, Any] | None:
    """Read the last-reported step/loss/timestamp for a run, or None if absent."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise OptionalDependencyError(
            "Heartbeat checks require huggingface_hub: pip install -e '.[training]'."
        ) from exc

    resolved = resolve_token(token)
    try:
        path = hf_hub_download(
            repo_id=hf_repo_id, filename=f"heartbeats/{run_id}.json", token=resolved
        )
    except _not_found_errors():
        return None
    return json.loads(Path(path).read_text())


CheckpointCallback = Callable[[int, float, dict[str, Any], dict[str, Any]], None]


def attach_to_run(
    hf_repo_id: str,
    checkpoint_every_steps: int,
    resume: bool,
    run_id: str,
    local_dir: Path | None = None,
) -> tuple[CheckpointCallback, dict[str, Any] | None, CheckpointManager]:
    """Build (checkpoint_cb, resume_state, manager) for a training loop's run().

    ``checkpoint_cb`` is safe to call every step -- it consults
    :meth:`CheckpointManager.should_checkpoint` itself and only pushes when the
    step/time cadence fires. ``manager`` is returned too so callers can push
    extra artifacts (e.g. plots via :meth:`CheckpointManager.push_artifact`)
    once training finishes.
    """

    local_dir = local_dir or Path(".gnsm_checkpoints") / hf_repo_id.replace("/", "__")
    resume_state = resume_from_hub(hf_repo_id, local_dir) if resume else None

    manager = CheckpointManager(
        CheckpointConfig(
            hf_repo_id=hf_repo_id,
            local_dir=local_dir,
            checkpoint_every_steps=checkpoint_every_steps,
        ),
        run_id=run_id,
    )

    def checkpoint_cb(
        step: int, loss: float, model_state: dict[str, Any], optimizer_state: dict[str, Any]
    ) -> None:
        if manager.should_checkpoint(step):
            manager.push(step, loss, {**model_state, "optimizer": optimizer_state})

    return checkpoint_cb, resume_state, manager
