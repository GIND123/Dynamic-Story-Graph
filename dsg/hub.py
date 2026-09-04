"""Push artefacts to the Hugging Face Hub.

Everything the project produces that is expensive to recompute -- the trained
adapter, the constructed dataset, the figures and result tables -- is mirrored
to the Hub, so a lost machine or a cancelled Modal run costs time rather than
results.

The token is read from the environment (``HF_TOKEN``) or a Modal secret of the
same name. It is never written to the repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ORG = os.environ.get("DSG_HF_ORG", "GOVINDFROM")
DATASET_REPO = f"{ORG}/dsg-state-continuation"
MODEL_REPO = f"{ORG}/dsg-writer"
ARTIFACT_REPO = f"{ORG}/dsg-artifacts"


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; cannot reach the Hub")
    return HfApi(token=token), token


def ensure_repo(repo_id: str, repo_type: str = "model", private: bool = False,
                token: str | None = None) -> str:
    api, tok = _api(token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private,
                    exist_ok=True, token=tok)
    return repo_id


def push_folder(
    local_dir: str | Path, repo_id: str, repo_type: str = "model",
    path_in_repo: str = "", token: str | None = None, message: str = "update",
) -> str:
    api, tok = _api(token)
    ensure_repo(repo_id, repo_type=repo_type, token=tok)
    api.upload_folder(
        folder_path=str(local_dir), repo_id=repo_id, repo_type=repo_type,
        path_in_repo=path_in_repo or None, token=tok, commit_message=message,
    )
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}"


def push_file(
    local_path: str | Path, repo_id: str, path_in_repo: str,
    repo_type: str = "model", token: str | None = None, message: str = "update",
) -> str:
    api, tok = _api(token)
    ensure_repo(repo_id, repo_type=repo_type, token=tok)
    api.upload_file(
        path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
        repo_id=repo_id, repo_type=repo_type, token=tok, commit_message=message,
    )
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/blob/main/{path_in_repo}"


def push_jsonl(
    rows: list[dict], repo_id: str, path_in_repo: str,
    repo_type: str = "dataset", token: str | None = None, message: str = "update",
) -> str:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(path_in_repo).name
        with local.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return push_file(local, repo_id, path_in_repo, repo_type, token, message)
