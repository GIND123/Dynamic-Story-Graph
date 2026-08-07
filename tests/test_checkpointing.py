import importlib.util
import time
from pathlib import Path

import pytest

from gnsm.training import checkpointing

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_HF = importlib.util.find_spec("huggingface_hub") is not None


def test_resolve_token_prefers_explicit_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert checkpointing.resolve_token("explicit") == "explicit"


def test_resolve_token_falls_back_to_hf_token_then_bare_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF", raising=False)
    monkeypatch.setenv("HF", "bare-hf-value")
    assert checkpointing.resolve_token() == "bare-hf-value"

    monkeypatch.setenv("HF_TOKEN", "hf-token-value")
    assert checkpointing.resolve_token() == "hf-token-value"


def test_resolve_token_raises_without_any_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF", raising=False)
    # Isolate from this repo's real .env (HF=...) so the test exercises the
    # "nothing found" path rather than python-dotenv's real, correct find-and-load.
    monkeypatch.setattr(checkpointing, "_load_dotenv_if_present", lambda: None)
    with pytest.raises(RuntimeError):
        checkpointing.resolve_token()


def _manager(tmp_path: Path, checkpoint_every_steps: int = 5) -> checkpointing.CheckpointManager:
    config = checkpointing.CheckpointConfig(
        hf_repo_id="acct/repo",
        local_dir=tmp_path,
        checkpoint_every_steps=checkpoint_every_steps,
        token="explicit-token",
    )
    return checkpointing.CheckpointManager(config, run_id="test-run")


def test_should_checkpoint_fires_on_step_cadence(tmp_path: Path) -> None:
    manager = _manager(tmp_path, checkpoint_every_steps=5)
    fired = [step for step in range(12) if manager.should_checkpoint(step)]
    assert fired == [5, 10]


def test_should_checkpoint_fires_on_time_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, checkpoint_every_steps=1_000_000)
    manager.config.checkpoint_every_seconds = 10.0

    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    manager._last_push_time = 0.0

    assert manager.should_checkpoint(1) is False
    clock["t"] = 11.0
    assert manager.should_checkpoint(1) is True


def test_attach_to_run_callback_only_pushes_on_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "explicit-token")
    checkpoint_cb, resume_state, manager = checkpointing.attach_to_run(
        "acct/repo", checkpoint_every_steps=4, resume=False, run_id="test-run", local_dir=tmp_path
    )
    assert resume_state is None
    assert manager.run_id == "test-run"

    pushed_steps: list[int] = []
    monkeypatch.setattr(
        checkpointing.CheckpointManager,
        "push",
        lambda self, step, loss, state: pushed_steps.append(step),
    )
    for step in range(9):
        checkpoint_cb(step, 0.1, {"encoder": {}}, {})
    assert pushed_steps == [4, 8]


@pytest.mark.skipif(not (HAS_TORCH and HAS_HF), reason="torch and huggingface_hub required")
def test_push_promotes_latest_only_after_folder_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads: list[tuple[str, str]] = []

    class FakeApi:
        def __init__(self, token: str) -> None:
            self.token = token

        def create_repo(self, repo_id: str, private: bool, exist_ok: bool) -> None:
            uploads.append(("create_repo", repo_id))

        def upload_folder(self, folder_path: str, repo_id: str, path_in_repo: str) -> None:
            uploads.append(("upload_folder", path_in_repo))

        def upload_file(self, path_or_fileobj: bytes, path_in_repo: str, repo_id: str) -> None:
            uploads.append(("upload_file", path_in_repo))

    monkeypatch.setattr(checkpointing, "_hf_api", lambda token: FakeApi(token))

    manager = _manager(tmp_path)
    manager.push(step=5, loss=0.42, state={"encoder": {"w": 1}})

    # The checkpoint folder must be uploaded before latest.json is promoted,
    # and the heartbeat is written last.
    assert uploads == [
        ("create_repo", "acct/repo"),
        ("upload_folder", "checkpoints/checkpoint-step-5"),
        ("upload_file", "latest.json"),
        ("upload_file", "heartbeats/test-run.json"),
    ]


@pytest.mark.skipif(not (HAS_TORCH and HAS_HF), reason="torch and huggingface_hub required")
def test_push_best_promotes_best_json_only_after_folder_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads: list[tuple[str, str]] = []

    class FakeApi:
        def __init__(self, token: str) -> None:
            self.token = token

        def create_repo(self, repo_id: str, private: bool, exist_ok: bool) -> None:
            uploads.append(("create_repo", repo_id))

        def upload_folder(self, folder_path: str, repo_id: str, path_in_repo: str) -> None:
            uploads.append(("upload_folder", path_in_repo))

        def upload_file(self, path_or_fileobj: bytes, path_in_repo: str, repo_id: str) -> None:
            uploads.append(("upload_file", path_in_repo))

    monkeypatch.setattr(checkpointing, "_hf_api", lambda token: FakeApi(token))

    manager = _manager(tmp_path)
    manager.push_best(step=12, val_loss=3.79, state={"encoder": {"w": 1}})

    assert uploads == [
        ("create_repo", "acct/repo"),
        ("upload_folder", "checkpoints/best"),
        ("upload_file", "best.json"),
    ]
    # push_best always writes to the same stable local path, so a second call
    # (a new best) overwrites rather than accumulating checkpoint-best-N dirs.
    assert (tmp_path / "best" / "state.pt").exists()
