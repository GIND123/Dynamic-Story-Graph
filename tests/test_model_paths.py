from pathlib import Path

import pytest

from gnsm.generation.huggingface import MODELS_DIR_ENV, local_or_hub


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv(MODELS_DIR_ENV, str(root))
    return root


def test_downloaded_repo_id_resolves_to_local_weights(models_dir: Path) -> None:
    local = models_dir / "llama-3.1-8b-instruct"
    local.mkdir()
    assert local_or_hub("meta-llama/Llama-3.1-8B-Instruct") == str(local)


def test_missing_repo_id_falls_back_to_the_hub(models_dir: Path) -> None:
    assert local_or_hub("Qwen/Qwen2.5-14B-Instruct") == "Qwen/Qwen2.5-14B-Instruct"


def test_explicit_path_is_never_rewritten(models_dir: Path, tmp_path: Path) -> None:
    # A directory outside the models root must survive, even when a same-named
    # directory exists inside it.
    (models_dir / "qwen2.5-7b-instruct").mkdir()
    elsewhere = tmp_path / "scratch" / "qwen2.5-7b-instruct"
    elsewhere.mkdir(parents=True)
    assert local_or_hub(str(elsewhere)) == str(elsewhere)
