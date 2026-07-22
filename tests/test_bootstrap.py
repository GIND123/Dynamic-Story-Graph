import os

import pytest

from gnsm.colab import bootstrap


def _clear_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in bootstrap._HF_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_ensure_hf_token_returns_false_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_hf_env(monkeypatch)
    # No google.colab in the test env, so detection falls through to "none".
    assert bootstrap.ensure_hf_token(verbose=False) is False


def test_ensure_hf_token_bridges_env_to_canonical_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_from_alias")
    assert bootstrap.ensure_hf_token(verbose=False) is True
    # Canonical names transformers / huggingface_hub read are now populated.
    assert os.environ["HF_TOKEN"] == "hf_from_alias"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_from_alias"


def test_detect_hf_token_prefers_primary_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    assert bootstrap.detect_hf_token() == "hf_primary"


def test_repo_root_points_at_package_parent() -> None:
    root = bootstrap.repo_root()
    assert (root / "gnsm" / "colab" / "bootstrap.py").exists()
