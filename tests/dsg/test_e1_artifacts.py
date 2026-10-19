"""Resolving saved E1 files: Hub first, local fallback, hash always checked."""

from __future__ import annotations

import hashlib
import json

import pytest

from dsg.eval import e1_artifacts as ea


def _manifest(tmp_path, content: bytes):
    m = {"repo": "r", "repo_type": "model", "revision": "abc",
         "files": [{"name": "run.json", "hub_path": "e1/run.json",
                    "sha256": hashlib.sha256(content).hexdigest()}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    return path


def _no_hub(manifest, entry):
    raise ConnectionError("offline")


def test_an_existing_path_is_used_as_is(tmp_path):
    f = tmp_path / "any.json"
    f.write_text("{}")
    assert ea.resolve(f, tmp_path / "missing.json") == f


def test_the_hub_copy_is_preferred(tmp_path):
    hub = tmp_path / "hub.json"
    hub.write_bytes(b"data")
    got = ea.resolve("run.json", _manifest(tmp_path, b"data"), fetch=lambda m, e: hub)
    assert got == hub


def test_falls_back_to_a_local_copy_with_a_matching_hash(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    (local / "run.json").write_bytes(b"data")
    monkeypatch.setenv("DSG_E1_LOCAL", str(local))
    got = ea.resolve("run.json", _manifest(tmp_path, b"data"), fetch=_no_hub)
    assert got == local / "run.json"


def test_a_copy_with_the_wrong_hash_is_refused(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    (local / "run.json").write_bytes(b"tampered")
    monkeypatch.setenv("DSG_E1_LOCAL", str(local))
    with pytest.raises(ea.ArtifactError, match="hash"):
        ea.resolve("run.json", _manifest(tmp_path, b"data"), fetch=_no_hub)


def test_an_unknown_name_is_an_error(tmp_path):
    with pytest.raises(ea.ArtifactError, match="neither a file"):
        ea.resolve("other.json", _manifest(tmp_path, b"data"), fetch=_no_hub)


def test_no_copy_anywhere_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("DSG_E1_LOCAL", str(tmp_path / "empty"))
    monkeypatch.setattr(ea, "LOCAL_DIRS", ())
    with pytest.raises(ea.ArtifactError, match="no copy"):
        ea.resolve("run.json", _manifest(tmp_path, b"data"), fetch=_no_hub)
