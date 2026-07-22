import importlib.util

import pytest

from gnsm import env

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_probe_runs_without_torch_and_reports_packages() -> None:
    report = env.probe()
    assert report.python_version
    assert report.platform
    # numpy and PyYAML are core dependencies, so they must be resolvable.
    assert report.packages["numpy"] is not None
    assert report.packages["yaml"] is not None
    assert "torch" in report.packages


def test_recommend_dtype_matrix() -> None:
    assert env.recommend_dtype(bf16_supported=False, cuda_available=False) == "float32"
    assert env.recommend_dtype(bf16_supported=True, cuda_available=False) == "float32"
    assert env.recommend_dtype(bf16_supported=False, cuda_available=True) == "float16"
    assert env.recommend_dtype(bf16_supported=True, cuda_available=True) == "bfloat16"


def test_format_report_is_stringable_and_has_verdict() -> None:
    text = env.format_report(env.probe())
    assert "GNSM environment report" in text
    assert "verdict" in text


def test_ready_flag_consistency() -> None:
    report = env.probe()
    expected = report.torch_installed and report.cuda_available and report.device_count > 0
    assert report.ready_for_gpu_training is expected


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_probe_reports_torch_when_available() -> None:
    report = env.probe()
    assert report.torch_installed is True
    assert report.torch_version is not None


def test_hf_token_present_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in env._HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert env.hf_token_present() is False
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    assert env.hf_token_present() is True


def test_report_never_contains_token_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_value")
    text = env.format_report(env.probe())
    assert "hf_supersecret_value" not in text
    assert "hf token" in text
