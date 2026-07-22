"""Runtime and CUDA diagnostics ("does this box have what GNSM needs?").

This module never hard-imports torch. It is safe to call on a laptop with only
the base dependencies installed and on a Colab GPU runtime with the full
training stack. ``gnsm doctor`` renders :func:`probe` as a readable table so the
first thing you run after cloning tells you exactly what the environment is.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import platform
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class GpuInfo:
    index: int
    name: str
    total_memory_gb: float
    capability: str


@dataclass(slots=True)
class EnvReport:
    python_version: str
    platform: str
    torch_installed: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    cudnn_version: int | None = None
    device_count: int = 0
    gpus: list[GpuInfo] = field(default_factory=list)
    bf16_supported: bool = False
    recommended_dtype: str = "float32"
    hf_token_present: bool = False
    packages: dict[str, str | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ready_for_gpu_training(self) -> bool:
        return self.torch_installed and self.cuda_available and self.device_count > 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# Optional packages we care about for the training / generation stack.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "datasets",
    "safetensors",
    "sentencepiece",
    "numpy",
    "yaml",
)


def _package_version(name: str) -> str | None:
    # ``yaml`` is imported as ``yaml`` but distributed as ``PyYAML``.
    distribution = {"yaml": "PyYAML"}.get(name, name)
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        pass
    try:  # fall back to a live import for editable / namespace installs
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def recommend_dtype(bf16_supported: bool, cuda_available: bool) -> str:
    if not cuda_available:
        return "float32"
    return "bfloat16" if bf16_supported else "float16"


# Environment names an already-active Hugging Face token may live under.
_HF_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


def hf_token_present() -> bool:
    """True if an HF token is active in the environment (value is never read out)."""
    return any(os.environ.get(name) for name in _HF_TOKEN_ENV_VARS)


def probe() -> EnvReport:
    """Inspect the current runtime without requiring any optional dependency."""

    report = EnvReport(
        python_version=platform.python_version(),
        platform=platform.platform(),
        hf_token_present=hf_token_present(),
        packages={name: _package_version(name) for name in _TRACKED_PACKAGES},
    )

    try:
        import torch
    except Exception:
        report.notes.append(
            "torch is not installed - run `pip install -r requirements-colab.txt` "
            "or `pip install -e '.[training]'` for GPU work."
        )
        return report

    report.torch_installed = True
    report.torch_version = torch.__version__
    report.cuda_version = torch.version.cuda
    report.cuda_available = bool(torch.cuda.is_available())

    if not report.cuda_available:
        report.notes.append(
            "torch is installed but CUDA is unavailable - on Colab set "
            "Runtime > Change runtime type > GPU, then restart the runtime."
        )
        report.recommended_dtype = "float32"
        return report

    cudnn = getattr(torch.backends, "cudnn", None)
    report.cudnn_version = cudnn.version() if cudnn is not None else None
    report.device_count = torch.cuda.device_count()
    for index in range(report.device_count):
        properties = torch.cuda.get_device_properties(index)
        report.gpus.append(
            GpuInfo(
                index=index,
                name=properties.name,
                total_memory_gb=round(properties.total_memory / (1024**3), 2),
                capability=f"sm_{properties.major}{properties.minor}",
            )
        )

    try:
        report.bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        report.bf16_supported = False
    report.recommended_dtype = recommend_dtype(report.bf16_supported, True)
    return report


def format_report(report: EnvReport) -> str:
    hf_status = "active (gated models reachable)" if report.hf_token_present else "not set"
    lines = [
        "GNSM environment report",
        "=" * 48,
        f"python            : {report.python_version}",
        f"platform          : {report.platform}",
        f"hf token          : {hf_status}",
        f"torch installed   : {report.torch_installed}",
    ]
    if report.torch_installed:
        lines += [
            f"torch version     : {report.torch_version}",
            f"cuda (torch build): {report.cuda_version or 'cpu-only build'}",
            f"cuda available    : {report.cuda_available}",
            f"cudnn version     : {report.cudnn_version}",
            f"device count      : {report.device_count}",
            f"bf16 supported    : {report.bf16_supported}",
            f"recommended dtype : {report.recommended_dtype}",
        ]
        for gpu in report.gpus:
            lines.append(
                f"  gpu[{gpu.index}]         : {gpu.name} "
                f"({gpu.total_memory_gb} GB, {gpu.capability})"
            )
    lines.append("-" * 48)
    lines.append("optional packages:")
    for name, version in report.packages.items():
        lines.append(f"  {name:<14}: {version or 'missing'}")
    if report.notes:
        lines.append("-" * 48)
        for note in report.notes:
            lines.append(f"note: {note}")
    lines.append("-" * 48)
    verdict = "READY for GPU training" if report.ready_for_gpu_training else "CPU-only (no GPU)"
    lines.append(f"verdict           : {verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="gnsm doctor", description="Report the GNSM runtime.")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="exit non-zero unless a CUDA device is available",
    )
    args = parser.parse_args(argv)

    report = probe()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    if args.require_gpu and not report.ready_for_gpu_training:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
