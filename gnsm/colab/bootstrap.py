"""One-shot environment bootstrap for Google Colab / cloud GPU runtimes.

Run it right after cloning::

    !python -m gnsm.colab.bootstrap

It uses only the standard library, so it works before anything is pip-installed.
Steps:
  1. print GPU info (`nvidia-smi`) if a GPU is attached;
  2. install a CUDA build of torch *only if torch is missing* (cu121 wheels);
  3. install the GNSM package (editable) plus ``requirements-colab.txt``;
  4. print the `gnsm doctor` runtime report.

Idempotent: re-running is safe and skips work that is already done.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# CUDA target for GNSM. cu121 wheels are compatible with Colab's driver and with
# CUDA 12.1-12.4 hosts. Bump to cu124 only if you intentionally move the target.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
CUDA_TAG = "cu121"


def _run(cmd: list[str], *, check: bool = True) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    if check and completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(cmd)}")
    return completed.returncode


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ


def repo_root() -> Path:
    # gnsm/colab/bootstrap.py -> repo root is two parents up from the package.
    return Path(__file__).resolve().parents[2]


def torch_present() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def show_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found - no GPU attached, or drivers unavailable.")
        print("On Colab: Runtime > Change runtime type > Hardware accelerator > GPU.")
        return
    _run(["nvidia-smi"], check=False)


def install_torch_if_missing() -> None:
    if torch_present():
        import torch

        print(f"torch already present: {torch.__version__} (cuda={torch.version.cuda}) - kept.")
        return
    print(f"torch missing - installing CUDA build ({CUDA_TAG}).")
    _run(
        [sys.executable, "-m", "pip", "install", "torch", "--index-url", TORCH_CUDA_INDEX],
        check=True,
    )


def install_stack(root: Path, *, editable: bool, requirements: bool) -> None:
    if requirements:
        req = root / "requirements-colab.txt"
        if req.exists():
            _run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)
        else:
            print(f"warning: {req} not found - skipping requirements install.")
    target = "-e" if editable else ""
    install_spec = str(root)
    cmd = [sys.executable, "-m", "pip", "install"]
    if target:
        cmd.append(target)
    cmd.append(install_spec)
    _run(cmd, check=True)


# Names that may already hold a token in the environment...
_HF_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)
# ...and Colab secret keys we will look under (kept broad, incl. lowercase "hf").
_HF_SECRET_NAMES = ("HF_TOKEN", "hf", "HF", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def detect_hf_token() -> str | None:
    """Find an already-provisioned HF token. Never prompts, never logs in."""
    for name in _HF_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    # Colab secrets are NOT auto-exported to the environment, so bridge them.
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    for name in _HF_SECRET_NAMES:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


def ensure_hf_token(*, verbose: bool = True) -> bool:
    """Activate an ambient HF token for this process (and its subprocesses).

    Reads from the environment or a Colab secret and re-exports it under the
    canonical names transformers / huggingface_hub read. There is no interactive
    login: if nothing is found, gated models simply stay out of reach. Run this
    from a *kernel* cell on Colab so later ``!python`` cells inherit the token.
    """
    token = detect_hf_token()
    if not token:
        if verbose:
            print(
                "HF token: none detected - public models only. Provide it via an "
                "HF_TOKEN env var, a prior `huggingface-cli login`, or a Colab "
                "secret named HF_TOKEN/hf. No prompt is shown."
            )
        return False
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ[name] = token
    if verbose:
        print("HF token: detected and active - gated models (Llama, Gemma, ...) are reachable.")
    return True


def maybe_set_hf_token() -> None:
    ensure_hf_token(verbose=True)


def run_doctor() -> int:
    return _run([sys.executable, "-m", "gnsm", "doctor"], check=False)


def bootstrap(
    *,
    editable: bool = True,
    install_torch: bool = True,
    requirements: bool = True,
    doctor: bool = True,
) -> int:
    root = repo_root()
    print("=" * 60)
    print(f"GNSM bootstrap  |  colab={in_colab()}  |  repo={root}")
    print("=" * 60)

    show_gpu()
    if install_torch:
        install_torch_if_missing()
    install_stack(root, editable=editable, requirements=requirements)
    maybe_set_hf_token()

    print("\n" + "=" * 60)
    print("bootstrap complete - runtime report:")
    print("=" * 60)
    code = run_doctor() if doctor else 0
    print(
        "\nNext:\n"
        "  python -m gnsm demo            # no-download end-to-end reference run\n"
        "  python -m gnsm smoke --json    # train neural stack on GPU (wiring check)\n"
        "  python -m gnsm doctor          # re-print the runtime report\n"
    )
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm.colab.bootstrap",
        description="Prepare a Colab / cloud GPU runtime to run GNSM.",
    )
    parser.add_argument(
        "--no-torch",
        dest="install_torch",
        action="store_false",
        help="do not install torch even if it is missing",
    )
    parser.add_argument(
        "--no-requirements",
        dest="requirements",
        action="store_false",
        help="skip requirements-colab.txt",
    )
    parser.add_argument(
        "--no-editable",
        dest="editable",
        action="store_false",
        help="install the package non-editable",
    )
    parser.add_argument(
        "--no-doctor", dest="doctor", action="store_false", help="skip the final doctor report"
    )
    args = parser.parse_args(argv)
    return bootstrap(
        editable=args.editable,
        install_torch=args.install_torch,
        requirements=args.requirements,
        doctor=args.doctor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
