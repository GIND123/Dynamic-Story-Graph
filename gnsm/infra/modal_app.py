"""Modal launcher for GNSM training, with periodic HF checkpoint push/resume
and a lightweight health check.

One-time setup (see the top-level README's "Run on Modal" section for the
full walkthrough):

    pip install modal
    modal setup                                   # authenticate, browser flow
    modal secret create hf-token HF=<hf-write-token>

Usage:

    # cheap pre-spend smoke test (a few seconds on the cheapest GPU tier)
    modal run gnsm/infra/modal_app.py --steps 5 --gpu t4 \
        --hf-repo GOVINDFROM/DNG-GNSM-test --checkpoint-every 2

    # a real run
    modal run gnsm/infra/modal_app.py --steps 5000 --gpu t4 \
        --hf-repo GOVINDFROM/DNG-GNSM --checkpoint-every 100

    # resume after a restart (by you or anyone else with repo access)
    modal run gnsm/infra/modal_app.py --steps 5000 --gpu t4 \
        --hf-repo GOVINDFROM/DNG-GNSM --checkpoint-every 100 --resume

    # is it still making progress? (no GPU spun up)
    modal run gnsm/infra/modal_app.py::health_check --run-id primary \
        --hf-repo GOVINDFROM/DNG-GNSM

    # real training on EvolvTrip (not synthetic), with early stopping and a
    # best-checkpoint push (checkpoints/best) every time val loss improves,
    # plus a train/val loss-curve PNG+PDF marking the best step. Needs
    # data/evolvtrip_data/all_books_current.json downloaded locally first --
    # see that module's docstring for the git clone command.
    modal run gnsm/infra/modal_app.py::evolvtrip --epochs 100 --batch-size 16 \
        --patience 10 --gpu t4 --hf-repo GOVINDFROM/DNG-GNSM --checkpoint-every 50

The default entrypoint (`main`) launches `gnsm.training.smoke`, a GPU wiring
check on a synthetic batch, not a research result — see that module's
docstring. `evolvtrip` launches `gnsm.training.train_state`, real supervised
training against real data. Both share the same `attach_to_run` checkpoint
hook from `gnsm.training.checkpointing`.
"""

from __future__ import annotations

import time

import modal

app = modal.App("gnsm-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.46",
        "accelerate>=1.1",
        "safetensors>=0.4.5",
        "einops>=0.8",
        "huggingface-hub>=0.26",
        "numpy>=1.26",
        "PyYAML>=6.0",
        "matplotlib>=3.9",
    )
    # Resolved via the local `gnsm` import (this repo installed editable, per
    # the README's `pip install -e ".[modal]"`), not a hand-computed path off
    # __file__ -- that broke when Modal re-imports this module inside the
    # remote container, where __file__ is flattened to /root/modal_app.py and
    # has no repo-root-relative parents.
    .add_local_python_source("gnsm")
)

# data/ is git-ignored, not part of the gnsm package, so add_local_python_source
# doesn't carry it -- it needs its own add_local_dir. Guarded by modal.is_local()
# so the __file__-relative path is only ever computed on the local machine
# (never re-evaluated inside the remote container, which is exactly what broke
# the gnsm sync above before that fix).
if modal.is_local():
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parents[2]
    image = image.add_local_dir(
        str(_repo_root / "data" / "evolvtrip_data"), remote_path="/root/data/evolvtrip_data"
    )

# `training/smoke.py` trains sub-10M-parameter modules on a synthetic batch —
# comfortably within T4's 16GB, no tensor-core throughput needed. T4 is
# Modal's cheapest CUDA tier. This is deliberately not hardcoded: pass
# --gpu a10 / --gpu a100 once real P2 training (real backbones, real batches)
# replaces the smoke test, so spend scales with the workload, not a guess.
DEFAULT_GPU = "T4"


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=3600,
    scaledown_window=120,  # tear the container down 120s after the last call
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_smoke(steps: int, hf_repo: str, checkpoint_every: int, resume: bool, run_id: str) -> dict:
    from gnsm.training.checkpointing import attach_to_run
    from gnsm.training.smoke import SmokeConfig
    from gnsm.training.smoke import run as run_smoke

    checkpoint_cb, resume_state, _manager = attach_to_run(
        hf_repo, checkpoint_every, resume, run_id=run_id
    )
    config = SmokeConfig(steps=steps, device="cuda")
    return run_smoke(config, checkpoint_cb=checkpoint_cb, resume_state=resume_state)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=3600,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_evolvtrip(
    epochs: int,
    batch_size: int,
    patience: int,
    hf_repo: str,
    checkpoint_every: int,
    resume: bool,
    run_id: str,
) -> dict:
    from pathlib import Path

    from gnsm.training.checkpointing import attach_to_run
    from gnsm.training.evolvtrip_adapter import load_examples
    from gnsm.training.plotting import plot_loss_curves
    from gnsm.training.train_state import TrainStateConfig
    from gnsm.training.train_state import run as run_train_state

    examples = load_examples(Path("/root/data/evolvtrip_data/all_books_current.json"))
    checkpoint_cb, resume_state, manager = attach_to_run(
        hf_repo, checkpoint_every, resume, run_id=run_id
    )
    config = TrainStateConfig(
        epochs=epochs, batch_size=batch_size, patience=patience, device="cuda"
    )

    def best_checkpoint_cb(
        step: int, val_loss: float, model_state: dict, optimizer_state: dict
    ) -> None:
        manager.push_best(step, val_loss, {**model_state, "optimizer": optimizer_state})

    def plot_cb(
        train_steps: list[int],
        train_losses: list[float],
        val_epochs: list[int],
        val_losses: list[float],
        best_step: int | None,
        best_val_loss: float | None,
    ) -> None:
        png_path, pdf_path = plot_loss_curves(
            train_steps,
            train_losses,
            Path("/tmp/plots"),
            stem=run_id,
            title="GNSM state encoder — EvolvTrip train/val loss",
            val_steps=val_epochs,
            val_losses=val_losses,
            best_step=best_step,
            best_val_loss=best_val_loss,
        )
        manager.push_artifact(png_path, f"plots/{run_id}/loss.png")
        manager.push_artifact(pdf_path, f"plots/{run_id}/loss.pdf")

    return run_train_state(
        config,
        examples,
        checkpoint_cb=checkpoint_cb,
        best_checkpoint_cb=best_checkpoint_cb,
        resume_state=resume_state,
        plot_cb=plot_cb,
    )


@app.function(image=image, timeout=60)
def health_check(run_id: str, hf_repo: str, stale_after_seconds: float = 1800.0) -> dict:
    """CPU-only, near-zero-cost check of whether a run is still progressing."""

    import time as _time

    from gnsm.training.checkpointing import read_heartbeat

    heartbeat = read_heartbeat(hf_repo, run_id)
    if heartbeat is None:
        return {"run_id": run_id, "status": "no_heartbeat_yet"}
    age = _time.time() - heartbeat["timestamp"]
    status = "healthy" if age < stale_after_seconds else "stalled"
    return {**heartbeat, "age_seconds": round(age, 1), "status": status}


@app.local_entrypoint()
def main(
    steps: int = 60,
    gpu: str = DEFAULT_GPU,
    hf_repo: str = "",
    checkpoint_every: int = 50,
    resume: bool = False,
    run_id: str = "primary",
) -> None:
    if not hf_repo:
        raise SystemExit("--hf-repo is required, e.g. --hf-repo GOVINDFROM/DNG-GNSM")

    fn = train_smoke.with_options(gpu=gpu)
    started = time.time()
    result = fn.remote(
        steps=steps,
        hf_repo=hf_repo,
        checkpoint_every=checkpoint_every,
        resume=resume,
        run_id=run_id,
    )
    print(f"gpu={gpu}  wall_time_s={time.time() - started:.1f}")
    print(result)


@app.local_entrypoint()
def evolvtrip(
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 10,
    gpu: str = DEFAULT_GPU,
    hf_repo: str = "",
    checkpoint_every: int = 50,
    resume: bool = False,
    run_id: str = "evolvtrip-primary",
) -> None:
    """Real training on EvolvTrip, not a synthetic batch, with early stopping
    (stops after `patience` epochs with no val-loss improvement) and the
    best-so-far checkpoint pushed to `checkpoints/best` in the HF repo every
    time val loss improves -- not just on the periodic --checkpoint-every
    cadence. Requires data/evolvtrip_data/all_books_current.json to exist
    locally first (`git clone
    https://huggingface.co/datasets/yangbh217/EvolvTrip data/evolvtrip_data`
    from the repo root) -- it's synced into the container at build time.
    """

    if not hf_repo:
        raise SystemExit("--hf-repo is required, e.g. --hf-repo GOVINDFROM/DNG-GNSM")

    fn = train_evolvtrip.with_options(gpu=gpu)
    started = time.time()
    result = fn.remote(
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        hf_repo=hf_repo,
        checkpoint_every=checkpoint_every,
        resume=resume,
        run_id=run_id,
    )
    print(f"gpu={gpu}  wall_time_s={time.time() - started:.1f}")
    print(result)
