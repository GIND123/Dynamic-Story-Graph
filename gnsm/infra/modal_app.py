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

    # real training on PDNC (28 novels, ~36K quote pairs -- much more data
    # than EvolvTrip, converges without overfitting). Needs data/pdnc/data/
    # downloaded locally first -- see gnsm/infra's pdnc() docstring.
    modal run gnsm/infra/modal_app.py::pdnc --epochs 30 --batch-size 64 \
        --patience 5 --gpu t4 --hf-repo GOVINDFROM/DNG-GNSM --checkpoint-every 500

The default entrypoint (`main`) launches `gnsm.training.smoke`, a GPU wiring
check on a synthetic batch, not a research result — see that module's
docstring. `evolvtrip` and `pdnc` launch `gnsm.training.train_state` (the
shared, dataset-agnostic training engine) against their own adapters. All
three share the same `attach_to_run` checkpoint hook from
`gnsm.training.checkpointing`, namespaced per run_id so different training
lines can share one HF repo without clobbering each other's
latest/best checkpoints.
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
    image = image.add_local_dir(
        str(_repo_root / "data" / "pdnc" / "data"), remote_path="/root/data/pdnc"
    )
    # pdnc_adapter.py reuses ingestion.pdnc.PDNCLoader (manuscript-memory-engine
    # isn't part of the gnsm package, so it needs its own sync); landed at a
    # fixed, known path rather than re-derived from __file__ inside the
    # container, same lesson as the gnsm sync above.
    image = image.add_local_dir(
        str(_repo_root / "manuscript-memory-engine" / "ingestion"),
        remote_path="/root/ingestion_pkg/ingestion",
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

    checkpoint_cb, _best_checkpoint_cb, resume_state, _manager = attach_to_run(
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
    from gnsm.training.evolvtrip_adapter import collate_batch, load_examples
    from gnsm.training.plotting import plot_loss_curves
    from gnsm.training.train_state import TrainStateConfig
    from gnsm.training.train_state import run as run_train_state

    examples = load_examples(Path("/root/data/evolvtrip_data/all_books_current.json"))
    checkpoint_cb, best_checkpoint_cb, resume_state, manager = attach_to_run(
        hf_repo, checkpoint_every, resume, run_id=run_id
    )
    config = TrainStateConfig(
        epochs=epochs, batch_size=batch_size, patience=patience, device="cuda"
    )

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
        collate_batch,
        checkpoint_cb=checkpoint_cb,
        best_checkpoint_cb=best_checkpoint_cb,
        resume_state=resume_state,
        plot_cb=plot_cb,
    )


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=7200,  # the whole seeds x conditions matrix runs in one container
    scaledown_window=120,
    secrets=[modal.Secret.from_name("hf-token")],
)
def adapter_experiment(
    seeds: str,
    conditions: str,
    epochs: int,
    batch_size: int,
    patience: int,
    lr: float,
    hf_repo: str,
    encoder_run_id: str,
    run_id: str,
    limit: int | None = None,
) -> dict:
    """Train one StatePrefixAdapter per (state condition, seed) and report the
    aggregate with CIs + a paired sign test.

    Deliberately one container for the whole matrix: the frozen LM and encoder
    load once instead of once per cell, which is the dominant cost.
    """

    import json
    from pathlib import Path

    from gnsm.eval.adapter_experiment import run_matrix
    from gnsm.training.checkpointing import (
        CheckpointConfig,
        CheckpointManager,
        resume_best_from_hub,
    )
    from gnsm.training.evolvtrip_adapter import load_examples

    examples = load_examples(Path("/root/data/evolvtrip_data/all_books_current.json"))
    if limit:
        examples = examples[:limit]

    encoder_state = resume_best_from_hub(
        hf_repo, encoder_run_id, Path("/tmp/encoder") / encoder_run_id
    )
    if encoder_state is None:
        raise SystemExit(
            f"No best checkpoint for encoder run_id {encoder_run_id!r} in {hf_repo!r}."
        )

    summary = run_matrix(
        examples,
        dict(encoder_state["encoder"]),
        seeds=[int(s) for s in seeds.split(",") if s.strip()],
        conditions=tuple(c.strip() for c in conditions.split(",") if c.strip()),
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        device="cuda",
    )
    summary["n_examples"] = len(examples)
    summary["encoder_run_id"] = encoder_run_id
    summary["encoder_step"] = encoder_state["step"]

    # Persist the raw summary next to the run's other artifacts so the numbers
    # in any write-up are traceable to a stored file, not just stdout.
    manager = CheckpointManager(
        CheckpointConfig(hf_repo_id=hf_repo, local_dir=Path("/tmp/experiment")), run_id=run_id
    )
    out_path = Path("/tmp/experiment") / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    manager.push_artifact(out_path, f"experiments/{run_id}/summary.json")
    return summary


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=3600,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("hf-token")],
)
def train_pdnc(
    epochs: int,
    batch_size: int,
    patience: int,
    hf_repo: str,
    checkpoint_every: int,
    resume: bool,
    run_id: str,
) -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/ingestion_pkg")

    from gnsm.training.checkpointing import attach_to_run
    from gnsm.training.pdnc_adapter import (
        ATTRIBUTE_CLASSES,
        DELTA_CLASSES,
        EDGE_TYPES,
        EMOTION_CLASSES,
        collate_batch,
        load_examples,
    )
    from gnsm.training.plotting import plot_loss_curves
    from gnsm.training.train_state import TrainStateConfig
    from gnsm.training.train_state import run as run_train_state

    examples = load_examples(Path("/root/data/pdnc"))
    checkpoint_cb, best_checkpoint_cb, resume_state, manager = attach_to_run(
        hf_repo, checkpoint_every, resume, run_id=run_id
    )
    config = TrainStateConfig(
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        nodes=4,
        edges_per_graph=3,
        device="cuda",
        edge_types=EDGE_TYPES,
        attribute_classes=ATTRIBUTE_CLASSES,
        emotion_classes=EMOTION_CLASSES,
        delta_classes=DELTA_CLASSES,
    )

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
            title="GNSM state encoder — PDNC train/val loss",
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
        collate_batch,
        checkpoint_cb=checkpoint_cb,
        best_checkpoint_cb=best_checkpoint_cb,
        resume_state=resume_state,
        plot_cb=plot_cb,
    )


@app.function(image=image, timeout=60, secrets=[modal.Secret.from_name("hf-token")])
def health_check(run_id: str, hf_repo: str, stale_after_seconds: float = 1800.0) -> dict:
    """CPU-only, near-zero-cost check of whether a run is still progressing."""

    import time as _time

    from gnsm.training.checkpointing import read_heartbeat

    heartbeat = read_heartbeat(hf_repo, run_id)
    if heartbeat is None:
        result = {"run_id": run_id, "status": "no_heartbeat_yet"}
        print(result)
        return result
    age = _time.time() - heartbeat["timestamp"]
    status = "healthy" if age < stale_after_seconds else "stalled"
    result = {**heartbeat, "age_seconds": round(age, 1), "status": status}
    print(result)
    return result


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


@app.local_entrypoint()
def pdnc(
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 5,
    gpu: str = DEFAULT_GPU,
    hf_repo: str = "",
    checkpoint_every: int = 500,
    resume: bool = False,
    run_id: str = "pdnc-primary",
) -> None:
    """Real training on PDNC (28 novels, ~36K consecutive same-speaker quote
    pairs -- see gnsm/training/pdnc_adapter.py's docstring). Requires
    data/pdnc/data/ to exist locally first (`git clone --depth 1
    https://github.com/Priya22/project-dialogism-novel-corpus.git data/pdnc`
    from the repo root) -- it's synced into the container at build time,
    along with manuscript-memory-engine/ingestion (reused for PDNC parsing).
    """

    if not hf_repo:
        raise SystemExit("--hf-repo is required, e.g. --hf-repo GOVINDFROM/DNG-GNSM")

    fn = train_pdnc.with_options(gpu=gpu)
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


@app.local_entrypoint()
def experiment(
    seeds: str = "0,1,2,3,4",
    conditions: str = "real,shuffled,zero",
    epochs: int = 30,
    batch_size: int = 8,
    patience: int = 5,
    lr: float = 1e-4,
    gpu: str = DEFAULT_GPU,
    hf_repo: str = "",
    encoder_run_id: str = "evolvtrip-v2-earlystop",
    run_id: str = "adapter-experiment-v1",
    limit: int = 0,
) -> None:
    """Stage C experiment: does conditioning on the learned narrative state
    beat conditioning on a state with no example-specific information?

    Trains one adapter per (condition, seed) against a frozen encoder
    checkpoint + frozen LM, then reports mean +/- bootstrap CI and an exact
    paired sign test. See gnsm/eval/adapter_experiment.py for the design.
    """

    if not hf_repo:
        raise SystemExit("--hf-repo is required, e.g. --hf-repo GOVINDFROM/DNG-GNSM")

    fn = adapter_experiment.with_options(gpu=gpu)
    started = time.time()
    result = fn.remote(
        seeds=seeds,
        conditions=conditions,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        hf_repo=hf_repo,
        encoder_run_id=encoder_run_id,
        run_id=run_id,
        limit=limit or None,
    )
    print(f"gpu={gpu}  wall_time_s={time.time() - started:.1f}")
    for condition, stats in result["per_condition"].items():
        print(
            f"{condition:>8}: {stats['mean']}  95% CI [{stats['ci_low']}, {stats['ci_high']}]  "
            f"n={stats['n_seeds']} seeds"
        )
    for name, comparison in result.get("comparisons", {}).items():
        print(f"{name}: mean_delta={comparison['mean_delta']}  p={comparison['p_value']}")
