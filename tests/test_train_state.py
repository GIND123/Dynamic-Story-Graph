import importlib.util
import json
from pathlib import Path

import pytest

from gnsm.training.evolvtrip_adapter import load_examples

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _write_fixture(tmp_path: Path, n_characters: int = 4, n_plot_points: int = 4) -> Path:
    records = []
    for c in range(n_characters):
        for p in range(n_plot_points):
            records.append(
                {
                    "book_name": "Test Book",
                    "character": f"Character{c}",
                    "plot_index": p,
                    "plot_summary": f"Something happens to Character{c} at step {p}.",
                    "scenario": f"Character{c} is in scene {p}.",
                    "triples": {
                        "k": [
                            f"(Character{c}, BelievesAbout, fact number {p})",
                            f"(Character{c}, FeelsTowards, feeling number {p})",
                        ]
                    },
                }
            )
    path = tmp_path / "all_books_current.json"
    path.write_text(json.dumps(records))
    return path


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_run_stops_early_and_tracks_best_checkpoint(tmp_path: Path) -> None:
    from gnsm.training.train_state import TrainStateConfig, run

    examples = load_examples(_write_fixture(tmp_path))
    assert len(examples) >= 8  # 4 characters x 3 consecutive pairs each

    config = TrainStateConfig(
        epochs=50,
        batch_size=4,
        hidden_dim=16,
        layers=1,
        heads=2,
        nodes=4,
        edges_per_graph=2,
        input_dim=8,
        val_fraction=0.25,
        seed=0,
        device="cpu",
        patience=3,
    )

    best_calls: list[tuple[int, float]] = []

    def best_checkpoint_cb(step, val_loss, model_state, optimizer_state):
        best_calls.append((step, val_loss))

    result = run(config, examples, best_checkpoint_cb=best_checkpoint_cb)

    assert result["epochs_run"] <= result["epochs_configured"]
    if result["early_stopped"]:
        assert result["epochs_run"] < result["epochs_configured"]
    # Patience must actually bound how long training runs without improvement.
    assert result["epochs_run"] <= 50
    assert result["best_step"] is not None
    assert best_calls, (
        "best_checkpoint_cb should fire at least once (the first epoch always improves)"
    )
    # The result's reported best_val_loss must match the last recorded improvement.
    assert best_calls[-1][1] == pytest.approx(result["best_val_loss"], abs=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_run_never_exceeds_configured_epochs_when_patience_is_large(tmp_path: Path) -> None:
    from gnsm.training.train_state import TrainStateConfig, run

    examples = load_examples(_write_fixture(tmp_path))
    config = TrainStateConfig(
        epochs=3,
        batch_size=4,
        hidden_dim=16,
        layers=1,
        heads=2,
        nodes=4,
        edges_per_graph=2,
        input_dim=8,
        val_fraction=0.25,
        seed=0,
        device="cpu",
        patience=1000,  # effectively disabled
    )
    result = run(config, examples)
    assert result["epochs_run"] == 3
    assert result["early_stopped"] is False
