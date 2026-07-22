import importlib.util

import pytest

from gnsm.training import smoke

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_plan_shapes_are_consistent() -> None:
    config = smoke.SmokeConfig(batch_size=4, nodes=5, edges_per_graph=7, steps=3)
    plan = smoke.plan(config)
    assert plan["graphs"] == 4
    assert plan["node_rows"] == 4 * 5
    assert plan["edge_rows"] == 4 * 7
    assert plan["delta_rows"] == 4
    assert plan["optimizer_steps"] == 3


def test_run_raises_clean_error_without_torch() -> None:
    if HAS_TORCH:
        pytest.skip("torch installed; clean-error path not exercised")
    from gnsm.exceptions import OptionalDependencyError

    with pytest.raises(OptionalDependencyError):
        smoke.run(smoke.SmokeConfig(steps=1))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_run_trains_and_reduces_loss_on_cpu() -> None:
    # Small, fast, deterministic; forced onto CPU so it runs in CI without a GPU.
    config = smoke.SmokeConfig(steps=40, batch_size=4, hidden_dim=64, device="cpu", seed=0)
    result = smoke.run(config)
    assert result["device"] == "cpu"
    assert result["loss_decreased"] is True
    assert result["trainable_parameters"] > 0
    assert result["nn_module_check"] is True
