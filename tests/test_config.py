from pathlib import Path

from gnsm.config import load_config


def test_default_config_loads() -> None:
    config = load_config(Path("gnsm/configs/default.yaml"))
    assert config.state.dimension == 128
    assert config.generation.backend == "template"
