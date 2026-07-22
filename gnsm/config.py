"""YAML configuration loading with strict, small runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gnsm.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    seed: int = 42
    max_regenerations: int = 2
    fail_on_warning: bool = False


@dataclass(frozen=True, slots=True)
class StateConfig:
    dimension: int = 128
    node_dimension: int = 128
    encoder: str = "hashing"
    transition: str = "rule_based"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    backend: str = "template"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    conditioning: str = "soft_prompt"
    max_new_tokens: int = 1024


@dataclass(frozen=True, slots=True)
class GNSMConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    state: StateConfig = field(default_factory=StateConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def load_config(path: str | Path) -> GNSMConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")
    return _from_mapping(raw)


def _from_mapping(raw: dict[str, Any]) -> GNSMConfig:
    allowed = {"runtime", "state", "generation"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigurationError(f"unknown configuration sections: {sorted(unknown)}")
    try:
        runtime = RuntimeConfig(**raw.get("runtime", {}))
        state = StateConfig(**raw.get("state", {}))
        generation = GenerationConfig(**raw.get("generation", {}))
    except TypeError as exc:
        raise ConfigurationError(f"invalid configuration field: {exc}") from exc
    if state.dimension <= 0 or state.node_dimension <= 0:
        raise ConfigurationError("state dimensions must be positive")
    if runtime.max_regenerations < 0:
        raise ConfigurationError("max_regenerations cannot be negative")
    return GNSMConfig(runtime=runtime, state=state, generation=generation)
